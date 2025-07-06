import streamlit as st
import os
from datetime import datetime
import uuid
import time
import requests
import re
import numpy as np
from typing import Dict, List, Any
from FlagEmbedding import BGEM3FlagModel
import chromadb
from chromadb.utils import embedding_functions
import random
import string
import hashlib
import hmac
import base64
import urllib.parse

# ==================== 蓝心API签名工具 ====================
def gen_nonce(length=8):
    chars = string.ascii_lowercase + string.digits
    return ''.join([random.choice(chars) for _ in range(length)])

def gen_canonical_query_string(params):
    if params:
        escape_uri = urllib.parse.quote
        raw = []
        for k in sorted(params.keys()):
            tmp_tuple = (escape_uri(k), escape_uri(str(params[k])))
            raw.append(tmp_tuple)
        s = "&".join("=".join(kv) for kv in raw)
        return s
    else:
        return ''

def gen_signature(app_secret, signing_string):
    bytes_secret = app_secret.encode('utf-8')
    hash_obj = hmac.new(bytes_secret, signing_string, hashlib.sha256)
    bytes_sig = base64.b64encode(hash_obj.digest())
    signature = str(bytes_sig, encoding='utf-8')
    return signature

def gen_sign_headers(app_id, app_key, method, uri, query):
    method = str(method).upper()
    uri = uri
    timestamp = str(int(time.time()))
    app_id = app_id
    app_key = app_key
    nonce = gen_nonce()
    canonical_query_string = gen_canonical_query_string(query)
    signed_headers_string = 'x-ai-gateway-app-id:{}\nx-ai-gateway-timestamp:{}\n' \
                            'x-ai-gateway-nonce:{}'.format(app_id, timestamp, nonce)
    signing_string = '{}\n{}\n{}\n{}\n{}\n{}'.format(method,
                                                     uri,
                                                     canonical_query_string,
                                                     app_id,
                                                     timestamp,
                                                     signed_headers_string)
    signing_string = signing_string.encode('utf-8')
    signature = gen_signature(app_key, signing_string)
    return {
        'X-AI-GATEWAY-APP-ID': app_id,
        'X-AI-GATEWAY-TIMESTAMP': timestamp,
        'X-AI-GATEWAY-NONCE': nonce,
        'X-AI-GATEWAY-SIGNED-HEADERS': "x-ai-gateway-app-id;x-ai-gateway-timestamp;x-ai-gateway-nonce",
        'X-AI-GATEWAY-SIGNATURE': signature
    }

# ==================== FactChecker类 ====================
class FactChecker:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.APP_ID = "2025468964"  # 替换为你的 APP_ID
        self.APP_KEY = "TbdyfqgdVNiNFhUA"  # 替换为你的 APP_KEY
        self.URI = "/vivogpt/completions"
        self.DOMAIN = "api-ai.vivo.com.cn"
        self.METHOD = "POST"

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 初始化嵌入模型
        try:
            self.embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        except Exception as e:
            st.error(f"加载BGE-M3模型错误: {str(e)}")
            self.embedding_model = None

        # 初始化Chroma本地知识库
        try:
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(
                name="my-knowledge-base",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
            )
        except Exception as e:
            st.error(f"加载Chroma错误: {str(e)}")
            self.collection = None

    def _call_lanxin_api(self, prompt: str) -> str:
        request_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        params = {"requestId": request_id}
        data = {
            "prompt": prompt,
            "model": self.model,
            "sessionId": session_id,
            "extra": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens
            }
        }
        headers = gen_sign_headers(self.APP_ID, self.APP_KEY, self.METHOD, self.URI, params)
        headers["Content-Type"] = "application/json"
        url = f"https://{self.DOMAIN}{self.URI}"
        try:
            response = requests.post(url, json=data, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            res_json = response.json()
            if res_json.get("code") == 0 and "data" in res_json:
                return res_json["data"].get("content", "")
            else:
                st.error(f"蓝心 API 返回错误: {res_json}")
                return ""
        except Exception as e:
            st.error(f"调用蓝心 API 失败: {str(e)}")
            return ""

    def extract_claim(self, text: str) -> str:
        system_prompt = """
            你是一个精确的声明提取助手。分析提供的新闻并总结其中心思想。
            将中心思想格式化为一个值得核查的陈述，即可以独立验证的声明并以中文回答。
            输出格式:
            claim: <claim>
        """
        prompt = f"{system_prompt}\n\n用户输入: {text}"
        claims_text = self._call_lanxin_api(prompt)
        claims = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)', claims_text, re.DOTALL)
        claims = [claim.strip() for claim in claims if claim.strip()]
        if not claims and claims_text.strip():
            claims = [line.strip() for line in claims_text.strip().split('\n') if line.strip()]
        return claims[0] if claims else text

    def extract_keyinformation(self, text: str) -> str:
        system_prompt = """
       你是一个精确的声明提取助手。请你对用户输入执行以下多层级分析：
       1.识别文本中涉及的所有实体（人物/组织/地点/时间）及其类型
       2.确定核心事件类型（政治/经济/社会/科技等类别）
       3.提取关键特征（来源可靠性/证据强度/时效性）
       4.总结中心思想并转化为可验证的声明

       输出格式要求：
       claim: <可验证的完整陈述>
       entities: <实体类型1:实体名称1, 实体类型2:实体名称2...>
       event_type: <事件主类别>
       key_features: <特征1:值1, 特征2:值2...>

       注意：请严格按照格式输出，不要添加额外解释
       """
        prompt = f"{system_prompt}\n\n用户输入: {text}"
        claims_text = self._call_lanxin_api(prompt)
        claim_match = re.search(r'claim:\s*(.*?)(?=\n\S+:|\Z)', claims_text, re.DOTALL)
        entities_match = re.search(r'entities:\s*(.*?)(?=\n\S+:|\Z)', claims_text, re.DOTALL)
        event_type_match = re.search(r'event_type:\s*(.*?)(?=\n\S+:|\Z)', claims_text, re.DOTALL)
        key_features_match = re.search(r'key_features:\s*(.*?)(?=\n\S+:|\Z)', claims_text, re.DOTALL)
        claim = claim_match.group(1).strip() if claim_match else "未提取到有效声明"
        entities = entities_match.group(1).strip() if entities_match else "无实体信息"
        event_type = event_type_match.group(1).strip() if event_type_match else "未知事件类型"
        key_features = key_features_match.group(1).strip() if key_features_match else "无特征信息"
        formatted_output = (
            f"claim: {claim}\n"
            f"entities: {entities}\n"
            f"event_type: {event_type}\n"
            f"key_features: {key_features}"
        )
        return formatted_output

    def search_local_knowledge(self, claim: str, top_k: int = 5) -> List[Dict[str, str]]:
        if not self.collection:
            return []
        try:
            results = self.collection.query(
                query_texts=[claim],
                n_results=top_k
            )
            evidence_docs = []
            for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
                evidence_docs.append({
                    'title': metadata.get('title', ''),
                    'url': metadata.get('url', 'Local Knowledge Base'),
                    'snippet': text
                })
            return evidence_docs
        except Exception as e:
            st.error(f"搜索本地知识库错误: {str(e)}")
            return []

    def search_evidence(self, claim: str, num_results: int = 5) -> List[Dict[str, str]]:
        try:
            from duckduckgo_search import DDGS
            ddgs = DDGS(timeout=60)
            results = list(ddgs.text(claim, max_results=num_results))
            external_evidence = []
            for result in results:
                external_evidence.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
            local_evidence = self.search_local_knowledge(claim, top_k=num_results)
            combined_evidence = external_evidence + local_evidence
            return combined_evidence
        except Exception as e:
            st.error(f"搜索证据错误: {str(e)}")
            return []

    def get_evidence_chunks(self, evidence_docs: List[Dict[str, str]], claim: str, chunk_size: int = 200,
                            chunk_overlap: int = 50, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.embedding_model:
            return [{
                'text': "证据排序不可用 - 无法加载BGE-M3模型",
                'source': "系统",
                'similarity': 0.0
            }]
        if not evidence_docs:
            return [{
                'text': "没有找到相关证据",
                'source': "系统",
                'similarity': 0.0
            }]
        try:
            all_chunks = []
            for doc in evidence_docs:
                all_chunks.append({
                    'text': doc['title'],
                    'source': doc['url'],
                })
                snippet = doc['snippet']
                if len(snippet) <= chunk_size:
                    all_chunks.append({
                        'text': snippet,
                        'source': doc['url'],
                    })
                else:
                    for i in range(0, len(snippet), chunk_size - chunk_overlap):
                        chunk_text = snippet[i:i + chunk_size]
                        if len(chunk_text) >= chunk_size // 2:
                            all_chunks.append({
                                'text': chunk_text,
                                'source': doc['url'],
                            })
            claim_embedding = self.embedding_model.encode(claim)['dense_vecs']
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)['dense_vecs']
            similarities = []
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = np.dot(claim_embedding, chunk_embedding) / (
                    np.linalg.norm(claim_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append(float(similarity))
            for i, similarity in enumerate(similarities):
                all_chunks[i]['similarity'] = similarity
            ranked_chunks = sorted(all_chunks, key=lambda x: x['similarity'], reverse=True)
            return ranked_chunks[:top_k]
        except Exception as e:
            st.error(f"证据排序错误: {str(e)}")
            return [{
                'text': f"证据排序错误: {str(e)}",
                'source': "系统",
                'similarity': 0.0
            }]

    def evaluate_claim(self, keyinformation: str, text: str, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
        system_prompt = """
        以下是您提供的英文内容的中文翻译：

---

### 1. **来源可信度验证（权重：40%）**  
   ✓ 官方来源（政府/学术/专业期刊） → ★★★★★  
   ✓ 主流媒体（法新社/新华社等） → ★★★☆☆  
   ✗ 个人博客/匿名平台 → ★☆☆☆☆  

### 2. **证据质量评估（权重：40%）**  
   │ 直接证据          │ 间接证据  
   ├─────────────────────┼─────────────────────┤  
   │ 实验数据          │ 专家证词  
   │ 统计报告          │ 历史案例  
   │ 现场影像          │ 文献引用  

### 3. **逻辑完整性审查（权重：20%）**  
   - 统计陷阱检测：样本量 <30  
   - 因果谬误识别：A→B 被曲解为 B→A  
   - 时间悖论验证：事件间隔 <2小时  

### 4. **二元判断标准**  
   **TRUE**: 总权重 ≥70% 且无核心矛盾  
   **FALSE**: 存在反证链/多重逻辑漏洞，或多数证据来自个人博客/匿名平台  

---

### **输出规范**  
**VERDICT**: [TRUE|FALSE|PARTIALLY TRUE]  
**EVIDENCE WEIGHT**: [总证据权重百分比]  
**REASONING**:  
1. 来源评级：□官方 □认证媒体 □可疑平台  
2. 关键证据: [E-12] NASA遥感数据 (2024Q3)  
3. 逻辑漏洞: 检测到样本选择偏差 (p=0.12)  
4. 时间有效性: ✓ 最新卫星影像 (2025-03)  

---

### **输入示例**:  
"月球背面存在外星人基地"

### **输出示例**:  
**VERDICT**: FALSE  
**EVIDENCE WEIGHT**: 92%  
**REASONING**:  
1. 来源评级: NASA官方数据 (权重: 50%)  
2. 关键证据: [E-3] 月球车扫描显示无异常结构  
3. 逻辑漏洞: 目击者证词属于轶事谬误  
4. 时间验证: 最新月球任务数据 (2024Q3)  

---

### 说明：
- 保留英文术语（如 VERDICT、TRUE/FALSE）以确保格式统一。
- 证据编号（如 [E-3]）和数据格式（如 2024Q3）保持原样。
- 使用中文标点符号（如 `✓`、`•`）增强可读性。
        """
        evidence_text = "\n\n".join([
            f"证据 {i+1} (相关性: {chunk['similarity']:.2f}):\n{chunk['text']}\n来源: {chunk['source']}"
            for i, chunk in enumerate(evidence_chunks)
        ])
        prompt = f"{system_prompt}\n\n关键信息: {keyinformation}\n\n{text}\n\n证据:\n{evidence_text}"
        result_text = self._call_lanxin_api(prompt)
        verdict_match = re.search(r'VERDICT:\s*(TRUE|FALSE)', result_text, re.IGNORECASE)
        verdict = verdict_match.group(1) if verdict_match else "UNVERIFIABLE"
        reasoning_match = re.search(r'REASONING:\s*(.*)', result_text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else result_text
        return {
            "verdict": verdict,
            "reasoning": reasoning
        }

# ==================== Streamlit前端 ====================
st.set_page_config(
    page_title="AI虚假新闻检测器",
    page_icon="🔍",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.title("AI虚假新闻检测器")
st.markdown("""
本应用程序使用蓝心大模型（vivo BlueLM）验证陈述的准确性。
请在下方输入需要核查的新闻，系统将检索网络证据进行新闻核查，无网络时将只进行本地知识库检索。
""")

with st.sidebar:
    st.header("配置")
    model_option = st.selectbox(
        "选择模型",
        ["vivo-BlueLM-TB-Pro", "vivo-BlueLM-TB"],
        index=0,
        help="使用 vivo 蓝心大模型"
    )
    with st.expander("高级设置"):
        temperature = st.slider("温度", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                               help="较低的值使响应更确定，较高的值使响应更具创造性")
        max_tokens = st.slider("最大响应长度", min_value=100, max_value=8000, value=1000, step=100,
                              help="响应中的最大标记数")
    st.divider()
    st.markdown("### 关于 ###")
    st.markdown("虚假新闻检测器:")
    st.markdown("1. 从新闻中提取核心声明")
    st.markdown("2. 在网络上和本地库搜索证据")
    st.markdown("3. 使用BGE-M3按相关性对证据进行排名")
    st.markdown("4. 基于证据提供结论")
    st.markdown("使用LLM、Streamlit、BGE-M3和RAG开发 ❤️❤️❤️")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("请在下方输入需要核查的新闻...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    assistant_message = st.chat_message("assistant")
    claim_placeholder = assistant_message.empty()
    information_placeholder = assistant_message.empty()
    evidence_placeholder = assistant_message.empty()
    verdict_placeholder = assistant_message.empty()

    fact_checker = FactChecker(model=model_option, temperature=temperature, max_tokens=max_tokens)

    claim_placeholder.markdown("### 🔍 正在提取新闻的核心声明...")
    claim = fact_checker.extract_claim(user_input)
    if "claim:" in claim.lower():
        claim = claim.split("claim:")[-1].strip()
    claim_placeholder.markdown(f"### 🔍 提取新闻的核心声明\n\n{claim}")

    information_placeholder.markdown(f"### 🔍 正在提取新闻的关键信息...")
    information = fact_checker.extract_keyinformation(user_input)
    information_placeholder.markdown(f"### 🔍 提取新闻的关键信息\n\n{information}")

    evidence_placeholder.markdown("### 🌐 正在搜索相关证据...")
    evidence_docs = fact_checker.search_evidence(claim)

    evidence_placeholder.markdown("### 🌐 正在分析证据相关性...")
    evidence_chunks = fact_checker.get_evidence_chunks(evidence_docs, claim)

    evidence_md = "### 🔗 证据来源\n\n"
    for j, chunk in enumerate(evidence_chunks[:-1]):
        evidence_md += f"**[{j+1}]:**\n"
        evidence_md += f"{chunk['text']}\n"
        evidence_md += f"来源: {chunk['source']}\n\n"
    evidence_placeholder.markdown(evidence_md)

    verdict_placeholder.markdown("### ⚖️ 正在评估声明真实性...")
    evaluation = fact_checker.evaluate_claim(information, user_input, evidence_chunks)

    verdict = evaluation["verdict"]
    if verdict.upper() == "TRUE":
        emoji = "✅"
        verdict_cn = "正确"
    elif verdict.upper() == "FALSE":
        emoji = "❌"
        verdict_cn = "错误"
    elif verdict.upper() == "PARTIALLY TRUE":
        emoji = "⚠️"
        verdict_cn = "部分正确"
    else:
        emoji = "❓"
        verdict_cn = "无法验证"

    verdict_md = f"### {emoji} 结论: {verdict_cn}\n\n"
    verdict_md += f"### 推理过程\n\n{evaluation['reasoning']}\n\n"
    verdict_placeholder.markdown(verdict_md)

    full_response = f"""
### 🔍 提取新闻的核心声明

{claim}

---

{information}

---

{evidence_md}

---

{verdict_md}
"""
    st.session_state.messages.append({"role": "assistant", "content": full_response})