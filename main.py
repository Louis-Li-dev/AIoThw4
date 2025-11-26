import streamlit as st
import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --- 設定頁面 ---
st.set_page_config(page_title="RAG 知識庫系統 (含分數顯示)", layout="wide", page_icon="🔢")
st.title("🔢 RAG 系統 (顯示相似度分數)")

# --- 核心路徑設定 ---
DB_PATH = "faiss_db_output"     # 向量資料庫儲存位置
DOCS_DIR = "source_data"        # 原始文件儲存位置

# 確保資料夾存在
os.makedirs(DOCS_DIR, exist_ok=True)

# --- 初始化 Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 載入模型 (快取) ---
@st.cache_resource
def load_embedding_model():
    # 使用 HuggingFace 的 Embedding 模型
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# --- 功能函式 ---

def save_uploaded_file(uploaded_file):
    """將上傳的檔案儲存到 DOCS_DIR"""
    file_path = os.path.join(DOCS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def generate_sample_data():
    """生成範例文件並存入 DOCS_DIR"""
    samples = {
        "AI研究社_介紹.txt": "AI研究社成立於2023年，社長是王小明。社課時間為每週五晚上7點，地點在資訊大樓305教室。我們的宗旨在於推廣生成式AI技術。",
        "登山社_活動規章.txt": "登山社安全規章：1. 參加百岳行程需具備基礎體能證明。 2. 裝備檢查未通過者不得上山。 3. 遇到颱風警報一律取消行程。費用部分：社員由社費補助20%，非社員全額自費。",
        "圖書館_借閱規則.txt": "圖書館開放時間為週一至週五 08:00-22:00。大學部學生可借閱10本書，借期30天。逾期罰款每日每本5元。遺失圖書需賠償原價之1.5倍。"
    }
    
    for filename, content in samples.items():
        path = os.path.join(DOCS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    
    return list(samples.keys())

def build_vector_db():
    """讀取 DOCS_DIR 中的所有檔案並建立向量庫"""
    embedding_model = load_embedding_model()
    documents = []
    
    # 掃描 DOCS_DIR 資料夾
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(('.txt', '.pdf'))]
    
    if not files:
        return False, "資料夾中沒有文件，請先上傳或生成資料。"

    progress_bar = st.progress(0, text="正在讀取檔案...")
    
    for i, file in enumerate(files):
        file_path = os.path.join(DOCS_DIR, file)
        try:
            if file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                continue
            
            # 載入並標記來源
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file # 確保 metadata 有檔名
            documents.extend(docs)
            
        except Exception as e:
            st.error(f"讀取 {file} 失敗: {e}")
        
        progress_bar.progress((i + 1) / len(files), text=f"已讀取: {file}")

    if not documents:
        return False, "沒有有效內容可建立索引。"

    # 切分與向量化
    progress_bar.progress(0.8, text="正在切分文本與建立索引...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    vectorstore.save_local(DB_PATH)
    
    progress_bar.progress(1.0, text="完成！")
    return True, f"成功建立資料庫！共包含 {len(files)} 份文件，切分為 {len(split_docs)} 個片段。"

def query_rag(query_text):
    """查詢向量資料庫並回傳分數"""
    if not os.path.exists(DB_PATH):
        return "⚠️ 請先建立資料庫 (請至左側 '建立知識庫' 分頁)"
    
    embedding_model = load_embedding_model()
    vectorstore = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    
    # 修改重點：使用 similarity_search_with_score
    # k=10 設大一點，以便盡可能顯示所有相關文件 (針對小資料集)
    results_with_score = vectorstore.similarity_search_with_score(query_text, k=10)
    
    response = f"🔍 **查詢內容**：{query_text}\n\n"
    response += "📊 **檢索結果 (按距離分數排序，越低代表越相似)**：\n\n"
    
    for i, (doc, score) in enumerate(results_with_score):
        # 嘗試從 metadata 獲取檔名
        source = doc.metadata.get('source', '未知來源')
        source_name = os.path.basename(source)
        
        # 格式化輸出
        response += f"**#{i+1} 來源**: `{source_name}`\n"
        response += f"🔴 **距離分數 (Distance Score)**: `{score:.5f}`\n" 
        response += f"📄 **內容片段**: {doc.page_content}\n\n---\n"
        
    return response

# --- 介面佈局 ---

tab1, tab2 = st.tabs(["📂 管理與瀏覽文件", "💬 AI 助手對話"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. 新增資料")
        st.info("上傳或生成檔案後，檔案會存入 `source_data` 資料夾。")
        
        # 生成範例按鈕
        if st.button("✨ 生成測試用文件"):
            files = generate_sample_data()
            st.success(f"已生成 {len(files)} 份文件。")
            st.rerun() 

        # 上傳按鈕
        uploaded_files = st.file_uploader("上傳新文件 (.txt, .pdf)", accept_multiple_files=True)
        if uploaded_files:
            for u_file in uploaded_files:
                save_uploaded_file(u_file)
            st.success(f"已儲存 {len(uploaded_files)} 份新文件。")
            st.rerun()

        st.divider()
        
        st.subheader("2. 建立/更新 資料庫")
        if st.button("🚀 重建 RAG 索引"):
            with st.spinner("正在處理..."):
                success, msg = build_vector_db()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    with col2:
        st.subheader("📚 檢視目前文件")
        st.caption(f"資料夾路徑: {DOCS_DIR}")
        
        existing_files = os.listdir(DOCS_DIR)
        
        if not existing_files:
            st.write("目前沒有任何文件。")
        else:
            for f in existing_files:
                file_path = os.path.join(DOCS_DIR, f)
                with st.expander(f"📄 {f}"):
                    if st.button("刪除", key=f"del_{f}"):
                        os.remove(file_path)
                        st.rerun()
                    
                    if f.endswith(".txt"):
                        with open(file_path, "r", encoding="utf-8") as _f:
                            st.text(_f.read())
                    else:
                        st.write("PDF 檔案僅支援預覽檔名與路徑。")

with tab2:
    st.header("與你的文件對話")
    
    if not os.path.exists(DB_PATH):
        st.warning("⚠️ 尚未偵測到向量資料庫，請先至「管理與瀏覽文件」分頁建立索引。")

    # 顯示歷史訊息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 處理輸入
    if prompt := st.chat_input("請輸入問題..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("搜尋並計算分數中..."):
                response = query_rag(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})