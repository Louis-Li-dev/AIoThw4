# RAG 知識庫系統 (含相似度分數顯示)

簡介
- 這是一個使用 Streamlit 建立的簡易 RAG（Retrieval-Augmented Generation）展示範例，會把 `source_data` 資料夾中的文本或 PDF 建立向量索引（FAISS），並顯示檢索結果和相似度/距離分數。

主要檔案
- `main.py`：Streamlit 應用程式主檔案。
- `requirements.txt`：建議的 Python 套件及版本。
- `source_data/`：原始文件放置目錄（上傳或由介面生成樣本）。
- `faiss_db_output/`：向量資料庫預設儲存路徑（建立索引後自動產生）。

快速開始（Windows / PowerShell）

1. 建議建立虛擬環境（可用 `venv` 或 Conda）：

```powershell
# 使用 venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 或使用 conda
conda create -n rag-env python=3.10 -y
conda activate rag-env
```

2. 安裝相依套件：

```powershell
pip install -r requirements.txt
```

3. 啟動應用程式：

```powershell
streamlit run main.py
```

功能說明
- 左側「管理與瀏覽文件」可用來上傳檔案、生成範例文件、以及建立/重建向量索引（FAISS）。
- 右側「AI 助手對話」可輸入問題並取得檢索片段與相似度分數顯示。

遇到常見問題（Streamlit 與 PyTorch 衝突）

- 問題描述：部份使用者會在啟動 Streamlit 時遇到類似錯誤：
  "RuntimeError: Tried to instantiate class '__path__._path'..."，或在文件監視（file watcher）階段因 PyTorch 的模組屬性存取而發生例外。

- 解法 A（推薦，較簡單）：停用 Streamlit 的檔案監視器（watcher）。在專案根目錄建立 `.streamlit/config.toml` 並填入：

```toml
# 停用內建檔案監視器以避免與某些大型套件衝突
server.fileWatcherType = "none"
```

  或使用已包含此專案的 `.streamlit/config.toml`（專案已提供）。停用 watcher 後再次執行 `streamlit run main.py`。

- 解法 B：改為安裝 CPU-only 的 PyTorch 版本（避免 CUDA 相關自訂模組注入），例如：

```powershell
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --upgrade
```

  或使用 Conda 安裝 CPU 版本，依系統與需求選擇適合的版本。

- 另外建議：若仍然出錯，試著更新 `streamlit` 至較新版本或暫時在啟動時使用 `--server.fileWatcherType none`（或將對應設定寫入 `.streamlit/config.toml`）。

開發與測試提示
- 若你在本機測試大量模型（使用 GPU），請注意套件版本的相容性（torch + torchvision + cuda）。
- 若希望減少下載體積與相容性問題，可優先使用 CPU-only 版本來驗證 Streamlit 應用程式邏輯。

其他資訊
- 範例文件可由介面按鈕 `✨ 生成測試用文件` 建立，會放到 `source_data/`。
- 建立索引後會產生 `faiss_db_output` 資料夾，請勿手動改動索引檔以免損壞資料庫。

如果你要我：
- 我可以幫你進一步更新 `requirements.txt`（選定 CPU-only torch 的版本），或
- 幫你執行並驗證 `streamlit run main.py`（若你允許我在你的環境跑指令）。

