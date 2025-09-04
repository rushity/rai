from flask import Flask, render_template, request, session, redirect, url_for
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import time
from typing import Optional

# ------------------ Config ------------------
DATA_DIR = "data"                 # folder with your PDFs/TXT/MD, etc.
PERSIST_DIR = "storage"           # where the index will be saved
CHECK_FILE = os.path.join(PERSIST_DIR, ".last_built")  # timestamp cache

app = Flask(__name__)
app.secret_key = "supersecretkey"  # needed for sessions

# Embedding & LLM (initialize once)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Ollama(model="gemma3:1b")  # make sure you've run: ollama pull gemma3:1b

# Globals (cached)
_index: Optional[VectorStoreIndex] = None
_query_engine = None

# ------------------ Helpers ------------------
def _latest_mtime(folder: str) -> float:
    """Return the newest modification time in a folder (0.0 if empty or missing)."""
    if not os.path.isdir(folder):
        return 0.0
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    files = [p for p in files if os.path.isfile(p)]
    if not files:
        return 0.0
    return max(os.path.getmtime(p) for p in files)

def _read_cached_build_time() -> float:
    if not os.path.exists(CHECK_FILE):
        return 0.0
    try:
        with open(CHECK_FILE, "r") as f:
            return float(f.read().strip())
    except Exception:
        return 0.0

def _write_cached_build_time(ts: float) -> None:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(CHECK_FILE, "w") as f:
        f.write(str(ts))

def _build_and_persist_index() -> VectorStoreIndex:
    print("🔄 Rebuilding index (embedding documents)…")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Persist to disk so next startup is instant
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    # Save the timestamp to avoid unnecessary rebuilds
    _write_cached_build_time(_latest_mtime(DATA_DIR))
    print("✅ Index built & persisted.")
    return index

def _load_index_from_disk() -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    return load_index_from_storage(storage_context)

def get_index_and_engine():
    """Load from cache or disk; rebuild only if /data changed."""
    global _index, _query_engine

    data_mtime = _latest_mtime(DATA_DIR)
    cached_mtime = _read_cached_build_time()

    # Case 1: We already have it in memory and data hasn't changed
    if _index is not None and data_mtime <= cached_mtime:
        return _index, _query_engine

    # Case 2: We have persisted index and it's still fresh → load from disk
    if os.path.isdir(PERSIST_DIR) and os.path.exists(CHECK_FILE) and data_mtime <= cached_mtime:
        print("📦 Loading index from disk cache…")
        _index = _load_index_from_disk()
    else:
        # Case 3: Need to (re)build
        if not os.path.isdir(DATA_DIR) or _latest_mtime(DATA_DIR) == 0.0:
            # No data yet; create an empty index so app stays responsive
            print("⚠️  No files in /data. Create the folder and add docs to enable Q&A.")
            # Build an empty index (optional). LlamaIndex needs docs to build;
            # We'll defer building until files exist.
            _index = None
            _query_engine = None
            return _index, _query_engine
        _index = _build_and_persist_index()

    # Build a lightweight query engine for faster responses
    _query_engine = _index.as_query_engine(response_mode="compact")
    return _index, _query_engine

# ------------------ Routes ------------------
@app.route("/", methods=["GET", "POST"])
def home():
    index, query_engine = get_index_and_engine()

    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            return redirect(url_for("home"))

        if query_engine is None:
            answer = "No data found in /data folder. Add files and refresh."
        else:
            # Query the index
            try:
                answer = str(query_engine.query(question))
            except Exception as e:
                answer = f"Query error: {e}"

        session["chat_history"].append({"q": question, "a": answer})
        session.modified = True
        return redirect(url_for("home"))

    return render_template("index.html", chat_history=session["chat_history"])

if __name__ == "__main__":
    # Keep debug logs without double-running heavy init
    app.run(debug=True, use_reloader=False)
