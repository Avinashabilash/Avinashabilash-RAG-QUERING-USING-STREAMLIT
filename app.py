import streamlit as st
import os
import streamlit as st
import numpy as np
import pdfplumber
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
# --- RAG Helper Functions ---
def load_pdf_text(path: str):
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text = p.extract_text() or ""
            pages.append(text)
    full = "\n\n".join(pages)
    return full, pages

def get_pdf_metadata(path: str):
    try:
        reader = PyPDF2.PdfReader(path)
        meta = reader.metadata or {}
        return {k.strip('/'): str(v) for k, v in meta.items() if v}
    except:
        return {}

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# --- RAG Store ---
class RagStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def _normalize(self, x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return x / norms

    def embed_chunks(self, chunks):
        embs = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        return self._normalize(embs)

    def build_index(self, chunks):
        self.chunks = chunks
        embs = self.embed_chunks(chunks)
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)

    def query(self, q, top_k=3):
        q_emb = self.model.encode([q], convert_to_numpy=True)
        q_emb = self._normalize(q_emb)
        D, I = self.index.search(q_emb, top_k)
        return [(int(idx), float(score)) for idx, score in zip(I[0], D[0]) if idx != -1]

def answer_query(store, chunks, query, k=3, threshold=0.25):
    results = store.query(query, top_k=k)
    sources = [(chunks[idx]["text"], score, chunks[idx]["page"]) for idx, score in results]
    if sources and sources[0][1] >= threshold:
        return {"type": "answer", "answer": sources[0][0], "sources": sources}
    else:
        return {"type": "related", "answer": None, "sources": sources}

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Q&A with RAG", page_icon="📄")

st.title("📄 PDF Q&A with RAG")
st.write("Upload a PDF and ask questions about its content.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Extracting text from PDF... Please wait ⏳")
    full_text, pages = load_pdf_text(pdf_path)
    st.success(f"✅ PDF Loaded! Found {len(pages)} pages.")

    # Chunk text
    all_chunks = []
    for i, p in enumerate(pages):
        for c in chunk_text(p, chunk_size=500, overlap=100):
            all_chunks.append({"text": c, "page": i})

    # Build RAG index
    st.info("Building embeddings and FAISS index... 🔍")
    store = RagStore()
    store.build_index([c["text"] for c in all_chunks])
    st.success(f"✅ RAG index built with {len(all_chunks)} chunks.")

    query = st.text_input("💬 Ask a question about your PDF:")

    if query:
        st.info("Thinking... 🧠")
        result = answer_query(store, all_chunks, query)
        if result["type"] == "answer":
            st.subheader("🟢 Answer:")
            st.write(result["answer"])
            st.subheader("📚 Sources:")
            for txt, score, page in result["sources"]:
                st.write(f"**Page {page+1}** | Score: {score:.3f}")
                st.caption(txt[:300] + "...")
        else:
            st.warning("No confident single answer. Showing related chunks:")
            for txt, score, page in result["sources"]:
                st.write(f"**Page {page+1}** | Score: {score:.3f}")
                st.caption(txt[:300] + "...")
