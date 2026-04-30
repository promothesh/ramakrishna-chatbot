"""
ingest.py
Loads all documents from docs/, splits into chunks, embeds them using a
multilingual sentence-transformers model, and stores in a local ChromaDB.

Run once (or re-run whenever docs change):
    python ingest.py
"""

import os
import glob
import time
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR        = "docs"
CHROMA_DIR      = "chroma_db"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 80
COLLECTION_NAME = "ramakrishna_books"

# ── Load documents ────────────────────────────────────────────────────────────
def load_documents(docs_dir: str):
    documents = []

    # PDF files
    for path in glob.glob(os.path.join(docs_dir, "*.pdf")):
        print(f"  Loading PDF: {os.path.basename(path)}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        print(f"    → {len(docs)} pages")
        documents.extend(docs)

    # TXT files
    for path in glob.glob(os.path.join(docs_dir, "*.txt")):
        print(f"  Loading TXT: {os.path.basename(path)}")
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        print(f"    → {len(docs)} document(s)")
        documents.extend(docs)

    return documents


# ── Split ─────────────────────────────────────────────────────────────────────
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("RAG Ingestion Pipeline")
    print("=" * 60)

    # 1. Load
    print(f"\n[1/3] Loading documents from '{DOCS_DIR}/'...")
    documents = load_documents(DOCS_DIR)
    print(f"  Total pages/documents loaded: {len(documents)}")

    if not documents:
        print("ERROR: No documents found. Add PDF or TXT files to docs/")
        return

    # 2. Split
    print(f"\n[2/3] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = split_documents(documents)
    print(f"  Total chunks: {len(chunks):,}")

    # Preview a sample chunk
    if chunks:
        sample = chunks[len(chunks) // 2]
        print(f"\n  Sample chunk (middle):")
        print(f"  Source : {sample.metadata.get('source', 'N/A')}")
        print(f"  Content: {sample.page_content[:200]}...")

    # 3. Embed & store
    print(f"\n[3/3] Embedding with '{EMBED_MODEL}' and storing in ChromaDB...")
    print("  (First run downloads the model ~420MB — this may take a few minutes)\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Build vector store in batches to avoid memory spikes
    BATCH = 500
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print(f"  Cleared existing '{CHROMA_DIR}/'")

    vectorstore = None
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        pct   = min(i + BATCH, len(chunks))
        print(f"  Embedding chunks {i+1}–{pct} / {len(chunks)}...", end="\r")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
                collection_name=COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)

    print(f"\n\n  ChromaDB saved to '{CHROMA_DIR}/'")

    # Sanity check — run a test query
    print("\n[TEST] Running a test retrieval query...")
    results = vectorstore.similarity_search("Who is Ramakrishna?", k=3)
    print(f"  Top-3 results for 'Who is Ramakrishna?':")
    for i, r in enumerate(results, 1):
        src = os.path.basename(r.metadata.get("source", "unknown"))
        print(f"  [{i}] ({src}) {r.page_content[:120]}...")

    print("\n✓ Ingestion complete!")
    print(f"  Documents : {len(documents)}")
    print(f"  Chunks    : {len(chunks):,}")
    print(f"  Vector DB : {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
