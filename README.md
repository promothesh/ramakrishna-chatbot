# 🪔 Sri Ramakrishna RAG Chatbot

A conversational AI chatbot grounded entirely in primary source texts about Sri Ramakrishna.  
It answers questions about his life, family, teachings, and spiritual legacy by retrieving and reasoning over two canonical books.

**Live app:** https://ramakrishna-chatbot-lksljhvf6z7bgn8ht6blfv.streamlit.app/

---

## Overview

This project builds a **Retrieval-Augmented Generation (RAG)** chatbot using LangChain, FAISS, and OpenAI's GPT-4o-mini. Rather than relying on the language model's general knowledge, every answer is grounded in passages retrieved from the source texts. The model is explicitly instructed not to guess — if the answer is not in the retrieved context, it says so.

---

## Source Texts

| Book | Author | How obtained |
|---|---|---|
| *Sri Ramakrishna: The Great Master* | Swami Saradananda | Scraped chapter-by-chapter from [rkmm.org](https://englishbooks.rkmm.org/s/lsr/m/sri-ramakrishna-the-great-master/) |
| *The Gospel of Sri Ramakrishna* | M. (Mahendranath Gupta) | Scraped in full from [ramakrishnavivekananda.info](https://www.ramakrishnavivekananda.info/gospel/gospel.htm) |

Both texts were saved as UTF-8 `.txt` files in `docs/`. Combined they span tens of thousands of paragraphs covering Sri Ramakrishna's biography, conversations with disciples, family background, and spiritual teachings.

---

## Architecture

```
User question
      │
      ▼
┌─────────────────────────────────┐
│   Multi-Query Generation (LLM)  │  → 4–7 expanded search queries
│   (nicknames, family terms,     │     (e.g. "Thakur" → also searches
│    alternate spellings)         │      "Gadadhar", "Sri Ramakrishna")
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   Hybrid Retriever              │
│   FAISS (semantic) +            │  → Up to 20 merged, deduplicated
│   BM25  (keyword)               │     passages
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   Biographical Filter           │  → Keeps chunks mentioning family
│   (for family questions)        │     cues: Khudiram, Chandra, chronology…
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   LLM Reranker                  │  → Selects top 4 most relevant passages,
│                                 │     distinguishing biographical facts
│                                 │     from devotional/spiritual mentions
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   Fact Extraction (LLM)         │  → For biographical/family questions,
│                                 │     extracts a direct factual answer
│                                 │     (e.g. from chronology: "1791 Birth
│                                 │      of Chandra Devi" → mother's name)
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   QA Answer Generation (LLM)   │  → Falls back to this only if fact
│   (strict context-only prompt)  │     extraction returns NOT_FOUND
└─────────────────────────────────┘
      │
      ▼
   Answer + source passages displayed in Streamlit UI
```

---

## Key Technical Decisions

### Heading-Aware Chunking (`ingest.py`)
Plain character-based chunking loses context — a passage about "His mother" has no way of knowing which chapter it belongs to. The ingestion pipeline detects heading lines (ALL-CAPS section titles, numbered sections like `1.2.3`, `VOLUME ...`) and prepends the current section heading to every chunk it produces. This means retrieval results carry structural context from the book.

### Hybrid Retrieval (FAISS + BM25)
- **FAISS** (semantic): Uses `paraphrase-multilingual-mpnet-base-v2` embeddings. Finds passages that are *conceptually similar* to the query even if exact words differ.
- **BM25** (keyword): Finds passages that contain *exact words* from the query. Critical for proper nouns, Sanskrit terms, and names (e.g. "Khudiram", "Kamarpukur") that embeddings may not handle well.
- Results from both are merged and deduplicated before reranking.

### Multi-Query Expansion
A single retrieval query often misses relevant passages due to nickname variation (Thakur, Sri Ramakrishna, Gadadhar), transliteration differences (Khudiram / Kshudiram), or phrasing mismatch. The chain asks the LLM to generate 4 complementary queries per user question, plus hard-coded fallback queries for family-term questions. All queries are retrieved in parallel and merged.

### LLM Reranker
After retrieval, up to 20 candidate passages are sent to the LLM with the question, and it selects the top 4 most useful. The reranker prompt specifically instructs it to distinguish *biological* family references from *spiritual* references (e.g. "Divine Mother" vs. Sri Ramakrishna's actual mother).

### Fact Extraction Step
The hardest class of failures involved answers that were *present* in the retrieved context but in an implicit, structured form — e.g., the chronology line `1791 Birth of Chandra Devi` does not explicitly say "Chandra Devi is Sri Ramakrishna's mother." A dedicated fact-extraction LLM pass runs before the main QA prompt. It is allowed to draw contextual inferences from chronology-style evidence (e.g., a woman listed in a Sri Ramakrishna life chronology alongside his father Khudiram is inferred to be his mother). The main QA prompt is only invoked as a fallback when the fact extractor returns `NOT_FOUND`.

---

## Benchmark Results

A biographical accuracy benchmark (`evaluate_bio_questions.py`) tests four core family/life questions:

| Question | Context hit | Answer hit |
|---|---|---|
| What is Thakur's mother's name? | ✅ | ✅ Chandra Devi |
| Who is Thakur's father? | ✅ | ✅ Khudiram |
| What is Thakur's childhood name? | ✅ | ✅ Gadadhar |
| When was Thakur born? | ✅ | ✅ February 18, 1836 |

**4/4 answer accuracy, 4/4 context accuracy.**

---

## Project Structure

```
Chatbot/
├── app.py                    # Streamlit chat UI
├── rag_chain.py              # Full RAG pipeline (retrieval → reranking → answer)
├── ingest.py                 # Document loading, heading-aware chunking, FAISS build
├── evaluate_bio_questions.py # Biographical accuracy benchmark
├── scrape_book.py            # Scraper for The Great Master
├── scrape_gospel.py          # Scraper for The Gospel of Sri Ramakrishna
├── verify_gospel.py          # Verifies gospel scrape completeness
├── docs/
│   ├── sri_ramakrishna_great_master.txt
│   └── gospel_of_sri_ramakrishna.txt
├── faiss_index/
│   ├── index.faiss           # FAISS vector index
│   ├── index.pkl             # FAISS metadata
│   └── chunks.pkl            # All chunks (used by BM25 retriever)
├── requirements.txt
└── .env.example
```

---

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/promothesh/ramakrishna-chatbot.git
cd ramakrishna-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

**Option A — `.env` file** (copy from example):
```bash
cp .env.example .env
# then edit .env and paste your key
```

**Option B — R users** (key already in `~/.Renviron`):  
The chain auto-reads `OPENAI_API_KEY` from `~/.Renviron` if no `.env` is present.

### 4. Build the index (first time only)
The FAISS index is committed to the repo, so this step is only needed if you change the source texts:
```bash
python ingest.py
```
This downloads the embedding model (~420 MB on first run), chunks both books, and writes `faiss_index/`.

### 5. Run the app
```bash
streamlit run app.py
```

---

## Deployment (Streamlit Community Cloud)

The app is deployed from the `master` branch of this private GitHub repository.  
Streamlit Cloud auto-redeploys on every push.

**Secret required:** Set `OPENAI_API_KEY` under **App settings → Secrets** in the Streamlit Cloud dashboard:
```toml
OPENAI_API_KEY = "sk-..."
```

The FAISS index (`faiss_index/`) is committed to git and loaded directly at startup — no separate build step is needed on the cloud.

---

## Stack

| Component | Library / Service |
|---|---|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| Vector store | FAISS (`faiss-cpu`) |
| Keyword search | BM25 (`rank-bm25`) |
| RAG orchestration | LangChain (LCEL) |
| UI | Streamlit |
| Hosting | Streamlit Community Cloud |

---

## Challenges & Solutions

| Challenge | Solution |
|---|---|
| Nickname variation ("Thakur" vs "Sri Ramakrishna" vs "Gadadhar") | LLM multi-query expansion generates 4–7 retrieval queries per question |
| Keyword terms like Sanskrit names not captured by semantic search | Hybrid FAISS + BM25 retrieval |
| Devotional "Divine Mother" references drowning out biographical family facts | LLM reranker prompt explicitly distinguishes biological vs. spiritual family references |
| Mother's name in chronology format, not an explicit sentence | Dedicated fact-extraction LLM pass with chronological inference rules |
| Large texts causing memory spikes during ingestion | Batch FAISS indexing (500 chunks per batch) |
| ChromaDB incompatible with Streamlit Cloud | Migrated to FAISS with committed index |
