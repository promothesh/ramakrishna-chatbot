"""
rag_chain.py
Builds a conversational RAG chain:
  - Loads the persisted ChromaDB vector store
  - Creates a history-aware retriever (rephrases follow-up questions)
  - Answers using ChatOpenAI, grounded strictly on retrieved context
"""

import os
import re

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ── Config ────────────────────────────────────────────────────────────────────
FAISS_DIR   = "faiss_index"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL   = "gpt-4o-mini"
RETRIEVER_K = 6
TEMPERATURE = 0.2


# ── Load API key from .Renviron (R environment) or .env ──────────────────────
def _load_api_key():
    """Read OPENAI_API_KEY from the environment, .env, or ~/.Renviron."""
    if os.environ.get("OPENAI_API_KEY"):
        return  # already set (e.g. Streamlit Cloud secrets or shell export)

    # Try .env in project root
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        _parse_env_file(env_path)
        if os.environ.get("OPENAI_API_KEY"):
            return

    # Fallback: R's ~/.Renviron
    renviron = os.path.expanduser("~/.Renviron")
    if os.path.exists(renviron):
        _parse_env_file(renviron)


def _parse_env_file(path: str):
    """Parse a .env / .Renviron file and set variables in os.environ."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$', line)
            if match:
                key, val = match.group(1), match.group(2).strip().strip('"').strip("'")
                os.environ.setdefault(key, val)


# ── Build retriever ───────────────────────────────────────────────────────────
def _build_retriever(faiss_dir: str = FAISS_DIR):
    import pickle
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from typing import List

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        faiss_dir, embeddings,
        allow_dangerous_deserialization=True,
    )

    chunks_path = os.path.join(faiss_dir, "chunks.pkl")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = RETRIEVER_K

    class HybridRetriever(BaseRetriever):
        """Merge FAISS semantic + BM25 keyword results, deduplicated."""
        def _get_relevant_documents(
            self, query: str,
            *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            faiss_docs = vectorstore.similarity_search(query, k=RETRIEVER_K)
            bm25_docs  = bm25.invoke(query)
            seen, merged = set(), []
            for doc in faiss_docs + bm25_docs:
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
            return merged[:RETRIEVER_K * 2]

    return HybridRetriever()


# ── Build chain ───────────────────────────────────────────────────────────────
def build_chain(faiss_dir: str = FAISS_DIR):
    """
    Returns a conversational RAG chain built with pure LCEL.
    Input  : {"input": str, "chat_history": list[BaseMessage]}
    Output : {"answer": str, "context": list[Document]}
    """
    _load_api_key()

    llm       = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    retriever = _build_retriever(faiss_dir)

    # ── Prompt 1: LLM multi-query retrieval ───────────────────────────
    # Instead of relying on one brittle search query, ask the LLM to
    # generate several complementary retrieval queries.
    multi_query_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You generate retrieval queries for a knowledge base about "
         "Sri Ramakrishna, Sarada Devi, Vivekananda, and related texts. "
         "Given the user's question and chat history, output 4 short search queries, "
         "one per line, with no numbering and no commentary.\n"
         "Requirements:\n"
         "- Make the question standalone if needed.\n"
         "- Expand nicknames, devotional titles, alternate spellings, and relationships.\n"
         "- You MAY use your own background knowledge to suggest likely names, family members, "
         "places, alternate transliterations, and titles that could appear in the source texts, "
         "but do not answer the question.\n"
         "- Include at least one literal keyword-heavy query.\n"
         "- Include at least one paraphrased conceptual query.\n"
         "- If the question is about a person's biological family, favor family terms "
         "such as mother, father, wife, brother, childhood name, birthplace, parents, early life, chronology.\n"
         "- Do not answer the question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    multi_query_chain = multi_query_prompt | llm | StrOutputParser()

    def get_search_queries(inputs: dict) -> list[str]:
        """Generate multiple retrieval queries from the user question."""
        raw = multi_query_chain.invoke(inputs)
        queries = [inputs["input"]]
        lowered = inputs["input"].lower()
        if any(term in lowered for term in ["mother", "father", "wife", "brother", "sister", "childhood", "born", "birth"]):
            fallback_queries = [
                "Sri Ramakrishna parents family early life chronology Kamarpukur Gadadhar",
                "biographical facts about Sri Ramakrishna family mother father childhood name birthplace",
            ]
            for q in fallback_queries:
                if q not in queries:
                    queries.append(q)
        for line in raw.splitlines():
            q = line.strip().lstrip("-*•").strip()
            if q and q not in queries:
                queries.append(q)
        return queries[:7]

    def get_context_documents(inputs: dict):
        """Retrieve across multiple LLM-generated queries and merge results."""
        merged = []
        seen = set()
        for query in inputs["search_queries"]:
            for doc in retriever.invoke(query):
                key = doc.page_content[:120]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged[:20]

    def filter_biographical_candidates(inputs: dict):
        """For family/biography questions, narrow to likely early-life family passages."""
        docs = inputs["context"]
        lowered = inputs["input"].lower()
        if not any(term in lowered for term in ["mother", "father", "wife", "brother", "sister", "parents", "childhood", "born", "birth"]):
            return docs

        preferred = []
        cues = [
            "gadadhar", "kamarpukur", "khudiram", "kshudiram",
            "chandra", "chandramani", "chandradevi", "parents",
            "master's mother", "master’s mother", "father’s death",
            "childhood", "chronology", "birth of",
        ]
        for doc in docs:
            text = doc.page_content.lower()
            if any(cue in text for cue in cues):
                preferred.append(doc)

        return preferred if preferred else docs

    rerank_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are reranking retrieved passages for a question about Sri Ramakrishna. "
         "Select the passage IDs that are MOST relevant for answering the question. "
         "Prefer literal biographical facts over devotional or tangential mentions. "
         "Distinguish biological family from spiritual references such as 'Divine Mother'. "
         "Return only a comma-separated list of IDs, e.g. 2,5,1"),
        ("human", "Question:\n{question}\n\nPassages:\n{passages}"),
    ])
    rerank_chain = rerank_prompt | llm | StrOutputParser()

    def rerank_documents(inputs: dict):
        """Use the LLM to pick the most relevant subset of retrieved docs."""
        docs = inputs["context"]
        if len(docs) <= 4:
            return docs

        passages = []
        for i, doc in enumerate(docs, 1):
            snippet = doc.page_content[:350].replace("\n", " ")
            passages.append(f"[{i}] {snippet}")

        raw = rerank_chain.invoke({
            "question": inputs["input"],
            "passages": "\n".join(passages),
        })

        chosen = []
        for part in raw.replace(" ", "").split(","):
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(docs):
                    chosen.append(idx - 1)

        # Fallback if parsing failed
        if not chosen:
            return docs[:4]

        # Deduplicate while preserving order
        ordered = []
        seen_idx = set()
        for idx in chosen:
            if idx not in seen_idx:
                seen_idx.add(idx)
                ordered.append(docs[idx])
        return ordered[:4]

    fact_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You extract concise factual answers from retrieved context about Sri Ramakrishna, "
         "his family, early life, and close associates. "
         "Use only the supplied context. "
         "Important rules:\n"
         "- Chronology lines, genealogical references, section headings, and short factual fragments ARE valid evidence.\n"
         "- For family questions, use biological family facts, not spiritual titles like Divine Mother.\n"
         "- A chronology of Sri Ramakrishna's life that lists 'Birth of [Name]' entries for a man and a woman\n"
         "  in the same generation as Khudiram (his father) is strong evidence that the woman is his mother.\n"
         "  If the context is titled 'CHRONOLOGY OF SRI RAMAKRISHNA'S LIFE' and lists births of Khudiram and a woman,\n"
         "  you may conclude the woman is Sri Ramakrishna's mother.\n"
         "- If the context clearly supports the answer — even through such contextual inference — "
         "return exactly one short sentence.\n"
         "- If the context does not support the answer at all, return exactly NOT_FOUND.\n"
         "- Do not add hedging, explanations, or commentary."),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ])
    fact_chain = fact_prompt | llm | StrOutputParser()

    def extract_biographical_fact(inputs: dict) -> str:
        """Extract a direct fact from context before general answer generation."""
        lowered = inputs["input"].lower()
        if not any(term in lowered for term in [
            "mother", "father", "wife", "brother", "sister",
            "parents", "childhood", "born", "birth", "name",
        ]):
            return "NOT_APPLICABLE"

        raw = fact_chain.invoke({
            "question": inputs["input"],
            "context": "\n\n".join(doc.page_content for doc in inputs["context"]),
        }).strip()
        return raw or "NOT_FOUND"

    # ── Prompt 2: answer strictly from retrieved context ──────────────────
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a knowledgeable assistant on Sri Ramakrishna, "
         "his life, teachings, and spiritual legacy. "
         "Answer the user's question using ONLY the context provided below. "
         "If the answer is not in the context, say so clearly — do not guess. "
         "Be thorough but concise. Use respectful, clear language.\n"
         "For biographical family questions, relevant evidence may appear in chronology lines, "
         "section headings, or short factual fragments rather than a full sentence. "
         "Use such evidence when it clearly identifies the person. "
         "Treat close spelling variants like Chandra Devi / Chandramani / Chandradevi as potentially "
         "the same person if the context indicates Sri Ramakrishna's early family background.\n\n"
         "CONTEXT:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ── Compose the full LCEL chain ───────────────────────────────────────
    # Step 1: generate multiple search queries
    # Step 2: retrieve and merge docs across those queries
    # Step 3: generate answer, pass through context for citation display
    chain = (
        RunnablePassthrough.assign(
            search_queries=RunnableLambda(get_search_queries)
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(get_context_documents)
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(filter_biographical_candidates)
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(rerank_documents)
        )
        | RunnablePassthrough.assign(
            extracted_fact=RunnableLambda(extract_biographical_fact)
        )
        | RunnablePassthrough.assign(
            answer=RunnableLambda(
                lambda x: (
                    x["extracted_fact"]
                    if x.get("extracted_fact") not in ("NOT_FOUND", "NOT_APPLICABLE")
                    else (qa_prompt | llm | StrOutputParser()).invoke({
                        "input":        x["input"],
                        "chat_history": x.get("chat_history", []),
                        "context":      format_docs(x["context"]),
                    })
                )
            )
        )
    )
    return chain


# ── Public helper ─────────────────────────────────────────────────────────────
def format_history(messages: list[dict]) -> list:
    """Convert Streamlit message dicts to LangChain message objects."""
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history
