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
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "ramakrishna_books"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL       = "gpt-4o-mini"
RETRIEVER_K     = 5        # number of chunks to retrieve per query
TEMPERATURE     = 0.2


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
def _build_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


# ── Build chain ───────────────────────────────────────────────────────────────
def build_chain():
    """
    Returns a conversational RAG chain built with pure LCEL.
    Input  : {"input": str, "chat_history": list[BaseMessage]}
    Output : {"answer": str, "context": list[Document]}
    """
    _load_api_key()

    llm       = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    retriever = _build_retriever()

    # ── Prompt 1: condense follow-up questions into standalone queries ─────
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and the latest user question, "
         "rewrite the question as a fully self-contained query. "
         "Do NOT answer it — only rephrase."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    def get_standalone_question(inputs: dict) -> str:
        """Return a standalone question, condensing if history exists."""
        if inputs.get("chat_history"):
            return condense_chain.invoke(inputs)
        return inputs["input"]

    # ── Prompt 2: answer strictly from retrieved context ──────────────────
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a knowledgeable assistant on Sri Ramakrishna, "
         "his life, teachings, and spiritual legacy. "
         "Answer the user's question using ONLY the context provided below. "
         "If the answer is not in the context, say so clearly — do not guess. "
         "Be thorough but concise. Use respectful, clear language.\n\n"
         "CONTEXT:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ── Compose the full LCEL chain ───────────────────────────────────────
    # Step 1: add standalone_question
    # Step 2: retrieve docs using the standalone question
    # Step 3: generate answer, pass through context for citation display
    chain = (
        RunnablePassthrough.assign(
            standalone_question=RunnableLambda(get_standalone_question)
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: retriever.invoke(x["standalone_question"]))
        )
        | RunnablePassthrough.assign(
            answer=RunnableLambda(
                lambda x: (
                    qa_prompt | llm | StrOutputParser()
                ).invoke({
                    "input":        x["input"],
                    "chat_history": x.get("chat_history", []),
                    "context":      format_docs(x["context"]),
                })
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
