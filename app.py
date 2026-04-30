"""
app.py  –  Streamlit chat UI for the Sri Ramakrishna RAG chatbot.

Run locally:
    streamlit run app.py
"""

import os
import streamlit as st
from rag_chain import build_chain, format_history

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sri Ramakrishna Chatbot",
    page_icon="🪔",
    layout="centered",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🪔 Sri Ramakrishna")
    st.caption("Ask anything about Sri Ramakrishna's life, teachings, and spiritual legacy.")

    st.divider()
    st.markdown("**Sources**")
    st.markdown("- *Sri Ramakrishna: The Great Master* — Swami Saradananda")
    st.markdown("- *The Gospel of Sri Ramakrishna* — M. (Mahendranath Gupta)")

    st.divider()
    show_sources = st.toggle("Show source passages", value=True)

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = None
        st.rerun()

    st.divider()
    st.caption("Powered by GPT-4o-mini + ChromaDB + sentence-transformers")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

# ── Load chain (once per session) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base and AI model…")
def load_chain():
    return build_chain()

try:
    chain = load_chain()
except Exception as e:
    st.error(f"**Failed to load the chain:** {e}")
    st.info("Make sure `OPENAI_API_KEY` is set and `chroma_db/` exists (run `python ingest.py` first).")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🪔 Sri Ramakrishna Chatbot")
st.caption("Ask questions about Sri Ramakrishna — answers are grounded in the source texts.")

# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📖 Sources ({len(msg['sources'])} passages)"):
                for i, src in enumerate(msg["sources"], 1):
                    book = os.path.basename(src["source"])
                    page = src.get("page", "")
                    label = f"**[{i}] {book}**" + (f" — page {page + 1}" if page != "" else "")
                    st.markdown(label)
                    st.markdown(f"> {src['content']}")
                    if i < len(msg["sources"]):
                        st.divider()

# ── Welcome message if no history yet ────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        welcome = (
            "Jai Sri Ramakrishna! 🙏 I can answer questions about Sri Ramakrishna's "
            "life, spiritual practices, teachings, and his relationships with disciples "
            "like Swami Vivekananda and the Holy Mother. What would you like to know?"
        )
        st.markdown(welcome)

# ── Chat input & response ─────────────────────────────────────────────────────
if question := st.chat_input("Ask about Sri Ramakrishna…"):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching the texts…"):
            try:
                chat_history = format_history(st.session_state.messages[:-1])
                result = chain.invoke({
                    "input": question,
                    "chat_history": chat_history,
                })

                answer = result.get("answer", "I could not find an answer in the source texts.")
                source_docs = result.get("context", [])

                # Format source metadata
                sources = []
                seen_content = set()
                for doc in source_docs:
                    snippet = doc.page_content[:300].strip()
                    if snippet not in seen_content:
                        seen_content.add(snippet)
                        sources.append({
                            "source":  doc.metadata.get("source", "Unknown"),
                            "page":    doc.metadata.get("page", ""),
                            "content": snippet,
                        })

                st.markdown(answer)

                if show_sources and sources:
                    with st.expander(f"📖 Sources ({len(sources)} passages)"):
                        for i, src in enumerate(sources, 1):
                            book = os.path.basename(src["source"])
                            page = src.get("page", "")
                            label = f"**[{i}] {book}**" + (f" — page {int(page) + 1}" if page != "" else "")
                            st.markdown(label)
                            st.markdown(f"> {src['content']}")
                            if i < len(sources):
                                st.divider()

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                err = f"⚠️ Error: {e}"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant", "content": err, "sources": []
                })
