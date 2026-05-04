"""
chat_ui.py
----------
Streamlit chat interface for the Local Wikipedia RAG system.

Features:
  - ChatGPT-style message history
  - Streaming LLM responses
  - "Show sources" expander per answer
  - Sidebar: system status, entity list, settings
  - Clear chat button
  - Query type indicator (person / place / both)
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.llm import (
    DEFAULT_MODEL,
    check_ollama_available,
    generate_answer,
    generate_answer_streaming,
)
from retrieval.router import classify_query, get_all_entity_names
from retrieval.vector_store import get_collection, get_collection_stats, query_store

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WikiRAG — Local AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* Main container */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 6rem;
    max-width: 900px;
}

/* Title area */
.rag-title {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}

.rag-title h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

.rag-title p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.3rem;
}

/* Chat messages */
.stChatMessage {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px);
    margin-bottom: 0.8rem !important;
}

/* User messages */
[data-testid="user-message"] {
    background: rgba(167, 139, 250, 0.1) !important;
    border-color: rgba(167, 139, 250, 0.2) !important;
}

/* Assistant messages */
[data-testid="assistant-message"] {
    background: rgba(96, 165, 250, 0.06) !important;
    border-color: rgba(96, 165, 250, 0.15) !important;
}

/* Source badge */
.source-badge {
    display: inline-block;
    background: rgba(52, 211, 153, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.3);
    color: #34d399;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px 4px 2px 0;
}

/* Query type badge */
.query-badge-person {
    display: inline-block;
    background: rgba(167, 139, 250, 0.15);
    border: 1px solid rgba(167, 139, 250, 0.3);
    color: #a78bfa;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.query-badge-place {
    display: inline-block;
    background: rgba(96, 165, 250, 0.15);
    border: 1px solid rgba(96, 165, 250, 0.3);
    color: #60a5fa;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.query-badge-both {
    display: inline-block;
    background: rgba(52, 211, 153, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.3);
    color: #34d399;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}

/* Status indicator */
.status-ok {
    color: #34d399;
    font-weight: 600;
}

.status-err {
    color: #f87171;
    font-weight: 600;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}

/* Input */
.stChatInput textarea {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collection" not in st.session_state:
        st.session_state.collection = None
    if "db_stats" not in st.session_state:
        st.session_state.db_stats = None
    if "ollama_status" not in st.session_state:
        st.session_state.ollama_status = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    if "n_results" not in st.session_state:
        st.session_state.n_results = 5
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "use_streaming" not in st.session_state:
        st.session_state.use_streaming = True


# ─────────────────────────────────────────────
# Load ChromaDB collection (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_collection():
    try:
        coll = get_collection()
        return coll, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 WikiRAG")
        st.markdown("*Local AI Assistant*")
        st.divider()

        # System status
        st.markdown("### System Status")

        # Ollama status
        available, msg = check_ollama_available(st.session_state.selected_model)
        if available:
            st.markdown('<p class="status-ok">✅ Ollama: Running</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-err">❌ Ollama: Offline</p>', unsafe_allow_html=True)
            st.caption(msg)

        # ChromaDB status
        coll, err = load_collection()
        if coll is not None:
            stats = get_collection_stats(coll)
            st.session_state.collection = coll
            st.session_state.db_stats = stats
            total = stats["total_chunks"]
            indexed = len(stats["indexed_sources"])
            st.markdown(f'<p class="status-ok">✅ ChromaDB: {total:,} chunks</p>', unsafe_allow_html=True)
            st.caption(f"{indexed} entities indexed")
        else:
            st.markdown('<p class="status-err">❌ ChromaDB: Error</p>', unsafe_allow_html=True)
            st.caption(str(err))

        st.divider()

        # Settings
        st.markdown("### ⚙️ Settings")
        st.session_state.selected_model = st.selectbox(
            "LLM Model",
            ["llama3.2:3b", "phi3", "mistral"],
            index=0,
        )
        st.session_state.n_results = st.slider(
            "Retrieved chunks", min_value=2, max_value=10, value=5
        )
        st.session_state.show_sources = st.toggle("Show source chunks", value=True)
        st.session_state.use_streaming = st.toggle("Streaming responses", value=True)

        st.divider()

        # Indexed entities
        if st.session_state.db_stats:
            sources = st.session_state.db_stats.get("indexed_sources", [])
            if sources:
                st.markdown("### 📚 Indexed Entities")
                persons = [s["name"] for s in sources if s["type"] == "person"]
                places = [s["name"] for s in sources if s["type"] == "place"]

                if persons:
                    with st.expander(f"👤 People ({len(persons)})"):
                        for p in persons:
                            st.caption(f"• {p}")
                if places:
                    with st.expander(f"🏛️ Places ({len(places)})"):
                        for p in places:
                            st.caption(f"• {p}")

        st.divider()

        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()

        # Ingest tip
        if st.session_state.db_stats and st.session_state.db_stats["total_chunks"] == 0:
            st.warning("No data indexed yet!\n\nRun:\n```\npython ingest/run_ingest.py\n```")


# ─────────────────────────────────────────────
# Example queries
# ─────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "Who was Albert Einstein?",
    "What did Marie Curie discover?",
    "Where is the Eiffel Tower located?",
    "Why is Nikola Tesla famous?",
    "What was the Colosseum used for?",
    "Compare Lionel Messi and Cristiano Ronaldo",
    "Which famous place is located in Turkey?",
    "Compare the Eiffel Tower and the Statue of Liberty",
    "Who is associated with electricity?",
    "Who is the president of Mars?",
]


# ─────────────────────────────────────────────
# Query processing
# ─────────────────────────────────────────────
def process_query(query: str, collection) -> tuple[str, list[dict], str]:
    """
    Full RAG pipeline: route → retrieve → generate.

    Returns:
        (answer, source_chunks, query_type)
    """
    # Route query
    query_type = classify_query(query)

    # Retrieve
    entity_type_filter = None if query_type == "both" else query_type
    chunks = query_store(
        query_text=query,
        entity_type=entity_type_filter,
        n_results=st.session_state.n_results,
        collection=collection,
    )

    # If "both" and no results, try without filter
    if not chunks:
        chunks = query_store(
            query_text=query,
            entity_type=None,
            n_results=st.session_state.n_results,
            collection=collection,
        )

    # Generate
    answer = generate_answer(
        query=query,
        context_chunks=chunks,
        model=st.session_state.selected_model,
    )

    return answer, chunks, query_type


def stream_query(query: str, collection):
    """
    Streaming version of process_query. Yields (token, chunks, query_type).
    First yields ("__meta__", chunks, query_type), then text tokens.
    """
    query_type = classify_query(query)
    entity_type_filter = None if query_type == "both" else query_type
    chunks = query_store(
        query_text=query,
        entity_type=entity_type_filter,
        n_results=st.session_state.n_results,
        collection=collection,
    )
    if not chunks:
        chunks = query_store(query_text=query, entity_type=None,
                             n_results=st.session_state.n_results, collection=collection)

    yield "__meta__", chunks, query_type

    for token in generate_answer_streaming(
        query=query,
        context_chunks=chunks,
        model=st.session_state.selected_model,
    ):
        yield token, None, None


# ─────────────────────────────────────────────
# Source chunks display
# ─────────────────────────────────────────────
def render_sources(chunks: list[dict]):
    """Renders retrieved source chunks in an expander."""
    if not chunks:
        st.caption("*No relevant chunks found in the knowledge base.*")
        return

    # Unique sources
    sources = list({c["source"] for c in chunks})
    source_html = " ".join(f'<span class="source-badge">📌 {s}</span>' for s in sources)
    st.markdown(source_html, unsafe_allow_html=True)

    st.markdown("---")
    for i, chunk in enumerate(chunks, 1):
        dist = chunk.get("distance", 0)
        similarity = max(0, 1 - dist)
        with st.expander(f"Chunk {i}: {chunk['source']} (relevance: {similarity:.1%})"):
            st.markdown(f"**Type:** {chunk['type'].title()} | **Chunk #{chunk['chunk_index']}**")
            st.text(chunk["text"][:600] + ("..." if len(chunk["text"]) > 600 else ""))


# ─────────────────────────────────────────────
# Query type badge
# ─────────────────────────────────────────────
def query_type_badge(query_type: str) -> str:
    icons = {"person": "👤", "place": "🏛️", "both": "🔍"}
    icon = icons.get(query_type, "🔍")
    return f'<span class="query-badge-{query_type}">{icon} {query_type.title()}</span>'


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────
def main():
    init_session_state()
    render_sidebar()

    # Title
    st.markdown("""
    <div class="rag-title">
        <h1>🧠 WikiRAG</h1>
        <p>Ask anything about famous people and places — powered by local AI</p>
    </div>
    """, unsafe_allow_html=True)

    collection = st.session_state.collection

    # Check if system is ready
    system_ready = collection is not None
    if not system_ready:
        st.error("⚠️ ChromaDB is not available. Check the sidebar for details.")

    # Welcome message
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown("#### 💡 Example questions to get started:")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUERIES[:8]):
            col = cols[i % 2]
            with col:
                if st.button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
        st.markdown("---")

    # Chat input
    if prompt := st.chat_input(
        "Ask about a famous person or place...",
        disabled=not system_ready,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Show query type badge
                if "query_type" in msg:
                    st.markdown(query_type_badge(msg["query_type"]), unsafe_allow_html=True)
                    st.markdown("")  # spacing
                st.markdown(msg["content"])
                # Sources
                if st.session_state.show_sources and "sources" in msg and msg["sources"]:
                    with st.expander(f"📎 View {len(msg['sources'])} source chunks"):
                        render_sources(msg["sources"])
            else:
                st.markdown(msg["content"])

    # Trigger generation if the last message is from the user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        prompt = st.session_state.messages[-1]["content"]

        # Generate response
        with st.chat_message("assistant"):
            if st.session_state.use_streaming:
                # Streaming mode
                query_type = None
                chunks = []
                response_tokens = []

                # Create placeholder for streaming text
                response_placeholder = st.empty()
                badge_placeholder = st.empty()

                try:
                    gen = stream_query(prompt, collection)
                    for token, meta_chunks, meta_qtype in gen:
                        if token == "__meta__":
                            chunks = meta_chunks
                            query_type = meta_qtype
                            badge_placeholder.markdown(
                                query_type_badge(query_type), unsafe_allow_html=True
                            )
                        else:
                            response_tokens.append(token)
                            response_placeholder.markdown("".join(response_tokens) + "▌")

                    full_response = "".join(response_tokens)
                    # Final render without cursor
                    badge_placeholder.markdown(
                        query_type_badge(query_type) if query_type else "", unsafe_allow_html=True
                    )
                    response_placeholder.markdown(full_response)

                except RuntimeError as e:
                    full_response = f"❌ Error: {e}"
                    query_type = "both"
                    chunks = []
                    response_placeholder.error(full_response)

            else:
                # Non-streaming mode
                query_type = classify_query(prompt)
                st.markdown(query_type_badge(query_type), unsafe_allow_html=True)
                st.markdown("")

                with st.spinner("Thinking..."):
                    try:
                        full_response, chunks, query_type = process_query(prompt, collection)
                    except RuntimeError as e:
                        full_response = f"❌ Error: {e}"
                        chunks = []

                st.markdown(full_response)

            # Show sources
            if st.session_state.show_sources and chunks:
                with st.expander(f"📎 View {len(chunks)} source chunks"):
                    render_sources(chunks)

        # Save to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": chunks,
            "query_type": query_type or "both",
        })


if __name__ == "__main__":
    main()
