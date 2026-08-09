"""
ResearchChatbot — Streamlit Dashboard
=====================================
Modern chat UI with file upload, Arxiv scraper, and RAG pipeline.
"""

import streamlit as st
import io
import os
import time
import torch
from pypdf import PdfReader

from RagPipeline.rag_pipeline import RagPipelineConnector
from RagPipeline.tools.text_cleaner import clean_text
from document_extractor.pdf_extractor_from_arxiv import ExtractPdf
from document_extractor.pdf_downloader import download_pdf


# ═══════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="ResearchChatbot",
    page_icon=":material/psychology:",
    layout="wide",
)

# ═══════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUGGESTIONS = {
    ":material/help: Summarize this paper": "Summarize the key findings of this paper",
    ":material/build: What are the methods used?": "What methods or techniques are used in this paper?",
    ":material/lightbulb: What are the main contributions?": "What are the main contributions of this paper?",
    ":material/bar_chart: Show me key results": "What are the key results and findings of this paper?",
}


# ═══════════════════════════════════════════════════════
# Cached pipeline builder
# ═══════════════════════════════════════════════════════

@st.cache_resource
def build_pipeline(pdf_text: str) -> RagPipelineConnector | None:
    """Build and cache the RAG pipeline for a given document text."""
    if not pdf_text or not pdf_text.strip():
        return None
    return RagPipelineConnector(
        pdf_text=pdf_text,
        top_k=5,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        device=DEVICE,
        embd_model_name=EMBED_MODEL_NAME,
        llm_device=DEVICE,
        llm_model_name=LLM_MODEL_NAME,
    )


# ═══════════════════════════════════════════════════════
# Helper: extract text from uploaded PDF bytes
# ═══════════════════════════════════════════════════════

def extract_text_from_bytes(uploaded_files) -> str:
    """Extract and clean text from a list of uploaded PDF files."""
    all_texts = []
    for f in uploaded_files:
        pdf_stream = io.BytesIO(f.getvalue() if hasattr(f, "getvalue") else f.read())
        reader = PdfReader(pdf_stream)
        raw_text = "".join(page.extract_text() for page in reader.pages)
        all_texts.append(clean_text(raw_text))
    return "\n\n".join(all_texts)


# ═══════════════════════════════════════════════════════
# Helper: scrape Arxiv, download PDFs, extract text
# ═══════════════════════════════════════════════════════

def scrape_and_extract(query: str, count: int, status_container) -> str | None:
    """Scrape Arxiv, download papers, return combined extracted text."""
    status_container.update(label=f"Searching Arxiv for '{query}'...", state="running")

    extractor = ExtractPdf(query=query, total_reasearch_paper=count)
    pdf_urls, authors_names, titles_names = extractor.extract_pdf()

    status_container.update(
        label=f"Found {len(pdf_urls)} papers. Downloading...", state="running"
    )

    all_text = []
    os.makedirs("docsContainer", exist_ok=True)

    for i, (url, title) in enumerate(zip(pdf_urls, titles_names)):
        # Sanitize filename
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80]
        filepath = f"docsContainer/{safe_title}.pdf"

        try:
            download_pdf(url, safe_title)
        except Exception:
            continue  # Skip failed downloads

        # Extract text from downloaded PDF
        try:
            reader = PdfReader(filepath)
            raw_text = "".join(page.extract_text() for page in reader.pages)
            all_text.append(clean_text(raw_text))
        except Exception:
            continue

        status_container.update(
            label=f"Downloaded {i+1}/{len(pdf_urls)}: {title[:50]}...",
            state="running",
        )

    if not all_text:
        status_container.update(
            label="No papers could be downloaded or extracted.",
            state="error",
        )
        return None

    status_container.update(
        label=f"Indexed {len(all_text)} papers successfully.",
        state="complete",
    )
    return "\n\n".join(all_text)


# ═══════════════════════════════════════════════════════
# Initialize session state
# ═══════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "paper_names" not in st.session_state:
    st.session_state.paper_names = []


# ═══════════════════════════════════════════════════════
# Sidebar — Paper management
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.header(":material/description: Papers")

    # ── Upload PDFs ──
    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button(":material/upload: Index uploaded papers", use_container_width=True):
            with st.status("Indexing papers...", expanded=True) as status:
                st.write("Extracting text from PDFs...")
                combined_text = extract_text_from_bytes(uploaded_files)
                if combined_text.strip():
                    st.session_state.pdf_text = combined_text
                    st.session_state.paper_names = [f.name for f in uploaded_files]
                    # Force pipeline rebuild by clearing cache for this text
                    build_pipeline.clear()
                    pipeline = build_pipeline(combined_text)
                    if pipeline:
                        st.session_state.pipeline_ready = True
                        st.session_state.messages = []  # Reset chat for new papers
                        status.update(
                            label=f"Ready — {len(st.session_state.paper_names)} papers indexed",
                            state="complete",
                        )
                    else:
                        status.update(label="Pipeline failed to build", state="error")
                else:
                    status.update(label="Could not extract text from PDFs", state="error")

    st.divider()

    st.header(":material/travel_explore: Arxiv search")

    arxiv_query = st.text_input(
        "Search query",
        placeholder="e.g. Reinforcement Learning",
        label_visibility="collapsed",
    )
    arxiv_count = st.selectbox(
        "Papers to fetch",
        options=[25, 50, 100, 200],
        index=0,
        label_visibility="collapsed",
    )

    if st.button(
        ":material/download: Scrape & index",
        use_container_width=True,
        disabled=not arxiv_query.strip(),
    ):
        with st.status("Scraping Arxiv...", expanded=True) as status:
            combined_text = scrape_and_extract(arxiv_query, arxiv_count, status)
            if combined_text:
                st.session_state.pdf_text = combined_text
                st.session_state.paper_names = [f"Arxiv: {arxiv_query} ({arxiv_count} papers)"]
                build_pipeline.clear()
                pipeline = build_pipeline(combined_text)
                if pipeline:
                    st.session_state.pipeline_ready = True
                    st.session_state.messages = []
                else:
                    status.update(label="Pipeline failed to build", state="error")

    # ── Indexed papers list ──
    if st.session_state.paper_names:
        st.divider()
        st.caption("Indexed papers:")
        for name in st.session_state.paper_names:
            st.markdown(f":material/check_circle: {name}")

    # ── Clear button ──
    if st.session_state.pipeline_ready:
        st.divider()
        if st.button(":material/delete: Clear & reset", use_container_width=True):
            build_pipeline.clear()
            st.session_state.messages = []
            st.session_state.pipeline_ready = False
            st.session_state.pdf_text = ""
            st.session_state.paper_names = []
            st.rerun()


# ═══════════════════════════════════════════════════════
# Main area — Chat UI
# ═══════════════════════════════════════════════════════

st.title(":material/psychology: ResearchChatbot")
st.caption("Ask questions about your research papers using local LLM + FAISS retrieval")

# ── Suggestion chips (only when chat is empty) ──
if not st.session_state.messages:
    st.markdown("##### Try asking:")
    selected = st.pills(
        "suggestions",
        list(SUGGESTIONS.keys()),
        label_visibility="collapsed",
    )
    if selected and st.session_state.pipeline_ready:
        prompt = SUGGESTIONS[selected]
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

# ── Chat history ──
for msg in st.session_state.messages:
    avatar = ":material/person:" if msg["role"] == "user" else ":material/robot_2:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# ── Chat input ──
prompt = st.chat_input(
    "Ask about your papers...",
    accept_file=True,
    file_type=["pdf"],
    submit_mode="disable",
)

if prompt:
    # Handle file attachments in chat — append to existing papers
    if prompt.files:
        with st.spinner("Indexing attached papers..."):
            new_text = extract_text_from_bytes(prompt.files)
            if new_text.strip():
                # Combine with existing text
                combined = (
                    st.session_state.pdf_text + "\n\n" + new_text
                    if st.session_state.pdf_text
                    else new_text
                )
                st.session_state.pdf_text = combined
                st.session_state.paper_names.extend(
                    [f.name for f in prompt.files]
                )
                build_pipeline.clear()
                pipeline = build_pipeline(combined)
                if pipeline:
                    st.session_state.pipeline_ready = True

    # ── User message ──
    user_text = prompt.text if prompt.text else "Summarize the attached paper"
    st.session_state.messages.append({"role": "user", "content": user_text})

    with st.chat_message("user", avatar=":material/person:"):
        st.write(user_text)

    # ── Assistant response ──
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        if not st.session_state.pipeline_ready:
            response = (
                "Please upload or scrape papers first using the sidebar, "
                "then ask your question."
            )
            st.write(response)
        else:
            try:
                with st.spinner("Thinking..."):
                    pipeline = build_pipeline(st.session_state.pdf_text)
                    if pipeline is None:
                        response = "Error: Could not initialize the pipeline."
                    else:
                        result = pipeline.query(user_text, generate_answer=True)
                        response = result.get("answer", "")
                        if not response or not response.strip():
                            response = (
                                "I couldn't generate an answer. "
                                "Try rephrasing your question."
                            )

                        # Append sources in an expander
                        sources = result.get("source_chunks", [])
                        if sources:
                            with st.expander(
                                f":material/book: {len(sources)} source chunks"
                            ):
                                for s in sources[:5]:
                                    st.caption(f"**Chunk {s['id']}**")
                                    st.write(s["text"][:300] + "...")

                st.write(response)
            except Exception as e:
                response = f"Error: {str(e)}"
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})