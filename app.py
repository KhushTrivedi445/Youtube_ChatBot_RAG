import streamlit as st
import os
import time
import requests
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv(find_dotenv())

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥", layout="centered")

# --- Custom CSS Theme ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    h1 {
        font-weight: 700;
        color: #f1f1f1;
    }
    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 10px 16px;
        margin-bottom: 8px;
    }
    [data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) {
        border: 1px solid #2a2e37;
    }
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #FF4B4B;
        background-color: transparent;
        color: #FF4B4B;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #FF4B4B;
        color: white;
        border-color: #FF4B4B;
    }
    /* Video card */
    .video-card {
        display: flex;
        align-items: center;
        gap: 12px;
        background-color: #1a1d24;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 12px;
        border: 1px solid #2a2e37;
    }
    .video-card img {
        border-radius: 8px;
        width: 100px;
    }
    .video-card .video-title {
        font-size: 14px;
        font-weight: 600;
        color: #f1f1f1;
        line-height: 1.3;
    }
    /* Caption timing */
    .response-time {
        font-size: 11px;
        color: #6b7280;
        margin-top: -6px;
    }
</style>
""", unsafe_allow_html=True)

if not os.getenv("HF_TOKEN") or not os.getenv("OPENROUTER_API_KEY"):
    st.error("Missing API Keys! Please check your .env file for HF_TOKEN and OPENROUTER_API_KEY.")
    st.stop()


# --- Helpers ---
def get_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        return parse_qs(query.query).get('v', [None])[0]
    return None


@st.cache_data(show_spinner=False)
def get_video_metadata(video_id):
    """Fetch title & thumbnail via YouTube oEmbed (no API key needed)."""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title": data.get("title", "Unknown title"),
                "author": data.get("author_name", ""),
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            }
    except Exception:
        pass
    return {
        "title": "Video",
        "author": "",
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    }


@st.cache_resource
def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="qwen/qwen-2.5-7b-instruct",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )


def generate_suggested_questions(llm, context):
    """Ask the LLM for 4 short starter questions based on the transcript."""
    try:
        prompt = (
            "Based on the following video transcript excerpt, suggest exactly 4 short, "
            "interesting questions a viewer might ask about this video. "
            "Reply with ONLY the 4 questions, one per line, no numbering, no extra text.\n\n"
            f"Transcript excerpt:\n{context[:3000]}"
        )
        result = llm.invoke(prompt).content
        questions = [q.strip("-•1234567890. ").strip() for q in result.split("\n") if q.strip()]
        return questions[:4]
    except Exception:
        return [
            "What is this video about?",
            "Summarize the key points.",
            "What are the main takeaways?",
            "Explain the most important part in simple terms."
        ]


# --- Session state init ---
defaults = {
    "ready": False,
    "messages": [],
    "video_id": None,
    "video_meta": None,
    "suggested_questions": [],
    "pending_query": None,
    "is_loading": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --- Sidebar ---
with st.sidebar:
    st.header("🎥 YouTube RAG Chatbot")
    st.markdown("Paste a YouTube link, index the transcript, then chat about the video.")

    is_loading = st.session_state.get("is_loading", False)

    # --- Current video section (shown once indexed, hidden while loading) ---
    if st.session_state.ready and st.session_state.video_meta and not is_loading:
        meta = st.session_state.video_meta
        st.markdown("**Current video**")
        st.markdown(
            "<span style='background-color:#16331e;color:#4ade80;padding:2px 8px;"
            "border-radius:6px;font-size:12px;font-weight:600;'>🟢 Indexed</span>",
            unsafe_allow_html=True
        )
        st.markdown(f"""
        <div class="video-card" style="margin-top:8px;">
            <img src="{meta['thumbnail']}" />
            <div>
                <div class="video-title">{meta['title'][:60]}{'...' if len(meta['title']) > 60 else ''}</div>
                <div style="font-size:12px;color:#9ca3af;margin-top:4px;">{meta['author']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        expander_label = "🔁 Load a different video"
    else:
        expander_label = "YouTube Video URL"

    # --- URL input (collapsed into expander once a video is indexed) ---
    if st.session_state.ready and not is_loading:
        with st.expander(expander_label, expanded=False):
            url = st.text_input("YouTube Video URL", label_visibility="collapsed",
                                 placeholder="Paste a new YouTube link...", key="url_input",
                                 on_change=lambda: None)
    elif not is_loading:
        url = st.text_input("YouTube Video URL", key="url_input", on_change=lambda: None)
    else:
        url = st.session_state.get("url_input", "")

    video_id = get_video_id(url) if url else None

    if video_id:
        st.session_state.video_id = video_id

    if not is_loading and video_id:
        button_label = "Start New Chat" if st.session_state.ready else "Start Chat"
        if st.button(button_label, use_container_width=True, key="start_chat_btn"):
            st.session_state.is_loading = True
            st.rerun()
    elif not is_loading and url and not video_id:
        st.error("Could not extract video ID from this URL.")

    # --- Loading state: do the actual indexing work ---
    if is_loading:
        with st.spinner("Fetching transcript and building index..."):
            try:
                video_id = st.session_state.video_id
                yt_api = YouTubeTranscriptApi()
                transcript_data = yt_api.fetch(video_id, languages=["en"])
                text = " ".join(chunk.text for chunk in transcript_data)

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

                embeddings = get_embeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)
                vectorstore.save_local(f"./vector_db_{video_id}")

                st.session_state.ready = True
                st.session_state.messages = []
                st.session_state.video_meta = get_video_metadata(video_id)

                llm = get_llm()
                st.session_state.suggested_questions = generate_suggested_questions(
                    llm, text
                )

                st.session_state.is_loading = False
                st.success("Indexed successfully! You can now chat.")
                st.rerun()
            except Exception as e:
                st.session_state.is_loading = False
                st.error(f"Error during indexing: {e}")



# --- Main area ---
st.title("Chat with the video")

if not st.session_state.ready:
    st.info("👈 Paste a YouTube URL and click **Start Chat** to get started.")
else:
    # --- Suggested questions (only before first message) ---
    if not st.session_state.messages and st.session_state.suggested_questions:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        for i, q in enumerate(st.session_state.suggested_questions):
            with cols[i % 2]:
                if st.button(q, key=f"suggested_{i}", use_container_width=True):
                    st.session_state.pending_query = q

    # --- Render chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "time" in msg:
                st.markdown(
                    f"<div class='response-time'>Answered in {msg['time']:.1f}s</div>",
                    unsafe_allow_html=True
                )

    # --- Chat input ---
    chat_query = st.chat_input("Ask something about the video...")
    query = st.session_state.pending_query or chat_query
    st.session_state.pending_query = None

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            # Typing indicator before generation
            for dots in ["Thinking", "Thinking.", "Thinking..", "Thinking..."]:
                placeholder.markdown(f"_{dots}_")
                time.sleep(0.15)

            start_time = time.time()
            try:
                embeddings = get_embeddings()
                vectorstore = FAISS.load_local(
                    f"./vector_db_{st.session_state.video_id}",
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                llm = get_llm()

                retrieved_docs = vectorstore.as_retriever().invoke(query)
                context = "\n".join([d.page_content for d in retrieved_docs])

                prompt = (
                    f"You are a helpful assistant answering questions about a YouTube video "
                    f"based on its transcript.\n\nContext:\n{context}\n\nQuestion: {query}"
                )

                response = llm.invoke(prompt).content
            except Exception as e:
                response = f"Sorry, something went wrong: {e}"
                retrieved_docs = []

            elapsed = time.time() - start_time

            # Streamed-looking output
            displayed = ""
            for word in response.split(" "):
                displayed += word + " "
                placeholder.markdown(displayed + "▌")
                time.sleep(0.02)
            placeholder.markdown(displayed)

            st.markdown(
                f"<div class='response-time'>Answered in {elapsed:.1f}s</div>",
                unsafe_allow_html=True
            )

            # Source citations
            if retrieved_docs:
                with st.expander("📄 Sources used"):
                    for i, d in enumerate(retrieved_docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.caption(d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""))

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "time": elapsed
        })