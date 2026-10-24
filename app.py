
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import build_rag_chain


# -------------------- CONFIGURATION --------------------

load_dotenv()

st.set_page_config(
    page_title="YouTube Transcript AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "style.css"

if css_path.exists():
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# -------------------- SESSION STATE --------------------

defaults = {
    "messages": [],
    "main_chain": None,
    "file_name": None,
    "file_bytes": None,
    "chunk_count": 0,
    "pending_question": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">🎬</div>
            <div>
                <h2>Transcript AI</h2>
                <p>Ask your video anything</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📚 How it works")
    st.caption(
        "Upload a YouTube transcript, then ask questions "
        "about its content."
    )

    if st.session_state.file_name:
        st.markdown("### 📄 Active transcript")
        st.caption(st.session_state.file_name)
        st.caption(f"{st.session_state.chunk_count} chunks indexed")

    st.markdown("---")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-footer">
            Built with Streamlit, LangChain, OpenAI and FAISS
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- MAIN HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">▶️ Youtube Assitant</div>
        <h1>Chat with your <span>YouTube transcript.</span></h1>
        <p>
            Upload your transcript, ask questions, and discover
            insights from your video.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- MAIN UPLOADER --------------------

st.markdown("### 📄 Upload your transcript")

uploaded_file = st.file_uploader(
    "Choose a YouTube transcript (.txt)",
    type=["txt"],
    key="main_transcript_uploader",
)

if uploaded_file is not None:
    current_bytes = uploaded_file.getvalue()

    if not current_bytes.strip():
        st.error("The uploaded file is empty.")

    elif (
        current_bytes != st.session_state.file_bytes
        or st.session_state.main_chain is None
    ):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            st.error(
                "OPENAI_API_KEY is missing. Check your .env file."
            )
        else:
            try:
                with st.spinner(
                    "Processing transcript and creating embeddings..."
                ):
                    transcript_text = current_bytes.decode(
                        "utf-8-sig",
                        errors="replace",
                    )

                    chain, chunk_count = build_rag_chain(
                        transcript_text=transcript_text,
                        api_key=api_key,
                    )

                st.session_state.main_chain = chain
                st.session_state.file_bytes = current_bytes
                st.session_state.file_name = uploaded_file.name
                st.session_state.chunk_count = chunk_count
                st.session_state.messages = []
                st.session_state.pending_question = None

                st.success(
                    f"Transcript ready! {chunk_count} chunks indexed."
                )

            except Exception as exc:
                st.session_state.main_chain = None
                st.error(f"Transcript processing failed: {exc}")


# -------------------- CHAT INTERFACE --------------------

if st.session_state.main_chain is None:

    st.info("Upload a .txt transcript above to start chatting.")

else:
    st.markdown(
        f"**📄 Active transcript:** {st.session_state.file_name}"
    )

    st.markdown("### 💡 What would you like to know?")

    suggestions = [
        "📝 Summarize the video",
        "🎯 Explain the main topic",
        "🔑 What are the key takeaways?",
    ]

    # All buttons submit a question to the SAME response area.
    cols = st.columns(3)

    for index, suggestion in enumerate(suggestions):
        if cols[index].button(
            suggestion,
            key=f"question_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = suggestion
            st.rerun()

    st.markdown("---")

    # One shared conversation area for every question.
    chat_area = st.container()

    with chat_area:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Process a button question or a typed question.
    pending_question = st.session_state.pending_question
    typed_question = st.chat_input(
        "Ask anything about your transcript..."
    )

    user_question = typed_question or pending_question

    if user_question:
        # Clear pending question before processing.
        st.session_state.pending_question = None

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        with chat_area:
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Finding your answer..."):
                        answer = st.session_state.main_chain.invoke(
                            user_question
                        )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as exc:
                    st.error(f"Could not generate an answer: {exc}")
