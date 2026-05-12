"""Streamlit UI: ask questions about a YouTube video using transcript + RAG from youtube_chatbot."""

import streamlit as st
from dotenv import load_dotenv

from youtube_chatbot import (
    CHAT_MODEL,
    EMBED_MODEL,
    ask_video,
    build_retriever,
    extract_video_id,
    get_video_transcript,
)

load_dotenv()

st.set_page_config(page_title="YouTube video Q&A", page_icon="▶️", layout="centered")


@st.cache_data(show_spinner="Fetching transcript…")
def cached_transcript(video_id: str) -> str:
    return get_video_transcript(video_id)


@st.cache_resource(show_spinner="Building search index (embeddings)…")
def cached_retriever(video_id: str):
    text = cached_transcript(video_id)
    return build_retriever(text)


def init_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    init_chat_state()

    st.title("YouTube video Q&A")
    st.caption(
        "Loads the video script (transcript), indexes it locally with Ollama embeddings, "
        "and answers using the same RAG flow as `youtube_chatbot.py`."
    )

    with st.sidebar:
        st.subheader("Video")
        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=…",
            help="Paste a full URL or youtu.be short link.",
        )
        load = st.button("Load video", type="primary")

        if load and url.strip():
            vid = extract_video_id(url.strip())
            if not vid:
                st.error("Could not read a video ID from that URL.")
            else:
                st.session_state.video_id = vid
                st.session_state.video_url = url.strip()
                st.session_state.messages = []
                st.success(f"Loaded video `{vid}`. Ask a question in the chat.")
        elif load and not url.strip():
            st.warning("Paste a URL first.")

        if st.session_state.get("video_id"):
            st.divider()
            st.text(f"Active video: {st.session_state.video_id}")
            with st.expander("Preview transcript (start)"):
                try:
                    preview = cached_transcript(st.session_state.video_id)[:2000]
                    st.text(preview + ("…" if len(preview) >= 2000 else ""))
                except Exception as e:
                    st.error(str(e))

            st.divider()
            st.markdown(
                f"**Models:** chat `{CHAT_MODEL}`, embeddings `{EMBED_MODEL}`  \n"
                "Ensure Ollama is running and models are pulled."
            )

    video_id = st.session_state.get("video_id")
    if not video_id:
        st.info("Paste a YouTube URL in the sidebar and click **Load video**.")
        return

    retriever = cached_retriever(video_id)

    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    if prompt := st.chat_input("Ask about this video…"):
        st.session_state.messages.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching transcript and generating answer…"):
                try:
                    answer = ask_video(retriever, prompt)
                except Exception as e:
                    answer = f"**Error:** {e}"
            st.markdown(answer)
        st.session_state.messages.append(("assistant", answer))


if __name__ == "__main__":
    main()
