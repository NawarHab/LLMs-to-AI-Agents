"""Gradio UI: ask questions about a YouTube video using transcript + RAG from youtube_chatbot."""

from __future__ import annotations

# Setup (if you see ModuleNotFoundError: No module named 'gradio', use the *same* python you run with):
#   python -m pip install "gradio>=4.0"
# Or from First_project, install everything in requirements.txt:
#   python -m pip install -r requirements.txt
# Run this app:
#   python Gradio_app.py
#
# If Gradio fails to import with:
#   ValueError: numpy.dtype size changed, may indicate binary incompatibility ...
# Gradio loads pandas; that error means pandas was built for a different numpy than the one
# installed (common after numpy 2.x upgrades). Fix with the same python you use to run:
#   python -m pip install --upgrade pandas
# If it still fails, reinstall both so wheels match:
#   python -m pip install --upgrade --force-reinstall numpy pandas
#
# If ModuleNotFoundError: No module named 'youtube_chatbot', your working directory may not
# be First_project; run:  cd path\to\First_project  then  python Gradio_app.py
#
# When clicking Load video / chat, Gradio may import matplotlib. If you see
#   AttributeError: _ARRAY_API not found
# or "compiled using NumPy 1.x cannot be run in NumPy 2.x", upgrade matplotlib for NumPy 2:
#   python -m pip install --upgrade "matplotlib>=3.9"
#
# Embeddings use Ollama. If you see: model "nomic-embed-text" not found (404), pull it:
#   ollama pull nomic-embed-text
# (Chat uses gemma3:1b — run: ollama pull gemma3:1b)

from typing import Any

import numpy  # noqa: F401  # before gradio: numpy/pandas ABI issues — see header comments
import gradio as gr
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

_transcript_cache: dict[str, str] = {}
_retriever_cache: dict[str, Any] = {}


def _cached_transcript(video_id: str) -> str:
    if video_id not in _transcript_cache:
        _transcript_cache[video_id] = get_video_transcript(video_id)
    return _transcript_cache[video_id]


def _cached_retriever(video_id: str):
    if video_id not in _retriever_cache:
        text = _cached_transcript(video_id)
        _retriever_cache[video_id] = build_retriever(text)
    return _retriever_cache[video_id]


def _models_markdown() -> str:
    return (
        f"**Models:** chat `{CHAT_MODEL}`, embeddings `{EMBED_MODEL}`  \n"
        "Ensure Ollama is running and models are pulled."
    )


def load_video(url: str) -> tuple:
    """Returns: video_id_state, chatbot, load_status, preview, active_video."""
    if not url or not url.strip():
        return None, [], "Paste a URL first.", "", ""
    vid = extract_video_id(url.strip())
    if not vid:
        return None, [], "Could not read a video ID from that URL.", "", ""
    try:
        full = _cached_transcript(vid)
        preview = full[:2000] + ("…" if len(full) >= 2000 else "")
    except Exception as e:
        return None, [], f"Transcript error: {e}", "", ""
    _ = _cached_retriever(vid)
    return (
        vid,
        [],
        f"Loaded video `{vid}`. Ask a question in the chat.",
        preview,
        f"Active video: `{vid}`",
    )


def _append_turn(history: list, user_text: str, assistant_text: str) -> list:
    """Gradio 6+ Chatbot uses messages: list of {role, content} dicts."""
    return history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def chat_reply(message: str, history: list, video_id: str | None) -> tuple:
    """Append one turn to history; returns (history, cleared_message_input)."""
    if not message or not message.strip():
        return history, ""
    text = message.strip()
    if not video_id:
        return _append_turn(
            history,
            text,
            "Paste a YouTube URL on the left and click **Load video** first.",
        ), ""
    try:
        retriever = _cached_retriever(video_id)
        answer = ask_video(retriever, text)
    except Exception as e:
        answer = f"**Error:** {e}"
    return _append_turn(history, text, answer), ""


def main() -> None:
    with gr.Blocks(title="YouTube video Q&A") as demo:
        video_id_state = gr.State(None)
        gr.Markdown("# YouTube video Q&A")
        gr.Markdown(
            "Loads the video script (transcript), indexes it locally with Ollama embeddings, "
            "and answers using the same RAG flow as `youtube_chatbot.py`."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Video")
                url_box = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=…",
                )
                load_btn = gr.Button("Load video", variant="primary")
                load_status = gr.Markdown()
                active_label = gr.Markdown()
                preview = gr.Textbox(
                    label="Preview transcript (start)",
                    lines=12,
                    max_lines=20,
                    interactive=False,
                )
                gr.Markdown(_models_markdown())

            with gr.Column(scale=2):
                gr.Markdown("*Paste a YouTube URL on the left and click **Load video** to start.*")
                chatbot = gr.Chatbot(label="Chat", height=480)
                msg = gr.Textbox(
                    label="Ask about this video",
                    placeholder="Ask about this video…",
                    lines=1,
                )
                send = gr.Button("Send", variant="secondary")

        load_btn.click(
            fn=load_video,
            inputs=[url_box],
            outputs=[
                video_id_state,
                chatbot,
                load_status,
                preview,
                active_label,
            ],
        )

        msg.submit(chat_reply, [msg, chatbot, video_id_state], [chatbot, msg])
        send.click(chat_reply, [msg, chatbot, video_id_state], [chatbot, msg])

    demo.launch()


if __name__ == "__main__":
    main()
