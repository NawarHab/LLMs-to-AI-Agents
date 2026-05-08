import argparse
import re
import sys
from typing import Optional

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Ollama models gemma3:1b

CHAT_MODEL = "gemma3:1b"
EMBED_MODEL = "nomic-embed-text"

# Url extraction


def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/shorts\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# Clean transcript from adds and sponsors


def clean_transcript(text: str) -> str:
    sponsor_keywords = [
        "sponsor",
        "sponsored",
        "thanks to",
        "today's sponsor",
        "brought to you by",
        "use code",
        "discount code",
        "affiliate",
        "promotion",
        "promo code",
    ]

    cleaned_parts = []
    for part in re.split(r"(?<=[.!?])\s+", text):
        low = part.lower()
        if any(keyword in low for keyword in sponsor_keywords):
            continue
        cleaned_parts.append(part)
    return " ".join(cleaned_parts).strip()


def get_video_transcript(video_id: str) -> str:
    # Support both old and new youtube-transcript-api styles.
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except AttributeError:
        transcript = YouTubeTranscriptApi().fetch(video_id)
    parts = []
    for chunk in transcript:
        if isinstance(chunk, dict):
            parts.append(chunk.get("text", ""))
        else:
            parts.append(getattr(chunk, "text", ""))
    text = " ".join(part for part in parts if part)
    cleaned = clean_transcript(text)
    if not cleaned:
        raise ValueError("Transcript was retrieved but became empty after cleaning.")
    return cleaned


# Retriever


def build_retriever(transcript_text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript_text)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def ask_video(retriever, question: str, model_name: str = CHAT_MODEL) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = (
        "You are a helpful assistant answering questions about a YouTube video transcript. "
        "Use only the provided context. If the answer is not in context, say you are not sure.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    llm = ChatOllama(model=model_name, temperature=0)
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Chat with a YouTube video's transcript."
    )
    parser.add_argument("--url", required=False, help="YouTube video URL")
    parser.add_argument(
        "--question",
        default="Summarize the video in 5 bullet points.",
        help="Question to ask about the video",
    )
    args = parser.parse_args()

    if args.url:
        url = args.url.strip()
    else:
        print("Paste YouTube URL: ", end="", file=sys.stderr, flush=True)
        url = input().strip()
    if not url:
        raise ValueError("No URL provided.")

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL. Could not extract video ID.")

    transcript_text = get_video_transcript(video_id)
    retriever = build_retriever(transcript_text)
    answer = ask_video(retriever, args.question)
    print(answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error:", exc)
        print("Tip: ensure Ollama is running and models are pulled:")
        print(f"  ollama pull {CHAT_MODEL}")
        print(f"  ollama pull {EMBED_MODEL}")
