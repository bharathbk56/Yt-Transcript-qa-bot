import html
import os
import time

import google.generativeai as genai
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SUPADATA_KEY = os.getenv("SUPADATA_API_KEY")
SUPADATA_TRANSCRIPT_URL = "https://api.supadata.ai/v1/transcript"
genai.configure(api_key=GEMINI_KEY)


def call_gemini(prompt, model_name="models/gemini-2.5-flash"):
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini Error] {e}"


def get_youtube_transcript(url, lang="en"):
    if not SUPADATA_KEY:
        st.error("Missing SUPADATA_API_KEY. Add it to your .env file.")
        return ""

    try:
        result = request_supadata_transcript(url, lang=lang)
        content = result.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return " ".join(
                item.get("text", "").strip()
                for item in content
                if isinstance(item, dict) and item.get("text")
            )

        st.error("Supadata returned an unexpected transcript format.")
    except requests.Timeout:
        st.error("Supadata transcript request timed out. Please try again.")
    except requests.RequestException as e:
        st.error(f"Supadata network error: {e}")
    except ValueError as e:
        st.error(str(e))
    return ""


def request_supadata_transcript(url, lang="en", max_poll_attempts=90):
    headers = {"x-api-key": SUPADATA_KEY}
    params = {
        "url": url,
        "lang": lang,
        "text": "true",
        "mode": "auto",
    }

    response = requests.get(
        SUPADATA_TRANSCRIPT_URL,
        headers=headers,
        params=params,
        timeout=65,
    )
    payload = parse_supadata_response(response)

    if response.status_code == 200:
        return payload

    if response.status_code == 202 and payload.get("jobId"):
        return poll_supadata_transcript_job(
            payload["jobId"],
            headers=headers,
            max_attempts=max_poll_attempts,
        )

    raise ValueError(get_supadata_error_message(response, payload))


def poll_supadata_transcript_job(job_id, headers, max_attempts):
    job_url = f"{SUPADATA_TRANSCRIPT_URL}/{job_id}"

    for _ in range(max_attempts):
        response = requests.get(job_url, headers=headers, timeout=30)
        payload = parse_supadata_response(response)

        if response.status_code == 200 and payload.get("status") == "completed":
            return payload.get("result") or payload

        if payload.get("status") == "failed":
            error = payload.get("error") or "Transcript job failed."
            raise ValueError(f"Supadata transcript job failed: {error}")

        if response.status_code not in (200, 202):
            raise ValueError(get_supadata_error_message(response, payload))

        time.sleep(1)

    raise ValueError("Supadata transcript job is still processing. Please try again shortly.")


def parse_supadata_response(response):
    try:
        return response.json()
    except ValueError as e:
        raise ValueError(f"Supadata returned a non-JSON response: HTTP {response.status_code}") from e


def get_supadata_error_message(response, payload):
    message = payload.get("message") or payload.get("error") or response.text
    return f"Supadata transcript error ({response.status_code}): {message}"


def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


def format_answer_html(answer_text):
    lines = [line.strip() for line in answer_text.splitlines()]
    parts = []
    in_list = False

    for line in lines:
        if not line:
            if in_list:
                parts.append("</ul>")
                in_list = False
            continue

        is_bullet = line.startswith("- ") or line.startswith("* ") or line.startswith("• ")
        is_numbered = (
            len(line) > 2
            and line[0].isdigit()
            and ". " in line[:4]
        )

        if is_bullet:
            content = line[2:].strip()
        elif line.startswith("• "):
            content = line[2:].strip()
        elif is_numbered:
            first_dot = line.find(".")
            content = line[first_dot + 1 :].strip()
        else:
            content = line

        if is_bullet or is_numbered or line.startswith("• "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{html.escape(content)}</li>")
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<p>{html.escape(content)}</p>")

    if in_list:
        parts.append("</ul>")

    return "".join(parts)


class SimpleRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.texts = [d["page_content"] for d in docs]
        self.vectorizer = TfidfVectorizer().fit(self.texts)
        self.vectors = self.vectorizer.transform(self.texts)

    def get_relevant(self, query, top_k=4):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.vectors)[0]
        idx = np.argsort(sims)[::-1][:top_k]
        return [self.docs[i] for i in idx]


st.set_page_config(page_title="AI Tutor", page_icon=":studio_microphone:", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;500;700&display=swap');

:root {
    --bg-a: #f7f3e8;
    --bg-b: #e8f0ea;
    --ink: #172125;
    --muted: #4f5c62;
    --accent: #1f8a70;
    --accent-strong: #15664f;
    --highlight: #d97706;
    --card: rgba(255, 255, 255, 0.72);
    --card-border: rgba(31, 138, 112, 0.20);
}

.stApp {
    background:
        radial-gradient(1200px 550px at 0% 0%, #ffe8c6 0%, transparent 55%),
        radial-gradient(900px 500px at 100% 100%, #d8f1e8 0%, transparent 65%),
        linear-gradient(145deg, var(--bg-a), var(--bg-b));
    color: var(--ink);
}

/* Hide Streamlit top bar and toolbar controls */
header[data-testid="stHeader"] { display: none; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stDecoration"] { display: none; }

[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    z-index: 0;
    filter: blur(4px);
    animation: floatBlob 8s ease-in-out infinite;
}

[data-testid="stAppViewContainer"]::before {
    top: -80px;
    right: 8%;
    background: rgba(217, 119, 6, 0.14);
}

[data-testid="stAppViewContainer"]::after {
    bottom: -100px;
    left: 6%;
    background: rgba(31, 138, 112, 0.14);
    animation-delay: 1.5s;
}

@keyframes floatBlob {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-14px); }
}

.block-container {
    max-width: 980px;
    padding-top: 2.2rem;
    padding-bottom: 2.2rem;
    position: relative;
    z-index: 1;
}

.hero {
    text-align: center;
    margin-bottom: 1.25rem;
    animation: reveal .45s ease-out;
}

.eyebrow {
    font-family: 'Manrope', sans-serif;
    color: var(--accent-strong);
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    font-size: 0.78rem;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--ink);
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 700;
    line-height: 1.08;
    margin: .4rem 0 .35rem 0;
}

.hero-subtitle {
    font-family: 'Manrope', sans-serif;
    color: var(--muted);
    font-size: 1.02rem;
    margin: 0 auto;
    max-width: 720px;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
    justify-content: center;
    margin: 1rem 0 1.5rem 0;
}

.chip {
    font-family: 'Manrope', sans-serif;
    color: var(--accent-strong);
    border: 1px solid rgba(31, 138, 112, .28);
    background: rgba(255, 255, 255, .56);
    border-radius: 999px;
    padding: .3rem .7rem;
    font-size: .76rem;
    font-weight: 700;
}

.panel {
    border: 1px solid var(--card-border);
    background: var(--card);
    border-radius: 20px;
    padding: 1rem 1rem .65rem 1rem;
    backdrop-filter: blur(6px);
    box-shadow: 0 14px 40px rgba(28, 50, 50, 0.08);
    margin-bottom: 1rem;
    animation: reveal .55s ease-out;
}

.panel-title {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--ink);
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0 0 .3rem 0;
}

.panel-caption {
    font-family: 'Manrope', sans-serif;
    color: var(--muted);
    font-size: .92rem;
    margin-bottom: .7rem;
}

.answer-box {
    border: 1px solid rgba(31, 138, 112, .35);
    border-left: 6px solid var(--accent);
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,.82), rgba(248,252,251,.72));
    color: #1f2a2f;
    padding: .95rem;
    font-family: 'Manrope', sans-serif;
    font-size: 1rem;
    line-height: 1.7;
    margin-top: .4rem;
    white-space: normal;
}

.stTextInput label {
    font-family: 'Manrope', sans-serif;
    font-weight: 600;
    color: #2d3c42;
}

.stTextInput > div > div {
    border-radius: 12px !important;
    border: 1px solid rgba(31, 138, 112, .45) !important;
    background: rgba(255, 255, 255, .88) !important;
}

.stTextInput input {
    font-family: 'Manrope', sans-serif !important;
    color: #1f2a2f !important;
}

.stTextInput input::placeholder {
    color: #5f6d73 !important;
}

.stButton > button {
    border-radius: 12px !important;
    border: none !important;
    background: linear-gradient(96deg, var(--accent), var(--accent-strong)) !important;
    color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    padding: .62rem .9rem !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(21, 102, 79, .25);
}

[data-testid="metric-container"] {
    border: 1px solid rgba(23, 102, 79, .2);
    background: rgba(255, 255, 255, .70);
    border-radius: 14px;
    padding: 6px 8px;
}

[data-testid="stMetricLabel"] {
    color: #2d3c42;
    font-family: 'Manrope', sans-serif;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #0f4d3e;
    font-family: 'Space Grotesk', sans-serif;
}

@keyframes reveal {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 768px) {
    .block-container { padding-top: 1.4rem; }
    .panel { padding: .85rem .75rem .45rem .75rem; }
    .chip-row { justify-content: flex-start; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<section class="hero">
  <div class="eyebrow">Transcript Tutor</div>
  <h1 class="hero-title">Ask Better Questions from Any YouTube Lesson</h1>
  <p class="hero-subtitle">
    Paste a video URL, build a transcript index, and get answers grounded in the actual lecture content.
  </p>
</section>
<div class="chip-row">
  <span class="chip">Context-Grounded Answers</span>
  <span class="chip">Fast Transcript Retrieval</span>
  <span class="chip">Lecture-Friendly Q&A</span>
</div>
""",
    unsafe_allow_html=True,
)

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "transcript_chars" not in st.session_state:
    st.session_state.transcript_chars = 0

st.markdown(
    """
<div class="panel">
  <h2 class="panel-title">Step 1: Process YouTube Video</h2>
  <p class="panel-caption">Extract transcript and prepare searchable chunks.</p>
</div>
""",
    unsafe_allow_html=True,
)

video_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    label_visibility="collapsed",
)

if st.button("Process Video", key="process", use_container_width=True):
    if video_url:
        with st.spinner("Fetching transcript..."):
            transcript_text = get_youtube_transcript(video_url)

        if transcript_text:
            save_transcript_to_file(transcript_text)

            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            docs_for_index = []
            for i, d in enumerate(docs):
                docs_for_index.append({"page_content": d.page_content, "metadata": {"chunk": i}})

            st.session_state.retriever = SimpleRetriever(docs_for_index)
            st.session_state.chunk_count = len(docs_for_index)
            st.session_state.transcript_chars = len(transcript_text)
            st.success("Transcript processed. Ask your question below.")
    else:
        st.warning("Please enter a valid URL.")

col_a, col_b = st.columns(2)
col_a.metric("Chunks Indexed", st.session_state.chunk_count)
col_b.metric("Transcript Size", f"{st.session_state.transcript_chars:,} chars")

st.markdown(
    """
<div class="panel" style="margin-top: 1rem;">
  <h2 class="panel-title">Step 2: Ask Questions</h2>
  <p class="panel-caption">Answers are generated only from retrieved transcript context.</p>
</div>
""",
    unsafe_allow_html=True,
)

question = st.text_input(
    "Question",
    placeholder="What are the three key concepts explained in this lecture?",
    label_visibility="collapsed",
)

if question and "retriever" in st.session_state:
    retriever = st.session_state.retriever
    top_chunks = retriever.get_relevant(question, top_k=4)
    context = "\n\n---\n\n".join([c["page_content"] for c in top_chunks])

    prompt = f"""
Use ONLY the transcript below to answer.
Format your response as detailed bullet points.
Keep the answer clear and structured:
- Start with a one-line definition.
- Then provide 4-8 bullet points with explanation.
- Add one short "Example" point at the end when possible.
If answer is not found in transcript, reply exactly: "Not available in the video".

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    with st.spinner("Thinking..."):
        answer = call_gemini(prompt)

    st.markdown('<div class="panel-title" style="margin-top: .7rem;">Answer</div>', unsafe_allow_html=True)
    answer_html = format_answer_html(answer)
    st.markdown(f'<div class="answer-box">{answer_html}</div>', unsafe_allow_html=True)
elif question:
    st.info("Process a video first.")
