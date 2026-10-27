"""FastAPI backend + static UI. Run:  uvicorn app.api.main:app --port 8000

One local user, so sessions live in memory (lost on restart). Only the vector index is on disk.
"""

import secrets
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app import config
from app.analysis.assessor import AnalysisError
from app.chat.answerer import LANGUAGES
from app.graph.analysis_graph import build_analysis_graph, run_analysis
from app.graph.chat_graph import ChatDeps, build_chat_graph
from app.ingestion.transcript import TranscriptError, parse_transcript
from app.ingestion.youtube import InvalidYouTubeURL, extract_video_id
from app.rag.store import E5Embeddings, build_index
from app.schemas import Level, Segment, SuitabilityReport

ROOT = Path(__file__).resolve().parent.parent.parent
UI_DIR = ROOT / "ui"
MAX_SESSIONS = 5
MAX_HISTORY_MESSAGES = 20
DEMO_VIDEO_ID_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # placeholder: demo links will not match the demo text

app = FastAPI(title="VideoFit AI")


# ---------------------------------------------------------------- services (replaceable in tests)
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = E5Embeddings(config.EMBEDDING_MODEL)
    return _embeddings


def make_analysis_graph():
    return build_analysis_graph(get_embeddings())


def make_chat_graph():
    return build_chat_graph(ChatDeps(embeddings=get_embeddings()))


# ---------------------------------------------------------------- sessions
@dataclass
class Session:
    video_id: str
    segments: list[Segment]
    duration_sec: float
    report: Optional[SuitabilityReport] = None
    messages: list[BaseMessage] = field(default_factory=list)


SESSIONS: "OrderedDict[str, Session]" = OrderedDict()


def get_session(session_id: str) -> Session:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found. Please load the video again.")
    return session


# ---------------------------------------------------------------- request models
class VideoRequest(BaseModel):
    url: str
    transcript: str


class AnalyzeRequest(BaseModel):
    session_id: str
    goal: str = Field(max_length=500)
    level: Level = "beginner"
    language: str = "English"
    topics: list[str] = Field(default_factory=list, max_length=12)
    comments: str = Field(default="", max_length=60_000)


class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=1000)
    language: str = "English"


class ResetRequest(BaseModel):
    session_id: str


def check_language(language: str) -> str:
    if language not in LANGUAGES:
        raise HTTPException(422, f"Language must be one of {', '.join(LANGUAGES)}.")
    return language


# ---------------------------------------------------------------- endpoints
@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "llm_key_set": bool(config.GROQ_API_KEY), "llm_model": config.LLM_MODEL,
            "embedding_model": config.EMBEDDING_MODEL, "languages": LANGUAGES}


@app.get("/api/sample")
def sample() -> dict:
    text = (ROOT / "samples" / "english_neural_networks_demo.txt").read_text(encoding="utf-8")
    return {"url": DEMO_VIDEO_ID_URL, "transcript": text,
            "note": "Demo text. The timestamp links point to a placeholder video, not to this lecture."}


@app.post("/api/video")
def load_video(req: VideoRequest) -> dict:
    try:
        video_id = extract_video_id(req.url)
        parsed = parse_transcript(req.transcript)
    except (InvalidYouTubeURL, TranscriptError) as exc:
        raise HTTPException(400, str(exc))

    try:
        chunks = build_index(video_id, parsed.segments, get_embeddings())
    except Exception as exc:
        raise HTTPException(500, f"Could not index the transcript ({type(exc).__name__}).")

    ends = [s.end_sec or s.start_sec or 0 for s in parsed.segments]
    duration = max(ends) if parsed.has_timestamps else 0.0
    session_id = secrets.token_urlsafe(8)
    SESSIONS[session_id] = Session(video_id=video_id, segments=parsed.segments, duration_sec=duration)
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)

    scripts: dict[str, int] = {}
    for s in parsed.segments:
        scripts[s.script] = scripts.get(s.script, 0) + 1
    return {"session_id": session_id, "video_id": video_id, "segments": len(parsed.segments), "chunks": chunks,
            "format": parsed.format, "has_timestamps": parsed.has_timestamps, "duration_sec": duration,
            "scripts": scripts, "warnings": parsed.warnings}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    session = get_session(req.session_id)
    check_language(req.language)
    try:
        report = run_analysis(make_analysis_graph(), session.video_id, session.segments, req.goal, req.level,
                              req.language, req.topics or None, req.comments or None)
    except AnalysisError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"The analysis failed ({type(exc).__name__}). Please try again.")
    session.report = report
    return {"report": report.model_dump(), "video_id": session.video_id, "duration_sec": session.duration_sec}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    session = get_session(req.session_id)
    check_language(req.language)
    try:
        final = make_chat_graph().invoke({
            "video_id": session.video_id, "language": req.language, "report": session.report,
            "messages": session.messages + [HumanMessage(content=req.question)],
        })
    except Exception as exc:
        raise HTTPException(502, f"The chat failed ({type(exc).__name__}). Please try again.")
    session.messages = final["messages"][-MAX_HISTORY_MESSAGES:]
    return {"answer": final["answer"].model_dump(), "understood_as": final.get("standalone_question", ""),
            "kind": final.get("kind", ""), "answer_language": final.get("answer_language", req.language)}


@app.post("/api/chat/reset")
def reset_chat(req: ResetRequest) -> dict:
    get_session(req.session_id).messages = []
    return {"ok": True}


# ---------------------------------------------------------------- UI
@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
