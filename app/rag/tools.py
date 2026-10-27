"""Tools the evidence-gathering agent may call.

Both are READ-ONLY, and their only argument is a short text string chosen by the model, so a
malicious transcript cannot make them change anything. The Pydantic classes below are the tool
schemas shown to the LLM (class name = tool name, docstring = description).
"""

from typing import Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from app.chat.answerer import format_context
from app.rag.store import search
from app.schemas import RetrievedChunk, SuitabilityReport


class SearchTranscript(BaseModel):
    """Search the video transcript for passages about a topic. Returns excerpts with segment IDs and timestamps."""

    query: str = Field(description="What to look for, e.g. 'backpropagation explanation' or 'CNN example'.")


class InspectTopicCoverage(BaseModel):
    """Look up how well the video covered a topic, from the completed analysis report."""

    topic: str = Field(description="Topic name or part of it, e.g. 'activation'.")


def run_search_transcript(query: str, video_id: str, embeddings: Embeddings, k: int = 4) -> tuple[str, list[RetrievedChunk]]:
    chunks = search(video_id, query, embeddings, k=k)
    return format_context(chunks) or "No excerpts found.", chunks


def run_inspect_topic_coverage(topic: str, report: Optional[SuitabilityReport]) -> str:
    if report is None:
        return "No analysis report is available yet. Use SearchTranscript instead."
    needle = topic.strip().lower()
    hits = [t for t in report.topics if needle and needle in t.topic.name.lower()]
    if not hits:
        return f"No checklist topic matches '{topic}'. Use SearchTranscript instead."
    return "\n".join(
        f"{t.topic.name}: {t.coverage}, depth {t.depth}, example: {t.has_example}. {t.reason}" for t in hits
    )
