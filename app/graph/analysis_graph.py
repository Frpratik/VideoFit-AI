"""Learning-suitability analysis workflow (LangGraph).

  (score -> comments -> END   when audience comments were supplied; otherwise score -> END)
  checklist -(failed, first try)-> checklist (retry)
            -(failed twice)-----> END (error)
            -> assess -(nothing assessed)-> END (error)
                      -> review -(failed)-> END (error)
                               -> score -> END

The node bodies reuse the tested helpers in app.analysis; this module only adds state and routing.
"""

from typing import Callable, Optional, TypedDict, Union

from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app import llm
from app.analysis import rubric
from app.analysis.comments import CommentError, analyze_comments
from app.analysis.assessor import (
    CAVEATS, DEFAULT_TOPICS, AnalysisError, assess_topic, holistic_review, make_checklist, normalize_topics,
)
from app.analysis.cache import AssessmentCache, transcript_hash
from app.schemas import HolisticAssessment, Level, Segment, SuitabilityReport, Topic, TopicResult

LlmCall = Callable[[type[BaseModel], str, str], BaseModel]
MAX_CHECKLIST_TRIES = 2


class AnalysisState(TypedDict, total=False):
    # inputs
    video_id: str
    segments: list[Segment]
    goal: str
    level: Level
    language: str
    custom_topics: Optional[list[Union[str, Topic]]]
    max_topics: int
    text_hash: str  # fingerprint of the transcript, used by the assessment cache
    # working data
    topics: list[Topic]
    results: list[TopicResult]
    holistic: HolisticAssessment
    warnings: list[str]
    attempt: int
    comments_text: Optional[str]
    # outputs
    report: SuitabilityReport
    error: Optional[str]


def build_analysis_graph(embeddings: Embeddings, call: LlmCall = llm.call_structured,
                         cache: Optional[AssessmentCache] = None):
    def checklist_node(state: AnalysisState) -> dict:
        """Use the learner's own topics, or ask the LLM for a checklist."""
        attempt = state.get("attempt", 0) + 1
        if not state["goal"].strip():
            return {"error": "Please describe your learning goal.", "attempt": MAX_CHECKLIST_TRIES}
        if state.get("custom_topics"):
            return {"topics": normalize_topics(state["custom_topics"]), "error": None, "attempt": attempt}
        try:
            topics = make_checklist(state["goal"], state["level"], call, state.get("max_topics", DEFAULT_TOPICS))
            return {"topics": topics, "error": None, "attempt": attempt}
        except Exception as exc:
            return {"error": f"Could not build the topic checklist ({type(exc).__name__}).", "attempt": attempt}

    def assess_node(state: AnalysisState) -> dict:
        """Assess every topic against its own retrieved evidence."""
        results = [assess_topic(t, state["video_id"], embeddings, state["language"], call,
                                cache=cache, text_hash=state.get("text_hash", "")) for t in state["topics"]]
        scored = [r for r in results if r.assessed]
        if not scored:
            return {"results": results, "error": "The language model failed on every topic. Please try again later."}
        warnings = []
        if len(scored) < len(results):
            warnings.append(f"{len(results) - len(scored)} topic(s) could not be assessed and were left out of the score.")
        return {"results": results, "warnings": warnings}

    def review_node(state: AnalysisState) -> dict:
        """One whole-video review: level fit, clarity, strengths, limitations, next topics."""
        scored = [r for r in state["results"] if r.assessed]
        try:
            holistic, sampled = holistic_review(state["segments"], state["goal"], state["level"], scored,
                                                state["language"], call)
        except Exception as exc:
            return {"error": f"The overall review failed ({type(exc).__name__}). Please try again."}
        warnings = list(state.get("warnings", []))
        if sampled:
            warnings.append("The transcript is long, so level fit and clarity were rated from an evenly spaced sample.")
        return {"holistic": holistic, "warnings": warnings}

    def score_node(state: AnalysisState) -> dict:
        """All arithmetic happens here, in Python."""
        scored, h = [r for r in state["results"] if r.assessed], state["holistic"]
        criteria = [
            rubric.coverage_criterion(scored), rubric.depth_criterion(scored), rubric.examples_criterion(scored),
            rubric.rating_criterion("level_fit", h.level_fit, h.level_reason),
            rubric.rating_criterion("clarity", h.clarity, h.clarity_reason),
        ]
        report = SuitabilityReport(
            goal=state["goal"], level=state["level"], overall_score=rubric.overall_score(criteria),
            criteria=criteria, topics=state["results"], strengths=h.strengths, limitations=h.limitations,
            recommended_next_topics=h.recommended_next_topics, caveats=CAVEATS, warnings=state.get("warnings", []),
            reused_topics=sum(r.from_cache for r in state["results"]),
        )
        return {"report": report}

    def comments_node(state: AnalysisState) -> dict:
        """Optional branch: summarize audience comments. A failure here never breaks the report."""
        report = state["report"]
        try:
            insights, warnings = analyze_comments(state["comments_text"], state["language"], call)
            return {"report": report.model_copy(update={
                "comment_insights": insights, "warnings": report.warnings + warnings})}
        except CommentError as exc:
            note = str(exc)
        except Exception as exc:
            note = f"Comment analysis failed ({type(exc).__name__})."
        return {"report": report.model_copy(update={"warnings": report.warnings + [note]})}

    def route_after_score(state: AnalysisState) -> str:
        return "comments" if (state.get("comments_text") or "").strip() else "end"

    def route_after_checklist(state: AnalysisState) -> str:
        if not state.get("error"):
            return "assess"
        return "checklist" if state["attempt"] < MAX_CHECKLIST_TRIES else "end"

    def stop_on_error(next_node: str):
        return lambda state: "end" if state.get("error") else next_node

    g = StateGraph(AnalysisState)
    g.add_node("checklist", checklist_node)
    g.add_node("assess", assess_node)
    g.add_node("review", review_node)
    g.add_node("score", score_node)
    g.add_node("comments", comments_node)
    g.add_edge(START, "checklist")
    g.add_conditional_edges("checklist", route_after_checklist, {"assess": "assess", "checklist": "checklist", "end": END})
    g.add_conditional_edges("assess", stop_on_error("review"), {"review": "review", "end": END})
    g.add_conditional_edges("review", stop_on_error("score"), {"score": "score", "end": END})
    g.add_conditional_edges("score", route_after_score, {"comments": "comments", "end": END})
    g.add_edge("comments", END)
    return g.compile()


def run_analysis(graph, video_id: str, segments: list[Segment], goal: str, level: Level,
                 language: str = "English", topics: Optional[list[Union[str, Topic]]] = None,
                 comments: Optional[str] = None, max_topics: int = DEFAULT_TOPICS) -> SuitabilityReport:
    """Invoke the graph and turn a recorded error into an AnalysisError."""
    final = graph.invoke({"video_id": video_id, "segments": segments, "goal": goal, "level": level,
                          "language": language, "custom_topics": topics, "comments_text": comments,
                          "max_topics": max_topics, "text_hash": transcript_hash(segments)})
    if final.get("error"):
        raise AnalysisError(final["error"])
    return final["report"]
