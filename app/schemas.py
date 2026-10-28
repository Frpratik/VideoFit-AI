"""Shared Pydantic data models. More models (reports, answers) are added in later phases."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

TranscriptFormat = Literal["srt", "vtt", "timestamped_text", "plain"]
Script = Literal["devanagari", "latin", "mixed", "unknown"]


class Segment(BaseModel):
    """One piece of the transcript. Original text and timestamps are never altered."""

    id: str
    start_sec: Optional[float] = None  # None when the transcript has no timestamps
    end_sec: Optional[float] = None    # None when unknown (e.g. the last pasted line)
    text: str
    script: Script = "unknown"         # writing system only; see detect_script


class Chunk(BaseModel):
    """Consecutive segments grouped for embedding. Keeps the original segments intact."""

    id: str
    segments: list[Segment]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def start_sec(self) -> Optional[float]:
        return self.segments[0].start_sec


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity, higher = more similar


class Citation(BaseModel):
    """A source shown to the user. All fields come from the real transcript, never from the LLM."""

    segment_id: str
    start_sec: Optional[float] = None
    timestamp: Optional[str] = None  # e.g. "1:02"
    url: Optional[str] = None        # clickable YouTube link at that time
    quote: str                       # the original segment text


class GroundedAnswer(BaseModel):
    """What we ask the LLM to return (structured output)."""

    evidence_found: bool = Field(description="True only if the excerpts contain the answer.")
    answer_from_transcript: str = Field(description="What the instructor said, based only on the excerpts.")
    general_knowledge_note: str = Field(default="", description="Optional extra explanation NOT from the video.")
    cited_segment_ids: list[str] = Field(default_factory=list, description="Segment IDs that support the answer.")


class ChatAnswer(BaseModel):
    """The validated answer returned to the user."""

    evidence_found: bool
    answer_from_transcript: str
    general_knowledge_note: str = ""
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


Coverage = Literal["covered", "partial", "not_detected"]
Depth = Literal["none", "mention", "explained", "in_depth"]
Level = Literal["beginner", "intermediate", "advanced"]


class Topic(BaseModel):
    name: str
    description: str = ""
    importance: Literal["core", "supporting"] = "core"


class TopicChecklist(BaseModel):
    """LLM output: what a video should cover for this learner."""

    topics: list[Topic]


class TopicAssessment(BaseModel):
    """LLM output for ONE topic, judged only from the retrieved excerpts."""

    coverage: Coverage
    depth: Depth
    has_example: bool = Field(description="True if the instructor gives a concrete example, demo or application.")
    reason: str
    evidence_segment_ids: list[str] = Field(default_factory=list)


class TopicResult(BaseModel):
    """A validated topic assessment. Citations are rebuilt from the real transcript."""

    topic: Topic
    coverage: Coverage
    depth: Depth
    has_example: bool
    reason: str
    citations: list[Citation] = Field(default_factory=list)
    assessed: bool = True  # False if the LLM failed for this topic (excluded from the score)
    from_cache: bool = False  # True if an earlier identical assessment was reused (no tokens spent)


class HolisticAssessment(BaseModel):
    """LLM output for judgements that need the whole video, not one topic."""

    level_fit: int = Field(ge=1, le=5, description="1 = badly matched to the learner's level, 5 = ideal.")
    level_reason: str
    clarity: int = Field(ge=1, le=5, description="Organization and clarity, 1 = confusing, 5 = excellent.")
    clarity_reason: str
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_next_topics: list[str] = Field(default_factory=list)


class CriterionScore(BaseModel):
    key: str
    label: str
    weight: float
    score: float  # 0-100
    explanation: str


class InsightItem(BaseModel):
    text: str
    example_comment: str = ""  # a real comment, verified to appear in the pasted text


class CommentInsights(BaseModel):
    """Audience feedback. Shown beside the report but NEVER used in the score."""

    recurring_questions: list[InsightItem] = Field(default_factory=list)
    praise: list[InsightItem] = Field(default_factory=list)
    difficulties: list[InsightItem] = Field(default_factory=list)
    complaints: list[InsightItem] = Field(default_factory=list)
    comments_analyzed: int = 0


class SuitabilityReport(BaseModel):
    goal: str
    level: Level
    overall_score: float  # 0-100, weighted in Python
    criteria: list[CriterionScore]
    topics: list[TopicResult]
    strengths: list[str]
    limitations: list[str]
    recommended_next_topics: list[str]
    caveats: list[str]
    warnings: list[str] = Field(default_factory=list)
    comment_insights: Optional[CommentInsights] = None
    reused_topics: int = 0  # how many topic assessments came from the cache


class QuestionPlan(BaseModel):
    """LLM output: rewrite the latest question so it stands alone, and say what kind it is."""

    standalone_question: str = Field(description="The question rewritten to be understandable without chat history.")
    kind: Literal["transcript", "coverage", "unrelated"] = Field(
        description="transcript = about what the video says; coverage = which topics the video "
        "covers or misses overall; unrelated = nothing to do with this video."
    )
    requested_language: Optional[Literal["English", "Hindi", "Marathi", "Hinglish"]] = Field(
        default=None,
        description="Set ONLY if the question explicitly asks for the answer in a language "
        "(e.g. 'explain in Hindi'). Otherwise null.",
    )


class ParseResult(BaseModel):
    """Everything the parser learned about the pasted transcript."""

    segments: list[Segment]
    format: TranscriptFormat
    has_timestamps: bool
    warnings: list[str] = Field(default_factory=list)
