import pytest
from fastapi.testclient import TestClient

from app.analysis import assessor
from app.analysis.assessor import AnalysisError, assess_topic, make_checklist
from app.analysis.cache import AssessmentCache, transcript_hash
from app.api import main
from app.chat.answerer import answer_question
from app.graph.analysis_graph import build_analysis_graph, run_analysis
from app.ingestion.transcript import parse_transcript
from app.rag import store
from app.schemas import (
    Chunk, HolisticAssessment, RetrievedChunk, Segment, Topic, TopicAssessment, TopicChecklist, TopicResult,
)
from tests.test_rag import FakeEmbeddings

VID = "dQw4w9WgXcQ"
TEXT = open("samples/english_youtube_paste.txt", encoding="utf-8").read()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(store.config, "CHROMA_DIR", str(tmp_path / "index"))
    segments = parse_transcript(TEXT).segments
    emb = FakeEmbeddings()
    store.build_index(VID, segments, emb)
    return segments, emb, tmp_path / "cache.json"


class CountingLlm:
    """Fake model that records how many times each kind of call happens."""

    def __init__(self):
        self.calls = []

    def __call__(self, schema, system, user):
        self.calls.append(schema.__name__)
        if schema is TopicAssessment:
            return TopicAssessment(coverage="covered", depth="explained", has_example=True, reason="r",
                                   evidence_segment_ids=["seg_0003"])
        if schema is HolisticAssessment:
            return HolisticAssessment(level_fit=4, level_reason="ok", clarity=4, clarity_reason="ok")
        if schema is TopicChecklist:
            return TopicChecklist(topics=[Topic(name=f"Topic {i}") for i in range(12)])
        raise AssertionError(schema)


# ---------------- topic cap ----------------
def test_checklist_is_capped_and_the_prompt_says_how_many():
    seen = {}

    def call(schema, system, user):
        seen["system"] = system
        return TopicChecklist(topics=[Topic(name=f"T{i}") for i in range(12)])

    assert len(make_checklist("goal", "beginner", call)) == assessor.DEFAULT_TOPICS == 8
    assert len(make_checklist("goal", "beginner", call, max_topics=5)) == 5 and "up to 5" in seen["system"]
    assert len(make_checklist("goal", "beginner", call, max_topics=99)) == assessor.MAX_TOPICS


# ---------------- cache ----------------
def test_second_assessment_comes_from_the_cache_without_calling_the_model(setup):
    segments, emb, path = setup
    cache, llm_call, h = AssessmentCache(path), CountingLlm(), transcript_hash(segments)
    topic = Topic(name="Neuron")
    first = assess_topic(topic, VID, emb, "English", llm_call, cache=cache, text_hash=h)
    second = assess_topic(topic, VID, emb, "English", llm_call, cache=cache, text_hash=h)
    assert llm_call.calls == ["TopicAssessment"] and not first.from_cache and second.from_cache
    assert second.coverage == first.coverage and second.citations == first.citations


@pytest.mark.parametrize("change", ["language", "video", "text", "topic", "model"])
def test_anything_that_can_change_the_answer_is_a_cache_miss(setup, monkeypatch, change):
    segments, emb, path = setup
    cache, llm_call, h = AssessmentCache(path), CountingLlm(), transcript_hash(segments)
    assess_topic(Topic(name="Neuron"), VID, emb, "English", llm_call, cache=cache, text_hash=h)
    kwargs = {"topic": Topic(name="Neuron"), "video_id": VID, "language": "English", "text_hash": h}
    if change == "language":
        kwargs["language"] = "Hindi"
    elif change == "video":
        store.build_index("aaaaaaaaaaa", segments, emb)
        kwargs["video_id"] = "aaaaaaaaaaa"
    elif change == "text":
        kwargs["text_hash"] = "different"
    elif change == "topic":
        kwargs["topic"] = Topic(name="Neuron", description="more specific")
    else:
        monkeypatch.setattr(assessor.config, "LLM_MODEL", "another-model")
    result = assess_topic(kwargs["topic"], kwargs["video_id"], emb, kwargs["language"], llm_call,
                          cache=cache, text_hash=kwargs["text_hash"])
    assert not result.from_cache and len(llm_call.calls) == 2


def test_failed_assessments_are_never_cached(setup):
    segments, emb, path = setup
    cache, h = AssessmentCache(path), transcript_hash(segments)

    def failing(schema, system, user):
        raise TimeoutError()

    assert not assess_topic(Topic(name="Neuron"), VID, emb, "English", failing, cache=cache, text_hash=h).assessed
    assert len(cache) == 0


def test_cache_survives_a_restart_and_a_corrupt_file_is_ignored(setup):
    segments, emb, path = setup
    h, llm_call = transcript_hash(segments), CountingLlm()
    assess_topic(Topic(name="Neuron"), VID, emb, "English", llm_call, cache=AssessmentCache(path), text_hash=h)
    reopened = AssessmentCache(path)
    assert len(reopened) == 1
    assert assess_topic(Topic(name="Neuron"), VID, emb, "English", llm_call, cache=reopened, text_hash=h).from_cache
    path.write_text("{ not json", encoding="utf-8")
    assert len(AssessmentCache(path)) == 0


def test_cached_topic_takes_the_callers_current_importance(setup):
    segments, emb, path = setup
    cache, h = AssessmentCache(path), transcript_hash(segments)
    assess_topic(Topic(name="Neuron", importance="core"), VID, emb, "English", CountingLlm(), cache=cache, text_hash=h)
    hit = assess_topic(Topic(name="Neuron", importance="supporting"), VID, emb, "English", CountingLlm(),
                       cache=cache, text_hash=h)
    assert hit.from_cache and hit.topic.importance == "supporting"


def test_rerunning_the_same_analysis_only_pays_for_the_overall_review(setup):
    segments, emb, path = setup
    llm_call, cache = CountingLlm(), AssessmentCache(path)
    graph = build_analysis_graph(emb, call=llm_call, cache=cache)
    topics = ["Neuron", "Gradient descent", "Dropout"]
    first = run_analysis(graph, VID, segments, "goal", "beginner", topics=topics)
    assert llm_call.calls.count("TopicAssessment") == 3 and first.reused_topics == 0
    llm_call.calls.clear()
    second = run_analysis(graph, VID, segments, "goal", "beginner", topics=topics)
    assert llm_call.calls == ["HolisticAssessment"] and second.reused_topics == 3
    assert second.overall_score == first.overall_score


def test_cache_key_changes_with_the_transcript():
    a = [Segment(id="seg_0001", start_sec=0, text="hello")]
    b = [Segment(id="seg_0001", start_sec=0, text="hello!")]
    assert transcript_hash(a) == transcript_hash(a) and transcript_hash(a) != transcript_hash(b)


# ---------------- API ----------------
@pytest.fixture
def client(setup, monkeypatch):
    segments, emb, path = setup
    llm_call = CountingLlm()
    monkeypatch.setattr(main, "get_embeddings", lambda: emb)
    monkeypatch.setattr(main, "make_analysis_graph", lambda: build_analysis_graph(emb, call=llm_call, cache=AssessmentCache(path)))
    main.SESSIONS.clear()
    return TestClient(main.app)


def test_analyze_reports_usage_and_validates_the_topic_count(client):
    sid = client.post("/api/video", json={"url": "https://youtu.be/dQw4w9WgXcQ", "transcript": TEXT}).json()["session_id"]
    body = {"session_id": sid, "goal": "learn", "max_topics": 5}
    ok = client.post("/api/analyze", json=body)
    assert ok.status_code == 200 and set(ok.json()["usage"]) == {"calls", "input_tokens", "output_tokens"}
    assert len(ok.json()["report"]["topics"]) == 5
    assert client.post("/api/analyze", json={**body, "max_topics": 2}).status_code == 422
    assert client.post("/api/analyze", json={**body, "max_topics": 13}).status_code == 422
    again = client.post("/api/analyze", json=body).json()["report"]
    assert again["reused_topics"] == 5


def test_full_topic_specs_keep_description_and_importance_and_hit_the_cache(client):
    sid = client.post("/api/video", json={"url": "https://youtu.be/dQw4w9WgXcQ", "transcript": TEXT}).json()["session_id"]
    specs = [{"name": "Neuron", "description": "weights and bias", "importance": "supporting"},
             {"name": "Gradient descent", "description": "", "importance": "core"}]
    body = {"session_id": sid, "goal": "learn", "topic_specs": specs}
    first = client.post("/api/analyze", json=body).json()["report"]
    second = client.post("/api/analyze", json=body).json()["report"]
    assert [t["topic"] for t in first["topics"]] == specs
    assert first["reused_topics"] == 0 and second["reused_topics"] == 2
    assert [c["score"] for c in first["criteria"][:3]] == [c["score"] for c in second["criteria"][:3]]
    # names only (what the textbox sends) is a different checklist: descriptions are missing, so no reuse
    plain = client.post("/api/analyze", json={"session_id": sid, "goal": "learn", "topics": ["Neuron"]}).json()["report"]
    assert plain["reused_topics"] == 0 and plain["topics"][0]["topic"]["importance"] == "core"


def test_normalize_topics_accepts_strings_and_topic_objects():
    out = assessor.normalize_topics([" A ", Topic(name="B", importance="supporting"), "  ", Topic(name=" ")])
    assert [(t.name, t.importance) for t in out] == [("A", "core"), ("B", "supporting")]


# ---------------- friendlier rate-limit message ----------------
def test_rate_limit_gets_its_own_message():
    class RateLimitError(Exception):
        pass

    def limited(system, user):
        raise RateLimitError("429")

    chunk = Chunk(id="c", segments=[Segment(id="seg_0001", start_sec=1, text="text")])
    answer = answer_question("q", [RetrievedChunk(chunk=chunk, score=1)], VID, llm_call=limited)
    assert "rate limit" in answer.answer_from_transcript and answer.warnings == ["LLM error: RateLimitError"]
