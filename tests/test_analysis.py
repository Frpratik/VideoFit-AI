import pytest

from app.analysis import rubric
from app.analysis.assessor import AnalysisError, analyze_video, sample_transcript
from app.ingestion.transcript import parse_transcript
from app.rag import store
from app.schemas import (
    HolisticAssessment, Segment, Topic, TopicAssessment, TopicChecklist, TopicResult,
)
from tests.test_rag import FakeEmbeddings

VID = "dQw4w9WgXcQ"


def result(name, importance, coverage, depth="none", example=False):
    return TopicResult(topic=Topic(name=name, importance=importance), coverage=coverage,
                       depth=depth, has_example=example, reason="r")


def test_weights_sum_to_one():
    assert sum(rubric.WEIGHTS.values()) == pytest.approx(1.0)


def test_rubric_hand_calculated_example():
    topics = [
        result("a", "core", "covered", "explained", True),
        result("b", "core", "partial", "mention", False),
        result("c", "supporting", "covered", "in_depth", True),
        result("d", "core", "not_detected"),
    ]
    cov = rubric.coverage_criterion(topics)
    assert cov.score == pytest.approx(57.1, abs=0.05)  # (2 + 1 + 1 + 0) / 7
    assert rubric.depth_criterion(topics).score == pytest.approx(65.0)
    assert rubric.examples_criterion(topics).score == pytest.approx(66.7, abs=0.05)
    criteria = [cov, rubric.depth_criterion(topics), rubric.examples_criterion(topics),
                rubric.rating_criterion("level_fit", 5, "x"), rubric.rating_criterion("clarity", 3, "x")]
    assert rubric.overall_score(criteria) == pytest.approx(64.1, abs=0.1)


def test_nothing_detected_scores_zero():
    topics = [result("a", "core", "not_detected")]
    assert rubric.coverage_criterion(topics).score == 0
    assert rubric.depth_criterion(topics).score == 0
    assert rubric.examples_criterion(topics).score == 0


def test_likert_mapping():
    assert [rubric.likert_to_100(x) for x in (1, 3, 5)] == [0, 50, 100]


def test_sample_transcript_short_and_long():
    short = [Segment(id=f"seg_{i}", start_sec=i, text="word") for i in range(5)]
    text, sampled = sample_transcript(short)
    assert not sampled and text.count("\n") == 4
    long = [Segment(id=f"seg_{i}", start_sec=i, text="w" * 50) for i in range(1000)]
    text, sampled = sample_transcript(long, max_chars=2000)
    assert sampled and len(text) <= 2200 and text.startswith("[0:00]")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(store.config, "CHROMA_DIR", str(tmp_path))
    segments = parse_transcript(open("samples/english_youtube_paste.txt", encoding="utf-8").read()).segments
    emb = FakeEmbeddings()
    store.build_index(VID, segments, emb)
    return segments, emb


def make_fake(assessments, fail_on=None):
    """Fake LLM: returns canned output per schema; topic name decides the assessment."""
    def call(schema, system, user):
        if schema is TopicChecklist:
            return TopicChecklist(topics=[Topic(name="Neuron"), Topic(name="Transformers"),
                                          Topic(name="Gradient descent", importance="supporting")])
        if schema is TopicAssessment:
            for name, value in assessments.items():
                if f"Topic: {name}" in user:
                    if fail_on == name:
                        raise TimeoutError("down")
                    return value
        if schema is HolisticAssessment:
            return HolisticAssessment(level_fit=4, level_reason="ok", clarity=4, clarity_reason="ok",
                                      strengths=["clear"], limitations=["short"], recommended_next_topics=["CNN"])
        raise AssertionError(f"unexpected call {schema}")
    return call


GOOD = {
    "Neuron": TopicAssessment(coverage="covered", depth="explained", has_example=True, reason="r",
                              evidence_segment_ids=["seg_0003"]),
    "Transformers": TopicAssessment(coverage="not_detected", depth="in_depth", has_example=True, reason="none"),
    "Gradient descent": TopicAssessment(coverage="partial", depth="mention", has_example=False, reason="r",
                                        evidence_segment_ids=["seg_0005", "seg_9999"]),
}


def test_full_analysis_end_to_end_with_fake_llm(setup):
    segments, emb = setup
    report = analyze_video(VID, segments, "learn deep learning", "beginner", emb, call=make_fake(GOOD))
    by_name = {t.topic.name: t for t in report.topics}
    assert by_name["Neuron"].citations[0].url.endswith("&t=14s")
    assert by_name["Transformers"].depth == "none" and not by_name["Transformers"].has_example  # normalised
    assert [c.segment_id for c in by_name["Gradient descent"].citations] == ["seg_0005"]  # invented ID dropped
    assert len(report.criteria) == 5 and 0 <= report.overall_score <= 100
    assert report.caveats and report.strengths == ["clear"]


def test_coverage_without_valid_evidence_is_downgraded(setup):
    segments, emb = setup
    bad = dict(GOOD)
    bad["Neuron"] = TopicAssessment(coverage="covered", depth="in_depth", has_example=True, reason="trust me",
                                    evidence_segment_ids=["seg_9999"])
    report = analyze_video(VID, segments, "goal", "beginner", emb, call=make_fake(bad))
    neuron = next(t for t in report.topics if t.topic.name == "Neuron")
    assert neuron.coverage == "not_detected" and "No valid transcript evidence" in neuron.reason


def test_failed_topic_excluded_and_reported(setup):
    segments, emb = setup
    report = analyze_video(VID, segments, "goal", "beginner", emb, call=make_fake(GOOD, fail_on="Neuron"))
    neuron = next(t for t in report.topics if t.topic.name == "Neuron")
    assert not neuron.assessed and any("could not be assessed" in w for w in report.warnings)
    assert "3 topics" not in report.criteria[0].explanation  # failed topic not counted


def test_all_topics_failing_raises(setup):
    segments, emb = setup
    def broken(schema, system, user):
        if schema is TopicChecklist:
            return TopicChecklist(topics=[Topic(name="A")])
        raise TimeoutError()
    with pytest.raises(AnalysisError):
        analyze_video(VID, segments, "goal", "beginner", emb, call=broken)


def test_empty_goal_and_custom_topics(setup):
    segments, emb = setup
    with pytest.raises(AnalysisError):
        analyze_video(VID, segments, "  ", "beginner", emb, call=make_fake(GOOD))
    report = analyze_video(VID, segments, "goal", "beginner", emb, topics=["Neuron"], call=make_fake(GOOD))
    assert [t.topic.name for t in report.topics] == ["Neuron"]
