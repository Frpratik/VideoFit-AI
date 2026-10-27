"""Scoring rules. All arithmetic happens here in Python, never inside the LLM.

The weights are an initial HEURISTIC, not a scientifically validated measure.
"""

from app.schemas import CriterionScore, TopicResult

WEIGHTS = {
    "topic_coverage": 0.40,
    "depth": 0.25,
    "examples": 0.15,
    "level_fit": 0.10,
    "clarity": 0.10,
}
LABELS = {
    "topic_coverage": "Required topic coverage",
    "depth": "Depth of explanations",
    "examples": "Practical examples and applications",
    "level_fit": "Suitability for the learner's level",
    "clarity": "Organization and clarity",
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "rubric weights must add up to 1"

COVERAGE_POINTS = {"covered": 1.0, "partial": 0.5, "not_detected": 0.0}
DEPTH_POINTS = {"none": 0.0, "mention": 0.25, "explained": 0.7, "in_depth": 1.0}
IMPORTANCE_WEIGHT = {"core": 2.0, "supporting": 1.0}


def likert_to_100(value: int) -> float:
    """Map a 1-5 rating to 0-100."""
    return (value - 1) / 4 * 100


def coverage_criterion(topics: list[TopicResult]) -> CriterionScore:
    total = sum(IMPORTANCE_WEIGHT[t.topic.importance] for t in topics)
    got = sum(IMPORTANCE_WEIGHT[t.topic.importance] * COVERAGE_POINTS[t.coverage] for t in topics)
    covered = sum(t.coverage == "covered" for t in topics)
    partial = sum(t.coverage == "partial" for t in topics)
    return _criterion(
        "topic_coverage", got / total * 100,
        f"{covered} covered, {partial} partial, {len(topics) - covered - partial} not detected "
        f"out of {len(topics)} topics (core topics count double).",
    )


def depth_criterion(topics: list[TopicResult]) -> CriterionScore:
    found = [t for t in topics if t.coverage != "not_detected"]
    if not found:
        return _criterion("depth", 0.0, "No required topic was detected, so depth could not be rated.")
    avg = sum(DEPTH_POINTS[t.depth] for t in found) / len(found)
    return _criterion("depth", avg * 100, f"Average depth over the {len(found)} detected topics.")


def examples_criterion(topics: list[TopicResult]) -> CriterionScore:
    found = [t for t in topics if t.coverage != "not_detected"]
    if not found:
        return _criterion("examples", 0.0, "No required topic was detected, so examples could not be rated.")
    with_example = sum(t.has_example for t in found)
    return _criterion("examples", with_example / len(found) * 100,
                      f"{with_example} of {len(found)} detected topics include a concrete example.")


def rating_criterion(key: str, rating: int, reason: str) -> CriterionScore:
    return _criterion(key, likert_to_100(rating), f"Rated {rating}/5. {reason}")


def overall_score(criteria: list[CriterionScore]) -> float:
    return round(sum(c.weight * c.score for c in criteria), 1)


def _criterion(key: str, score: float, explanation: str) -> CriterionScore:
    return CriterionScore(key=key, label=LABELS[key], weight=WEIGHTS[key],
                          score=round(score, 1), explanation=explanation)
