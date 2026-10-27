import pytest

from app import llm
from app.schemas import QuestionPlan


def patch(monkeypatch, outcomes):
    calls = []

    def fake(schema, system, user, temperature):
        calls.append(1)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm, "_invoke_once", fake)
    monkeypatch.setattr(llm, "RETRY_DELAY", 0)
    monkeypatch.setattr(llm.config, "GROQ_API_KEY", "test-key")
    return calls


def test_retries_once_after_a_failure(monkeypatch):
    ok = QuestionPlan(standalone_question="q", kind="transcript")
    calls = patch(monkeypatch, [ValueError("tool_use_failed"), ok])
    assert llm.call_structured(QuestionPlan, "s", "u") is ok and len(calls) == 2


def test_raises_after_two_failures(monkeypatch):
    calls = patch(monkeypatch, [ValueError("a"), ValueError("b")])
    with pytest.raises(ValueError, match="b"):
        llm.call_structured(QuestionPlan, "s", "u")
    assert len(calls) == 2


def test_missing_key_fails_fast(monkeypatch):
    monkeypatch.setattr(llm.config, "GROQ_API_KEY", "")
    with pytest.raises(RuntimeError):
        llm.call_structured(QuestionPlan, "s", "u")
