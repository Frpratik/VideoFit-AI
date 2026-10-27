import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.analysis.assessor import AnalysisError, analyze_video
from app.chat.answerer import MESSAGES
from app.graph.analysis_graph import build_analysis_graph, run_analysis
from app.graph.chat_graph import ChatDeps, build_chat_graph
from app.ingestion.transcript import parse_transcript
from app.rag import store
from app.schemas import GroundedAnswer, QuestionPlan, Topic, TopicChecklist
from tests.test_analysis import GOOD, make_fake
from tests.test_rag import FakeEmbeddings

VID = "dQw4w9WgXcQ"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(store.config, "CHROMA_DIR", str(tmp_path))
    segments = parse_transcript(open("samples/english_youtube_paste.txt", encoding="utf-8").read()).segments
    emb = FakeEmbeddings()
    store.build_index(VID, segments, emb)
    return segments, emb


# ---------------- analysis graph ----------------

def test_graph_matches_plain_function(setup):
    segments, emb = setup
    plain = analyze_video(VID, segments, "goal", "beginner", emb, call=make_fake(GOOD))
    graph = build_analysis_graph(emb, call=make_fake(GOOD))
    assert run_analysis(graph, VID, segments, "goal", "beginner").model_dump() == plain.model_dump()


def test_checklist_retries_once_then_succeeds(setup):
    segments, emb = setup
    calls = {"n": 0}
    good = make_fake(GOOD)

    def flaky(schema, system, user):
        if schema is TopicChecklist:
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError()
        return good(schema, system, user)

    report = run_analysis(build_analysis_graph(emb, call=flaky), VID, segments, "goal", "beginner")
    assert calls["n"] == 2 and report.topics


def test_checklist_fails_twice_gives_error(setup):
    segments, emb = setup

    def always_down(schema, system, user):
        raise TimeoutError()

    with pytest.raises(AnalysisError, match="checklist"):
        run_analysis(build_analysis_graph(emb, call=always_down), VID, segments, "goal", "beginner")


def test_graph_rejects_empty_goal(setup):
    segments, emb = setup
    with pytest.raises(AnalysisError, match="goal"):
        run_analysis(build_analysis_graph(emb, call=make_fake(GOOD)), VID, segments, "  ", "beginner")


# ---------------- chat graph ----------------

def tool_call_message(query):
    return AIMessage(content="", tool_calls=[{"name": "SearchTranscript", "args": {"query": query}, "id": "t1"}])


def make_deps(emb, kind="transcript", agent_script=None, answers=None, plans=None):
    """agent_script: list of AIMessages returned in order. answers: list of GroundedAnswer in order."""
    agent_script, answers = list(agent_script or [AIMessage(content="DONE")]), list(answers or [])
    seen = {"plans": [], "agent_calls": 0, "answer_calls": 0}

    def structured(schema, system, user):
        seen["plans"].append(user)
        return QuestionPlan(standalone_question="What is a neuron?", kind=kind)

    def tool_llm(messages, with_cov):
        seen["agent_calls"] += 1
        msg = agent_script.pop(0) if len(agent_script) > 1 else agent_script[0]
        # a real model returns a NEW message every time; add_messages merges messages that share an id
        return msg.model_copy(update={"id": None})

    def answer_call(system, user):
        seen["answer_calls"] += 1
        return answers.pop(0)

    return ChatDeps(embeddings=emb, structured_call=structured, tool_llm=tool_llm, answer_call=answer_call), seen


def run_chat(graph, question, history=(), report=None, language="English"):
    msgs = list(history) + [HumanMessage(content=question)]
    return graph.invoke({"video_id": VID, "language": language, "report": report, "messages": msgs})


GOOD_ANSWER = GroundedAnswer(evidence_found=True, answer_from_transcript="Neurons multiply inputs by weights.",
                             cited_segment_ids=["seg_0003"])
NO_ANSWER = GroundedAnswer(evidence_found=False, answer_from_transcript="Not covered.")


def test_chat_happy_path_with_tool_call_and_history(setup):
    _, emb = setup
    deps, seen = make_deps(emb, agent_script=[tool_call_message("neuron"), AIMessage(content="DONE")],
                           answers=[GOOD_ANSWER])
    final = run_chat(build_chat_graph(deps), "What does a neuron do?")
    assert final["answer"].citations[0].url.endswith("&t=14s")
    assert seen["agent_calls"] == 2 and final["retrieved"]
    assert [m.type for m in final["messages"]] == ["human", "ai"]  # history grows by one turn


def test_followup_receives_previous_turns_in_plan(setup):
    _, emb = setup
    deps, seen = make_deps(emb, answers=[GOOD_ANSWER])
    history = [HumanMessage(content="What is a neuron?"), AIMessage(content="It multiplies inputs.")]
    run_chat(build_chat_graph(deps), "Explain it more", history=history)
    assert "user: What is a neuron?" in seen["plans"][0] and "assistant: It multiplies inputs." in seen["plans"][0]


def test_unrelated_question_skips_retrieval(setup):
    _, emb = setup
    deps, seen = make_deps(emb, kind="unrelated")
    final = run_chat(build_chat_graph(deps), "What's the weather?")
    assert not final["answer"].evidence_found
    assert seen["agent_calls"] == 0 and seen["answer_calls"] == 0


def test_no_evidence_triggers_exactly_one_retry(setup):
    _, emb = setup
    deps, seen = make_deps(emb, answers=[NO_ANSWER, NO_ANSWER, NO_ANSWER])
    final = run_chat(build_chat_graph(deps), "Explain transformers")
    assert seen["answer_calls"] == 2 and final["attempt"] == 2  # tried twice, then stopped


def test_retry_can_succeed(setup):
    _, emb = setup
    deps, seen = make_deps(emb, answers=[NO_ANSWER, GOOD_ANSWER])
    final = run_chat(build_chat_graph(deps), "neuron?")
    assert final["answer"].evidence_found and seen["answer_calls"] == 2


def test_runaway_agent_is_stopped(setup):
    _, emb = setup
    deps, seen = make_deps(emb, agent_script=[tool_call_message("x")], answers=[GOOD_ANSWER])
    run_chat(build_chat_graph(deps), "q")
    assert seen["agent_calls"] == 3  # MAX_TOOL_ROUNDS, then forced to answer


def test_agent_that_never_searches_gets_fallback_search(setup):
    _, emb = setup
    deps, _ = make_deps(emb, answers=[GOOD_ANSWER])
    final = run_chat(build_chat_graph(deps), "neuron")
    assert final["retrieved"]  # answer node searched on its own


def test_llm_failure_does_not_retry(setup):
    _, emb = setup

    def boom(system, user):
        raise TimeoutError()

    deps, seen = make_deps(emb)
    deps.answer_call = boom
    final = run_chat(build_chat_graph(deps), "neuron")
    assert final["attempt"] == 1 and final["answer"].warnings[0].startswith("LLM error")


def test_coverage_question_uses_report_without_llm(setup):
    segments, emb = setup
    report = analyze_video(VID, segments, "goal", "beginner", emb, call=make_fake(GOOD))
    deps, seen = make_deps(emb, kind="coverage")
    final = run_chat(build_chat_graph(deps), "Which topics are missing?", report=report)
    assert "Not detected in the supplied transcript: Transformers" in final["answer"].answer_from_transcript
    assert seen["answer_calls"] == 0


def test_coverage_question_without_report_asks_for_analysis(setup):
    _, emb = setup
    deps, _ = make_deps(emb, kind="coverage")
    final = run_chat(build_chat_graph(deps), "Which topics are missing?")
    assert "run the analysis" in final["answer"].answer_from_transcript


def test_language_requested_in_question_overrides_dropdown(setup):
    _, emb = setup
    systems = []
    deps, _ = make_deps(emb)
    deps.structured_call = lambda schema, system, user: QuestionPlan(
        standalone_question="Explain a neuron", kind="transcript", requested_language="Hindi")

    def answer_call(system, user):
        systems.append(system)
        return GOOD_ANSWER

    deps.answer_call = answer_call
    final = run_chat(build_chat_graph(deps), "Explain a neuron in Hindi", language="English")
    assert final["answer_language"] == "Hindi" and "Write the answer in Hindi" in systems[0]


def test_dropdown_language_used_when_question_names_none(setup):
    _, emb = setup
    deps, _ = make_deps(emb, answers=[GOOD_ANSWER])
    final = run_chat(build_chat_graph(deps), "What is a neuron?", language="Marathi")
    assert final["answer_language"] == "Marathi"


@pytest.mark.parametrize("language", ["Hindi", "Marathi", "Hinglish"])
def test_fixed_messages_are_localized(setup, language):
    _, emb = setup
    deps, _ = make_deps(emb, kind="unrelated")
    final = run_chat(build_chat_graph(deps), "weather?", language=language)
    assert final["answer"].answer_from_transcript == MESSAGES["unrelated"][language]
    assert final["answer"].answer_from_transcript != MESSAGES["unrelated"]["English"]


def test_every_message_exists_in_all_languages():
    for key, by_language in MESSAGES.items():
        assert set(by_language) == {"English", "Hindi", "Marathi", "Hinglish"}, key


def test_planner_failure_falls_back_to_transcript_path(setup):
    _, emb = setup
    deps, _ = make_deps(emb, answers=[GOOD_ANSWER])

    def broken(schema, system, user):
        raise TimeoutError()

    deps.structured_call = broken
    final = run_chat(build_chat_graph(deps), "What does a neuron do?")
    assert final["kind"] == "transcript" and final["answer"].evidence_found
