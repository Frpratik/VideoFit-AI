import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.analysis.comments import CommentError, analyze_comments, parse_comments
from app.api import main
from app.graph.analysis_graph import build_analysis_graph
from app.graph.chat_graph import ChatDeps, build_chat_graph
from app.rag import store
from app.schemas import CommentInsights, GroundedAnswer, InsightItem, QuestionPlan
from tests.test_analysis import GOOD, make_fake
from tests.test_rag import FakeEmbeddings

COMMENTS = """Great explanation, thank you!
Can you make a video on backpropagation math?
The part about learning rate was confusing for me
Audio quality is bad in the second half
Great explanation, thank you!
ok"""


def fake_insights(schema, system, user):
    return CommentInsights(
        recurring_questions=[InsightItem(text="Wants backprop math", example_comment="Can you make a video on backpropagation math?")],
        praise=[InsightItem(text="Clear explanation", example_comment="THIS COMMENT WAS INVENTED")],
    )


# ---------------- comments ----------------
def test_parse_comments_dedupes_and_drops_tiny_lines():
    comments, warnings = parse_comments(COMMENTS)
    assert len(comments) == 4 and not warnings


def test_parse_comments_caps_length_and_count():
    comments, warnings = parse_comments("\n".join(f"comment number {i} " + "x" * 600 for i in range(200)))
    assert len(comments) == 150 and all(len(c) <= 400 for c in comments) and warnings


def test_invented_example_comments_are_removed():
    insights, warnings = analyze_comments(COMMENTS, "English", fake_insights)
    assert insights.recurring_questions[0].example_comment.startswith("Can you make")
    assert insights.praise[0].example_comment == "" and "could not be verified" in warnings[0]
    assert insights.comments_analyzed == 4


def test_too_few_comments_rejected():
    with pytest.raises(CommentError):
        analyze_comments("only one comment here", "English", fake_insights)


# ---------------- API ----------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store.config, "CHROMA_DIR", str(tmp_path))
    emb = FakeEmbeddings()
    analysis_llm = make_fake(GOOD)

    def combined(schema, system, user):
        return fake_insights(schema, system, user) if schema is CommentInsights else analysis_llm(schema, system, user)

    monkeypatch.setattr(main, "get_embeddings", lambda: emb)
    monkeypatch.setattr(main, "make_analysis_graph", lambda: build_analysis_graph(emb, call=combined))
    deps = ChatDeps(
        embeddings=emb,
        structured_call=lambda schema, system, user: QuestionPlan(standalone_question="What is a neuron?", kind="transcript"),
        tool_llm=lambda messages, cov: AIMessage(content="DONE"),
        answer_call=lambda system, user: GroundedAnswer(evidence_found=True, answer_from_transcript="Neurons multiply inputs.",
                                                        cited_segment_ids=["seg_0003"]),
    )
    monkeypatch.setattr(main, "make_chat_graph", lambda: build_chat_graph(deps))
    main.SESSIONS.clear()
    return TestClient(main.app)


URL = "https://youtu.be/dQw4w9WgXcQ"
TEXT = open("samples/english_youtube_paste.txt", encoding="utf-8").read()


def load(client):
    r = client.post("/api/video", json={"url": URL, "transcript": TEXT})
    assert r.status_code == 200, r.text
    return r.json()


def test_health_and_ui_are_served(client):
    assert client.get("/api/health").json()["languages"] == ["English", "Hindi", "Marathi", "Hinglish"]
    page = client.get("/")
    assert page.status_code == 200 and "VideoFit" in page.text
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/style.css").status_code == 200


def test_demo_sample_endpoint(client):
    assert "neural" in client.get("/api/sample").json()["transcript"].lower()


@pytest.mark.parametrize("payload,message", [
    ({"url": "https://example.com", "transcript": TEXT}, "valid YouTube"),
    ({"url": URL, "transcript": "   "}, "empty"),
])
def test_bad_video_input_is_a_400_with_a_readable_message(client, payload, message):
    r = client.post("/api/video", json=payload)
    assert r.status_code == 400 and message in r.json()["detail"]


def test_load_analyze_and_chat_flow(client):
    info = load(client)
    assert info["has_timestamps"] and info["segments"] == 5 and info["duration_sec"] > 0

    r = client.post("/api/analyze", json={"session_id": info["session_id"], "goal": "learn deep learning",
                                          "level": "beginner", "language": "English", "comments": COMMENTS})
    assert r.status_code == 200, r.text
    report = r.json()["report"]
    assert 0 <= report["overall_score"] <= 100 and len(report["criteria"]) == 5
    assert report["comment_insights"]["comments_analyzed"] == 4
    assert report["topics"][0]["citations"][0]["start_sec"] is not None

    # comments must not change the score
    r2 = client.post("/api/analyze", json={"session_id": info["session_id"], "goal": "learn deep learning",
                                           "level": "beginner", "language": "English"})
    assert r2.json()["report"]["overall_score"] == report["overall_score"]
    assert r2.json()["report"]["comment_insights"] is None

    chat = client.post("/api/chat", json={"session_id": info["session_id"], "question": "What does a neuron do?"})
    assert chat.status_code == 200
    body = chat.json()
    assert body["answer"]["evidence_found"] and body["answer"]["citations"][0]["url"].endswith("&t=14s")
    assert client.post("/api/chat/reset", json={"session_id": info["session_id"]}).json() == {"ok": True}


def test_unknown_session_and_bad_language(client):
    assert client.post("/api/chat", json={"session_id": "nope", "question": "hi"}).status_code == 404
    sid = load(client)["session_id"]
    assert client.post("/api/chat", json={"session_id": sid, "question": "hi", "language": "Klingon"}).status_code == 422
    assert client.post("/api/chat", json={"session_id": sid, "question": ""}).status_code == 422


def test_empty_goal_is_a_422(client):
    sid = load(client)["session_id"]
    r = client.post("/api/analyze", json={"session_id": sid, "goal": "  "})
    assert r.status_code == 422 and "goal" in r.json()["detail"]


def test_only_five_sessions_are_kept(client):
    ids = [load(client)["session_id"] for _ in range(7)]
    assert len(main.SESSIONS) == 5 and ids[0] not in main.SESSIONS and ids[-1] in main.SESSIONS
