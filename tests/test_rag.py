import hashlib

import pytest
from langchain_core.embeddings import Embeddings

from app.chat.answerer import answer_question, build_citations, format_context
from app.ingestion.transcript import parse_transcript
from app.rag import store
from app.rag.chunking import chunk_segments
from app.schemas import Chunk, GroundedAnswer, RetrievedChunk, Segment

VID = "dQw4w9WgXcQ"


def seg(i, start, text):
    return Segment(id=f"seg_{i:04d}", start_sec=start, text=text)


class FakeEmbeddings(Embeddings):
    """Bag-of-words hashing embeddings: deterministic, no model download."""

    def _vec(self, text):
        v = [0.0] * 64
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_chunking_overlap_and_order():
    segs = [seg(i, i * 5, "x" * 200) for i in range(1, 7)]
    chunks = chunk_segments(segs, max_chars=450, overlap=1)
    assert [s.id for s in chunks[0].segments] == ["seg_0001", "seg_0002"]
    assert chunks[1].segments[0].id == "seg_0002"  # overlap
    assert chunks[-1].segments[-1].id == "seg_0006"  # nothing lost
    assert all(c.segments for c in chunks)


def test_chunking_empty_and_single():
    assert chunk_segments([]) == []
    assert len(chunk_segments([seg(1, 0, "hello")])) == 1


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    monkeypatch.setattr(store.config, "CHROMA_DIR", str(tmp_path))
    raw = open("samples/english_youtube_paste.txt", encoding="utf-8").read()
    segments = parse_transcript(raw).segments
    emb = FakeEmbeddings()
    store.build_index(VID, segments, emb)
    return segments, emb


def test_search_returns_original_segments_with_timestamps(indexed):
    segments, emb = indexed
    results = store.search(VID, "gradient descent simple example", emb, k=2)
    top = results[0].chunk
    assert any("gradient descent" in s.text for s in top.segments)
    original = {s.id: s for s in segments}
    for s in top.segments:  # ids, text and times survive the vector store round trip
        assert s == original[s.id]


def test_rebuild_does_not_mix_old_data(indexed):
    _, emb = indexed
    store.build_index(VID, [seg(1, 0, "totally different transcript")], emb)
    results = store.search(VID, "gradient descent", emb, k=5)
    assert len(results) == 1


def retrieved_for_tests():
    chunk = Chunk(id="chunk_0001", segments=[seg(1, 65, "we cover CNNs"), seg(2, 70, "filters slide over images")])
    return [RetrievedChunk(chunk=chunk, score=0.9)]


def test_build_citations_drops_invented_ids():
    cites, warnings = build_citations(["seg_0002", "seg_9999", "seg_0002"], retrieved_for_tests(), VID)
    assert len(cites) == 1  # invented and duplicate removed
    assert cites[0].quote == "filters slide over images"  # quote comes from the transcript
    assert cites[0].timestamp == "1:10"
    assert cites[0].url == f"https://www.youtube.com/watch?v={VID}&t=70s"
    assert "Removed 1" in warnings[0]


def test_citation_without_timestamp_has_no_link():
    chunk = Chunk(id="c", segments=[Segment(id="seg_0001", text="plain text")])
    cites, _ = build_citations(["seg_0001"], [RetrievedChunk(chunk=chunk, score=0.5)], VID)
    assert cites[0].url is None and cites[0].timestamp is None


def test_format_context_lists_each_segment_once():
    c1 = Chunk(id="a", segments=[seg(1, 0, "one"), seg(2, 5, "two")])
    c2 = Chunk(id="b", segments=[seg(2, 5, "two"), seg(3, 9, "three")])
    ctx = format_context([RetrievedChunk(chunk=c1, score=1), RetrievedChunk(chunk=c2, score=1)])
    assert ctx.count("seg_0002") == 1 and "[seg_0003 | 0:09] three" in ctx


def test_answer_question_validates_and_passes_language_and_history():
    seen = {}

    def fake_llm(system, user):
        seen["system"], seen["user"] = system, user
        return GroundedAnswer(
            evidence_found=True, answer_from_transcript="CNNs use filters.",
            cited_segment_ids=["seg_0002", "seg_0042"],
        )

    ans = answer_question("What is a CNN?", retrieved_for_tests(), VID, "Hindi",
                          history=[("user", "hi")], llm_call=fake_llm)
    assert "Hindi" in seen["system"] and "user: hi" in seen["user"]
    assert [c.segment_id for c in ans.citations] == ["seg_0002"]
    assert ans.warnings


def test_empty_answer_gets_default_refusal():
    def empty(system, user):
        return GroundedAnswer(evidence_found=True, answer_from_transcript="  ")

    ans = answer_question("q", retrieved_for_tests(), VID, llm_call=empty)
    assert not ans.evidence_found and "does not contain" in ans.answer_from_transcript


def test_answer_question_handles_llm_failure():
    def broken(system, user):
        raise TimeoutError("boom secret transcript text")

    ans = answer_question("q", retrieved_for_tests(), VID, llm_call=broken)
    assert not ans.evidence_found
    assert "TimeoutError" in ans.warnings[0] and "secret" not in ans.warnings[0]


def test_answer_question_no_retrieval_does_not_call_llm():
    def must_not_call(system, user):
        raise AssertionError("LLM should not be called")

    ans = answer_question("q", [], VID, llm_call=must_not_call)
    assert not ans.evidence_found and not ans.citations
