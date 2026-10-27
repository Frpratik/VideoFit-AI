"""Embeddings + Chroma vector store. One collection per video.

Chroma metadata can only hold simple values, so each chunk's segments (id, times, original
text) are stored as a JSON string and rebuilt into Segment objects on retrieval.
"""

import json
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app import config
from app.rag.chunking import MAX_CHARS, chunk_segments
from app.schemas import Chunk, RetrievedChunk, Segment


class E5Embeddings(Embeddings):
    """Local multilingual embeddings. E5 models need 'query: ' / 'passage: ' prefixes."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from langchain_huggingface import HuggingFaceEmbeddings

        self._prefixed = "e5" in model_name.lower()
        self._model = HuggingFaceEmbeddings(
            model_name=model_name, encode_kwargs={"normalize_embeddings": True}
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._prefixed:
            texts = ["passage: " + t for t in texts]
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query("query: " + text if self._prefixed else text)


def collection_name(video_id: str) -> str:
    """Chroma names must start and end with a letter or digit."""
    return f"vf_{video_id}_v1"


def _open(video_id: str, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=collection_name(video_id),
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(video_id: str, segments: list[Segment], embeddings: Embeddings, max_chars: int = MAX_CHARS) -> int:
    """(Re)build the vector index for a video. Returns the number of chunks stored."""
    store = _open(video_id, embeddings)
    store.delete_collection()  # always start clean so old transcripts never mix in
    store = _open(video_id, embeddings)

    chunks = chunk_segments(segments, max_chars=max_chars)
    store.add_texts(
        texts=[c.text for c in chunks],
        ids=[c.id for c in chunks],
        metadatas=[{"segments": json.dumps([s.model_dump() for s in c.segments], ensure_ascii=False)} for c in chunks],
    )
    return len(chunks)


def search(video_id: str, query: str, embeddings: Embeddings, k: int = 5) -> list[RetrievedChunk]:
    """Return the k chunks most similar to the query, best first."""
    store = _open(video_id, embeddings)
    results = store.similarity_search_with_score(query, k=k)
    found = []
    for doc, distance in results:
        segments = [Segment(**s) for s in json.loads(doc.metadata["segments"])]
        found.append(RetrievedChunk(chunk=Chunk(id=_chunk_id(doc, segments), segments=segments), score=1 - distance))
    return found


def _chunk_id(doc, segments: list[Segment]) -> str:
    return doc.id or segments[0].id
