"""Group transcript segments into chunks for embedding.

Segments from YouTube are tiny (a few words). Embedding each alone loses context, so we
group consecutive segments up to a size limit, repeating the last segment of a chunk at
the start of the next one (overlap) so an idea split at a boundary is still found.
"""

from app.schemas import Chunk, Segment

MAX_CHARS = 500


def chunk_segments(segments: list[Segment], max_chars: int = MAX_CHARS, overlap: int = 1) -> list[Chunk]:
    """Split segments into chunks of at most ~max_chars characters (a single long segment stays whole)."""
    chunks: list[Chunk] = []
    current: list[Segment] = []
    size = 0

    for seg in segments:
        if current and size + len(seg.text) > max_chars:
            chunks.append(Chunk(id=f"chunk_{len(chunks) + 1:04d}", segments=current))
            current = current[-overlap:] if overlap else []
            size = sum(len(s.text) for s in current)
        current.append(seg)
        size += len(seg.text)

    if current:  # always holds at least one segment not yet in a chunk
        chunks.append(Chunk(id=f"chunk_{len(chunks) + 1:04d}", segments=current))
    return chunks
