"""Command-line demo of Phase 3.

Example:
  python -m scripts.ask samples/english_youtube_paste.txt "https://youtu.be/dQw4w9WgXcQ" "What is gradient descent?" English
"""

import sys

from app.chat.answerer import answer_question
from app.ingestion.transcript import parse_transcript
from app.ingestion.youtube import extract_video_id
from app.rag.store import E5Embeddings, build_index, search


def main(path: str, url: str, question: str, language: str = "English") -> None:
    video_id = extract_video_id(url)
    parsed = parse_transcript(open(path, encoding="utf-8").read())
    for warning in parsed.warnings:
        print("WARNING:", warning)

    embeddings = E5Embeddings()
    print(f"Indexed {build_index(video_id, parsed.segments, embeddings)} chunks")

    retrieved = search(video_id, question, embeddings, k=4)
    for r in retrieved:
        print(f"  retrieved {r.chunk.id} score={r.score:.3f}")

    ans = answer_question(question, retrieved, video_id, language)
    print("\nEvidence found:", ans.evidence_found)
    print("Answer:", ans.answer_from_transcript)
    if ans.general_knowledge_note:
        print("Outside the video:", ans.general_knowledge_note)
    for c in ans.citations:
        print(f"  [{c.timestamp}] {c.url}\n      \"{c.quote}\"")
    for w in ans.warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
