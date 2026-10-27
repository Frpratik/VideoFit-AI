"""Command-line demo of Phase 4.

Example:
  python -m scripts.analyze samples/english_youtube_paste.txt "https://youtu.be/dQw4w9WgXcQ" "Learn deep learning basics" beginner English
"""

import sys

from app.analysis.assessor import analyze_video
from app.ingestion.transcript import parse_transcript
from app.ingestion.youtube import extract_video_id
from app.rag.store import E5Embeddings, build_index


def main(path: str, url: str, goal: str, level: str = "beginner", language: str = "English") -> None:
    video_id = extract_video_id(url)
    parsed = parse_transcript(open(path, encoding="utf-8").read())
    embeddings = E5Embeddings()
    build_index(video_id, parsed.segments, embeddings)

    report = analyze_video(video_id, parsed.segments, goal, level, embeddings, language)

    print(f"\nOVERALL SCORE: {report.overall_score}/100  (goal: {goal}, level: {level})")
    for c in report.criteria:
        print(f"  {c.label:<40} {c.score:>5.1f}  x{c.weight:.2f}  {c.explanation}")
    print("\nTOPICS")
    for t in report.topics:
        marks = f"{t.coverage}/{t.depth}" + ("/example" if t.has_example else "")
        print(f"  [{t.topic.importance}] {t.topic.name}: {marks}\n      {t.reason}")
        for c in t.citations:
            print(f"      - [{c.timestamp}] {c.url}  \"{c.quote}\"")
    for title, items in [("STRENGTHS", report.strengths), ("LIMITATIONS", report.limitations),
                         ("NEXT TOPICS", report.recommended_next_topics), ("WARNINGS", report.warnings),
                         ("CAVEATS", report.caveats)]:
        print(f"\n{title}")
        for item in items:
            print("  -", item)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:6])
