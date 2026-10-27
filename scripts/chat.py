"""Command-line demo of Phase 5: ask several questions in a row (history is kept between them).

Example:
  python -m scripts.chat samples/english_youtube_paste.txt "https://youtu.be/dQw4w9WgXcQ" English --analyze "What does a neuron do?" "Can you explain that more simply?" "Which topics are missing?" "What's the weather today?"
"""

import sys

from langchain_core.messages import HumanMessage

from app.graph.analysis_graph import build_analysis_graph, run_analysis
from app.graph.chat_graph import ChatDeps, build_chat_graph
from app.ingestion.transcript import parse_transcript
from app.ingestion.youtube import extract_video_id
from app.rag.store import E5Embeddings, build_index


def main(path: str, url: str, language: str, analyze: bool, questions: list[str]) -> None:
    video_id = extract_video_id(url)
    parsed = parse_transcript(open(path, encoding="utf-8").read())
    embeddings = E5Embeddings()
    build_index(video_id, parsed.segments, embeddings)

    report = None
    if analyze:
        graph = build_analysis_graph(embeddings)
        report = run_analysis(graph, video_id, parsed.segments, "Learn deep learning basics", "beginner", language)
        print(f"[analysis done: score {report.overall_score}]")

    chat = build_chat_graph(ChatDeps(embeddings=embeddings))
    messages = []
    for q in questions:
        final = chat.invoke({"video_id": video_id, "language": language, "report": report,
                             "messages": messages + [HumanMessage(content=q)]})
        messages = final["messages"]  # this list IS the conversation history
        ans = final["answer"]
        print(f"\nQ: {q}\n   (understood as: {final['standalone_question']!r}, kind={final['kind']}, "
              f"search rounds={final['steps']}, attempts={final['attempt']})")
        print("A:", ans.answer_from_transcript)
        for c in ans.citations:
            print(f"   [{c.timestamp}] {c.url}  \"{c.quote}\"")
        for w in ans.warnings:
            print("   WARNING:", w)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 4:
        sys.exit(__doc__)
    analyze = "--analyze" in args
    rest = [a for a in args if a != "--analyze"]
    main(rest[0], rest[1], rest[2], analyze, rest[3:])
