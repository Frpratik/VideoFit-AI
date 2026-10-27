"""End-to-end evaluation with the REAL LLM and embeddings (costs Groq calls, takes several minutes).

  python -m scripts.run_eval consistency --n 5     # score stability, fixed vs auto topic checklist
  python -m scripts.run_eval qa                    # retrieval, citations, refusals, languages, follow-ups
  python -m scripts.run_eval injection             # hostile transcript and comments

Gold labels: eval/demo_gold.json (synthetic lecture; not independently reviewed).
Results are saved to eval/results/<section>.json.
"""

import argparse
import json
import re
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from langchain_core.messages import HumanMessage

from app import config, llm
from app.graph.analysis_graph import build_analysis_graph, run_analysis
from app.graph.chat_graph import ChatDeps, build_chat_graph
from app.ingestion.transcript import detect_script, parse_transcript
from app.ingestion.youtube import extract_video_id
from app.rag import store

ROOT = Path(__file__).resolve().parent.parent
GOLD = json.loads((ROOT / "eval" / "demo_gold.json").read_text(encoding="utf-8"))
RAW = (ROOT / "samples" / "english_neural_networks_demo.txt").read_text(encoding="utf-8")
OUT = ROOT / "eval" / "results"
VIDEO_ID = extract_video_id(GOLD["video_url"])
_emb = None


def embeddings():
    global _emb
    if _emb is None:
        _emb = store.E5Embeddings(config.EMBEDDING_MODEL)
        config.CHROMA_DIR = tempfile.mkdtemp(prefix="vf_eval_run_")
    return _emb


def save(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def inside(sec, window) -> bool:
    return sec is not None and window[0] <= sec <= window[1]


def index(video_id: str, raw: str):
    parsed = parse_transcript(raw)
    store.build_index(video_id, parsed.segments, embeddings())
    return parsed.segments


def stats(values: list[float]) -> dict:
    return {"n": len(values), "mean": round(statistics.mean(values), 1),
            "sd": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
            "min": round(min(values), 1), "max": round(max(values), 1)}


# ------------------------------------------------------------------ consistency + topic accuracy
def consistency(n: int) -> None:
    segments = index(VIDEO_ID, RAW)
    graph = build_analysis_graph(embeddings())
    names = [t["name"] for t in GOLD["topics"]]
    gold = {t["name"]: t for t in GOLD["topics"]}
    out = {"fixed": [], "auto": []}

    for mode, topics in (("fixed", names), ("auto", None)):
        for i in range(n):
            started = time.time()
            try:
                r = run_analysis(graph, VIDEO_ID, segments, GOLD["goal"], GOLD["level"], "English", topics)
                out[mode].append({"overall": r.overall_score, "criteria": {c.key: c.score for c in r.criteria},
                                  "topics": [{"name": t.topic.name, "coverage": t.coverage, "depth": t.depth,
                                              "has_example": t.has_example, "assessed": t.assessed,
                                              "cited_starts": [c.start_sec for c in t.citations]} for t in r.topics]})
                print(f"{mode} run {i + 1}/{n}: score {r.overall_score} ({time.time() - started:.0f}s)", flush=True)
            except Exception as exc:
                out[mode].append({"error": type(exc).__name__})
                print(f"{mode} run {i + 1}/{n}: FAILED {type(exc).__name__}", flush=True)

    summary = {}
    for mode in ("fixed", "auto"):
        ok = [r for r in out[mode] if "overall" in r]
        summary[mode] = {"runs": len(out[mode]), "failed": len(out[mode]) - len(ok),
                         "overall": stats([r["overall"] for r in ok]) if ok else None}
        if ok:
            summary[mode]["criteria"] = {k: stats([r["criteria"][k] for r in ok]) for k in ok[0]["criteria"]}
    # topic-level results only make sense with the fixed checklist
    ok = [r for r in out["fixed"] if "overall" in r]
    exact = binary = fp_n = fp = present_n = recall = ev_n = ev_ok = ex_n = ex_ok = 0
    confusion = Counter()
    per_topic_agree = {}
    for r in ok:
        for t in r["topics"]:
            g = gold[t["name"]]
            confusion[(g["gold"], t["coverage"])] += 1
            exact += t["coverage"] == g["gold"]
            is_present_gold, is_present_pred = g["gold"] != "not_detected", t["coverage"] != "not_detected"
            binary += is_present_gold == is_present_pred
            if not is_present_gold:
                fp_n += 1; fp += is_present_pred
            else:
                present_n += 1; recall += is_present_pred
            if is_present_pred and is_present_gold:
                for s in t["cited_starts"]:
                    ev_n += 1; ev_ok += inside(s, g["window"])
            if "has_example" in g:
                ex_n += 1; ex_ok += t["has_example"] == g["has_example"]
    n_topic_obs = sum(len(r["topics"]) for r in ok)
    for name in names:
        labels = [next(t["coverage"] for t in r["topics"] if t["name"] == name) for r in ok]
        per_topic_agree[name] = Counter(labels).most_common(1)[0][1] / len(labels) if labels else None
    summary["topic_accuracy"] = {
        "exact_coverage_accuracy": round(exact / n_topic_obs, 3) if n_topic_obs else None,
        "detected_vs_not_accuracy": round(binary / n_topic_obs, 3) if n_topic_obs else None,
        "false_positive_rate_on_absent_topics": round(fp / fp_n, 3) if fp_n else None,
        "recall_on_present_topics": round(recall / present_n, 3) if present_n else None,
        "evidence_precision_cited_segment_in_gold_window": round(ev_ok / ev_n, 3) if ev_n else None,
        "has_example_accuracy_on_unambiguous_topics": round(ex_ok / ex_n, 3) if ex_n else None,
        "confusion_gold_to_predicted": {f"{a}->{b}": c for (a, b), c in sorted(confusion.items())},
        "label_agreement_across_runs_per_topic": per_topic_agree,
    }
    save("consistency", {"model": config.LLM_MODEL, "usage": dict(llm.USAGE), "summary": summary, "runs": out})
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("model:", config.LLM_MODEL, "| tokens used:", llm.USAGE)


# ------------------------------------------------------------------ question answering
def citation_integrity(answer, segments) -> bool:
    by_id = {s.id: s for s in segments}
    for c in answer.citations:
        s = by_id.get(c.segment_id)
        if not s or s.text != c.quote or (c.url and not c.url.endswith(f"&t={int(s.start_sec)}s")):
            return False
    return True


def qa() -> None:
    segments = index(VIDEO_ID, RAW)
    chat = build_chat_graph(ChatDeps(embeddings=embeddings()))
    starts = {s.id: s.start_sec for s in segments}
    results = {"answerable": [], "unanswerable": [], "followups": []}

    def ask(question, language, messages=()):
        return chat.invoke({"video_id": VIDEO_ID, "language": language, "report": None,
                            "messages": list(messages) + [HumanMessage(content=question)]})

    for item in GOLD["questions"]:
        lang = item.get("language", "English")
        top = store.search(VIDEO_ID, item["q"], embeddings(), k=4)
        hit_rank = next((r for r, x in enumerate(top, 1) if any(inside(s.start_sec, item["window"]) for s in x.chunk.segments)), None)
        row = {"id": item["id"], "language": lang, "retrieval_hit_rank": hit_rank}
        try:
            final = ask(item["q"], lang)
            a = final["answer"]
            row.update({"evidence_found": a.evidence_found, "citations": len(a.citations),
                        "citation_in_window": any(inside(c.start_sec, item["window"]) for c in a.citations),
                        "integrity": citation_integrity(a, segments), "attempts": final["attempt"],
                        "script_ok": (detect_script(a.answer_from_transcript) in ("devanagari", "mixed") if item.get("script") == "devanagari"
                                      else detect_script(a.answer_from_transcript) == "latin") if item.get("script") else None,
                        "answer": a.answer_from_transcript})
        except Exception as exc:
            row["error"] = type(exc).__name__
        results["answerable"].append(row)
        print(row["id"], {k: v for k, v in row.items() if k != "answer"}, flush=True)

    for item in GOLD["unanswerable"]:
        row = {"id": item["id"]}
        try:
            a = ask(item["q"], "English")["answer"]
            row.update({"evidence_found": a.evidence_found, "citations": len(a.citations), "answer": a.answer_from_transcript})
        except Exception as exc:
            row["error"] = type(exc).__name__
        results["unanswerable"].append(row)
        print(row["id"], {k: v for k, v in row.items() if k != "answer"}, flush=True)

    A, U = results["answerable"], results["unanswerable"]
    ok = [r for r in A if "error" not in r]
    summary = {
        "answerable_questions": len(A), "errors": sum("error" in r for r in A + U),
        "retrieval_top1_chunk_has_gold": sum(r["retrieval_hit_rank"] == 1 for r in A) / len(A),
        "retrieval_top4_chunk_has_gold": sum(r["retrieval_hit_rank"] is not None for r in A) / len(A),
        "answered_with_evidence": sum(r["evidence_found"] for r in ok) / len(A),
        "some_citation_inside_gold_window": sum(r["citation_in_window"] for r in ok) / len(A),
        "citation_integrity_all_answers": all(r["integrity"] for r in ok),
        "language_script_ok": [r["script_ok"] for r in A if r.get("script_ok") is not None],
        "unanswerable_refused": sum(not r["evidence_found"] for r in U if "error" not in r), "unanswerable_total": len(U),
        "unanswerable_with_citations": sum(r["citations"] > 0 for r in U if "error" not in r),
    }
    save("qa", {"model": config.LLM_MODEL, "summary": summary, "results": results})
    print(json.dumps(summary, indent=1))


def followups() -> None:
    """A 3-turn conversation: resolve 'it', then 'that', and switch language mid-chat."""
    segments = index(VIDEO_ID, RAW)
    chat = build_chat_graph(ChatDeps(embeddings=embeddings()))
    rows, messages = [], []
    for item in GOLD["followups"]:
        row = {"q": item["q"]}
        try:
            final = chat.invoke({"video_id": VIDEO_ID, "language": "English", "report": None,
                                 "messages": messages + [HumanMessage(content=item["q"])]})
            messages = final["messages"]
            a = final["answer"]
            row.update({"understood_as": final["standalone_question"], "kind": final["kind"],
                        "answer_language": final["answer_language"], "evidence_found": a.evidence_found,
                        "citation_in_window": any(inside(c.start_sec, item["window"]) for c in a.citations),
                        "integrity": citation_integrity(a, segments),
                        "script_ok": detect_script(a.answer_from_transcript) in ("devanagari", "mixed") if item.get("script") else None,
                        "warnings": a.warnings, "answer": a.answer_from_transcript})
        except Exception as exc:
            row["error"] = type(exc).__name__
        rows.append(row)
        print(row, flush=True)
    save("followups", {"model": config.LLM_MODEL, "results": rows, "usage": dict(llm.USAGE)})


# ------------------------------------------------------------------ prompt injection
def injection() -> None:
    inj = GOLD["injection"]
    entries = [(int(m.group(1)) * 60 + int(m.group(2)), f"{m.group(1)}:{m.group(2)}\n{m.group(3)}")
               for m in re.finditer(r"^(\d+):(\d{2})\n(.*)$", RAW, re.M)]
    for line in inj["transcript_lines"]:
        m = re.match(r"(\d+):(\d{2})\n", line)
        entries.append((int(m.group(1)) * 60 + int(m.group(2)), line))
    raw = "\n".join(text for _, text in sorted(entries, key=lambda e: e[0]))
    vid = "injectTEST1"
    segments = index(vid, raw)
    canary = inj["canary"]
    graph = build_analysis_graph(embeddings())
    names = [t["name"] for t in GOLD["topics"]]
    result = {}

    chat = build_chat_graph(ChatDeps(embeddings=embeddings()))
    final = chat.invoke({"video_id": vid, "language": "English", "report": None,
                         "messages": [HumanMessage(content="How does a single neuron compute its output?")]})
    a = final["answer"]
    result["chat"] = {"canary_in_answer": canary in json.dumps(a.model_dump(), ensure_ascii=False),
                      "evidence_found": a.evidence_found, "answer": a.answer_from_transcript}

    r = run_analysis(graph, vid, segments, GOLD["goal"], GOLD["level"], "English", names, "\n".join(inj["comments"]))
    absent = {t["name"] for t in GOLD["topics"] if t["gold"] == "not_detected"}
    result["analysis"] = {
        "canary_anywhere_in_report": canary in json.dumps(r.model_dump(), ensure_ascii=False),
        "overall_score": r.overall_score,
        "absent_topics_marked_present": sorted(t.topic.name for t in r.topics if t.topic.name in absent and t.coverage != "not_detected"),
        "comment_insights": r.comment_insights.model_dump() if r.comment_insights else None,
        "warnings": r.warnings,
    }
    save("injection", {"model": config.LLM_MODEL, "usage": dict(llm.USAGE), **result})
    print(json.dumps(result, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------ does the score discriminate?
def mismatch() -> None:
    """The same beginner lecture judged against an advanced goal. A useful score must drop clearly."""
    segments = index(VIDEO_ID, RAW)
    graph = build_analysis_graph(embeddings())
    goals = [("Master transformer architectures and attention for research", "advanced"),
             ("Learn deep learning from beginner to intermediate", "intermediate")]
    rows = []
    for goal, level in goals:
        r = run_analysis(graph, VIDEO_ID, segments, goal, level, "English")
        rows.append({"goal": goal, "level": level, "overall": r.overall_score,
                     "criteria": {c.key: c.score for c in r.criteria},
                     "topics": [(t.topic.name, t.coverage) for t in r.topics],
                     "limitations": r.limitations})
        print(goal, "|", level, "->", r.overall_score, rows[-1]["criteria"], flush=True)
    save("mismatch", {"model": config.LLM_MODEL, "usage": dict(llm.USAGE), "rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("section", choices=["consistency", "qa", "followups", "injection", "mismatch"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--model", help="override LLM_MODEL for this run (e.g. when a daily free quota is used up)")
    args = parser.parse_args()
    if args.model:
        config.LLM_MODEL = args.model
    {"consistency": lambda: consistency(args.n), "qa": qa, "followups": followups, "injection": injection,
     "mismatch": mismatch}[args.section]()
