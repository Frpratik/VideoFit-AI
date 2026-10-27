"""Measure multilingual retrieval quality on eval/multilingual_set.json.

For every transcript language x query style, each question must retrieve the chunk that holds its
gold segment. Metrics: hit@1, hit@3 and MRR over the top 5 chunks.

Example:
  python -m scripts.eval_retrieval intfloat/multilingual-e5-small --chunk-chars 250 500
"""

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from app import config
from app.rag import store
from app.schemas import Segment

DATA = Path(__file__).resolve().parent.parent / "eval" / "multilingual_set.json"
TRANSCRIPT_LANGS = ["en", "hi", "mr", "hinglish"]
QUERY_STYLES = ["en", "hi", "mr", "hinglish", "mixed"]


def load_segments(data: dict, lang: str) -> list[Segment]:
    return [
        Segment(id=f"seg_{i + 1:04d}", start_sec=i * 10.0, end_sec=i * 10.0 + 10, text=c["segments"][lang])
        for i, c in enumerate(data["concepts"])
    ]


def evaluate(model_name: str, chunk_chars: int, data: dict, embeddings) -> dict:
    config.CHROMA_DIR = tempfile.mkdtemp(prefix="vf_eval_")
    scores = defaultdict(lambda: {"n": 0, "hit1": 0, "hit3": 0, "mrr": 0.0})
    for t_lang in TRANSCRIPT_LANGS:
        segments = load_segments(data, t_lang)
        store.build_index(f"eval_{t_lang}", segments, embeddings, max_chars=chunk_chars)
        for i, concept in enumerate(data["concepts"]):
            gold = f"seg_{i + 1:04d}"
            for q_style in QUERY_STYLES:
                results = store.search(f"eval_{t_lang}", concept["queries"][q_style], embeddings, k=5)
                rank = next((r for r, item in enumerate(results, 1) if gold in {s.id for s in item.chunk.segments}), None)
                for key in ((t_lang, q_style), ("ALL", q_style), (t_lang, "ALL"), ("ALL", "ALL")):
                    s = scores[key]
                    s["n"] += 1
                    s["hit1"] += rank == 1
                    s["hit3"] += rank is not None and rank <= 3
                    s["mrr"] += 1 / rank if rank else 0.0
    return {f"{k[0]}|{k[1]}": {"n": v["n"], "hit@1": v["hit1"] / v["n"], "hit@3": v["hit3"] / v["n"],
                              "mrr": v["mrr"] / v["n"]} for k, v in scores.items()}


def print_table(model: str, chunk_chars: int, result: dict) -> None:
    print(f"\n=== {model} | chunk {chunk_chars} chars | hit@1 (rows = transcript language, columns = query style) ===")
    print(f"{'':10}" + "".join(f"{q:>10}" for q in QUERY_STYLES + ["ALL"]))
    for t in TRANSCRIPT_LANGS + ["ALL"]:
        print(f"{t:10}" + "".join(f"{result[f'{t}|{q}']['hit@1']:>10.2f}" for q in QUERY_STYLES + ["ALL"]))
    overall = result["ALL|ALL"]
    print(f"overall: hit@1={overall['hit@1']:.3f}  hit@3={overall['hit@3']:.3f}  MRR={overall['mrr']:.3f}  (n={overall['n']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--chunk-chars", type=int, nargs="+", default=[250, 500])
    parser.add_argument("--out", default="eval/retrieval_results.json")
    args = parser.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))
    out_path = Path(args.out)
    all_results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for model in args.models:
        embeddings = store.E5Embeddings(model)
        for chars in args.chunk_chars:
            result = evaluate(model, chars, data, embeddings)
            all_results[f"{model}|{chars}"] = result
            print_table(model, chars, result)
        out_path.write_text(json.dumps(all_results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
