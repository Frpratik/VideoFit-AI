"""Does adding an English rewording of the question improve retrieval?

For each non-English query we ask the LLM for an English version (cached in
eval/query_translations.json), search with both, and fuse the two rankings with reciprocal rank fusion.

Example:
  python -m scripts.eval_query_rewrite intfloat/multilingual-e5-small BAAI/bge-m3 --chunk-chars 250
"""

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from app import config, llm
from app.rag import store
from scripts.eval_retrieval import DATA, QUERY_STYLES, TRANSCRIPT_LANGS, load_segments

CACHE = DATA.parent / "query_translations.json"


class EnglishQuery(BaseModel):
    english_query: str


def translations(data: dict) -> dict:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for concept in data["concepts"]:
        for style in QUERY_STYLES:
            key = f"{concept['id']}|{style}"
            if style != "en" and key not in cache:
                result = llm.call_structured(
                    EnglishQuery,
                    "Rewrite the user's question about a deep learning video as a short, clear English search "
                    "query. Keep technical terms. Return it in the english_query field.",
                    concept["queries"][style],
                )
                cache[key] = result.english_query
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] += 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--chunk-chars", type=int, default=250)
    args = parser.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))
    english = translations(data)
    for model in args.models:
        embeddings = store.E5Embeddings(model)
        config.CHROMA_DIR = tempfile.mkdtemp(prefix="vf_eval_")
        stats = defaultdict(lambda: {"n": 0, "base1": 0, "fuse1": 0, "baseR": 0.0, "fuseR": 0.0})
        for t_lang in TRANSCRIPT_LANGS:
            vid = f"eval_{t_lang}"
            store.build_index(vid, load_segments(data, t_lang), embeddings, max_chars=args.chunk_chars)
            for i, concept in enumerate(data["concepts"]):
                gold = f"seg_{i + 1:04d}"
                for style in QUERY_STYLES:
                    if style == "en":
                        continue
                    base = store.search(vid, concept["queries"][style], embeddings, k=5)
                    extra = store.search(vid, english[f"{concept['id']}|{style}"], embeddings, k=5)
                    holds = {r.chunk.id: gold in {s.id for s in r.chunk.segments} for r in base + extra}
                    base_ids = [r.chunk.id for r in base]
                    fused_ids = rrf([base_ids, [r.chunk.id for r in extra]])[:5]
                    s = stats[style]
                    s["n"] += 1
                    for tag, ids in (("base", base_ids), ("fuse", fused_ids)):
                        rank = next((r for r, c in enumerate(ids, 1) if holds[c]), None)
                        s[f"{tag}1"] += rank == 1
                        s[f"{tag}R"] += 1 / rank if rank else 0.0
        print(f"\n=== {model} | chunk {args.chunk_chars} | original query vs original+English rewording (RRF) ===")
        print(f"{'query style':12}{'n':>5}{'hit@1 base':>12}{'hit@1 fused':>13}{'MRR base':>10}{'MRR fused':>11}")
        for style, s in stats.items():
            print(f"{style:12}{s['n']:>5}{s['base1'] / s['n']:>12.2f}{s['fuse1'] / s['n']:>13.2f}"
                  f"{s['baseR'] / s['n']:>10.3f}{s['fuseR'] / s['n']:>11.3f}")


if __name__ == "__main__":
    main()
