# Evaluation data

Phase 8 end-to-end results (with caveats) are in [RESULTS.md](RESULTS.md); gold labels for the demo lecture are in
`demo_gold.json`; run them with `python -m scripts.run_eval ...`. The multilingual retrieval study below is separate.

`multilingual_set.json` is a **synthetic** parallel corpus (16 deep-learning concepts x English, Hindi,
Marathi, Hinglish, with 5 query styles each). It was written by the developer's assistant, not copied from
YouTube, and has **not yet been reviewed by a native Hindi/Marathi speaker**. Treat every number as an
indicative result on a small, easy set, not as accuracy on real videos.

Run:

```bash
python -m scripts.eval_retrieval intfloat/multilingual-e5-small --chunk-chars 250 500
python -m scripts.eval_query_rewrite intfloat/multilingual-e5-small --chunk-chars 250
```

Results are written to `retrieval_results.json`; LLM query rewordings are cached in `query_translations.json`.
