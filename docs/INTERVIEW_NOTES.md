# Interview notes

Honest talking points for explaining this project. Everything here is backed by the code or by `eval/RESULTS.md`.

## 60-second pitch

"VideoFit AI tells a learner whether a YouTube video fits their goal. They paste the transcript, and I parse it into
timestamped segments, embed them locally, and store them in Chroma. One feature scores the video against a topic
checklist built from their goal, with every judgement tied to a cited transcript line. The other is a chat where each
answer links to the exact timestamp. The part I'm most careful about is trust: the model only names segment IDs, and
my code builds the quote, time and link, so it cannot invent a citation. All scoring arithmetic is in Python."

## Walk through the design

1. **Parsing** (`ingestion/transcript.py`): text, SRT and VTT into segments. Missing or invalid timestamps degrade with
   a warning, never silently.
2. **RAG** (`rag/`): about 500-character chunks with one segment of overlap, local `multilingual-e5-small`, Chroma.
3. **Grounded answers** (`chat/answerer.py`): structured output, ID-only citations, validation in Python.
4. **LangGraph** (`graph/`): two workflows with real branching (route, tool loop, retry, optional comments branch).
5. **Scoring** (`analysis/rubric.py`): weighted heuristic, computed in Python and unit tested against a hand
   calculation.

## Decisions and trade-offs

- **Why LangGraph at all?** The chat has genuine branching: plan, route, tool loop, retry. A plain function chain
  would hide that. I did not force simple steps into nodes.
- **Why local embeddings?** Free, private, multilingual. I compared four models on a small synthetic set (e5-small
  86%, bge-m3 91% hit@1 but 11x slower), and kept e5-small as the default with bge-m3 as an option.
- **Why ID-only citations?** An LLM can fabricate a plausible quote or time. Checking IDs against what was retrieved
  makes fabrication impossible for citations.
- **Why a custom HTML front end?** Control over the look, no build step. The cost is writing the UI by hand.
- **Why no Docker or CI?** Nothing here needs them yet. I would add them when someone else has to run it.
- **Why is the planner a separate LLM call?** It resolves "explain that more" into a standalone question and routes
  off-topic questions before spending a search.

## Things that broke, and what I learned

- Groq sometimes answers in plain text instead of calling the structured-output tool. I added one retry with a delay.
- LangGraph's `add_messages` merges messages that share an ID, which made a test fake loop incorrectly. Real models
  return fresh messages.
- Control characters, an unpunctuated 250 KB line, and times like `12:75` all slipped through the parser until I wrote
  failure-mode tests.
- A text file saved with a byte-order mark hid the first timestamp.
- The free Groq tier has a daily token limit. I measured about 16,500 tokens per analysis, and running the evaluation
  used up one model's daily allowance.
- Auto-generated topic checklists change between runs, moving the score by about 10 points. A fixed checklist gave a
  standard deviation of 1.0, so the app lets you reuse it.

## Questions to expect, and honest answers

- **How accurate is it?** On one synthetic lecture with labels I wrote myself: 100% on present vs not detected, 92% on
  exact coverage label. I would not claim that on real videos. I have not tested real transcripts.
- **Is the score valid?** It is a heuristic with unvalidated weights. It does respond sensibly: the same lecture
  scored 28 against an advanced goal and about 69 against a beginner one.
- **How do you stop hallucinations?** Retrieval first, a refusal path, ID-only citations checked in code. I did not
  measure whether every sentence in an answer is supported by its citations.
- **Prompt injection?** Untrusted text sits in delimited blocks and the tools are read-only. In one test a hostile
  transcript did not change the output, but one absent topic flipped to present in that run, and I could not tell
  whether that was the injection or noise.
- **What would you do next?** Test real transcripts and long videos, add a human-reviewed benchmark and a faithfulness
  check, stream analysis progress, and persist reports.
