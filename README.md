# DGUV Standards RAG

A retrieval system for German occupational-safety regulations for electrical work
(DGUV *Vorschriften*, *Regeln*, *Informationen*) that cites document and page for every
statement it makes.

**Status: work in progress.** Retrieval, evaluation and one of two planned retrieval
experiments are done and measured. What isn't done is at the bottom, with reasons.

## Core objective

In electrical work, inaccurate information is a safety risk. The goal is not a bot that
sounds authoritative — it's a bot whose every claim can be checked in seconds.

**Zero tolerance for hallucination**, ahead of voice input, tools or an agentic layer.

1. **Every statement carries a source** — document *and* page.
2. **One question often needs several sources.** The regulations cross-reference
   constantly, so a single-document answer is often incomplete.
3. **"I don't know" is a valid answer.** If the context doesn't contain the answer, the
   system says so instead of filling the gap.

Requirements 1 and 2 came from the end user — an electrician who had fed 75 PDFs into an
off-the-shelf chatbot and stopped using it because the answers drifted.

Originally aimed at DIN VDE standards; those are copyrighted, so the project moved to the
openly published DGUV rulebook covering the same domain.

## Corpus

15 documents, ~950 pages, 1,872 indexed passages — chosen to be hard rather than
convenient: 203-070, -071 and -072 all cover recurring inspections. If ranking matters
anywhere, it matters between documents a reader would confuse.

The PDFs aren't in this repo. They're freely downloadable from
[publikationen.dguv.de](https://publikationen.dguv.de) — redistribution is another matter.

## Pipeline

```
PDFs → extraction → chunking → BGE-M3 → ChromaDB
Question → dense retrieval → [cross-encoder re-ranking] → LLM → answer + sources
```

Everything except the answer-generation call runs locally: PyMuPDF, sentence-transformers
(BGE-M3, BGE-reranker-v2-m3), ChromaDB, LangChain + OpenAI `gpt-5.4-mini`.

**Working:** text and table extraction · chunking with content-derived ids, so
re-indexing is idempotent · local dense retrieval · citations naming document and page,
with the source list filtered to what the answer actually cites · cross-encoder
re-ranking, measured · a self-generated evaluation set.

**Known gaps:** the chunker doesn't recognise `§`, so DGUV *Vorschriften* barely get
split at all — `vorschrift3.pdf` becomes six chunks for 27 paragraphs. The image branch
is indexed on every run and queried by nothing. `data_cleaning.py` isn't wired in.

## Evaluation

Measuring retrieval means knowing the correct passage for every test question — by hand,
across 950 pages, that's days of work. So the task is reversed: take a passage whose
location is known, have a model write the question it answers. The correct answer is then
known by construction.

Each passage yields a **colloquial** question, a **precise** one naming the DGUV
designation, and a **verbatim quote** — used both as the answer key and as a check that
the model didn't invent its evidence. The gold label is document + page + quote, never
the chunk id, so the set survives a change of chunking. 64 entries, 128 questions.

**Both errors it has found so far were in my own work**, and both produced entirely
plausible numbers:

- **40 % of gold entries rejected as unsupported** — not hallucination but typesetting.
  German regulations break words across lines and the PDF keeps an invisible soft hyphen
  (`Unterneh¬men`). The same artefact sits in a third of the indexed corpus.
- **79 % of the precise questions named the target file**, some the page. The test set was
  handing the system the answer. After the fix, an apparent 17.3-point advantage for
  precise questions collapsed to 1.5 — and the conclusion drawn from it was withdrawn.

Neither would have surfaced from a summary table.

## Result: cross-encoder re-ranking

| | Baseline | Re-ranked |
|---|---|---|
| Recall@5 | 0.766 / 0.781 | **0.906 / 0.859** |
| MRR | 0.557 / 0.608 | **0.778 / 0.759** |
| Median latency | 0.2 s | **30.0 s** |

*(colloquial / precise questions)*

Recall@20 is unchanged at 0.938 / 0.906 — that's the control. The candidate pool doesn't
change, only its order; had that number moved, nothing else could be trusted. MRR moves
twice as far as Recall@5, which is why both are reported: Recall@5 can't see a passage
moving from rank 4 to rank 1.

Re-ranking captures ~81 % of the available headroom, so the method is close to exhausted.
At 30 s per query on CPU it isn't usable interactively. Both results were predicted in
writing before being measured.

## Running it

Key in `.env` as `OPENAI_API_KEY`, install `requirements.txt`, DGUV PDFs into `data/`:

```bash
python src/build_index.py --reset
python src/goldset_builder.py
python src/run_eval.py --config mvp_baseline
python src/run_eval.py --config rerank
python src/demo.py           # example answers, including a refusal
python test_metrics.py
```

## To do

- [ ] **Measure hybrid search.** An embedding blurs exact strings — `203-071` looks almost
      like `203-072` — where a lexical index matches them exactly. 95 % of the precise
      test questions name such a designation, so the effect should be visible if it exists.
- [ ] **Measure the abstention rate.** Refusal works observably and 14 verified
      unanswerable questions are prepared, but there's no number yet.
- [ ] **Measure multi-source answers.** Requirement 2 holds in practice, but each gold
      entry has one correct location, so no metric captures it.
- [ ] **Fix chunking for `§` documents and enforce a minimum chunk size.** Deferred on
      purpose: measured headroom here is 6–9 points against 12–17 for re-ranking.
- [ ] **Cut re-ranking latency.** Smaller cross-encoder, 10 candidates instead of 20, or a
      GPU — each brings it under a second.
- [ ] **Decide on the image branch.** CLIP is a poor fit for technical drawings and German
      queries; figure captions embedded as text look more promising.
- [ ] Web UI and deployment.