# DGUV Standards RAG

A retrieval system for German occupational-safety regulations for electrical work
(DGUV *Vorschriften*, *Regeln*, *Informationen*) that cites document and page for every
statement it makes.

**Status: work in progress.** Retrieval, evaluation and both planned retrieval
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
                            └→ tokenisation → BM25 index

Question → dense retrieval ┐
        → BM25 retrieval   ┴→ rank fusion → [cross-encoder] → LLM → answer + sources
```

Everything except the answer-generation call runs locally: PyMuPDF, sentence-transformers
(BGE-M3, BGE-reranker-v2-m3), ChromaDB, LangChain + OpenAI `gpt-5.4-mini`. BM25 and the
fusion are written out rather than imported — the formula is short, and the part worth
owning is the tokenizer, which every general-purpose tokenizer gets wrong here.

**Working:** text and table extraction · chunking with content-derived ids, so
re-indexing is idempotent · dense, lexical and fused retrieval, all local · citations
naming document and page, with the source list filtered to what the answer actually
cites · cross-encoder re-ranking · a self-generated evaluation set · both retrieval
experiments measured against it.

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

## Results

Two experiments, four configurations, one gold set. Every result was predicted in
writing before it was measured — see `artifacts/results/prediction_hybrid.md`.

| | Baseline | Hybrid | Re-rank | Hybrid + re-rank |
|---|---|---|---|---|
| Recall@5 | 0.766 / 0.781 | 0.844 / 0.828 | 0.906 / 0.859 | **0.922 / 0.906** |
| Recall@20 | 0.938 / 0.906 | 0.953 / 0.969 | 0.938 / 0.906 | **0.953 / 0.969** |
| MRR | 0.557 / 0.608 | 0.645 / 0.691 | 0.778 / 0.759 | 0.769 / **0.797** |
| Median latency | 0.3 s | **0.3 s** | 30.3 s | 25.1 s |
| Found nothing at all | 4 / 6 | 3 / **2** | 4 / 6 | 3 / **2** |

*(colloquial / precise questions, 64 each)*

### The control

Re-ranking reorders a candidate pool without changing it, so Recall@20 must not move.
It doesn't — 0.938 → 0.938 and 0.953 → 0.953, in both configurations. Had it moved,
nothing else in the table could be trusted.

For hybrid search the same number is not a control but the measurement: fusion changes
which passages reach the pool in the first place.

### What re-ranking does

Recall@5 improves by 14 points, MRR by 22 — MRR moves further because Recall@5 cannot
see a passage travel from rank 4 to rank 1. The method is close to exhausted: it
converts 94–97 % of whatever its pool contains. At 30 s per query on CPU it is not
usable interactively.

### What hybrid search does — and the prediction it broke

The prediction was that the lexical branch would help precise questions more than
colloquial ones, since 95 % of the precise questions name a DGUV designation and BM25
matches such strings exactly where an embedding blurs `203-071` into `203-072`.

On Recall@5 the opposite happened: **+7.8 points colloquial against +4.7 precise.**

Recall@20 shows why, and splits exactly as predicted: **+1.6 colloquial against +6.2
precise.** The two question forms are being helped by two different mechanisms.

- **Precise questions: BM25 finds passages dense retrieval never had.** The pool grows.
  This is the rare-term effect the experiment was built to test. In this corpus the
  distinguishing terms are not only designations — `Schutzleiter` appears in 129
  passages and `Schutzpotentialausgleichsleiter` in 2, and SELV, PELV and FELV differ
  by one letter while denoting three different protective measures. An embedding places
  them close together because they *are* close; a lexical index treats them as the
  different strings they also are.
- **Colloquial questions: BM25 reorders what was already there.** The pool barely
  grows, but the top 5 improves sharply. The correct passage was already in the pool at
  rank 6–20; what was missing was a reason to promote it. Rank fusion supplies one —
  not by understanding the question, but by having two methods with unrelated failure
  modes point at the same passage. Agreement is the evidence, and RRF is what converts
  it into rank.

The falsification criterion written down beforehand was itself imprecise: it named
Recall@5, which conflates finding with ordering. Recall@20 isolates finding, and that
is where the predicted split appears. The prediction was wrong about the metric, not
about the mechanism.

### What the combination shows

Stacking both is best everywhere on Recall@5, and the reason is worth stating precisely.
The cross-encoder converts a near-constant fraction of whatever it is given:

| | pool (Recall@20) | delivered (Recall@5) | converted |
|---|---|---|---|
| re-rank, colloquial | 0.938 | 0.906 | 96.7 % |
| hybrid + re-rank, colloquial | 0.953 | 0.922 | 96.7 % |
| re-rank, precise | 0.906 | 0.859 | 94.8 % |
| hybrid + re-rank, precise | 0.969 | 0.906 | 93.5 % |

**The entire advantage of the combination comes from the better pool, not from better
ranking.** That splits the system into two independent levers — fusion decides what is
available, the cross-encoder converts a fixed share of it — and says where further work
belongs: the re-ranker is at its ceiling, the pool is not. Which is the argument for
fixing chunking next.

### Which one would ship

**Hybrid alone.** It captures 56–60 % of the re-ranker's improvement at 0.3 s instead of
30 s, and it more than halves the questions that retrieve nothing usable at all
(6 → 2 on precise questions). The combination is the quality ceiling, not the product.

### A bias in these numbers

The precise question form overstates the benefit. 95 % of those questions name a DGUV
designation because that is how they were generated — but the intended user is an
electrician who knows the documents and is unlikely to quote a designation verbatim.
That question form describes a user who does not exist, so the gain measured there is
an upper bound rather than an expectation for real use. The colloquial number is the
honest one, and it happens to be the larger of the two.

## Running it

Key in `.env` as `OPENAI_API_KEY`, install `requirements.txt`, DGUV PDFs into `data/`:

```bash
python src/build_index.py --reset
python src/goldset_builder.py
python src/run_eval.py --config mvp_baseline
python src/run_eval.py --config hybrid
python src/run_eval.py --config rerank         # ~30 s per question on CPU
python src/run_eval.py --config hybrid_rerank
python src/demo.py           # example answers, including a refusal
python test_metrics.py
python test_hybrid_search.py
```

The lexical index is built from `artifacts/chunks.json`, which `build_index.py` exports.
Re-index and it has to be rebuilt, or the two retrieval branches search different
corpora and the fusion between them means nothing.

## To do

- [ ] **Measure the abstention rate.** Refusal works observably and 14 verified
      unanswerable questions are prepared, but there's no number yet.
- [ ] **Measure multi-source answers.** Requirement 2 holds in practice, but each gold
      entry has one correct location, so no metric captures it.
- [ ] **Fix chunking for `§` documents, filter front matter, enforce a minimum chunk
      size.** Now the highest-value work left: the results above show the cross-encoder
      is at its ceiling and the candidate pool is not, and chunking is a pool lever.
- [ ] **Cut re-ranking latency.** Smaller cross-encoder, 10 candidates instead of 20, or a
      GPU — each brings it under a second.
- [ ] **Decide on the image branch.** CLIP is a poor fit for technical drawings and German
      queries; figure captions embedded as text look more promising.
- [ ] Web UI and deployment.