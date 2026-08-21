# DGUV Standards RAG

A retrieval system for German occupational-safety regulations for electrical work
(DGUV *Vorschriften*, *Regeln*, *Informationen*) that cites document and page for every
statement it makes.

**Status: working, measured, incomplete.** Two retrieval experiments and one text
experiment are done and tested for significance. What isn't done is at the bottom, with
reasons.

## Core objective

In electrical work, inaccurate information is a safety risk. The goal is not a bot that
sounds authoritative — it's a bot whose every claim can be checked in seconds.

1. **Every statement carries a source** — document *and* page.
2. **One question often needs several sources.** The regulations cross-reference
   constantly, so a single-document answer is often incomplete.
3. **"I don't know" is a valid answer.** Measured: it refuses **14 of 14** questions the
   corpus cannot answer.

Requirements 1 and 2 came from the end user — an electrician who had fed 75 PDFs into an
off-the-shelf chatbot and stopped using it because the answers drifted.

Originally aimed at DIN VDE standards; those are copyrighted, so the project moved to the
openly published DGUV rulebook covering the same domain.

## Corpus

15 documents, ~950 pages, 1,092 indexed passages — chosen to be hard rather than
convenient: 203-070, -071 and -072 all cover recurring inspections. If ranking matters
anywhere, it matters between documents a reader would confuse.

The PDFs aren't in this repo. They're freely downloadable from
[publikationen.dguv.de](https://publikationen.dguv.de) — redistribution is another matter.

## Pipeline

```
PDFs → extraction → cleaning → chunking → BGE-M3 → ChromaDB
                                       └→ tokenisation → BM25 index

Question → dense retrieval ┐
        → BM25 retrieval   ┴→ rank fusion → [cross-encoder] → LLM → answer + sources
```

Everything except the answer-generation call runs locally: PyMuPDF, sentence-transformers
(BGE-M3, BGE-reranker-v2-m3), ChromaDB, LangChain + OpenAI `gpt-5.4-mini`. BM25 and the
rank fusion are written out rather than imported — the formula is twenty lines, and the
part worth owning is the tokenizer, which every general-purpose tokenizer gets wrong here
by splitting `203-071` into two terms.

Document titles are read from the PDFs themselves, so any DGUV document dropped into
`data/` is cited by name rather than by filename.

## Evaluation

Measuring retrieval means knowing the correct passage for every test question — by hand,
across 950 pages, that's days of work. So the task is reversed: take a passage whose
location is known, have a model write the question it answers. The correct answer is then
known by construction.

Each passage yields a **colloquial** question, a **precise** one naming the DGUV
designation, and a **verbatim quote** — used both as the answer key and as a check that
the model didn't invent its evidence. The gold label is document + page + quote, never
the chunk id, so the set survives a change of chunking. It did: the chunker was rewritten
mid-project and every gold entry still resolved. **136 entries, 272 questions.**

Because both variants answer the same questions, differences are tested **paired**
(McNemar's exact test, computed from the binomial distribution rather than imported).

### Four measurement errors, all in my own work

Each produced a plausible number, and none would have surfaced from a summary table.

**1 — 40 % of gold entries rejected as unsupported.** Not hallucination but typesetting.
German regulations break words across lines and the PDF keeps an invisible soft hyphen
(`Unterneh¬men`). The same artefact sat in 49 % of the indexed corpus.

**2 — 79 % of the precise questions named the target file**, some the page. The test set
was handing the system the answer. After the fix, an apparent 17.3-point advantage for
precise questions collapsed to 1.5, and the conclusion drawn from it was withdrawn.

**3 — A conclusion built on noise.** At 64 gold entries, hybrid search appeared to help
colloquial questions more than precise ones, and a mechanism was written up to explain
it. Doubling the gold set reversed the finding. The effect had been five questions.

That error is why the paired test exists. At 64 questions, a *perfect* result of 5 fixed
and 0 broken still gives p = 0.06 — effects of this size were unprovable in principle,
whatever they were. Six discordant questions is the minimum that can reach p < 0.05.

**4 — Abstention measured at 0.14 when it was 1.00.** Refusals were detected by matching
phrases in the answer text. Seven phrases recognised 2 of 14; extending the list to
eleven recognised 8. The misses were *"diese Frage"* for *"die Frage"*, *"Ihre Frage"*,
and a subject inserted before the negation — German has no fixed word order for this
sentence, so every addition created new ways to miss.

Published as 0.14, the system would have looked like it invented answers to 86 % of
questions it had in fact refused. The fix was not a longer list: the model now marks its
own refusals with a token, so detection is an exact comparison. That also lets a refusal
suppress its own source list — which phrase matching could never do, because by then the
answer was already written.

## Results

136 gold entries, colloquial / precise, all against the same index.

| | Baseline | **Hybrid** | Re-rank | Hybrid + re-rank |
|---|---|---|---|---|
| Recall@5 | 0.765 / 0.772 | 0.846 / 0.838 | 0.875 / 0.853 | **0.912 / 0.949** |
| Recall@20 | 0.934 / 0.875 | 0.956 / 0.978 | 0.934 / 0.875 | 0.956 / 0.978 |
| MRR | 0.578 / 0.598 | 0.641 / 0.693 | 0.730 / 0.751 | **0.748 / 0.825** |
| Median latency | 0.3 s | **0.3 s** | 20.5 s | 21.0 s |
| Found nothing at all | 9 / 17 | 6 / **3** | 9 / 17 | 6 / **3** |

### The control

Re-ranking reorders a candidate pool without changing it, so Recall@20 must not move.
Measured twice, question by question:

```
baseline → rerank          0 fixed, 0 broken
hybrid   → hybrid_rerank   0 fixed, 0 broken
```

Not one question moved. Had that number drifted, nothing else in the table could be
trusted. For hybrid search the same number is not a control but the measurement, because
fusion changes which passages reach the pool at all.

### Two levers, not one

The clearest result is the last row of the table. **Questions that retrieve nothing
usable fall from 17 to 3 on precise questions — and the cross-encoder changes that number
not at all.** It cannot rank a passage that was never retrieved.

- **Fusion decides what is findable.** BM25 matches exact strings where an embedding
  blurs them. In this corpus the distinguishing terms are not only designations:
  `Schutzleiter` appears in 129 passages and `Schutzpotentialausgleichsleiter` in 2,
  while SELV, PELV and FELV differ by one letter and denote three different protective
  measures. Paired result on precise questions: **15 fixed, 1 broken, p = 0.001**.
- **The cross-encoder decides what reaches the top five.** On the dense pool it is not
  distinguishable from hybrid search at 1/60th the latency (p = 0.45 / 0.77). On the
  hybrid pool it is (p = 0.049 / < 0.001).

They are complementary, and neither substitutes for the other. **Re-ranking alone is
dominated**: 20 seconds for a result that 0.3 seconds also delivers.

`mvp_baseline → hybrid_rerank` on precise questions: **24 fixed, 0 broken, p < 0.001.**
Not one question got worse.

### What would ship

**Hybrid alone.** It captures most of the available gain at 0.3 s instead of 21 s, and
cuts unanswerable retrievals by 82 %. The stacked configuration is the quality ceiling,
not the product.

### A bias in these numbers

The precise question form overstates the benefit: 95 % of those questions name a DGUV
designation, because that is how they were generated. The intended user is an electrician
who knows the documents and is unlikely to quote a designation verbatim. That question
form describes a user who does not exist, so the gain measured there is an upper bound.
The colloquial column is the honest one.

## An experiment that changed nothing

Text cleaning and chunking were rebuilt: soft hyphens and control characters removed,
hyphenated line breaks rejoined without damaging compounds like `DGUV-Regel` or
`203-071`, tables of contents and imprints dropped, fragments merged. The corpus went
from 1,522 passages to 1,092, the median passage from 786 to 1,250 characters, and
passages under 300 characters from 31 % to 7 %.

Measured paired against the previous index, on the same 136 questions:

```
baseline  colloquial@5    6 fixed,  8 broken   p = 0.79
          precise@5      13 fixed,  7 broken   p = 0.26
hybrid    colloquial@5    3 fixed,  3 broken   p = 1.00
          precise@5       6 fixed,  2 broken   p = 0.29
```

**No effect on retrieval, in either direction.** It was kept anyway, for reasons that are
not retrieval: the cleaned text is what a reader sees beside a citation, three
dependencies and a dead image branch went with it, and the gold-set generator had been
rejecting 40 % of its own output over those same artefacts.

The claim in this README is therefore not "cleaning improved retrieval". It is "cleaning
left retrieval unchanged — measured, not assumed".

## Abstention

14 questions the corpus cannot answer, in three categories: references to DIN VDE
standards the documents cite but do not reproduce, unrelated subjects, and near misses
where the corpus holds something adjacent and wrong.

| | rate |
|---|---|
| `din_vde_verweis` | 1.00 |
| `fachfremd` | 1.00 |
| `nah_dran` | 1.00 |
| **overall** | **1.00** (14/14) |

`nah_dran` is the category that matters: the corpus specifies minimum cross-sections for
a *different* kind of conductor, and answering with those numbers would be confident,
well-cited and wrong.

## Running it

Key in `.env` as `OPENAI_API_KEY`, install `requirements.txt`, DGUV PDFs into `data/`:

```bash
python src/build_index.py --reset
python src/document_titles.py
python src/goldset_builder.py --per-document 6
python src/run_eval.py --config mvp_baseline
python src/run_eval.py --config hybrid --abstention
python src/run_eval.py --config rerank            # ~20 s per question on CPU
python src/run_eval.py --config hybrid_rerank
python src/compare_runs.py mvp_baseline hybrid    # paired significance test
python src/demo.py                                # example answers, including a refusal
python test_metrics.py
python test_hybrid_search.py
python src/data_cleaning.py
```

The lexical index is built from `artifacts/chunks.json`, which `build_index.py` exports.
Re-index and it is rebuilt with it, or the two retrieval branches would search different
corpora and the fusion between them would mean nothing.

## To do

- [ ] **Fix page attribution.** Merged chunks carry the page number of their first
      element, so a passage continuing onto the next page is cited one page early. In a
      system whose promise is verifiability, this is the most expensive open bug.
- [ ] **Deploy.** Now realistic: the shipping configuration answers in 0.3 s and needs no
      cross-encoder. Blocked on a question rather than on effort — serving passages from
      these documents publicly is redistribution, and that needs deciding, not assuming.
- [ ] **Measure multi-source answers.** Requirement 2 holds in practice, but each gold
      entry has one correct location, so no metric captures it.
- [ ] **Grow the gold set past 136.** Effects smaller than about six questions remain
      unprovable, and that is now the binding constraint on every further experiment.
- [ ] **Cut re-ranking latency.** Smaller cross-encoder, 10 candidates instead of 20, or
      a GPU — each brings 21 s under a second.
- [ ] Web UI, and figure captions embedded as text (CLIP was removed: it is a poor fit
      for technical drawings and German queries, and nothing queried it).
