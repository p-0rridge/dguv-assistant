# Prediction: hybrid search

Written **before** `hybrid.json` and `hybrid_rerank.json` existed, and left unedited
afterwards. A result explained after the fact can always be explained; the point of
this file is that it cannot be.

## Baseline being predicted against

`mvp_baseline.json`, dense retrieval only (colloquial / precise):

| | colloquial | precise |
|---|---|---|
| Recall@5 | 0.766 | 0.781 |
| Recall@20 | 0.938 | 0.906 |

## Prediction

**Little change in either question form, slightly more for the precise questions,
because those contain more technical terminology.**

A second, more optimistic prediction was on the table — a clear gain on precise
questions, on the grounds that 95 % of them name a DGUV designation and BM25 matches
such strings exactly where an embedding blurs them. Both are recorded, so whichever
turns out closer, it was written down first.

## What the argument rests on

The lexical branch can only contribute where a query term is rare in the corpus. In
this corpus the distinguishing terms are not only designations:

| term | passages containing it (of 1522) | IDF |
|---|---|---|
| Schutzleiter | 129 | 2.47 |
| Schutzpotentialausgleichsleiter | 2 | 6.63 |
| SELV / PELV / FELV | 33 / 19 / 5 | 3.83 / 4.38 / 5.72 |
| 203-071 / 203-072 | 18 / 9 | 4.44 / 5.13 |

An embedding places these close together because they *are* semantically close. The
system's existing near-miss failure — right cross-sections, wrong type of conductor —
is exactly this case.

## What would count as the prediction failing

- **A drop in Recall@20.** Fusion would then be displacing correct passages that dense
  retrieval had already found, and the two branches would not be complementary.
- **A gain on colloquial questions as large as on precise ones.** That would suggest
  the improvement is not coming from exact term matching at all, and the stated
  mechanism would be wrong even if the number is good.

## Known bias in this measurement

Recall@20 is **not** a control here, unlike in the re-ranking experiment. Fusion
changes which passages enter the candidate pool, so this number is the measurement
rather than the check on it.

And the result will overstate the benefit. 95 % of the precise gold questions name a
DGUV designation, because that is how they were generated — but the actual end user is
an electrician who knows the documents and is unlikely to quote a designation verbatim.
The precise question form describes a user who does not exist. Whatever gain appears
there is an upper bound, not an expectation for real use.
