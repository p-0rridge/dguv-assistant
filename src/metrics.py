"""
Retrieval metrics and the rule that decides what counts as a hit.

Deliberately written as pure functions rather than a class: each one maps an input to
an output with no hidden state, which is what makes them unit-testable - and a silent
off-by-one in the rank counting would corrupt every number in the evaluation without
ever raising an error.

Each question in the gold set has exactly one correct location. That has two
consequences worth stating out loud, because they shape how the results must be read:
  - Recall@k is then identical to Hit Rate@k: either the one correct chunk is among the
    first k results or it is not.
  - nDCG@k collapses to 1 / log2(rank + 1), which is a close relative of the reciprocal
    rank. Both are reported, but they carry nearly the same information here.
"""
import math
import re

# PDF text carries typographic artefacts that are invisible to a reader but not to a
# string comparison: soft hyphens marking a line break inside a word, non-breaking
# spaces after a paragraph symbol, hyphens left over from justified typesetting. A
# language model quoting such a passage silently reproduces the word as a reader sees
# it, so a raw comparison rejects correct quotations. These patterns undo the artefacts
# before comparing.
_NBSP = "\u00a0"
_SOFT_HYPHEN = re.compile(r"\u00ad\s*")
_LINE_HYPHEN = re.compile(r"(\w)[-\u2010\u2011]\s*\n\s*(\w)")
_DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014]")
_QUOTES = re.compile(r"[\u201a\u201e\u201c\u201d\u2018\u2019]")
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """
    Reduce text to a form in which two renderings of the same wording compare equal.

    Removes soft hyphens, rejoins words broken across a line by a hyphen, replaces
    non-breaking spaces, unifies dash and quote variants, lowercases and collapses
    whitespace. Wording itself is never changed, so a genuinely different sentence still
    fails to match - only presentation differences are neutralised.

    This matters twice over: it decides which generated gold entries survive validation,
    and it decides whether a retrieved chunk counts as a hit during evaluation.
    """
    text = text.replace(_NBSP, " ")
    text = _SOFT_HYPHEN.sub("", text)
    text = _LINE_HYPHEN.sub(r"\1\2", text)
    text = _DASHES.sub("-", text)
    text = _QUOTES.sub('"', text)
    return _WHITESPACE.sub(" ", text.lower()).strip()


def is_relevant(chunk: dict, gold: dict) -> bool:
    """
    Decide whether a retrieved chunk satisfies a gold answer.
    chunk: A retrieval result with keys text and metadata (source_file, page_number)
    gold: A gold entry with keys source_file, page_number, answer_snippet
    returns: True if this chunk counts as the correct answer location

    The rule is: right document, and either the right page or the verbatim snippet
    contained in the text. The snippet clause matters because a chunk is tagged with the
    page of its first element, so a section running across a page break gets the earlier
    page number - without the snippet fallback those cases would be scored as misses.
    Anchoring on document and page rather than on chunk_id is what lets the same gold set
    survive a change to the chunking strategy.
    """
    if chunk["metadata"].get("source_file") != gold["source_file"]:
        return False
    if chunk["metadata"].get("page_number") == gold["page_number"]:
        return True
    snippet = normalise(gold.get("answer_snippet", ""))
    return bool(snippet) and snippet in normalise(chunk["text"])


def hits_from_results(results: list[dict], gold: dict) -> list[bool]:
    """
    Turn a ranked result list into a ranked list of booleans.
    results: Retrieved chunks, best first
    gold: The gold entry for this question
    returns: One bool per result, in the same order
    """
    return [is_relevant(chunk, gold) for chunk in results]


def recall_at_k(hits: list[bool], k: int) -> float:
    """
    1.0 if the correct location appears within the first k results, else 0.0.
    With a single relevant item per question this equals Hit Rate@k. Averaged over all
    questions it describes the candidate pool - the quantity hybrid search should move.
    """
    return 1.0 if any(hits[:k]) else 0.0


def reciprocal_rank(hits: list[bool]) -> float:
    """
    1 / rank of the first correct result, counting from 1; 0.0 if there is none.
    This is the metric a re-ranker is supposed to improve: it cannot add a document the
    retriever never fetched, it can only move the right one further up.
    """
    for index, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / index
    return 0.0


def ndcg_at_k(hits: list[bool], k: int) -> float:
    """
    Normalised discounted cumulative gain over the first k results.
    With exactly one relevant item the ideal ranking puts it first, so IDCG is 1 and the
    result reduces to 1 / log2(rank + 1).
    """
    for index, hit in enumerate(hits[:k], start=1):
        if hit:
            return 1.0 / math.log2(index + 1)
    return 0.0


def mean(values: list[float]) -> float:
    """Arithmetic mean, returning 0.0 for an empty list instead of raising."""
    return sum(values) / len(values) if values else 0.0
