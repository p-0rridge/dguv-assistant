"""
Why this file? The metrics, and the rule that decides what counts as a hit.

Pure functions rather than a class: no hidden state, so each is unit-testable. A silent
off-by-one in the rank counting would corrupt every number in the evaluation without
ever raising an error.

Each gold question has exactly one correct location, so Recall@k equals Hit Rate@k, and
nDCG@k reduces to 1 / log2(rank + 1) - close kin to the reciprocal rank.
"""
import math
import re

# PDF text carries artefacts invisible to a reader but not to a string comparison. A
# model quoting such a passage reproduces the word as a reader sees it, so an untreated
# comparison rejects correct quotations.
_NBSP = " "
_SOFT_HYPHEN = re.compile(r"­\s*")
_LINE_HYPHEN = re.compile(r"(\w)[-‐‑]\s*\n\s*(\w)")
_DASHES = re.compile(r"[‐‑‒–—]")
_QUOTES = re.compile(r"[‚„“”‘’]")
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """
    Reduce text to a form in which two renderings of the same wording compare equal.

    Wording is never changed, so a genuinely different sentence still fails to match.
    Used twice: it decides which gold entries survive validation, and whether a
    retrieved chunk counts as a hit ----> goldset_builder, hybrid_search
    """
    text = text.replace(_NBSP, " ")
    text = _SOFT_HYPHEN.sub("", text)
    text = _LINE_HYPHEN.sub(r"\1\2", text)
    text = _DASHES.sub("-", text)
    text = _QUOTES.sub('"', text)
    return _WHITESPACE.sub(" ", text.lower()).strip()


def is_relevant(chunk: dict, gold: dict) -> bool:
    """
    Right document, and either the right page or the verbatim snippet in the text.

    The snippet clause matters because a chunk carries the page of its first element, so
    a section crossing a page break gets the earlier number. Anchoring on document and
    page rather than chunk_id is what lets one gold set survive a change of chunking.
    """
    if chunk["metadata"].get("source_file") != gold["source_file"]:
        return False
    if chunk["metadata"].get("page_number") == gold["page_number"]:
        return True
    snippet = normalise(gold.get("answer_snippet", ""))
    return bool(snippet) and snippet in normalise(chunk["text"])


def hits_from_results(results: list[dict], gold: dict) -> list[bool]:
    """A ranked result list as a ranked list of booleans."""
    return [is_relevant(chunk, gold) for chunk in results]


def recall_at_k(hits: list[bool], k: int) -> float:
    """1.0 if the correct location is within the first k results. Describes the pool."""
    return 1.0 if any(hits[:k]) else 0.0


def reciprocal_rank(hits: list[bool]) -> float:
    """
    1 / rank of the first correct result, 0.0 if there is none.
    What a re-ranker improves: it cannot add a passage, only move the right one up.
    """
    for index, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / index
    return 0.0


def ndcg_at_k(hits: list[bool], k: int) -> float:
    """With one relevant item the ideal ranking puts it first, so IDCG is 1."""
    for index, hit in enumerate(hits[:k], start=1):
        if hit:
            return 1.0 / math.log2(index + 1)
    return 0.0


def mean(values: list[float]) -> float:
    """Arithmetic mean, 0.0 for an empty list instead of raising."""
    return sum(values) / len(values) if values else 0.0
