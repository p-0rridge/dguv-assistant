"""
Unit tests for metrics.py.

Run from the project root:  python test_metrics.py

These cover the failure modes that stay silent otherwise: ranks counted from 0 instead
of 1, a hit sitting just outside k, and an empty result list. A metric that is wrong in
one of those ways still returns a plausible-looking number, so every later comparison
would be quietly corrupted.
"""
import math
import sys

sys.path.insert(0, "src")

from metrics import (
    hits_from_results,
    is_relevant,
    mean,
    ndcg_at_k,
    normalise,
    recall_at_k,
    reciprocal_rank,
)

passed = failed = 0


def check(label: str, got, want) -> None:
    """Compare one result against its expectation and record the outcome."""
    global passed, failed
    is_ok = got == want or (isinstance(want, float) and abs(got - want) < 1e-9)
    print(("PASS " if is_ok else "FAIL ") + f"{label:52s} got={got!r} want={want!r}")
    passed += is_ok
    failed += not is_ok


def chunk(source_file: str, page: int, text: str = "some text") -> dict:
    """Build a minimal retrieval result for testing."""
    return {"text": text, "metadata": {"source_file": source_file, "page_number": page}}


GOLD = {
    "source_file": "203-071.pdf",
    "page_number": 12,
    "answer_snippet": "Prüffristen  für\nortsveränderliche Betriebsmittel",
}

# --- is_relevant: the definition of a hit ------------------------------------
check("right file, right page", is_relevant(chunk("203-071.pdf", 12), GOLD), True)
check("right file, wrong page", is_relevant(chunk("203-071.pdf", 40), GOLD), False)
check("wrong file, right page", is_relevant(chunk("203-072.pdf", 12), GOLD), False)
check(
    "wrong page but snippet present",
    is_relevant(
        chunk("203-071.pdf", 11, "... PRÜFFRISTEN FÜR ortsveränderliche   Betriebsmittel ..."),
        GOLD,
    ),
    True,
)
check(
    "empty snippet disables the fallback",
    is_relevant(chunk("203-071.pdf", 11), {**GOLD, "answer_snippet": ""}),
    False,
)

# --- text normalisation ------------------------------------------------------
# These are the artefacts that caused 40% of generated gold entries to be rejected:
# the model quotes a passage the way a reader sees it, the raw PDF text does not.
check(
    "soft hyphen inside a word is ignored",
    normalise("Unterneh\u00ad\nmen") == normalise("Unternehmen"),
    True,
)
check(
    "non-breaking space compares equal to a space",
    normalise("\u00a7\u00a02 Absatz\u00a01") == normalise("§ 2 Absatz 1"),
    True,
)
check(
    "word broken by a hyphen at a line break is rejoined",
    normalise("Betriebs-\nmittel") == normalise("Betriebsmittel"),
    True,
)
check("dash variants unified", normalise("A \u2013 B") == normalise("A - B"), True)
check("quote variants unified", normalise("\u201eGrundsätze\u201c") == normalise('"Grundsätze"'), True)
check(
    "different wording still fails to match",
    normalise("alle zwölf Monate") == normalise("alle sechs Monate"),
    False,
)

# --- rank metrics ------------------------------------------------------------
hit_first = [True, False, False, False, False]
hit_third = [False, False, True, False, False]
hit_none = [False] * 5

check("recall@5, hit at rank 3", recall_at_k(hit_third, 5), 1.0)
check("recall@2, hit at rank 3 is outside k", recall_at_k(hit_third, 2), 0.0)
check("recall@5, no hit", recall_at_k(hit_none, 5), 0.0)

check("reciprocal rank, rank 1", reciprocal_rank(hit_first), 1.0)
check("reciprocal rank, rank 3", reciprocal_rank(hit_third), 1 / 3)
check("reciprocal rank, no hit", reciprocal_rank(hit_none), 0.0)

check("ndcg@5, rank 1", ndcg_at_k(hit_first, 5), 1.0)
check("ndcg@5, rank 3", ndcg_at_k(hit_third, 5), 1 / math.log2(4))
check("ndcg@2, rank 3 is outside k", ndcg_at_k(hit_third, 2), 0.0)
check(
    "ndcg decreases as the hit moves down",
    ndcg_at_k([True], 5) > ndcg_at_k([False, True], 5) > ndcg_at_k([False, False, True], 5),
    True,
)

# --- end to end --------------------------------------------------------------
results = [chunk("203-072.pdf", 5), chunk("203-071.pdf", 12), chunk("100-001.pdf", 3)]
check("hits_from_results marks rank 2", hits_from_results(results, GOLD), [False, True, False])
check("mean of an empty list", mean([]), 0.0)
check("mean", mean([1.0, 0.0, 0.5]), 0.5)

# --- additional test: chunk id stability ------------------------------------------------------
from data_preprocessing import MultiModalPreprocessor

make_id = MultiModalPreprocessor.make_chunk_id

BASE_CHUNK = {
    "source_file": "203-071.pdf",
    "page_number": 12,
    "type": "NarrativeText",
    "text": "Prüffristen für ortsveränderliche Betriebsmittel",
}

check("chunk id is stable across runs", make_id(BASE_CHUNK), make_id(dict(BASE_CHUNK)))
check(
    "chunk id changes with the page",
    make_id(BASE_CHUNK) != make_id({**BASE_CHUNK, "page_number": 13}),
    True,
)
check(
    "chunk id changes with the text",
    make_id(BASE_CHUNK) != make_id({**BASE_CHUNK, "text": BASE_CHUNK["text"] + "."}),
    True,
)
check(
    "chunk id changes with the source file",
    make_id(BASE_CHUNK) != make_id({**BASE_CHUNK, "source_file": "203-072.pdf"}),
    True,
)
check("chunk id is 16 characters", len(make_id(BASE_CHUNK)), 16)


print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)