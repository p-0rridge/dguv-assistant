"""
Unit tests for tokenisation and rank fusion.

These two pieces are testable without a model, an index or a network call, which is why
they are tested and the retrieval stack around them is not: a wrong tokenizer produces
a lexical index that is silently wrong on part of the corpus, and an off-by-one in the
fusion reorders every result without ever raising an error. Both would show up as
plausible numbers rather than as a failure.

Run with:  python test_hybrid_search.py
"""
import sys

sys.path.insert(0, "src")

from hybrid_search import (
    BM25Index,
    STOPWORDS,
    analyse,
    filter_tokens,
    reciprocal_rank_fusion,
    tokenize,
)

SOFT_HYPHEN = "­"


# --- Tokenizer ---------------------------------------------------------------
# A DGUV designation must survive as a single term. This is the whole reason the
# lexical branch exists: an embedding places 203-071 and 203-072 close together,
# and splitting them here would throw away the one thing BM25 does better.
assert tokenize("203-071") == ["203-071"]
assert tokenize("DGUV Information 203-071") == ["dguv", "information", "203-071"]
assert tokenize("DIN VDE 0100-540") == ["din", "vde", "0100-540"]

# SELV, PELV and FELV differ by one letter and denote three different protective
# measures. They must never collapse into one another.
assert tokenize("SELV")[0] != tokenize("PELV")[0]
assert analyse("SELV und PELV") == ["selv", "pelv"]

# The paragraph symbol is joined to its number, so § 5 and § 7 stay distinguishable.
# Dropping the symbol would leave the bare "5", which the length filter then removes -
# the section number would disappear entirely.
assert tokenize("§ 5") == ["§5"]
assert tokenize("§5") == ["§5"]
assert tokenize("§ 5") != tokenize("§ 7")

# Soft hyphens and hyphenated line breaks are typesetting artefacts and must be
# repaired, or a third of the corpus indexes as fragments that match nothing.
assert tokenize(f"Unterneh{SOFT_HYPHEN}men") == ["unternehmen"]
assert tokenize("Unterneh-\nmen") == ["unternehmen"]

# ... but a plain hyphen between two words is NOT repaired, and that is deliberate:
# the same rule that would merge "unterneh-men" would destroy "203-071". The signal
# is the soft hyphen or the line break, never the hyphen on its own.
assert tokenize("unterneh-men") == ["unterneh-men"]

# Case is normalised, so a question and the regulation it quotes compare equal.
assert tokenize("Prüfung") == tokenize("PRÜFUNG") == ["prüfung"]

# Characters outside the German alphabet are not part of a token. Rare in this corpus,
# but it means an accented foreign word is truncated rather than kept - a known and
# accepted limitation, recorded here so it is a decision and not a surprise.
assert "à" not in tokenize("à")
assert tokenize("café") == ["caf"]

# A trailing hyphen never becomes a token of its own.
assert tokenize("Prüfung - Fristen") == ["prüfung", "fristen"]


# --- Stopword filter ---------------------------------------------------------
# Question words are the reason this list exists. "warum" is filtered even though it
# never appears in the corpus, because the filter is applied to the query as well.
assert "warum" in STOPWORDS
assert filter_tokens(["warum", "prüffrist"]) == ["prüffrist"]
assert analyse("Warum muss geprüft werden?") == ["muss", "geprüft"]

# Modal verbs must survive. In German regulatory language muss, soll, kann and darf are
# defined levels of obligation - filtering them would erase the difference between a
# requirement and a recommendation, which in a safety corpus is the difference that
# matters most.
for modal in ("muss", "soll", "kann", "darf"):
    assert modal not in STOPWORDS, f"{modal} is a level of obligation, not filler"
    assert modal in analyse(f"Der Unternehmer {modal} prüfen lassen.")

# Single characters carry no information and are everywhere: numbering and list markers
# mean "2" occurs in half of all passages.
assert filter_tokens(["2", "b", "prüfung"]) == ["prüfung"]

# Technical terms are never filtered, however rare or however common.
for term in ("selv", "schutzpotentialausgleichsleiter", "prüfung", "unternehmer"):
    assert term not in STOPWORDS

# The filter treats query and document identically - if it did not, a term could be
# indexed but never searchable.
assert analyse("Wann prüfen?") == filter_tokens(tokenize("Wann prüfen?"))


# --- Reciprocal rank fusion --------------------------------------------------
def _results(*chunk_ids):
    """A ranked list in the shape every retriever returns."""
    return [{"chunk_id": cid, "text": "", "metadata": {}, "score": 1.0} for cid in chunk_ids]


# The worked example: agreement beats a single confident list. B is first in neither
# list and still wins, because A is far down in one of them.
dense = _results("A", "x1", "x2", "x3", "x4", "B")
lexical = _results("y1", "y2", "B", *[f"z{i}" for i in range(46)], "A")
fused = reciprocal_rank_fusion([dense, lexical])
assert fused[0]["chunk_id"] == "B"
assert fused[0]["score"] > fused[1]["score"]

# A chunk found by only one branch still ranks, it simply collects one contribution.
only_dense = reciprocal_rank_fusion([_results("A"), _results("B")])
assert {r["chunk_id"] for r in only_dense} == {"A", "B"}

# Rank 1 in both branches beats rank 1 in one branch alone.
both = reciprocal_rank_fusion([_results("A", "B"), _results("A", "C")])
assert both[0]["chunk_id"] == "A"

# The fusion assumes ids are unique within a list. A duplicate would accumulate
# contributions and climb for no reason, so a branch returning duplicates is a bug in
# that branch - this test records the assumption rather than defending against it.
duplicated = reciprocal_rank_fusion([_results("A", "A", "A"), _results("B")])
assert duplicated[0]["chunk_id"] == "A"

# An empty branch is a normal event: a question containing no indexed term produces no
# lexical results at all, and fusion must then simply pass the other branch through.
assert reciprocal_rank_fusion([_results("A", "B"), []])[0]["chunk_id"] == "A"
assert reciprocal_rank_fusion([[], []]) == []

# Both branch scores are preserved, so it stays possible to see afterwards which branch
# contributed what.
merged = reciprocal_rank_fusion([_results("A"), _results("A")])[0]
assert "score_branch_0" in merged and "score_branch_1" in merged
assert merged["rank_branch_0"] == 1 and merged["rank_branch_1"] == 1


# --- BM25 over the real corpus -----------------------------------------------
# One end-to-end check against the actual index. Skipped rather than failed when the
# corpus has not been built, so the pure unit tests above stay runnable anywhere.
import config

if config.CHUNKS_FILE.exists():
    index = BM25Index.from_file(config.CHUNKS_FILE)

    assert index.doc_count > 0
    assert index.average_length > 0

    # A rare designation must retrieve its own document. This is the claim the whole
    # hybrid branch rests on, checked against real data rather than a fixture.
    top = index.search("DGUV Regel 103-011", k=1)[0]
    assert top["metadata"]["source_file"].startswith("103-011")

    # Results carry the shape every other retriever produces, so downstream code cannot
    # tell which branch a chunk came from.
    assert set(top) >= {"chunk_id", "text", "metadata", "score"}
    assert set(top["metadata"]) >= {"source_file", "page_number"}

    # A query of nothing but stopwords has no terms left and returns nothing, rather
    # than returning arbitrary passages with meaningless scores.
    assert index.search("und oder aber", k=5) == []

    # A term absent from the corpus contributes nothing instead of raising: an
    # unanswerable question is normal operation in this system.
    assert index.search("xyzzy", k=5) == []

    print(f"BM25 checked against {index.doc_count} real passages.")
else:
    print("chunks.json not found - skipped the corpus tests, ran the unit tests only.")

print("All tests passed.")
