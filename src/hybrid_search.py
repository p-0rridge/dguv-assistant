"""
Lexical retrieval (BM25) and the fusion that combines it with dense retrieval.

Why this exists at all: a dense embedding encodes meaning, which is what makes it good
at paraphrase and bad at precision. In this corpus that weakness is concrete and
safety-relevant. "Schutzleiter" appears in 129 passages, "Schutzpotentialausgleichs-
leiter" in 2; SELV, PELV and FELV differ by one letter and mean three different things.
An embedding places such terms close together because they *are* semantically close.
A lexical index treats them as what they also are: different strings.

The two branches are complementary rather than competing, so neither replaces the
other. Dense retrieval finds the approximately right passage; BM25 distinguishes the
approximately right one from the exactly right one.

Three pieces, deliberately separable so each can be tested on its own:
    tokenize / filter_tokens  - text to terms, the part carrying domain judgement
    BM25Index                 - lexical scoring over the exported chunks
    reciprocal_rank_fusion    - merging two ranked lists without comparing their scores
"""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from metrics import normalise

# --- Tokenisation ------------------------------------------------------------
# Joins a paragraph symbol to the number that follows it, so "§ 5" becomes the single
# token "§5". The alternative - dropping § and keeping "5" - loses the distinction
# between § 5 and § 7, which in DGUV Vorschriften is often the whole question. Applied
# before the main pattern because by then the space is gone.
PARAGRAPH_PATTERN = re.compile(r"§\s*(\d+[a-z]?)")

# A token is a run of letters, digits or §, optionally continued across hyphens.
# The hyphen rule is the reason "203-071" survives as one term instead of splitting
# into "203" and "071", which is precisely the confusion this module exists to prevent.
# Requiring characters *after* each hyphen also means a hyphen left at a line break
# never ends up as a token of its own.
# The pattern assumes lowercase input, which normalise() guarantees.
TOKEN_PATTERN = re.compile(r"[a-zäöüß0-9§]+(?:-[a-zäöüß0-9]+)*")

# Words removed before indexing and before searching.
#
# The list is a blocklist, not an allowlist: IDF already discovers which terms are
# informative, by measuring how rare they are. It needs help in exactly one place -
# where a word is rare for a reason that has nothing to do with meaning.
#
# That place is question words. Regulations are written as statements, so "wann" occurs
# in 2 of 1522 passages and "welches" in 2 - the same document frequency as the most
# specific technical term in the corpus, and therefore the same IDF. Without this list,
# every question beginning "Wann..." would rank those two passages first.
#
# Modal verbs are deliberately absent. In German regulatory language muss, soll, kann
# and darf are defined levels of obligation, not filler: treating them as stopwords
# would erase the difference between a requirement and a recommendation.
STOPWORDS = frozenset([
    # Question words - rare in declarative text, therefore over-weighted by IDF.
    "wer", "wen", "wem", "wessen", "wie", "was", "wo", "wann", "warum", "wieso",
    "weshalb", "weswegen", "wohin", "woher", "wofür", "womit", "worauf", "worin",
    "wobei", "wozu", "wieviel", "wieviele",
    "welche", "welcher", "welches", "welchen", "welchem",
    # Conjunctions and particles.
    "und", "oder", "aber", "denn", "doch", "nur", "auch", "noch", "schon", "sehr",
    "viel", "mehr", "meisten", "sowie", "sowohl", "beziehungsweise", "bzw",
    # Articles, pronouns, prepositions, auxiliaries. IDF neutralises these on its own,
    # so they barely affect ranking - but they do count towards passage length, and
    # BM25 penalises long passages. Removing them is what makes the length correction
    # measure content rather than grammar.
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einer",
    "eines", "einem", "dieser", "diese", "dieses", "diesen", "diesem",
    "in", "im", "an", "am", "auf", "aus", "bei", "beim", "mit", "nach", "von", "vom",
    "vor", "zu", "zum", "zur", "über", "unter", "durch", "für", "gegen", "ohne", "um",
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden", "sein",
    "hat", "haben", "hatte", "hatten",
    "nicht", "kein", "keine", "keinen", "als", "wenn", "dass", "sich", "es", "man",
    "ich", "sie", "ihr", "ihre", "ihren", "seine", "seinen", "dabei", "damit", "dazu",
])

# Single characters carry no lexical information here and are numerous: list markers
# and numbering mean "2" appears in 50% of all passages and "b" in 35%.
MIN_TOKEN_LENGTH = 2


def tokenize(text: str) -> list[str]:
    """
    Split text into terms, without removing anything.

    Runs metrics.normalise() first, and that is not cosmetic. Roughly a third of this
    corpus contains soft hyphens marking line breaks inside words. Untreated,
    "Unterneh<shy>men" tokenises as "unterneh" plus "men" - two terms that occur nowhere
    else, receive maximum IDF, and never match a query for "Unternehmen". The index
    would be silently wrong on a third of the corpus while every number still looked
    plausible.

    Reusing normalise() also means retrieval and evaluation judge the same text: one
    definition, two uses.

    Kept separate from filter_tokens() so document frequencies can be measured over the
    unfiltered vocabulary - a stopword list cannot be reviewed once its words are gone.
    """
    text = normalise(text)
    text = PARAGRAPH_PATTERN.sub(r"§\1", text)
    return TOKEN_PATTERN.findall(text)


def filter_tokens(tokens: list[str]) -> list[str]:
    """Drop stopwords and single characters. Applied identically to documents and queries."""
    return [
        token for token in tokens
        if len(token) >= MIN_TOKEN_LENGTH and token not in STOPWORDS
    ]


def analyse(text: str) -> list[str]:
    """Full pipeline from raw text to indexable terms."""
    return filter_tokens(tokenize(text))


# --- BM25 --------------------------------------------------------------------
class BM25Index:
    """
    Lexical scoring over the exported chunks.

    BM25 is a term-frequency score with three corrections, and all three are visible in
    _score():
      1. Rare terms count more (IDF). This is the correction that matters here: it is
         why a passage containing "203-071" outranks one containing "Prüfung".
      2. Repetition saturates. A term occurring twenty times does not make a passage
         twice as relevant as one where it occurs ten times; k1 controls how quickly
         the benefit flattens.
      3. Long passages are normalised. Without this a long passage would win simply by
         containing more words; b controls how strongly length is discounted.

    k1 and b are left at the values the literature settled on. Tuning them against this
    gold set would be a separate experiment, and one that risks fitting the test data.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, chunks: list[dict]):
        """
        chunks: Chunk records as exported to artifacts/chunks.json, each with chunk_id,
            text, source_file, page_number and type

        Must be built from the same chunks that populate the vector store, or the two
        branches would be searching different corpora and the fusion would be
        meaningless. Rebuild after every re-index.
        """
        self.chunks = chunks
        self.doc_lengths: list[int] = []

        # Inverted index: term -> {document position: how often the term occurs there}.
        # Scoring then only visits documents that actually contain a query term, rather
        # than all 1522 of them.
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)

        for position, chunk in enumerate(chunks):
            terms = analyse(chunk["text"])
            self.doc_lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self.postings[term][position] = frequency

        self.doc_count = len(chunks)
        self.average_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )

    @classmethod
    def from_file(cls, path: Path) -> "BM25Index":
        """
        Build the index from artifacts/chunks.json, skipping image records.

        encoding="utf-8" is not optional: the default on Windows is cp1252, and this
        corpus is full of umlauts and paragraph symbols.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run build_index.py first - it exports the chunks "
                f"that the lexical index is built from."
            )
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        return cls([record for record in records if record.get("type") != "Image"])

    def _idf(self, term: str) -> float:
        """
        Inverse document frequency: how much a term's presence tells us.

        The +0.5 smoothing and the outer log(1 + x) are the standard formulation. The
        alternative form without the 1 + goes negative for terms appearing in more than
        half the documents, which would mean a passage is penalised for containing a
        common query word - unwanted, and hard to notice once summed.
        """
        document_frequency = len(self.postings.get(term, {}))
        if document_frequency == 0:
            return 0.0
        return math.log(
            1 + (self.doc_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def search(self, query: str, k: int) -> list[dict]:
        """
        Return the k best-matching chunks, best first, in the shape every Retriever uses.

        A query term absent from the corpus contributes nothing rather than raising:
        an unanswerable question is a normal event in this system, not an error.
        """
        terms = analyse(query)
        if not terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for position, frequency in postings.items():
                length_ratio = self.doc_lengths[position] / self.average_length
                saturation = frequency * (self.K1 + 1)
                normaliser = frequency + self.K1 * (1 - self.B + self.B * length_ratio)
                scores[position] += idf * saturation / normaliser

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [self._as_result(position, score) for position, score in ranked]

    def _as_result(self, position: int, score: float) -> dict:
        """
        Convert an internal document position back into a retrieval result.

        The single place where a position becomes a chunk_id. Keeping the conversion in
        one method is deliberate: an off-by-one between the scoring loop and the chunk
        list would return real passages with real-looking scores under the wrong ids,
        and nothing downstream could detect it.
        """
        chunk = self.chunks[position]
        return {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": {
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "type": chunk.get("type"),
            },
            "score": float(score),
        }


# --- Fusion ------------------------------------------------------------------
def reciprocal_rank_fusion(result_lists: list[list[dict]], rrf_k: int = 60) -> list[dict]:
    """
    Merge several ranked lists into one, using positions rather than scores.

    result_lists: Ranked lists, best first, each item carrying a chunk_id
    rrf_k: Damping constant, 60 in the original paper
    returns: One ranked list, best first, with score replaced by the fused value

    Cosine similarities lie between 0 and 1; BM25 scores are unbounded and depend on the
    corpus and the query. Adding them would be meaningless, and normalising them would
    make the result depend on the score distribution of whichever query happened to be
    asked. Ranks are the one thing the two branches express comparably.

    Each list contributes 1 / (rrf_k + rank). The constant flattens the top of the
    curve: without it rank 1 would score 1.0 against rank 2's 0.5, and a single
    confident list would dominate. With rrf_k = 60 the two differ by less than 2%, so
    appearing high in *both* lists beats topping one of them:

        rank 1 in one list, rank 50 in the other -> 1/61 + 1/110  = 0.0255
        rank 5 in one list, rank  3 in the other -> 1/65 + 1/63   = 0.0313  wins

    That is the intended behaviour, not a side effect: RRF rewards agreement.

    The original score is preserved per branch, so it stays possible to see afterwards
    which branch found a passage the other one missed.
    """
    fused: dict[str, float] = defaultdict(float)
    seen: dict[str, dict] = {}

    for list_index, results in enumerate(result_lists):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            fused[chunk_id] += 1 / (rrf_k + rank)
            if chunk_id not in seen:
                seen[chunk_id] = dict(result)
            seen[chunk_id][f"score_branch_{list_index}"] = result["score"]
            seen[chunk_id][f"rank_branch_{list_index}"] = rank

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    output = []
    for chunk_id, score in ordered:
        result = seen[chunk_id]
        result["score"] = score
        output.append(result)
    return output
