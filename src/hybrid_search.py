"""
Why this file? A dense embedding encodes meaning, which makes it good at paraphrase and
bad at precision - it places "Schutzleiter" and "Schutzpotentialausgleichsleiter" close
together, and SELV next to PELV. A lexical index treats them as the different strings
they also are. The two branches fail in different places, which is the whole premise.

Three parts, separable so each is testable on its own:
    tokenize / filter_tokens  - text to terms, the part carrying domain judgement
    BM25Index                 - lexical scoring over the exported chunks
    reciprocal_rank_fusion    - merging ranked lists without comparing their scores
"""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from metrics import normalise

# Joins "§ 5" into one token. Dropping § would leave a bare "5", which the length filter
# then removes - and in DGUV Vorschriften the section number is often the whole question.
PARAGRAPH_PATTERN = re.compile(r"§\s*(\d+[a-z]?)")

# Letters, digits or §, continued across hyphens. The hyphen rule is why "203-071"
# survives as one term instead of splitting into "203" and "071" - the exact confusion
# this module exists to prevent. Assumes lowercase input, which normalise() guarantees.
TOKEN_PATTERN = re.compile(r"[a-zäöüß0-9§]+(?:-[a-zäöüß0-9]+)*")

# A blocklist, not an allowlist: IDF already discovers which terms are informative by
# measuring how rare they are. It needs help in one place only - where a word is rare
# for a reason that has nothing to do with meaning. That place is question words:
# regulations are written as statements, so "wann" is as rare as the most specific
# technical term in the corpus, and IDF weights it accordingly.
#
# Modal verbs are deliberately absent. In German regulatory language muss, soll, kann
# and darf are defined levels of obligation, not filler.
STOPWORDS = frozenset([
    # Question words - rare in declarative text, therefore over-weighted by IDF
    "wer", "wen", "wem", "wessen", "wie", "was", "wo", "wann", "warum", "wieso",
    "weshalb", "weswegen", "wohin", "woher", "wofür", "womit", "worauf", "worin",
    "wobei", "wozu", "wieviel", "wieviele",
    "welche", "welcher", "welches", "welchen", "welchem",
    # Conjunctions and particles
    "und", "oder", "aber", "denn", "doch", "nur", "auch", "noch", "schon", "sehr",
    "viel", "mehr", "meisten", "sowie", "sowohl", "beziehungsweise", "bzw",
    # Articles, pronouns, prepositions, auxiliaries. IDF neutralises these anyway, but
    # they count towards passage length, and BM25 penalises long passages.
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einer",
    "eines", "einem", "dieser", "diese", "dieses", "diesen", "diesem",
    "in", "im", "an", "am", "auf", "aus", "bei", "beim", "mit", "nach", "von", "vom",
    "vor", "zu", "zum", "zur", "über", "unter", "durch", "für", "gegen", "ohne", "um",
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden", "sein",
    "hat", "haben", "hatte", "hatten",
    "nicht", "kein", "keine", "keinen", "als", "wenn", "dass", "sich", "es", "man",
    "ich", "sie", "ihr", "ihre", "ihren", "seine", "seinen", "dabei", "damit", "dazu",
])

MIN_TOKEN_LENGTH = 2 # List markers and numbering make single characters very frequent


def tokenize(text: str) -> list[str]:
    """
    Split text into terms, removing nothing.

    normalise() first, and that is not cosmetic: an untreated soft hyphen turns
    "Unterneh<shy>men" into two terms that occur nowhere else, receive maximum IDF and
    never match "Unternehmen". It also means retrieval and evaluation judge the same
    text - one definition, two uses.

    Separate from filter_tokens() so document frequencies stay measurable over the
    unfiltered vocabulary; a stopword list cannot be reviewed once its words are gone.
    """
    text = normalise(text)
    text = PARAGRAPH_PATTERN.sub(r"§\1", text)
    return TOKEN_PATTERN.findall(text)


def filter_tokens(tokens: list[str]) -> list[str]:
    """Drop stopwords and single characters. Applied to documents and queries alike."""
    return [
        token for token in tokens
        if len(token) >= MIN_TOKEN_LENGTH and token not in STOPWORDS
    ]


def analyse(text: str) -> list[str]:
    """Raw text to indexable terms."""
    return filter_tokens(tokenize(text))


class BM25Index:
    """
    Lexical scoring over the exported chunks.

    Three corrections, all visible in search():
      1. rare terms count more (IDF) - the one that matters here
      2. repetition saturates (k1)
      3. long passages are normalised (b)

    k1 and b stay at the values the literature settled on. Tuning them against this gold
    set would be a separate experiment, and one that risks fitting the test data.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, chunks: list[dict]):
        """
        chunks: records as exported to artifacts/chunks.json

        Must be built from the same chunks that populate the vector store, or the two
        branches search different corpora and the fusion means nothing.
        """
        self.chunks = chunks
        self.doc_lengths: list[int] = []
        # term -> {document position: frequency}, so scoring only visits documents that
        # actually contain a query term.
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
        """Build the index from chunks.json. utf-8 is not optional on Windows."""
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run build_index.py first - it exports the chunks "
                f"the lexical index is built from."
            )
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        return cls([record for record in records if record.get("type") != "Image"])

    def _idf(self, term: str) -> float:
        """
        How much a term's presence tells us.

        log(1 + x), not the textbook form: that one goes negative for terms in more than
        half the documents, so a passage would be penalised for containing a common
        query word - unwanted, and hard to notice once summed.
        """
        document_frequency = len(self.postings.get(term, {}))
        if document_frequency == 0:
            return 0.0
        return math.log(
            1 + (self.doc_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def search(self, query: str, k: int) -> list[dict]:
        """The k best-matching chunks, best first, in the shape every Retriever uses."""
        terms = analyse(query)
        if not terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self.postings.get(term)
            if not postings:
                continue # An unknown term contributes nothing rather than raising
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
        Turn an internal position back into a retrieval result.

        The single place where a position becomes a chunk_id. An off-by-one here would
        return real passages with real-looking scores under the wrong ids, and nothing
        downstream could detect it.
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


def reciprocal_rank_fusion(result_lists: list[list[dict]], rrf_k: int = 60) -> list[dict]:
    """
    Merge ranked lists by position, not by score ----> HybridRetriever

    Cosine similarities lie in [0, 1]; BM25 scores are unbounded and depend on corpus
    and query. Adding them is meaningless, and normalising them makes the result depend
    on whichever query happened to be asked. Ranks are the one comparable thing.

    Each list contributes 1 / (rrf_k + rank). The constant flattens the top of the
    curve, so appearing high in *both* lists beats topping one of them - RRF rewards
    agreement. Assumes ids are unique within a list.
    """
    fused: dict[str, float] = defaultdict(float)
    seen: dict[str, dict] = {}

    for list_index, results in enumerate(result_lists):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            fused[chunk_id] += 1 / (rrf_k + rank)
            if chunk_id not in seen:
                seen[chunk_id] = dict(result)
            # Per-branch score and rank are kept, so it stays visible afterwards which
            # branch found a passage the other one missed.
            seen[chunk_id][f"score_branch_{list_index}"] = result["score"]
            seen[chunk_id][f"rank_branch_{list_index}"] = rank

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    output = []
    for chunk_id, score in ordered:
        result = seen[chunk_id]
        result["score"] = score
        output.append(result)
    return output
