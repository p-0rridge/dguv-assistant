"""
Retrieval strategies behind one common interface.

Everything downstream - the RAG engine, the evaluation harness, later the chat app -
talks to a Retriever and never to a vector store directly. That single seam is what
makes the planned comparison possible: swapping dense-only for hybrid, or wrapping a
re-ranker around either of them, changes which object is constructed and nothing else.
"""
from abc import ABC, abstractmethod

import torch
from sentence_transformers import CrossEncoder

import config as config_module
from config import RetrievalConfig
from data_preprocessing import MultiModalPreprocessor
from hybrid_search import BM25Index, reciprocal_rank_fusion


class Retriever(ABC):
    """
    Common contract for all retrieval strategies.

    retrieve() must return a list of dicts ordered best-first, each with:
        chunk_id: str   - stable id, shared across all retrievers and the exported chunks
        text:     str   - the chunk text
        metadata: dict  - at minimum source_file and page_number
        score:    float - higher is better; comparable within one retriever, not across

    The score is deliberately not normalised across strategies. Cosine similarities and
    cross-encoder logits live on different scales, and pretending otherwise would hide
    exactly the difference the evaluation is supposed to measure. Where results from two
    retrievers have to be combined, ranks are fused rather than scores.
    """

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[dict]:
        """Return the k most relevant chunks for a query, best first."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used in logs and result files."""


class DenseRetriever(Retriever):
    """
    Pure semantic search: embed the query with BGE-M3 and take the nearest chunks by
    cosine similarity. This is the MVP baseline - the variant every later measurement
    is compared against.
    """

    def __init__(self, preprocessor: MultiModalPreprocessor):
        """
        preprocessor: An initialised MultiModalPreprocessor owning the embedding model
            and the Chroma collections
        """
        self.preprocessor = preprocessor

    @property
    def name(self) -> str:
        return "dense"

    def retrieve(self, query: str, k: int) -> list[dict]:
        """Return the k nearest chunks by cosine similarity, best first."""
        return self.preprocessor.search_text(query, k=k)


class HybridRetriever(Retriever):
    """
    Runs dense and lexical retrieval side by side and fuses their rankings.

    The two branches fail in different places, which is the entire premise. The dense
    branch handles paraphrase - a question worded nothing like the regulation - and
    blurs terms that differ by a few characters. The lexical branch does the opposite:
    it matches "SELV" and not "PELV", and returns near-noise for a question containing
    no distinctive term. Fusing them is a bet that these blind spots do not overlap.

    Unlike re-ranking, this changes *which* passages reach the candidate pool, not only
    their order. Recall@candidate_k is therefore no longer a control here - it is the
    quantity under measurement.
    """

    def __init__(
        self,
        dense: Retriever,
        index: BM25Index,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ):
        """
        dense: The semantic branch, normally a DenseRetriever
        index: The lexical branch, built from the same chunks as the vector store
        rrf_k: Fusion constant, from config
        candidate_k: How deep each branch is read before fusing. Fixed at construction
            for the same reason as in RerankingRetriever: the evaluation asks for 20
            results and the answer engine for 5, and both have to fuse the same pool or
            they are not the same system. Fusing only the top 5 of each would also give
            the fusion almost nothing to work with - a passage cannot be rescued by
            agreement if neither list was read far enough to contain it.
        """
        self.dense = dense
        self.index = index
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    @property
    def name(self) -> str:
        return "hybrid"

    def retrieve(self, query: str, k: int) -> list[dict]:
        """Fetch from both branches, fuse by rank, return the best k."""
        depth = max(k, self.candidate_k)
        dense_results = self.dense.retrieve(query, depth)
        lexical_results = self.index.search(query, depth)
        fused = reciprocal_rank_fusion([dense_results, lexical_results], rrf_k=self.rrf_k)
        return fused[:k]


class RerankingRetriever(Retriever):
    """
    Re-scores the candidates of another retriever with a cross-encoder.

    A bi-encoder like BGE-M3 embeds query and passage separately, so it can only compare
    two summaries of meaning. A cross-encoder reads query and passage together in one
    pass and can therefore judge whether the passage actually answers *this* question -
    at the cost of one forward pass per candidate, which is why it can only be applied
    to a shortlist rather than to the whole collection.

    Deliberately a wrapper rather than a flag: re-ranking is a stage that operates on any
    candidate list, whatever produced it. The same class will wrap the hybrid retriever
    without a line changing here.
    """

    def __init__(
        self,
        base: Retriever,
        model_name: str,
        candidate_k: int,
        batch_size: int = 8,
        max_length: int = 512,
    ):
        """
        base: The retriever supplying the candidates
        model_name: Cross-encoder checkpoint
        candidate_k: How many candidates to score, regardless of how many are requested.
            Fixing this at construction keeps the measurement honest: the evaluation asks
            for 20 results and the answer engine for 5, but both must re-rank the same
            pool, or the two would not be comparing the same system.
        max_length: Token budget per query-passage pair. 512 covers 99% of the chunks in
            this corpus; raising it doubles the cost per pair for the benefit of a
            handful of outliers, which on CPU is the difference between a run that
            finishes and one that appears to hang.
        """
        self.base = base
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    @property
    def name(self) -> str:
        return f"{self.base.name}+rerank"

    def retrieve(self, query: str, k: int) -> list[dict]:
        """
        Fetch candidates from the base retriever, re-score them, return the best k.

        Always fetches at least candidate_k, so asking for the top 5 still re-ranks the
        full shortlist. The dense score is kept alongside the new one, which makes it
        possible to see afterwards which chunks the re-ranker actually moved.
        """
        candidates = self.base.retrieve(query, max(k, self.candidate_k))
        if not candidates:
            return []

        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        for candidate, score in zip(candidates, scores):
            candidate["dense_score"] = candidate["score"]
            candidate["score"] = float(score)

        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        return candidates[:k]


def build_retriever(config: RetrievalConfig, preprocessor: MultiModalPreprocessor) -> Retriever:
    """
    Assemble the retrieval stack described by a config.
    config: One of the named variants from config.py
    preprocessor: Provides the embedding model and the vector store
    returns: A ready-to-use Retriever

    Single place where "which variant am I running" is decided, so the evaluation
    harness and the chat app can never drift apart.
    """
    retriever: Retriever = DenseRetriever(preprocessor)

    if config.use_bm25:
        retriever = HybridRetriever(
            dense=retriever,
            index=BM25Index.from_file(config_module.CHUNKS_FILE),
            rrf_k=config.rrf_k,
            candidate_k=config.candidate_k,
        )

    if config.use_reranker:
        retriever = RerankingRetriever(
            base=retriever,
            model_name=config.reranker_model,
            candidate_k=config.candidate_k,
        )

    return retriever