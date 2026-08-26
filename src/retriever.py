"""
Why this file? One interface in front of every retrieval strategy. Everything
downstream talks to a Retriever and never to a vector store directly, so swapping
dense-only for hybrid, or wrapping a re-ranker around either, changes which object is
constructed and nothing else. That seam is what makes the comparison possible.
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
    retrieve() returns dicts, best first: chunk_id, text, metadata, score.

    Scores are deliberately not normalised across strategies - cosine similarities and
    cross-encoder logits live on different scales, and pretending otherwise would hide
    the difference the evaluation exists to measure. Where two result lists have to be
    combined, ranks are fused rather than scores.
    """

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[dict]:
        """The k most relevant chunks, best first."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used in logs and result files."""


class DenseRetriever(Retriever):
    """Semantic search only. The baseline every later measurement is compared against."""

    def __init__(self, preprocessor: MultiModalPreprocessor):
        self.preprocessor = preprocessor

    @property
    def name(self) -> str:
        return "dense"

    def retrieve(self, query: str, k: int) -> list[dict]:
        return self.preprocessor.search_text(query, k=k)


class HybridRetriever(Retriever):
    """
    Dense and lexical retrieval side by side, rankings fused.

    The dense branch handles paraphrase and blurs terms differing by a few characters.
    The lexical branch does the opposite: it separates SELV from PELV, and returns
    near-noise for a question with no distinctive term. Fusing them bets that the two
    blind spots do not overlap.

    Unlike re-ranking, this changes *which* passages reach the pool, so Recall@k is the
    measurement here rather than the control.
    """

    def __init__(
        self,
        dense: Retriever,
        index: BM25Index, # Built from the same chunks as the vector store
        rrf_k: int = 60,
        candidate_k: int = 20, # Depth read from each branch before fusing
    ):
        self.dense = dense
        self.index = index
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    @property
    def name(self) -> str:
        return "hybrid"

    def retrieve(self, query: str, k: int) -> list[dict]:
        """Fetch from both branches, fuse by rank, return the best k."""
        # max() so the evaluation (k=20) and the answer engine (k=5) fuse the same pool.
        # Fusing only the top 5 would also leave the fusion nothing to work with: a
        # passage cannot rise through agreement if neither list was read far enough.
        depth = max(k, self.candidate_k)
        dense_results = self.dense.retrieve(query, depth)
        lexical_results = self.index.search(query, depth)
        fused = reciprocal_rank_fusion([dense_results, lexical_results], rrf_k=self.rrf_k)
        return fused[:k]


class RerankingRetriever(Retriever):
    """
    Re-scores another retriever's candidates with a cross-encoder.

    A bi-encoder embeds query and passage separately and can only compare two summaries
    of meaning. A cross-encoder reads both together and judges whether the passage
    answers *this* question - at one forward pass per candidate, which is why it only
    ever sees a shortlist.

    A wrapper rather than a flag: re-ranking operates on any candidate list, whatever
    produced it, so it wraps the hybrid retriever without a line changing here.
    """

    def __init__(
        self,
        base: Retriever, # Supplies the candidates
        model_name: str,
        candidate_k: int, # Scored regardless of how many are requested
        batch_size: int = 8,
        max_length: int = 512, # Covers almost every chunk; raising it doubles cost per pair
    ):
        self.base = base
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    @property
    def name(self) -> str:
        return f"{self.base.name}+rerank"

    def retrieve(self, query: str, k: int) -> list[dict]:
        """Fetch candidates, re-score them, return the best k."""
        candidates = self.base.retrieve(query, max(k, self.candidate_k))
        if not candidates:
            return []

        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        for candidate, score in zip(candidates, scores):
            candidate["dense_score"] = candidate["score"] # Kept, so moved chunks stay visible
            candidate["score"] = float(score)

        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        return candidates[:k]


def build_retriever(config: RetrievalConfig, preprocessor: MultiModalPreprocessor) -> Retriever:
    """
    Assemble the stack a config describes ----> run_eval, demo, rag_engine

    The single place where "which variant am I running" is decided, so the evaluation
    harness and the answer engine can never drift apart.
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
