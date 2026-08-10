"""
Retrieval strategies behind one common interface.

Everything downstream - the RAG engine, the evaluation harness, later the chat app -
talks to a Retriever and never to a vector store directly. That single seam is what
makes the planned comparison possible: swapping dense-only for hybrid, or wrapping a
re-ranker around either of them, changes which object is constructed and nothing else.
"""
from abc import ABC, abstractmethod

from config import RetrievalConfig
from data_preprocessing import MultiModalPreprocessor


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

    Its known weakness is the reason hybrid search is being tested: an embedding model
    represents meaning, not surface form. A query naming an exact designation such as
    "DGUV Information 203-071" has no reliable advantage over a nearly identical
    document, because the vectors of those two documents are almost the same.
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
        # Day 2: wrap dense + lexical in a HybridRetriever fusing both rankings via RRF.
        raise NotImplementedError("Hybrid search is not implemented yet.")

    if config.use_reranker:
        # Day 2: wrap the above in a RerankingRetriever re-scoring candidates with a
        # cross-encoder. Deliberately a wrapper, not a flag: re-ranking is a stage that
        # operates on any candidate list, whatever produced it.
        raise NotImplementedError("Re-ranking is not implemented yet.")

    return retriever