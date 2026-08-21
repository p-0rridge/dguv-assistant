"""

Why this file? Central configuration: Every stage of this system (dense-only MVP, hybrid search,
hybrid + re-ranker) has to be reproducible and comparable. Keeping each stage as a
named, immutable config object.
"""
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"              # full evaluation corpus
DATA_DEV_DIR = BASE_DIR / "data_dev"      # 3 documents, for fast debugging only
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_DEV_DIR = BASE_DIR / "chroma_db_dev"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHUNKS_FILE = ARTIFACTS_DIR / "chunks.json"
GOLDSET_FILE = ARTIFACTS_DIR / "goldset.json"
TITLES_FILE = ARTIFACTS_DIR / "document_titles.json"
UNANSWERABLE_FILE = ARTIFACTS_DIR / "unanswerable.json"
RESULTS_DIR = ARTIFACTS_DIR / "results"


def corpus_paths(corpus: str = "full") -> tuple[Path, Path]:
    if corpus == "dev": #smaller debugging corpus --> against 3 documents: faster
        return DATA_DEV_DIR, CHROMA_DEV_DIR
    if corpus == "full": #full evaluation corpus
        return DATA_DIR, CHROMA_DIR
    raise ValueError(f"Unknown corpus '{corpus}'. Use 'full' or 'dev'.")


# Models ------------------------------------------------------------------
# Check OpenAI's current
# list before a run; gpt-4.1-mini is a well-established fallback. 
# (A light model is enough for both jobs here.)
ANSWER_MODEL = "gpt-5.4-mini"
GOLDSET_MODEL = "gpt-5.4-mini"


# Retrieval variants ------------------------------------------------------
@dataclass(frozen=True) #makes instances immutable
class RetrievalConfig:
    name: str # Identifies the run, and names the results file ----> run_eval, compare_runs
    top_k: int = 5 # Chunks handed to the LLM as context ----> rag_engine
    candidate_k: int = 20 # Pool depth, and where recall is measured ----> retriever, evaluator
    use_bm25: bool = False # Add the lexical branch and fuse it ----> build_retriever
    rrf_k: int = 60 # Fusion constant, unchanged from the RRF paper ----> HybridRetriever
    use_reranker: bool = False # Re-score the candidates ----> build_retriever
    reranker_model: str = "BAAI/bge-reranker-v2-m3" # ----> RerankingRetriever


MVP_BASELINE = RetrievalConfig(name="mvp_baseline") 

RERANK = RetrievalConfig(name="rerank", use_reranker=True) 

HYBRID = RetrievalConfig(name="hybrid", use_bm25=True)

HYBRID_RERANK = RetrievalConfig(name="hybrid_rerank", use_bm25=True, use_reranker=True)

CONFIGS = {
    variant.name: variant
    for variant in (MVP_BASELINE, RERANK, HYBRID, HYBRID_RERANK)
}


def get_config(name: str) -> RetrievalConfig:
    """Look up a named config, failing loudly on typos rather than silently defaulting."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {sorted(CONFIGS)}")
    return CONFIGS[name]