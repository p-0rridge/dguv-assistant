"""
Central configuration: paths and named retrieval variants.

Why this file exists: every stage of this system (dense-only MVP, hybrid search,
hybrid + re-ranker) has to be reproducible and comparable. Keeping each stage as a
named, immutable config object - instead of as flags scattered through the code -
means an evaluation run can be identified by a single string, and a result recorded
today stays interpretable tomorrow.
"""
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"              # full evaluation corpus
DATA_DEV_DIR = BASE_DIR / "data_dev"      # 3 documents, for fast debugging only
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_DEV_DIR = BASE_DIR / "chroma_db_dev"
IMAGES_DIR = BASE_DIR / "extracted_images"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHUNKS_FILE = ARTIFACTS_DIR / "chunks.json"
GOLDSET_FILE = ARTIFACTS_DIR / "goldset.json"
UNANSWERABLE_FILE = ARTIFACTS_DIR / "unanswerable.json"
RESULTS_DIR = ARTIFACTS_DIR / "results"


def corpus_paths(corpus: str = "full") -> tuple[Path, Path]:
    """
    Resolve which document folder and which vector store to use.
    corpus: "full" for the evaluation corpus, "dev" for the small debugging corpus
    returns: (data_dir, chroma_dir)

    Two separate Chroma directories on purpose: debugging against three documents
    rebuilds in ~2 minutes instead of ~40, and it can never contaminate the index
    the evaluation numbers were measured on.
    """
    if corpus == "dev":
        return DATA_DEV_DIR, CHROMA_DEV_DIR
    if corpus == "full":
        return DATA_DIR, CHROMA_DIR
    raise ValueError(f"Unknown corpus '{corpus}'. Use 'full' or 'dev'.")


# --- Retrieval variants ------------------------------------------------------
@dataclass(frozen=True)
class RetrievalConfig:
    """
    One retrieval variant. frozen=True makes instances immutable: a config cannot be
    modified halfway through an evaluation run, so a result file can always be traced
    back to exactly these settings.

    name: Identifier, also used as the results filename
    top_k: Chunks finally handed to the LLM as context
    candidate_k: Chunks pulled from the store before re-ranking. Also the depth at
        which recall is measured, so Recall@candidate_k describes the candidate pool
        independently of how it is later reordered.
    use_bm25: Add a lexical BM25 branch and fuse it with the dense results
    rrf_k: Reciprocal Rank Fusion constant, taken unchanged from the original paper
    use_reranker: Re-score the candidates with a cross-encoder
    reranker_model: Cross-encoder matching the BGE-M3 embedding model
    """
    name: str
    top_k: int = 5
    candidate_k: int = 20
    use_bm25: bool = False
    rrf_k: int = 60
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


# The frozen baseline. Everything measured later is measured against this.
# Deliberately unchanged from the original MVP: dense-only retrieval, top 5 chunks.
MVP_BASELINE = RetrievalConfig(name="mvp_baseline")

# Same candidate pool, re-ordered by a cross-encoder. Recall@20 must therefore stay
# identical to the baseline - if it moves, something other than the ranking changed.
RERANK = RetrievalConfig(name="rerank", use_reranker=True)

# Dense and lexical retrieval fused by Reciprocal Rank Fusion. Unlike RERANK this
# changes which chunks enter the candidate pool, so Recall@20 is expected to move -
# here it is the measurement rather than the control.
HYBRID = RetrievalConfig(name="hybrid", use_bm25=True)

# Both improvements at once: fusion decides the shortlist, the cross-encoder orders it.
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