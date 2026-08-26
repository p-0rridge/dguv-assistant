"""
Why this file? Turns extracted elements into chunks, embeds them with BGE-M3 and writes
them to the local vector store. Also exports chunks.json, which everything downstream
reads: the gold set, the lexical index and any inspection of the chunking.
"""
import hashlib
import json
import re
from pathlib import Path

import chromadb
import tiktoken
import torch
from sentence_transformers import SentenceTransformer

from data_loader import PDFDocumentLoader

# German section and annex numbering ("6.4.3.7", "B.3", "Anhang C", "Tabelle 6.1"), so
# chunks break at structural boundaries instead of at an arbitrary character count.
SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(\d+(\.\d+){1,4}\b|[A-Z]\.\d+(\.\d+)*\b|Anhang\s+\S+|Tabelle\s+\S+|Bild\s+\S+)"
)


class MultiModalPreprocessor:

    def __init__(
        self,
        persist_dir: Path, # Local folder ChromaDB writes to. No server, no network port
        text_collection_name: str = "text_chunks",
        bge_model_name: str = "BAAI/bge-m3",
        max_text_chunk_tokens: int = 600, # Upper bound per chunk
        min_chunk_chars: int = 300, # Below this a chunk is a fragment and gets merged
        embed_batch_size: int = 16, # Chunks per forward pass; batching only affects speed
    ):
        """
        Chunk extracted PDF elements and embed them with BGE-M3.

        BGE-M3 rather than a shared text/image model: its 8192-token context lets
        chunking follow document structure instead of the model's limit.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_text_chunk_tokens = max_text_chunk_tokens
        self.min_chunk_chars = min_chunk_chars
        self.embed_batch_size = embed_batch_size

        # Not BGE-M3's own tokenizer, but close enough to size chunks under its limit.
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

        self.bge_model = SentenceTransformer(bge_model_name, device=self.device)
        self.bge_model.max_seq_length = 8192 # Or sentence-transformers falls back to less
        if hasattr(self.bge_model, "get_embedding_dimension"):
            self.text_embedding_dim = self.bge_model.get_embedding_dimension()
        else: # Older sentence-transformers
            self.text_embedding_dim = self.bge_model.get_sentence_embedding_dimension()

        # hnsw:space="cosine" set explicitly - Chroma defaults to squared L2. With
        # normalised embeddings the ranking is the same either way, but only cosine
        # distance is readable as a number, which an abstention threshold would need.
        # Applied only when the collection is created; changing it means re-indexing.
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.text_collection = client.get_or_create_collection(
            name=text_collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.token_encoder.encode(text))

    @staticmethod
    def make_chunk_id(chunk: dict) -> str:
        """
        Stable id derived from the chunk's own content, not a uuid.

        Makes indexing idempotent, lets the dense and lexical retrievers refer to the
        same chunk by the same key, and keeps evaluation runs comparable across days.
        A collision means identical text on the same page - a genuine duplicate.
        """
        raw = f"{chunk['source_file']}|{chunk['page_number']}|{chunk['type']}|{chunk['text']}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def chunk_text_elements(self, texts: list[dict]) -> list[dict]:
        """Group text elements into chunks, breaking at section headings."""
        chunks = []
        buffer_texts = []
        buffer_tokens = 0
        buffer_page = None

        def flush_buffer():
            if buffer_texts:
                chunks.append({
                    "type": "NarrativeText",
                    "text": "\n\n".join(buffer_texts),
                    "page_number": buffer_page,
                })

        for el in texts:
            el_text = el["text"]
            el_tokens = self._count_tokens(el_text)
            is_new_section = bool(SECTION_HEADING_PATTERN.match(el_text))

            # A heading only breaks once the buffer can stand on its own - otherwise a
            # run of consecutive headings produces one useless chunk each.
            buffer_length = sum(len(text) for text in buffer_texts)
            would_overflow = buffer_tokens + el_tokens > self.max_text_chunk_tokens
            breaks_here = would_overflow or (is_new_section and buffer_length >= self.min_chunk_chars)
            if buffer_texts and breaks_here:
                flush_buffer()
                buffer_texts, buffer_tokens, buffer_page = [], 0, None

            buffer_texts.append(el_text)
            buffer_tokens += el_tokens
            buffer_page = buffer_page or el["page_number"]

            if buffer_tokens > self.max_text_chunk_tokens:
                flush_buffer() # An oversized element becomes its own chunk, never split
                buffer_texts, buffer_tokens, buffer_page = [], 0, None

        flush_buffer()
        return self._merge_short_chunks(chunks)

    def _merge_short_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Fold fragments into the chunk that follows them.

        Forward, not backward: a fragment is usually a heading, which belongs to the
        section it opens. The merged chunk keeps the earlier page number.
        """
        merged: list[dict] = []
        pending: dict | None = None

        for chunk in chunks:
            if pending:
                chunk = {
                    "type": chunk["type"],
                    "text": f"{pending['text']}\n\n{chunk['text']}",
                    "page_number": pending["page_number"],
                }
                pending = None
            if len(chunk["text"]) < self.min_chunk_chars:
                pending = chunk
                continue
            merged.append(chunk)

        if pending: # Nothing follows it, so it joins the previous chunk rather than vanishing
            if merged:
                merged[-1]["text"] += f"\n\n{pending['text']}"
            else:
                merged.append(pending)
        return merged

    def chunk_tables(self, tables: list[dict]) -> list[dict]:
        """One chunk per table; tables are never merged with surrounding text."""
        return [
            {"type": "Table", "text": el["text"], "page_number": el["page_number"]}
            for el in tables
        ]

    def build_chunks(self, document: dict) -> list[dict]:
        """All chunks for one loaded document, tagged with source_file and chunk_id."""
        chunks = (
            self.chunk_text_elements(document["texts"])
            + self.chunk_tables(document["tables"])
        )
        for chunk in chunks:
            chunk["source_file"] = document["file_name"]
            chunk["chunk_id"] = self.make_chunk_id(chunk) # Assigned once, used everywhere
        return chunks

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        """Embed one string. Queries get BGE-M3's instruction prefix, passages do not."""
        if is_query:
            text = f"Represent this sentence for searching relevant passages: {text}"
        embedding = self.bge_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many passages in batches. Same vectors as embed_text, much faster."""
        embeddings = self.bge_model.encode(
            texts,
            batch_size=self.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return [vector.tolist() for vector in embeddings]

    def generate_embeddings(self, chunks: list[dict]) -> list[dict]:
        """Attach an embedding to every chunk."""
        if chunks:
            vectors = self.embed_texts([c["text"] for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                chunk["embedding"] = vector
        return chunks

    def add_to_vectorstore(self, chunks: list[dict]) -> None:
        """
        Write embedded chunks to ChromaDB, keyed by chunk_id.

        upsert() rather than add(): with content-derived ids a re-run overwrites each
        chunk in place. add() plus random ids silently doubled the collection on every
        run, which changes what "top 5" means and makes measurements incomparable.
        """
        text_chunks = self._deduplicate(chunks)
        if not text_chunks:
            return

        self.text_collection.upsert(
            ids=[c["chunk_id"] for c in text_chunks],
            embeddings=[c["embedding"] for c in text_chunks],
            documents=[c["text"] for c in text_chunks],
            metadatas=[
                {
                    "chunk_id": c["chunk_id"],
                    "type": c["type"],
                    "page_number": c["page_number"],
                    "source_file": c["source_file"],
                }
                for c in text_chunks
            ],
        )

    @staticmethod
    def _deduplicate(chunks: list[dict]) -> list[dict]:
        """One upsert call must never carry the same id twice."""
        seen = {}
        for chunk in chunks:
            seen.setdefault(chunk["chunk_id"], chunk)
        return list(seen.values())

    def process_document(self, document: dict) -> list[dict]:
        """Chunk, embed and index one loaded document."""
        chunks = self.build_chunks(document)
        chunks = self.generate_embeddings(chunks)
        self.add_to_vectorstore(chunks)
        return chunks

    @staticmethod
    def export_chunks(chunks: list[dict], path: Path) -> None:
        """
        Write chunks to JSON without their embeddings ----> goldset_builder, hybrid_search

        The shared source of truth for everything after indexing, so none of it has to
        re-run the slow PDF-and-embedding pipeline.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = [{k: v for k, v in chunk.items() if k != "embedding"} for chunk in chunks]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, ensure_ascii=False, indent=2)

    def search_text(self, query: str, k: int = 5) -> list[dict]:
        """
        Dense semantic search ----> DenseRetriever
        returns: chunk_id, text, metadata, score - best first

        score is cosine similarity in [0, 1], derived from the distance Chroma returns.
        """
        query_embedding = self.embed_text(query, is_query=True)
        results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "chunk_id": chunk_id,
                "text": document,
                "metadata": metadata,
                "score": 1.0 - distance,
            }
            for chunk_id, document, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]


if __name__ == "__main__":
    # Smoke test only. The real entry point is build_index.py, which also exports chunks.json.
    BASE_DIR = Path(__file__).resolve().parent.parent

    loader = PDFDocumentLoader()
    preprocessor = MultiModalPreprocessor(persist_dir=BASE_DIR / "chroma_db")

    docs = loader.load_directory(BASE_DIR / "data")
    for doc in docs:
        chunks = preprocessor.process_document(doc)
        print(f"Indexed: {doc['file_name']} ({len(chunks)} chunks)")
