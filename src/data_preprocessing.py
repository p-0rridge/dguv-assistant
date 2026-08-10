import hashlib
import json
import re
from pathlib import Path

import chromadb
import open_clip
import tiktoken
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from data_loader import PDFDocumentLoader

# Matches the German norm's section/annex numbering (e.g. "6.4.3.7", "B.3", "Anhang C", "Tabelle 6.1")
# so text chunking can break at structural boundaries instead of at an arbitrary character count.
SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(\d+(\.\d+){1,4}\b|[A-Z]\.\d+(\.\d+)*\b|Anhang\s+\S+|Tabelle\s+\S+|Bild\s+\S+)"
)


class MultiModalPreprocessor:

    def __init__(
        self,
        persist_dir: Path,
        text_collection_name: str = "text_chunks",
        image_collection_name: str = "image_chunks",
        bge_model_name: str = "BAAI/bge-m3",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        max_text_chunk_tokens: int = 600,
        embed_batch_size: int = 16,
    ):
        """
        Chunk extracted PDF elements and embed them with two separate models:
        BGE-M3 for text/tables (long context, strong German support), and CLIP for
        images (shared text/image space, so a text query can retrieve an image directly).
        persist_dir: Local folder ChromaDB writes to on disk. No server, no network
            port, nothing to install beyond the pip packages above.
        embed_batch_size: How many chunks BGE-M3 encodes in one forward pass. Encoding
            one chunk at a time wastes most of the available compute; batching is the
            single cheapest speedup for a full re-index and changes nothing about the
            resulting vectors.

        Two embedding spaces on purpose: a single CLIP model would force text chunks
        down to CLIP's ~77-token limit, far too small for structured legal text, while
        BGE-M3 handles up to 8192 - so chunking can follow document structure instead
        of the embedding model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_text_chunk_tokens = max_text_chunk_tokens
        self.embed_batch_size = embed_batch_size

        # Token counting for chunk sizing. This is a fast, good-enough estimate; it is not
        # BGE-M3's own tokenizer, but close enough to size chunks safely under its 8192-token limit.
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

        # --- Text/table embedding model (BGE-M3) ---
        self.bge_model = SentenceTransformer(bge_model_name, device=self.device)
        # BGE-M3 ships with an 8192-token context; make sure sentence-transformers
        # actually uses it instead of falling back to a shorter default.
        self.bge_model.max_seq_length = 8192
        # get_embedding_dimension() is the current sentence-transformers API name;
        # fall back to the older get_sentence_embedding_dimension() on older versions.
        if hasattr(self.bge_model, "get_embedding_dimension"):
            self.text_embedding_dim = self.bge_model.get_embedding_dimension()
        else:
            self.text_embedding_dim = self.bge_model.get_sentence_embedding_dimension()

        # --- Image embedding model (CLIP) ---
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

        # --- Local, server-less vector store ---
        # hnsw:space="cosine" is set explicitly: Chroma otherwise defaults to squared L2.
        # With normalised embeddings both produce the same *ranking*, but only cosine
        # distance is interpretable as a number - and readable scores are needed later to
        # set an abstention threshold ("say I don't know below X").
        # NOTE: this metadata is only applied when the collection is first created. An
        # existing collection keeps whatever space it was built with, so switching this
        # requires deleting persist_dir and re-indexing.
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.text_collection = client.get_or_create_collection(
            name=text_collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.image_collection = client.get_or_create_collection(
            name=image_collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _count_tokens(self, text: str) -> int:
        """Estimate the token count of a text string."""
        return len(self.token_encoder.encode(text))

    @staticmethod
    def make_chunk_id(chunk: dict) -> str:
        """
        Build a stable, content-derived id for a chunk.
        chunk: A chunk dict that already carries source_file, page_number, type and text
        returns: 16-character hex digest

        Why content-derived instead of uuid4: the id has to be the same on every run.
        That makes indexing idempotent (re-running never duplicates a chunk), lets the
        dense and the lexical retriever refer to the same chunk by the same key when
        their result lists are fused, and keeps evaluation runs comparable across days.

        Two chunks colliding on this id means identical text on the same page of the
        same document - a genuine duplicate, which should collapse into one entry.
        """
        raw = f"{chunk['source_file']}|{chunk['page_number']}|{chunk['type']}|{chunk['text']}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def chunk_text_elements(self, texts: list[dict]) -> list[dict]:
        """
        Group narrative text elements into chunks, breaking at section headings and
        never exceeding max_text_chunk_tokens.
        texts: List of NarrativeText element dicts, as returned by categorize_elements
        returns: List of chunk dicts with keys type, text, page_number
        """
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

            # Start a new chunk if this element opens a new section, or the buffer
            # would otherwise exceed the target token budget.
            would_overflow = buffer_tokens + el_tokens > self.max_text_chunk_tokens
            if buffer_texts and (is_new_section or would_overflow):
                flush_buffer()
                buffer_texts, buffer_tokens, buffer_page = [], 0, None

            buffer_texts.append(el_text)
            buffer_tokens += el_tokens
            buffer_page = buffer_page or el["page_number"]

            # A single element longer than the budget still becomes its own chunk
            # (never split mid-element), and is flushed immediately.
            if buffer_tokens > self.max_text_chunk_tokens:
                flush_buffer()
                buffer_texts, buffer_tokens, buffer_page = [], 0, None

        flush_buffer()
        return chunks

    def chunk_tables(self, tables: list[dict]) -> list[dict]:
        """Each table becomes its own chunk; tables are never merged with surrounding text."""
        return [
            {"type": "Table", "text": el["text"], "page_number": el["page_number"]}
            for el in tables
        ]

    def chunk_images(self, images: list[dict]) -> list[dict]:
        """Each extracted page image becomes its own chunk, keyed by its saved file path."""
        return [
            {
                "type": "Image",
                "text": el["text"],
                "page_number": el["page_number"],
                "image_path": el["image_path"],
            }
            for el in images
        ]

    def build_chunks(self, document: dict) -> list[dict]:
        """
        Build the full set of chunks for a single loaded document.
        document: A dict as returned by PDFDocumentLoader.load_single_pdf
        returns: List of chunk dicts (text/table/image), tagged with source_file and chunk_id
        """
        chunks = (
            self.chunk_text_elements(document["texts"])
            + self.chunk_tables(document["tables"])
            + self.chunk_images(document["images"])
        )
        for chunk in chunks:
            chunk["source_file"] = document["file_name"]
            chunk.setdefault("image_path", None)
            # Assigned here, once, so every downstream consumer (vector store, exported
            # JSON, lexical index, evaluation) refers to a chunk by the same key.
            chunk["chunk_id"] = self.make_chunk_id(chunk)
        return chunks

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        """
        Embed a single text string with BGE-M3.
        is_query: BGE-M3's retrieval quality improves slightly when queries carry a short
            instruction prefix; passages (the chunks we index) are embedded as-is.
        """
        if is_query:
            text = f"Represent this sentence for searching relevant passages: {text}"
        embedding = self.bge_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed many passages in batches. Identical output to calling embed_text in a loop,
        but several times faster on CPU because the model processes embed_batch_size
        chunks per forward pass instead of one.
        """
        embeddings = self.bge_model.encode(
            texts,
            batch_size=self.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return [vector.tolist() for vector in embeddings]

    def embed_image(self, image_path: str) -> list[float]:
        """Embed an image file with the CLIP image tower."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def generate_embeddings(self, chunks: list[dict]) -> list[dict]:
        """
        Attach an embedding to each chunk: text/table chunks go through BGE-M3 in
        batches, image chunks are embedded individually via CLIP.
        chunks: List of chunk dicts, as returned by build_chunks
        returns: The same list, with an added "embedding" key per chunk
        """
        text_chunks = [c for c in chunks if c["type"] != "Image"]
        if text_chunks:
            vectors = self.embed_texts([c["text"] for c in text_chunks])
            for chunk, vector in zip(text_chunks, vectors):
                chunk["embedding"] = vector

        for chunk in chunks:
            if chunk["type"] == "Image":
                chunk["embedding"] = self.embed_image(chunk["image_path"])
        return chunks

    def add_to_vectorstore(self, chunks: list[dict]) -> None:
        """
        Write embedded chunks into the local ChromaDB collections, keyed by chunk_id.
        chunks: List of chunk dicts, as returned by generate_embeddings

        Uses upsert() rather than add(): with content-derived ids, re-running the whole
        pipeline overwrites each chunk in place instead of inserting a second copy. With
        add() plus random ids, every re-run silently doubled the collection - which
        quietly changes what "top 5 results" means and makes measurements incomparable.
        """
        text_chunks = self._deduplicate([c for c in chunks if c["type"] in ("NarrativeText", "Table")])
        image_chunks = self._deduplicate([c for c in chunks if c["type"] == "Image"])

        if text_chunks:
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

        if image_chunks:
            self.image_collection.upsert(
                ids=[c["chunk_id"] for c in image_chunks],
                embeddings=[c["embedding"] for c in image_chunks],
                # Chroma documents must be text; the raw image lives at metadata["image_path"].
                documents=[c["text"] for c in image_chunks],
                metadatas=[
                    {
                        "chunk_id": c["chunk_id"],
                        "page_number": c["page_number"],
                        "source_file": c["source_file"],
                        "image_path": c["image_path"],
                    }
                    for c in image_chunks
                ],
            )

    @staticmethod
    def _deduplicate(chunks: list[dict]) -> list[dict]:
        """Drop chunks sharing a chunk_id, so one upsert call never carries the same id twice."""
        seen = {}
        for chunk in chunks:
            seen.setdefault(chunk["chunk_id"], chunk)
        return list(seen.values())

    def process_document(self, document: dict) -> list[dict]:
        """Run the full chunking + embedding + indexing pipeline for one loaded document."""
        chunks = self.build_chunks(document)
        chunks = self.generate_embeddings(chunks)
        self.add_to_vectorstore(chunks)
        return chunks

    @staticmethod
    def export_chunks(chunks: list[dict], path: Path) -> None:
        """
        Write all chunks to a JSON file, without their embeddings.
        chunks: List of chunk dicts
        path: Destination file, created together with any missing parent folders

        This file is the shared source of truth for everything that follows: the
        evaluation questions are generated from it, the lexical index is built from it,
        and chunking changes can be inspected by diffing it - none of which requires
        re-running the slow PDF-and-embedding pipeline.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = [{k: v for k, v in chunk.items() if k != "embedding"} for chunk in chunks]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, ensure_ascii=False, indent=2)

    def search_text(self, query: str, k: int = 5) -> list[dict]:
        """
        Dense semantic search over the text collection.
        query: Natural-language search query
        k: Number of results to return
        returns: List of dicts with keys chunk_id, text, metadata, score - ordered best first

        score is a cosine similarity in [0, 1] (1 = identical direction), derived from the
        cosine distance Chroma returns. It is reported here because later stages need it:
        the fusion step needs a per-retriever ranking, and the abstention rule needs a
        number it can threshold.
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
    # Kept only as a smoke test for this module. The real pipeline entry point is
    # build_index.py, which also writes artifacts/chunks.json.
    BASE_DIR = Path(__file__).resolve().parent.parent

    loader = PDFDocumentLoader(output_dir=BASE_DIR / "extracted_images")
    preprocessor = MultiModalPreprocessor(persist_dir=BASE_DIR / "chroma_db")

    docs = loader.load_directory(BASE_DIR / "data")
    for doc in docs:
        chunks = preprocessor.process_document(doc)
        print(f"Indexed: {doc['file_name']} ({len(chunks)} chunks)")