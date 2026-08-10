# pip install "sentence-transformers>=3" "chromadb==1.5.9" torch "open-clip-torch==2.32.0" "pillow==12.3.0" "tiktoken==0.13.0"
import re
import uuid
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
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_text_chunk_tokens = max_text_chunk_tokens

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
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.text_collection = client.get_or_create_collection(name=text_collection_name)
        self.image_collection = client.get_or_create_collection(name=image_collection_name)

    def _count_tokens(self, text: str) -> int:
        """Estimate the token count of a text string."""
        return len(self.token_encoder.encode(text))

    def chunk_text_elements(self, texts: list[dict]) -> list[dict]:
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
        chunks = (
            self.chunk_text_elements(document["texts"])
            + self.chunk_tables(document["tables"])
            + self.chunk_images(document["images"])
        )
        for chunk in chunks:
            chunk["source_file"] = document["file_name"]
            chunk.setdefault("image_path", None)
        return chunks

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        if is_query:
            text = f"Represent this sentence for searching relevant passages: {text}"
        embedding = self.bge_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_image(self, image_path: str) -> list[float]:
        """Embed an image file with the CLIP image tower."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def generate_embeddings(self, chunks: list[dict]) -> list[dict]:
        for chunk in chunks:
            if chunk["type"] == "Image":
                chunk["embedding"] = self.embed_image(chunk["image_path"])
            else:
                chunk["embedding"] = self.embed_text(chunk["text"])
        return chunks

    def add_to_vectorstore(self, chunks: list[dict]) -> None:
        text_chunks = [c for c in chunks if c["type"] in ("NarrativeText", "Table")]
        image_chunks = [c for c in chunks if c["type"] == "Image"]

        if text_chunks:
            self.text_collection.add(
                ids=[str(uuid.uuid4()) for _ in text_chunks],
                embeddings=[c["embedding"] for c in text_chunks],
                documents=[c["text"] for c in text_chunks],
                metadatas=[
                    {"type": c["type"], "page_number": c["page_number"], "source_file": c["source_file"]}
                    for c in text_chunks
                ],
            )

        if image_chunks:
            self.image_collection.add(
                ids=[str(uuid.uuid4()) for _ in image_chunks],
                embeddings=[c["embedding"] for c in image_chunks],
                # Chroma documents must be text; the raw image lives at metadata["image_path"].
                documents=[c["text"] for c in image_chunks],
                metadatas=[
                    {"page_number": c["page_number"], "source_file": c["source_file"], "image_path": c["image_path"]}
                    for c in image_chunks
                ],
            )

    def process_document(self, document: dict) -> list[dict]:
        """Run the full chunking + embedding + indexing pipeline for one loaded document."""
        chunks = self.build_chunks(document)
        chunks = self.generate_embeddings(chunks)
        self.add_to_vectorstore(chunks)
        return chunks

    def search_text(self, query: str, k: int = 5) -> list[dict]:
        """
        Plain dense semantic search over the text collection (no BM25/lexical component -
        see the trade-off note in __init__). Good starting point until hybrid search
        becomes necessary.
        query: Natural-language search query
        k: Number of results to return
        returns: List of matched documents with their metadata
        """
        query_embedding = self.embed_text(query, is_query=True)
        results = self.text_collection.query(query_embeddings=[query_embedding], n_results=k)
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    SUB_DIR = BASE_DIR / "data"
    IMAGES_DIR = BASE_DIR / "extracted_images"
    PERSIST_DIR = BASE_DIR / "chroma_db"

    loader = PDFDocumentLoader(output_dir=IMAGES_DIR)
    preprocessor = MultiModalPreprocessor(persist_dir=PERSIST_DIR)

    docs = loader.load_directory(SUB_DIR)
    for doc in docs:
        preprocessor.process_document(doc)
        print(f"Indexed: {doc['file_name']}")
