"""
Single entry point for building the search index.

Usage:
    python src/build_index.py                      # full corpus
    python src/build_index.py --corpus dev         # small corpus, for fast debugging
    python src/build_index.py --reset              # delete the store first, then rebuild

Rebuilding is safe to repeat: chunk ids are content-derived and written with upsert,
so a second run overwrites in place instead of duplicating. --reset is only needed when
documents were removed from the corpus, since upsert never deletes anything.
"""
import argparse
import shutil
import statistics

import config
from data_loader import PDFDocumentLoader
from data_preprocessing import MultiModalPreprocessor


class IndexBuilder:
    """Loads a corpus, chunks and embeds it, writes it to the vector store and to JSON."""

    def __init__(self, corpus: str = "full", reset: bool = False):
        """
        corpus: "full" for the evaluation corpus, "dev" for the small debugging corpus
        reset: Delete the vector store before building, for a guaranteed clean state
        """
        self.corpus = corpus
        self.data_dir, self.chroma_dir = config.corpus_paths(corpus)
        # Separate output files so a quick dev run can never overwrite the chunks the
        # evaluation questions were generated from.
        self.chunks_file = (
            config.CHUNKS_FILE if corpus == "full"
            else config.ARTIFACTS_DIR / "chunks_dev.json"
        )

        if reset and self.chroma_dir.exists():
            print(f"Removing existing vector store at {self.chroma_dir}")
            shutil.rmtree(self.chroma_dir)

        self.loader = PDFDocumentLoader(output_dir=config.IMAGES_DIR)
        self.preprocessor = MultiModalPreprocessor(persist_dir=self.chroma_dir)

    def run(self) -> list[dict]:
        """Process every PDF in the corpus and return all chunks."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Corpus folder not found: {self.data_dir}")

        documents = self.loader.load_directory(self.data_dir)
        if not documents:
            raise FileNotFoundError(f"No PDFs found in {self.data_dir}")

        all_chunks = []
        for document in documents:
            chunks = self.preprocessor.process_document(document)
            all_chunks.extend(chunks)
            print(f"  indexed {document['file_name']}: {len(chunks)} chunks")

        self.preprocessor.export_chunks(all_chunks, self.chunks_file)
        print(f"\nWrote {len(all_chunks)} chunks to {self.chunks_file}")
        return all_chunks

    def report(self, chunks: list[dict]) -> None:
        """
        Print a per-document breakdown of the index.

        The median token count per chunk is the interesting column: chunking is supposed
        to break at section headings, so a healthy document produces chunks of varying,
        mostly moderate length. A median sitting close to max_text_chunk_tokens means no
        headings were recognised and the text was only ever cut when the token budget ran
        out - which is a chunking failure, not a property of the document.
        """
        encoder = self.preprocessor.token_encoder
        cap = self.preprocessor.max_text_chunk_tokens

        by_document: dict[str, list[dict]] = {}
        for chunk in chunks:
            by_document.setdefault(chunk["source_file"], []).append(chunk)

        header = f"{'Document':<34}{'chunks':>8}{'tables':>8}{'images':>8}{'median tok':>12}{'at cap %':>10}"
        print("\n" + header)
        print("-" * len(header))

        for source_file, document_chunks in sorted(by_document.items()):
            text_chunks = [c for c in document_chunks if c["type"] == "NarrativeText"]
            token_counts = [len(encoder.encode(c["text"])) for c in text_chunks] or [0]
            at_cap = sum(1 for count in token_counts if count >= cap * 0.9)

            print(
                f"{source_file[:33]:<34}"
                f"{len(document_chunks):>8}"
                f"{sum(1 for c in document_chunks if c['type'] == 'Table'):>8}"
                f"{sum(1 for c in document_chunks if c['type'] == 'Image'):>8}"
                f"{statistics.median(token_counts):>12.0f}"
                f"{100 * at_cap / len(token_counts):>9.0f}%"
            )

        print("-" * len(header))
        print(f"{'TOTAL':<34}{len(chunks):>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DGUV search index.")
    parser.add_argument("--corpus", choices=["full", "dev"], default="full")
    parser.add_argument("--reset", action="store_true", help="delete the vector store first")
    args = parser.parse_args()

    builder = IndexBuilder(corpus=args.corpus, reset=args.reset)
    chunks = builder.run()
    builder.report(chunks)


if __name__ == "__main__":
    main()