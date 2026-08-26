"""
Why this file? PDFs in, text and table elements out. The one place raw PDF text enters
the system, so cleaning happens here and everything downstream sees the same text.
"""
from pathlib import Path

import fitz  # PyMuPDF

from data_cleaning import TextCleaner


class PDFDocumentLoader:

    def __init__(self, cleaner: TextCleaner | None = None, drop_boilerplate: bool = True):
        self.cleaner = cleaner or TextCleaner() # Repairs typesetting artefacts ----> data_cleaning
        self.drop_boilerplate = drop_boilerplate # Drop tables of contents, imprints, numbering

    def _accept(self, text: str, element_type: str) -> str | None:
        """Clean a block, or return None if it carries no content."""
        cleaned = self.cleaner.clean(text)
        if not cleaned:
            return None
        if self.drop_boilerplate and self.cleaner.is_boilerplate(cleaned, element_type):
            return None
        return cleaned

    def extract_pdf_elements(self, file_path: Path) -> list[dict]:
        """Every table and text block of one document, cleaned, in reading order."""
        doc = fitz.open(file_path)
        processed_elements = []

        for page_num, page in enumerate(doc, start=1):
            table_rects = []

            for tab in page.find_tables(): # Tables first, as markdown, so structure survives
                text = self._accept(tab.to_markdown(), "Table")
                table_rects.append(fitz.Rect(tab.bbox)) # Recorded even if the table is dropped
                if text:
                    processed_elements.append({
                        "type": "Table",
                        "text": text,
                        "page_number": page_num,
                    })

            for block in page.get_text("blocks"):
                block_rect = fitz.Rect(block[:4])
                if any(block_rect.intersects(rect) for rect in table_rects):
                    continue # Already captured as a table; otherwise every cell is indexed twice
                text = self._accept(block[4], "NarrativeText")
                if text:
                    processed_elements.append({
                        "type": "NarrativeText",
                        "text": text,
                        "page_number": page_num,
                    })

        doc.close()
        return processed_elements

    @staticmethod
    def categorize_elements(processed_elements: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split elements into (texts, tables)."""
        texts = [el for el in processed_elements if el["type"] == "NarrativeText"]
        tables = [el for el in processed_elements if el["type"] == "Table"]
        return texts, tables

    def load_single_pdf(self, file_path: Path) -> dict:
        """One PDF in the structure the preprocessor expects."""
        processed_elements = self.extract_pdf_elements(file_path)
        texts, tables = self.categorize_elements(processed_elements)

        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "content": "\n\n".join(el["text"] for el in processed_elements if el["text"]),
            "elements": processed_elements,
            "texts": texts,
            "tables": tables,
        }

    def load_directory(self, dir_path: Path) -> list[dict]:
        """Load every PDF below a directory ----> build_index"""
        documents = []
        # rglob descends into subdirectories - a stray folder under data/ gets indexed too,
        # which has caused an unrelated corpus to be measured before.
        for pdf_file in sorted(dir_path.rglob("*.pdf")):
            if not pdf_file.name.startswith("."):
                print(f"Processing: {pdf_file.name}")
                documents.append(self.load_single_pdf(pdf_file))
        return documents


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    docs = PDFDocumentLoader().load_directory(BASE_DIR / "data")
    print(f"\n{len(docs)} documents, "
          f"{sum(len(d['texts']) for d in docs)} text blocks, "
          f"{sum(len(d['tables']) for d in docs)} tables")
