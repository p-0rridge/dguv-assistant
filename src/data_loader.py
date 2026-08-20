"""
Extracts text and tables from PDFs with PyMuPDF.

Page images were extracted here until it became clear they carried nothing: each one
was stored with the placeholder text "[Complete page diagram page N]" and was reachable
only through CLIP, which nothing queried. The figure captions - "Bild 2-1: Körperreaktion
im Zeit-Stromdiagramm" and the like - are part of the page text and are indexed anyway,
so removing the image branch cost no retrievable content and removed three dependencies.
"""
from pathlib import Path

import fitz  # PyMuPDF


class PDFDocumentLoader:
    """Turns a PDF into a flat list of text and table elements, page by page."""

    def extract_pdf_elements(self, file_path: Path) -> list[dict]:
        """
        Extract every table and text block of a document.
        returns: Elements with type, text and page_number, in reading order
        """
        doc = fitz.open(file_path)
        processed_elements = []

        for page_num, page in enumerate(doc, start=1):
            table_rects = []

            # Tables first, as markdown, so their structure survives into the chunk.
            for tab in page.find_tables():
                table_markdown = tab.to_markdown()
                if table_markdown.strip():
                    processed_elements.append({
                        "type": "Table",
                        "text": table_markdown,
                        "page_number": page_num,
                    })
                    table_rects.append(fitz.Rect(tab.bbox))

            # Then the text blocks, skipping any that sit inside a table already
            # captured above - otherwise every table cell would be indexed twice, once
            # as structured markdown and once as loose text.
            for block in page.get_text("blocks"):
                block_rect = fitz.Rect(block[:4])
                inside_table = any(block_rect.intersects(rect) for rect in table_rects)
                if not inside_table and block[4].strip():
                    processed_elements.append({
                        "type": "NarrativeText",
                        "text": block[4].strip(),
                        "page_number": page_num,
                    })

        doc.close()
        return processed_elements

    @staticmethod
    def categorize_elements(processed_elements: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split elements into text and tables."""
        texts = [el for el in processed_elements if el["type"] == "NarrativeText"]
        tables = [el for el in processed_elements if el["type"] == "Table"]
        return texts, tables

    def load_single_pdf(self, file_path: Path) -> dict:
        """Load one PDF into the structure the preprocessor expects."""
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
        """
        Load every PDF below a directory.

        rglob descends into subdirectories, so anything left in a folder under data/
        is indexed too - that has caused an unrelated corpus to be measured before.
        """
        documents = []
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
