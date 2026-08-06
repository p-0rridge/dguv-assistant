import logging
import os
from pathlib import Path
import fitz  # PyMuPDF


class PDFDocumentLoader:

    def __init__(self, output_dir: Path | None = None):
        """Output directory"""
        self.output_dir = output_dir

    def load_single_pdf(self, file_path: Path) -> dict:
        """
        Extracts structured elements (text, markdown tables, cropped diagrams) 
        from PDFs safely.
        """
        images_dir = None
        if self.output_dir:
            images_dir = self.output_dir / file_path.stem
            images_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(file_path)
        processed_elements = []

        for page_num, page in enumerate(doc, start=1):
            table_rects = []

            #extract tables as markdown
            tabs = page.find_tables()
            for tab_idx, tab in enumerate(tabs, start=1):
                table_markdown = tab.to_markdown()
                if table_markdown.strip():
                    processed_elements.append({
                        "type": "Table",
                        "text": table_markdown,
                        "page_number": page_num,
                        "image_path": None,
                    })
                    table_rects.append(fitz.Rect(tab.bbox))

            #extract only text blocks
            blocks = page.get_text("blocks")
            for b in blocks:
                block_rect = fitz.Rect(b[:4])
                is_inside_table = any(block_rect.intersects(tr) for tr in table_rects)
                
                if not is_inside_table and b[4].strip():
                    processed_elements.append({
                        "type": "NarrativeText",
                        "text": b[4].strip(),
                        "page_number": page_num,
                        "image_path": None,
                    })

#if there are images: extract whole page screenshot
            if self.output_dir:
                #check if there are images or drawings on the page
                has_images = len(page.get_images()) > 0
                has_drawings = any(d["rect"].width > 150 and d["rect"].height > 150 for d in page.get_drawings())

                if has_images or has_drawings:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    img_filename = f"page_{page_num}_full.png"
                    img_save_path = images_dir / img_filename
                    pix.save(img_save_path)

                    processed_elements.append({
                        "type": "Image",
                        "text": f"[Complete page diagram page {page_num}]",
                        "page_number": page_num,
                        "image_path": str(img_save_path),
                    })

        doc.close()

        full_text = "\n\n".join([el["text"] for el in processed_elements if el["text"]])

        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "content": full_text,
            "elements": processed_elements,
        }

    def load_directory(self, dir_path: Path) -> list[dict]:
        """Recursively loads all PDF files within a given directory."""
        documents = []
        for pdf_file in dir_path.rglob("*.pdf"):
            if not pdf_file.name.startswith("."):
                print(f"Processing: {pdf_file.name}")
                doc_data = self.load_single_pdf(pdf_file)
                documents.append(doc_data)

        return documents


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    SUB_DIR = BASE_DIR / "data"


    loader = PDFDocumentLoader(output_dir=None) #for now we work without images
    docs = loader.load_directory(SUB_DIR)

