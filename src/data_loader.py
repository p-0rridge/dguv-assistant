from pathlib import Path
from pypdf import PdfReader


class PDFDocumentLoader:

    def load_single_pdf(self, file_path: Path) -> dict: 
        """read in text and metadata"""
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

        return {
            "file_name": file_path.name,
            "content": text,
        }

    def load_directory(self, dir_path: Path) -> list[dict]:
        """load directory"""
        documents = []
        for pdf_file in dir_path.rglob("*.pdf"):
            if not pdf_file.name.startswith("."):
                doc_data = self.load_single_pdf(pdf_file)
                documents.append(doc_data)

        return documents


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    """for this quick demo we only load the data from 'HR policies' ---> 3 documents"""
    subset_dir = (
        BASE_DIR
        / "documents"
        / "documents"
        / "documents"
        / "internal_docs_by_area"
        / "HR_policies"
    )

    loader = PDFDocumentLoader()
    docs = loader.load_directory(subset_dir)