import re
import tiktoken
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_loader import PDFDocumentLoader

class TextPreprocessor:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 20, model_encoding: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.tokenizer = tiktoken.get_encoding(model_encoding)

        self.separators = [
            r"\n(?=\d+\.\d+\s)",  # separate at 1.1, 1.2 etc.
            r"\n(?=\d+\.\s)",     # otherwise at 1., 2. ...
            "\n\n",               # otherwise at white lines
            "\n",                 # otherwise at linebreaks
            " ",                  
            "",
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.tiktoken_len,
            separators=self.separators,
            is_separator_regex=True
        )



    def clean_text(self, text: str) -> str:
        """clean text from multiple whitespaces and linebreaks"""
        lines = [line.strip() for line in text.splitlines()]
        cleaned_text = "\n".join(lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        return cleaned_text.strip()

    def tiktoken_len(self, text):
        tokens = self.tokenizer.encode(
        text,
        disallowed_special=()
        )
        return len(tokens)
    

    def chunk_text(self, text: str) -> list[str]:
        chunks = self.text_splitter.split_text(text)
        return chunks

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
    preprocessor = TextPreprocessor()
    docs = loader.load_directory(subset_dir)

    processed_docs = []

    for doc in docs: 
        cleaned = preprocessor.clean_text(doc["content"])
        chunks = preprocessor.chunk_text(cleaned)

        processed_docs.append({
            "file_name" : doc["file_name"],
            "chunks": chunks,
            "total_chunks" : len(chunks)

        })
