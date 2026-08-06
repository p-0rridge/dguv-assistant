import re
from pathlib import Path
from pygments.lexer import combined
from spellchecker import SpellChecker


from data_loader import PDFDocumentLoader

class TextCleaner:
    def __init__(self, language: str = 'de'):
        self.spellchecker = SpellChecker(language=language)  # Initialize the spell checker for German

    def fix_split_words(self, text: str) -> str:
        """fix corrupted word separations"""

        def replace_func(match):
            """helper function for fix_split_words"""
            w1 = match.group(1)
            w2 = match.group(2)
            combined = w1 + w2

            if combined.lower() in self.spellchecker:
                return combined

            if w1.lower() not in self.spellchecker and combined.lower() in self.spellchecker:
                return combined #if german compound word is valid, return the combined word
            
            return f"{w1} {w2}"


        pattern = r"\b([a-zA-ZäöüÄÖÜß]{2,})\s+([a-zA-ZäöüÄÖÜß]{2,})\b" #describes two parts of German words

        return re.sub(pattern, replace_func, text)


    def basic_clean(self, text: str) -> str:
        """clean text from multiple linebreaks"""
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text) #more than 2 line breaks to 2 line breaks

        """clean paragraphes"""

        paragraphs = text.split("\n\n")
        cleaned_paragraphs = []

        for paragraph in paragraphs:
            paragraph = self.fix_split_words(paragraph)  # fix split words in the paragraph
            paragraph_line = re.sub(r"(?<!\n)\n(?!\n)", " ", paragraph) #convert single line breaks within paragraphs to space
            paragraph_line = re.sub(r"[ \t]+", " ", paragraph_line)
            cleaned_paragraphs.append(paragraph_line.strip())

        return "\n\n".join(cleaned_paragraphs)
  


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    subset_dir = (
        BASE_DIR
        / "data/"
    )

 
   
    loader = PDFDocumentLoader()
    cleaner = TextCleaner()
    docs = loader.load_directory(subset_dir)

    cleaned_docs = []

    for doc in docs: 
        cleaned = cleaner.basic_clean(doc["content"])

        cleaned_docs.append({
            "file_name" : doc["file_name"],
            "content": cleaned
        })

    print(cleaned_docs[0]["content"][:5000]) 