"""
Why this file? PDF extraction leaves typesetting artefacts that a reader never sees but
a string comparison does. This module repairs them and changes wording in no other way,
because the text it produces is what gets indexed and quoted back beside a citation.

Not the same job as metrics.normalise(), which flattens text for comparison only.
"""
import re

SOFT_HYPHEN = re.compile(r"­\s*") # Invisible break inside a word; the word closes up
SUSPENDED = re.compile(r"(\w)[-‐‑]\s*\n\s*(und|oder|bzw|sowie)\b") # "Mess- und ..." stays apart
LINE_HYPHEN = re.compile(r"(\w)[-‐‑]\s*\n\s*([a-zäöüß])") # Lower case after = word break, join
COMPOUND_HYPHEN = re.compile(r"(\w)[-‐‑]\s*\n\s*([A-ZÄÖÜ0-9])") # Capital or digit = keep hyphen
FIXED_SPACES = re.compile(r"[    ]") # Non-breaking and thin spaces, e.g. inside "§ 5"
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]") # Backspace, form feed, friends
DOT_LEADER = re.compile(r"\.{4,}") # Table-of-contents leader
TRAILING_SPACES = re.compile(r"[ \t]+(\n|$)")
SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
BLANK_LINES = re.compile(r"\n{3,}")

BOILERPLATE_START = re.compile(r"^\s*(Impressum|Herausgegeben von|Bildnachweis|ISBN)\b", re.I)
MIN_LETTER_RATIO = 0.35 # Below this a block is numbering, a page header or a running foot


class TextCleaner:

    def clean(self, text: str) -> str:
        """Repair one extracted block. Presentation only, never wording."""
        text = SOFT_HYPHEN.sub("", text)
        text = SUSPENDED.sub(r"\1- \2", text)
        text = LINE_HYPHEN.sub(r"\1\2", text)
        text = COMPOUND_HYPHEN.sub(r"\1-\2", text) # Keeps DGUV-Regel and 203-071 intact
        text = FIXED_SPACES.sub(" ", text)
        text = CONTROL_CHARS.sub("", text)
        text = TRAILING_SPACES.sub(r"\1", text)
        text = SINGLE_NEWLINE.sub(" ", text) # Where the line ended, not a paragraph break
        text = BLANK_LINES.sub("\n\n", text)
        return text.strip()

    @staticmethod
    def is_boilerplate(text: str, element_type: str = "NarrativeText") -> bool:
        """
        Front matter or table of contents rather than content ----> data_loader

        Deliberately narrow: every rule here deletes text permanently, and an unindexed
        passage cannot be found or reported missing.
        """
        stripped = text.strip()
        if not stripped:
            return True
        if DOT_LEADER.search(stripped):
            return True
        if BOILERPLATE_START.match(stripped): # Only at the start, so a body mention survives
            return True
        if element_type == "Table": # A table of current ratings is legitimately all digits
            return False
        letters = sum(character.isalpha() for character in stripped)
        return letters / len(stripped) < MIN_LETTER_RATIO


if __name__ == "__main__":
    cleaner = TextCleaner()

    # Artefacts that must be repaired
    assert cleaner.clean("Unterneh­men") == "Unternehmen"
    assert cleaner.clean("Unterneh-\nmen") == "Unternehmen"
    assert cleaner.clean("Text mit\x08 Steuerzeichen") == "Text mit Steuerzeichen"
    assert cleaner.clean("§ 5") == "§ 5"

    # Wording that must survive untouched
    assert cleaner.clean("DGUV-\nRegel") == "DGUV-Regel"
    assert cleaner.clean("Mess-\nund Prüfmittel") == "Mess- und Prüfmittel"
    assert cleaner.clean("203-071") == "203-071"
    assert cleaner.clean("DGUV Information 203-\n071") == "DGUV Information 203-071"

    # Blocks that carry no content
    assert cleaner.is_boilerplate("1.2 Anwendungsbereich..................... 7")
    assert cleaner.is_boilerplate("Impressum\nHerausgegeben von: DGUV")
    assert cleaner.is_boilerplate("2.1.3")
    assert not cleaner.is_boilerplate("Der Unternehmer hat dafür zu sorgen, dass geprüft wird.")
    assert not cleaner.is_boilerplate("|104 0A 0A 15A 25A 63A 35A 32A|2A|6A 4A|", "Table")

    print("data_cleaning: all checks passed.")
