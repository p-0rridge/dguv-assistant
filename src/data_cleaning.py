"""
Repairs typesetting artefacts in text extracted from PDFs.

German regulations are justified and hyphenated, and the PDF preserves that layout with
characters a reader never sees. Measured on this corpus of 1522 passages:

    soft hyphens              49 %
    hyphen at a line break    23 %
    non-breaking spaces       25 %
    control characters         9 %

Untreated, "Unterneh<shy>men" is two tokens that occur nowhere else and match nothing,
and a verbatim quotation fails to compare equal to its own source. Both effects have
already cost this project a wrong conclusion: 40 % of generated gold entries were once
rejected as unsupported, which looked like model hallucination and was typesetting.

Related but not the same as metrics.normalise(): that function flattens text for
comparison and may lowercase or discard freely, because its output is never read by
anyone. This module produces the text that gets indexed, quoted back to a user and
shown next to a citation, so it repairs artefacts and changes nothing else.

An earlier version also rejoined words split across a space using a German spell
checker. It was removed: the heuristic altered wording it could not verify, which in a
safety corpus is the one thing text preparation must never do.
"""
import re

# A soft hyphen marks a permitted break inside a word. Where the line actually broke,
# a newline follows it; both go, and the word closes up.
SOFT_HYPHEN = re.compile(r"­\s*")

# A visible hyphen before a line break is ambiguous, and German orthography resolves it:
# a hyphenated compound continues with a capital ("DGUV-\nRegel"), a hyphenated word
# break continues in lower case ("Unterneh-\nmen"). Only the lower-case case is joined,
# so compounds survive intact - which matters here, because "DGUV-Regel" and "203-071"
# are exactly the strings the lexical index depends on.
LINE_HYPHEN = re.compile(r"(\w)[-‐‑]\s*\n\s*([a-zäöüß])")

# The exception to that rule: a suspended hyphen ("Mess- und Prüfmittel") also continues
# in lower case, but the parts are separate words and must not be joined.
SUSPENDED = re.compile(r"(\w)[-‐‑]\s*\n\s*(und|oder|bzw|sowie)\b")

# A compound broken at its own hyphen. The hyphen belongs to the word and stays; only
# the line break goes, so "DGUV-\nRegel" becomes "DGUV-Regel" and not "DGUV- Regel".
# Digits are included for the case this project cannot afford to get wrong: a document
# designation split across a line, "203-\n071", has to come back as one string.
COMPOUND_HYPHEN = re.compile(r"(\w)[-‐‑]\s*\n\s*([A-ZÄÖÜ0-9])")

# Non-breaking and thin spaces separate a paragraph symbol from its number ("§ 5").
# They are spaces to a reader and distinct characters to a string comparison.
FIXED_SPACES = re.compile(r"[    ]")

# Backspace, form feed and friends survive PDF extraction and belong to no word.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Four or more dots in a row is a table-of-contents leader.
DOT_LEADER = re.compile(r"\.{4,}")

TRAILING_SPACES = re.compile(r"[ \t]+(\n|$)")
SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
BLANK_LINES = re.compile(r"\n{3,}")


class TextCleaner:
    """Removes PDF typesetting artefacts without altering wording."""

    def clean(self, text: str) -> str:
        """Repair one extracted text block. Wording is never changed, only presentation."""
        text = SOFT_HYPHEN.sub("", text)
        text = SUSPENDED.sub(r"\1- \2", text)
        text = LINE_HYPHEN.sub(r"\1\2", text)
        text = COMPOUND_HYPHEN.sub(r"\1-\2", text)
        text = FIXED_SPACES.sub(" ", text)
        text = CONTROL_CHARS.sub("", text)
        text = TRAILING_SPACES.sub(r"\1", text)
        # A single newline inside a block is where the line happened to end, not a
        # paragraph break. Collapsing it matters because this text is quoted back to a
        # reader beside a citation; a blank line still separates paragraphs.
        text = SINGLE_NEWLINE.sub(" ", text)
        text = BLANK_LINES.sub("\n\n", text)
        return text.strip()

    @staticmethod
    def is_boilerplate(text: str, element_type: str = "NarrativeText") -> bool:
        """
        Decide whether a block is front matter or a table of contents rather than content.
        element_type: "Table" blocks skip the letter-ratio rule - a table of current
            ratings is legitimately almost all digits, and an early version of this
            function proposed deleting an 8,641-character one.

        Deliberately narrow. Every rule deletes text permanently, and removing one real
        passage costs more than leaving ten useless ones: an unindexed passage cannot be
        found, and nothing downstream can report that it ever existed.
        """
        stripped = text.strip()
        if not stripped:
            return True

        # Table-of-contents lines: dot leaders pointing at a page number.
        if DOT_LEADER.search(stripped):
            return True

        # Imprint pages. Matched only at the very start of a block, so a body passage
        # mentioning a publisher is left alone.
        if re.match(r"^\s*(Impressum|Herausgegeben von|Bildnachweis|ISBN)\b",
                    stripped, re.IGNORECASE):
            return True

        if element_type == "Table":
            return False

        # Blocks that are mostly digits and punctuation: page headers, numbering
        # columns, running feet, bare section numbers like "2.1.3".
        letters = sum(character.isalpha() for character in stripped)
        return letters / len(stripped) < 0.35


if __name__ == "__main__":
    cleaner = TextCleaner()
    # Artefacts that must be repaired.
    assert cleaner.clean("Unterneh­men") == "Unternehmen"
    assert cleaner.clean("Unterneh-\nmen") == "Unternehmen"
    assert cleaner.clean("Text mit\x08 Steuerzeichen") == "Text mit Steuerzeichen"
    assert cleaner.clean("§ 5") == "§ 5"

    # Wording that must survive untouched. Joining these would change words, which is
    # the one thing this module may not do - and "DGUV-Regel" and "203-071" are exactly
    # the strings the lexical index depends on.
    assert cleaner.clean("DGUV-\nRegel") == "DGUV-Regel"
    assert cleaner.clean("Mess-\nund Prüfmittel") == "Mess- und Prüfmittel"
    assert cleaner.clean("203-071") == "203-071"
    assert cleaner.clean("DGUV Information 203-\n071") == "DGUV Information 203-071"

    # Blocks that carry no content.
    assert cleaner.is_boilerplate("1.2 Anwendungsbereich..................... 7")
    assert cleaner.is_boilerplate("Impressum\nHerausgegeben von: DGUV")
    assert cleaner.is_boilerplate("2.1.3")
    assert not cleaner.is_boilerplate("Der Unternehmer hat dafür zu sorgen, dass geprüft wird.")

    # A table of current ratings is almost all digits and is not a numbering block.
    # Found by running the filter over the real corpus: an earlier version of
    # is_boilerplate proposed deleting an 8,641-character table.
    assert not cleaner.is_boilerplate("|104 0A 0A 15A 25A 63A 35A 32A|2A|6A 4A|", "Table")

    print("data_cleaning: all checks passed.")
