"""
Reads a readable title for every PDF in the corpus, so answers can cite
"DGUV Information 203-071 - Wiederkehrende Prüfungen" instead of "203-071.pdf".

Titles are derived from the documents themselves rather than from a hand-written list.
The point of the system is that a user can drop any DGUV PDF into data/ and search it;
a mapping maintained by hand would work for these fifteen files and for nothing else.

Two sources, in order of reliability:
  1. The PDF's own title metadata. Populated in 13 of the 15 documents here, and
     accurate where present.
  2. The cover page, when the metadata is empty. Best effort: covers put the publisher,
     the date and the title in no fixed order, so this is a heuristic and is allowed to
     produce something imperfect. It never produces nothing - the filename is the last
     resort.

Deliberately not part of the index. Storing the title on every chunk would mean a
re-index whenever a title changed, and titles are display text, not retrieval data.

Usage:
    python src/document_titles.py           # write artifacts/document_titles.json
"""
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

import config
from data_cleaning import TextCleaner

DESIGNATION = re.compile(r"DGUV\s+(?:Regel|Information|Vorschrift|Grundsatz)\s+[\w\-]+", re.I)
QUOTES = re.compile(r'[„“”"»«]')

# Cover-page lines that are never part of a title: dates, publisher names, standing
# imprint text. Anything not excluded here can still end up in a fallback title, which
# is the intended trade-off - a slightly wordy title beats a missing one.
COVER_NOISE = re.compile(
    r"^(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November"
    r"|Dezember|Ausgabe|Stand|Fassung|Auflage)\b"
    r"|Berufsgenossenschaft?\b|Unfallversicherung|gesetzliche|Holz und Metall"
    r"|Deutsche Gesetzliche|^BG\w*$",
    re.IGNORECASE,
)


class TitleReader:
    """Extracts a display title from a PDF."""

    def __init__(self, cleaner: TextCleaner | None = None):
        self.cleaner = cleaner or TextCleaner()

    def read(self, path: Path) -> str:
        """Return the best available title for one PDF, never empty."""
        doc = fitz.open(path)
        try:
            title = self._from_metadata(doc)
            if len(QUOTES.sub("", title).strip()) < 12:
                title = self._from_cover(doc)
        finally:
            doc.close()

        title = self._tidy(title)
        return title or path.stem

    def _from_metadata(self, doc) -> str:
        raw = (doc.metadata or {}).get("title") or ""
        return self._undouble(self.cleaner.clean(raw.replace("\r", " ").replace("\n", " ")))

    def _from_cover(self, doc) -> str:
        """
        Assemble a title from the first page.
        Keeps the DGUV designation if one appears, plus the first few substantial lines
        that are not dates or publisher names.
        """
        lines = [self.cleaner.clean(line) for line in doc[0].get_text().splitlines()]
        lines = [
            line for line in lines
            if len(line) > 3 and not re.fullmatch(r"[\d\W]+", line)
        ]
        designation = next((line for line in lines if DESIGNATION.search(line)), "")
        body = [
            line for line in lines[:10]
            if line != designation and len(line) > 12 and not COVER_NOISE.search(line)
        ]
        return " ".join(([designation] if designation else []) + body[:3])

    @staticmethod
    def _undouble(text: str) -> str:
        """
        Drop a title that the PDF producer stored twice in a row.
        One document here carries its title duplicated inside the metadata field.
        """
        half = len(text) // 2
        for cut in range(half - 8, half + 9):
            if 0 < cut < len(text) and text[:cut].strip() == text[cut:].strip():
                return text[:cut].strip()

        first = DESIGNATION.search(text)
        if first:
            second = DESIGNATION.search(text, first.end())
            if second:
                return text[:second.start()].strip()
        return text

    @staticmethod
    def _tidy(text: str) -> str:
        """
        Render as "DGUV Information 203-071 - Wiederkehrende Prüfungen ...".

        Quotes are dropped rather than balanced. Several metadata titles are truncated
        mid-phrase, so a closing quote is missing and a lone opening one survives into
        the citation, where it reads like a corrupted name. A separator between the
        designation and the subject is more useful than the original punctuation, and
        it makes every document in the corpus render the same way.
        """
        text = re.sub(r"\s+", " ", QUOTES.sub("", text)).strip(" -–")

        match = DESIGNATION.match(text)
        if not match:
            return text

        designation = match.group(0)
        rest = text[match.end():].strip(" -–:")
        # Covers repeat the designation, and a fallback title can pick it up twice.
        rest = re.sub(r"^DGUV\s+(?:Regel|Information|Vorschrift)\s*", "", rest, flags=re.I)
        return f"{designation} - {rest}".strip(" -") if rest else designation


def build_titles(data_dir: Path, output_file: Path) -> dict[str, str]:
    """Read every PDF below data_dir and write filename -> title as JSON."""
    reader = TitleReader()
    titles = {path.name: reader.read(path) for path in sorted(data_dir.rglob("*.pdf"))}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(titles, handle, ensure_ascii=False, indent=2)
    return titles


def load_titles(path: Path | None = None) -> dict[str, str]:
    """
    Read the title map, returning an empty one if it has not been built.

    Missing titles are not an error: every caller falls back to the filename, so the
    system stays usable on a corpus this has never been run against.
    """
    path = path or config.TITLES_FILE
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def describe(source_file: str, titles: dict[str, str]) -> str:
    """Render one document for display, falling back to the filename."""
    return titles.get(source_file) or source_file


if __name__ == "__main__":
    titles = build_titles(config.DATA_DIR, config.TITLES_FILE)
    width = max((len(name) for name in titles), default=0)
    for name, title in titles.items():
        print(f"  {name[:38]:<{min(width, 38)}}  {title}")
    print(f"\n{len(titles)} titles written to {config.TITLES_FILE}")
