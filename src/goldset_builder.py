"""
Builds the evaluation gold set by reversing the retrieval task: a known passage is
given to a model, which writes the question it answers, so the correct location is
known by construction. Each passage yields two questions - one colloquial, one naming
the document designation - plus a verbatim snippet used both as the answer key and as
a check that the model did not invent its evidence.

The gold label is source_file + page_number + snippet, never the chunk_id, so the same
gold set survives a change to the chunking strategy.

Rationale and known limitations: presentation/baseline_results.md
Cost: one API call per selected chunk (~75 by default).

Usage:
    python src/goldset_builder.py
    python src/goldset_builder.py --per-document 3
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI

import config
from config import GOLDSET_MODEL
from metrics import normalise

load_dotenv(find_dotenv())

# Matches table-of-contents style lines ("1.2 Anwendungsbereich ....... 7") and pages
# that are mostly dot leaders. Such chunks contain headings but no answerable content.
TOC_PATTERN = re.compile(r"\.{4,}")

GENERATION_PROMPT = """Du erhältst einen Auszug aus einem deutschen DGUV-Dokument.

Dokument: {source_file}

--- AUSZUG ---
{chunk_text}
--- ENDE AUSZUG ---

Formuliere daraus zwei Prüffragen und ein Belegzitat. Antworte ausschließlich mit \
einem JSON-Objekt, ohne Markdown-Codeblock, mit genau diesen Schlüsseln:

{{
  "question_colloquial": "...",
  "question_precise": "...",
  "answer_snippet": "..."
}}

Beide Fragen müssen für sich allein verständlich sein - sie werden später ohne diesen \
Auszug an ein Suchsystem gestellt. Verweise wie "hier", "in diesem Dokument", "diese \
Anlage" oder "der genannte Abschnitt" sind verboten; benenne den Gegenstand ausdrücklich.

Nenne in keiner der beiden Fragen einen Dateinamen, eine Dateiendung oder eine \
Seitenzahl, und verwende keine Wendungen wie "laut Auszug" oder "nach Dokument X.pdf". \
Solche Angaben kennt ein Fragesteller nicht.

"question_colloquial": So, wie eine Elektrofachkraft auf der Baustelle mündlich fragen \
würde. Alltagssprache, keine Normbezeichnung, keine Abschnittsnummer. Übernimm möglichst \
wenige seltene Fachbegriffe wörtlich aus dem Auszug.

"question_precise": Fachsprachlich, mit der DGUV-Bezeichnung in natürlicher Form \
(zum Beispiel "DGUV Regel 103-011" oder "DGUV Information 203-071") und - falls im \
Auszug erkennbar - der Abschnittsnummer.

"answer_snippet": Ein wörtliches, unverändertes Zitat von 5 bis 15 Wörtern aus dem \
Auszug, das die Antwort belegt. Nichts hinzufügen, nichts weglassen, nichts korrigieren.
"""


class GoldsetBuilder:
    """Selects chunks, generates question pairs for them and validates the result."""

    def __init__(
        self,
        model: str = GOLDSET_MODEL,
        openai_api_key: str | None = None,
        per_document: int = 5,
        min_chars: int = 300,
        seed: int = 42,
    ):
        """
        model: OpenAI chat model used to generate the questions
        openai_api_key: Falls back to the OPENAI_API_KEY environment variable
        per_document: How many chunks to sample from each document. Sampling per
            document rather than across the whole corpus keeps a 132-page document from
            dominating the gold set, which would turn the evaluation into a measurement
            of how well that single document is found.
        min_chars: Shortest chunk still considered usable. Below this a chunk is usually
            a heading or a fragment, and any question about it would be unanswerable.
        seed: Fixes the sampling so the same corpus always produces the same gold set
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key found. Pass openai_api_key= or set OPENAI_API_KEY in your .env file."
            )

        self.per_document = per_document
        self.min_chars = min_chars
        self.random = random.Random(seed)
        # temperature=0: the gold set should be reproducible, not creative.
        self.llm = ChatOpenAI(api_key=api_key, model=model, temperature=0.0)

    @staticmethod
    def load_chunks(path: Path) -> list[dict]:
        """Read the chunks exported by build_index.py."""
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run build_index.py first.")
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def is_usable(self, chunk: dict) -> bool:
        """
        Decide whether a chunk can carry an answerable question.
        Filters out images (no text to ask about), fragments, and table-of-contents or
        index pages, which are full of headings but contain no statements.
        """
        if chunk["type"] not in ("NarrativeText", "Table"):
            return False
        text = chunk["text"]
        if len(text) < self.min_chars:
            return False
        if TOC_PATTERN.search(text):
            return False
        # A chunk that is mostly digits and punctuation is a table of contents, a page
        # index or a numbering block rather than prose.
        letters = sum(character.isalpha() for character in text)
        return letters / len(text) > 0.55

    def select_chunks(self, chunks: list[dict], exclude: set[str] = frozenset()) -> list[dict]:
        """
        Pick a stratified sample: up to per_document usable chunks from each document.
        exclude: chunk ids already in the gold set, so an extension adds new passages
            instead of resampling the old ones.
        """
        by_document: dict[str, list[dict]] = {}
        for chunk in chunks:
            if self.is_usable(chunk) and chunk["chunk_id"] not in exclude:
                by_document.setdefault(chunk["source_file"], []).append(chunk)

        selected = []
        for source_file in sorted(by_document):
            candidates = by_document[source_file]
            take = min(self.per_document, len(candidates))
            selected.extend(self.random.sample(candidates, take))
            if take < self.per_document:
                print(f"  note: only {take} usable chunks in {source_file}")

        return sorted(selected, key=lambda c: (c["source_file"], c["page_number"] or 0))

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """
        Turn the model's reply into a dict, tolerating a markdown code fence around it.
        Raises ValueError if the reply is not usable JSON, so the caller can skip it.
        """
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"Reply was not valid JSON: {error}") from error
        if not isinstance(parsed, dict):
            raise ValueError("Reply was valid JSON but not an object.")
        return parsed

    def generate_entry(self, chunk: dict, entry_id: str) -> dict | None:
        """
        Produce one gold-set entry for a chunk, or None if generation or validation failed.
        chunk: A usable chunk dict from chunks.json
        entry_id: Identifier written into the entry
        """
        # The page number is deliberately not passed to the model: it must not be able
        # to name the answer's location in the question it writes.
        prompt = GENERATION_PROMPT.format(
            source_file=chunk["source_file"],
            chunk_text=chunk["text"],
        )
        try:
            reply = self.llm.invoke(prompt).content
            parsed = self._parse_response(reply)
        except Exception as error:  # noqa: BLE001 - one bad chunk must not stop the run
            print(f"  skipped {chunk['chunk_id']}: {error}")
            return None

        entry = {
            "id": entry_id,
            "chunk_id": chunk["chunk_id"],
            "source_file": chunk["source_file"],
            "page_number": chunk["page_number"],
            "chunk_type": chunk["type"],
            "question_colloquial": parsed.get("question_colloquial", "").strip(),
            "question_precise": parsed.get("question_precise", "").strip(),
            "answer_snippet": parsed.get("answer_snippet", "").strip(),
        }

        problem = self.validate(entry, chunk)
        if problem:
            print(f"  rejected {chunk['chunk_id']}: {problem}")
            return None
        return entry

    @staticmethod
    def validate(entry: dict, chunk: dict) -> str | None:
        """
        Check one generated entry, returning a reason string if it must be discarded.

        The snippet check is the important one: it has to occur verbatim in the chunk.
        If it does not, the model paraphrased or invented the evidence, and the entry
        would silently weaken every measurement built on it. Comparison runs through the
        same normalisation the evaluation uses, so whitespace differences are tolerated
        but wording differences are not.
        """
        if not entry["question_colloquial"] or not entry["question_precise"]:
            return "missing question"
        if not entry["answer_snippet"]:
            return "missing snippet"
        if len(entry["answer_snippet"].split()) < 4:
            return "snippet too short to be evidence"
        if normalise(entry["answer_snippet"]) not in normalise(chunk["text"]):
            return "snippet not found verbatim in the chunk"
        if entry["question_colloquial"] == entry["question_precise"]:
            return "both questions are identical"
        return None

    def build(self, chunks_file: Path, output_file: Path, extend: bool = False) -> list[dict]:
        """
        Run the whole process and write the gold set to disk.

        extend: keep the existing entries and add new ones for passages not yet used.
            Adding rather than regenerating is what keeps earlier measurements
            comparable - a regenerated set would be a different test, and every number
            recorded against the old one would have to be thrown away. It also matters
            for the reason the set is being extended at all: below roughly six
            disagreeing questions no paired comparison can reach significance however
            clean the result, so the set has to grow rather than change.
        """
        chunks = self.load_chunks(chunks_file)
        print(f"Loaded {len(chunks)} chunks from {chunks_file}")

        existing: list[dict] = []
        if extend and output_file.exists():
            with output_file.open(encoding="utf-8") as handle:
                existing = json.load(handle)
            print(f"Extending the existing gold set of {len(existing)} entries")

        used = {entry["chunk_id"] for entry in existing}
        selected = self.select_chunks(chunks, exclude=used)
        print(f"Selected {len(selected)} chunks across {len({c['source_file'] for c in selected})} documents\n")

        entries = list(existing)
        for index, chunk in enumerate(selected, start=1):
            print(f"[{index}/{len(selected)}] {chunk['source_file']} p.{chunk['page_number']}")
            entry = self.generate_entry(chunk, entry_id=f"gold-{len(entries) + 1:03d}")
            if entry:
                entries.append(entry)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, ensure_ascii=False, indent=2)

        added = len(entries) - len(existing)
        rejected = len(selected) - added
        print(f"\nWrote {len(entries)} entries to {output_file} (+{added} new, {rejected} rejected)")
        print(f"That is {2 * len(entries)} test questions: one colloquial and one precise per entry.")
        return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the evaluation gold set.")
    parser.add_argument("--per-document", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extend", action="store_true",
                        help="add to the existing gold set instead of replacing it")
    args = parser.parse_args()

    builder = GoldsetBuilder(per_document=args.per_document, seed=args.seed)
    builder.build(config.CHUNKS_FILE, config.GOLDSET_FILE, extend=args.extend)


if __name__ == "__main__":
    main()