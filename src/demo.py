"""
Runs a handful of questions through the full pipeline and prints the answers in a form
that is readable on a slide or in a screenshot.

The last question is deliberately unanswerable from this corpus: a system whose selling
point is refusing to invent has to be shown refusing, not only answering.

Usage:
    python src/demo.py
    python src/demo.py --config rerank     # slower, but the better ranking
"""
import argparse
import os
import textwrap

# Quiet the model libraries so the output is clean enough to screenshot.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import config
from data_preprocessing import MultiModalPreprocessor
from rag_engine import RAGEngine
from retriever import build_retriever

WIDTH = 78

QUESTIONS = [
    "Wer darf ortsveränderliche elektrische Betriebsmittel prüfen?",
    "Wer darf elektrische Betriebsmittel instand setzen?",
    "In welchen Abständen sind ortsveränderliche elektrische Betriebsmittel zu prüfen?",
    # Not answerable, and nothing in the corpus is even adjacent: a clean refusal.
    "Welche Schutzmaßnahmen gelten, wenn bei Abbrucharbeiten Asbest gefunden wird?",
    # Not answerable either, but the corpus does contain cross-sections for a *different*
    # kind of conductor. The near miss - right numbers, wrong question - is the failure
    # mode that matters in a safety context.
    "Welche Mindestquerschnitte für Schutzleiter legt DIN VDE 0100-540 fest?",
]


def render(index: int, question: str, result: dict) -> str:
    """Format one question and its answer as a block of plain text."""
    lines = ["", "=" * WIDTH, f"FRAGE {index}", "-" * WIDTH]
    lines += textwrap.wrap(question, WIDTH)
    lines += ["", "ANTWORT", "-" * WIDTH]
    for paragraph in result["answer"].split("\n"):
        lines += textwrap.wrap(paragraph, WIDTH) if paragraph.strip() else [""]
    lines += ["", "QUELLEN (in der Antwort zitiert)", "-" * WIDTH]
    if result["sources"]:
        for source in result["sources"]:
            lines.append(f"  · {source['source_file']}, Seite {source['page_number']}")
    else:
        # A refusal cites nothing, so it has no sources. Showing what was retrieved
        # anyway would claim evidence the answer explicitly says it does not have.
        lines.append("  keine - die Antwort stützt sich auf keine Textstelle")
        retrieved = sorted(
            {(c["metadata"].get("source_file"), c["metadata"].get("page_number")) for c in result["chunks"]}
        )
        lines += ["", "  abgerufen, aber nicht verwendet:"]
        lines += [f"    {name}, Seite {page}" for name, page in retrieved]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print example answers for the slides.")
    parser.add_argument("--config", default="mvp_baseline")
    parser.add_argument("--corpus", choices=["full", "dev"], default="full")
    args = parser.parse_args()

    variant = config.get_config(args.config)
    _, chroma_dir = config.corpus_paths(args.corpus)

    preprocessor = MultiModalPreprocessor(persist_dir=chroma_dir)
    engine = RAGEngine(
        retriever=build_retriever(variant, preprocessor),
        top_k=variant.top_k,
    )

    blocks = []
    for index, question in enumerate(QUESTIONS, start=1):
        block = render(index, question, engine.answer(question))
        print(block, flush=True)
        blocks.append(block)

    out = config.ARTIFACTS_DIR / "demo_answers.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print("\n" + "=" * WIDTH)
    print(f"Also written to {out}")


if __name__ == "__main__":
    main()