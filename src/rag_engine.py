"""
Why this file? Turns retrieved passages into an answer that carries its sources, and
refuses when the context does not support one.

The engine never touches the vector store - it holds a Retriever - so swapping the
retrieval strategy leaves answer generation untouched and keeps comparisons honest.
"""
import os
import re

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import ANSWER_MODEL
from document_titles import load_titles
from retriever import Retriever

load_dotenv(find_dotenv())

CITATION_PATTERN = re.compile(r"\[([^\[\]]+?)\s*,\s*Seite\s*(\d+)\s*\]", re.IGNORECASE)

# The model marks its own refusals, because matching refusal phrases in free German text
# does not work: every phrase added to a list created new ways to miss one. An emitted
# token is an exact comparison, and it lets a refusal suppress its own source list.
REFUSAL_TOKEN = "[KEINE_ANTWORT]"
REFUSAL_PATTERN = re.compile(r"^\s*\[\s*KEINE_ANTWORT\s*\]\s*", re.IGNORECASE)

# Document *and* page: across fifteen documents "Seite 43" alone identifies nothing, and
# a regulation is only useful if the reader can go and check the clause.
SYSTEM_PROMPT = """Du bist ein Assistent, der Fragen ausschließlich anhand des \
bereitgestellten Kontexts aus technischen Normdokumenten beantwortet.

Regeln:
- Nutze ausschließlich die Informationen aus dem Kontext. Wenn die Antwort nicht im \
Kontext enthalten ist, sage das explizit - erfinde nichts.
- Belege jede inhaltliche Aussage mit Dokument und Seitenzahl in eckigen Klammern, \
genau in der Form [dateiname.pdf, Seite 12]. Übernimm den Dateinamen unverändert so, \
wie er im Kontext über dem jeweiligen Abschnitt steht.
- Wenn der Kontext die gestellte Frage nicht beantwortet, beginne deine Antwort mit \
genau [KEINE_ANTWORT] und sage anschließend in einem Satz, dass der Kontext die Frage \
nicht beantwortet. Führe dann keine Belegstellen an und keine thematisch verwandten \
Angaben, die eine andere Frage beantworten würden.
- Antworte auf Deutsch, klar und knapp.

Kontext:
{context}
"""


class RAGEngine:

    def __init__(
        self,
        retriever: Retriever, # Dense, hybrid or re-ranked - the engine does not care
        openai_api_key: str | None = None,
        model: str = ANSWER_MODEL,
        temperature: float = 0.0,
        top_k: int = 5,
    ):
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key found. Pass openai_api_key= or set OPENAI_API_KEY in your .env file."
            )

        self.retriever = retriever
        self.top_k = top_k
        # Display only. The model keeps citing filenames, which is what the context shows
        # and what CITATION_PATTERN parses; titles are attached afterwards. Keeping the
        # two apart means renaming a document cannot break citation parsing.
        self.titles = load_titles()

        self.llm = ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def retrieve(self, query: str) -> list[dict]:
        """The top_k most relevant chunks for a query."""
        return self.retriever.retrieve(query, k=self.top_k)

    def build_context(self, chunks: list[dict]) -> str:
        """Label each passage exactly as the model is asked to cite it back."""
        blocks = []
        for chunk in chunks:
            meta = chunk["metadata"]
            label = f"[{meta.get('source_file', '?')}, Seite {meta.get('page_number', '?')}]"
            blocks.append(f"{label}\n{chunk['text']}")
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _parse_citations(answer: str) -> set[tuple[str, int]]:
        """The (document, page) pairs an answer actually cites."""
        found = set()
        for name, page in CITATION_PATTERN.findall(answer):
            name = name.strip()
            if not name.lower().endswith(".pdf"): # Tolerate a missing extension
                name = f"{name}.pdf"
            found.add((name, int(page)))
        return found

    @staticmethod
    def _split_refusal(answer: str) -> tuple[bool, str]:
        """Separate the refusal marker from the text a reader sees."""
        if REFUSAL_PATTERN.match(answer):
            return True, REFUSAL_PATTERN.sub("", answer, count=1).strip()
        return False, answer

    def build_sources(self, chunks: list[dict], answer: str | None = None) -> list[dict]:
        """
        Exactly the passages the answer cites, and nothing else.

        Listing everything retrieved would overstate the evidence: a one-sentence answer
        would appear to rest on five documents. There is deliberately no fallback -
        showing passages the answer did not use is the overstatement this system exists
        to avoid.
        """
        seen = {}
        for chunk in chunks:
            meta = chunk["metadata"]
            source_file = meta.get("source_file")
            key = (source_file, meta.get("page_number"))
            seen[key] = {
                "source_file": source_file,
                "page_number": meta.get("page_number"),
                "title": self.titles.get(source_file) or source_file,
            }

        if answer is not None:
            cited = self._parse_citations(answer)
            seen = {key: value for key, value in seen.items() if key in cited}

        return sorted(seen.values(), key=lambda s: (s["source_file"] or "", s["page_number"] or 0))

    def answer(self, question: str) -> dict:
        """
        Retrieve, generate, attach sources.
        returns: answer, abstained, sources, chunks
        """
        chunks = self.retrieve(question)
        if not chunks:
            return {
                "answer": "Dazu habe ich keine relevanten Informationen in den Dokumenten gefunden.",
                "abstained": True,
                "sources": [],
                "chunks": [],
            }

        context = self.build_context(chunks)
        raw = self.chain.invoke({"context": context, "question": question})
        abstained, answer_text = self._split_refusal(raw)

        # A refusal cites nothing, whatever the model wrote. Observed: asked about a
        # topic outside the corpus, it refused and appended five citations - which would
        # render as "here is the answer, backed by five documents".
        return {
            "answer": answer_text,
            "abstained": abstained,
            "sources": [] if abstained else self.build_sources(chunks, answer=answer_text),
            "chunks": chunks,
        }


if __name__ == "__main__":
    import config
    from data_preprocessing import MultiModalPreprocessor
    from retriever import build_retriever

    _, chroma_dir = config.corpus_paths("full") # Assumes build_index.py has run
    variant = config.MVP_BASELINE

    preprocessor = MultiModalPreprocessor(persist_dir=chroma_dir)
    engine = RAGEngine(
        retriever=build_retriever(variant, preprocessor),
        top_k=variant.top_k,
    )

    question = "In welchen Abständen müssen ortsveränderliche elektrische Betriebsmittel geprüft werden?"
    result = engine.answer(question)

    print("Frage:", question)
    print("\nAntwort:\n", result["answer"])
    print("\nQuellen:")
    for src in result["sources"]:
        print(f"  - {src['source_file']}, Seite {src['page_number']}")
