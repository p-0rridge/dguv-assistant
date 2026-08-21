# pip install "langchain-openai>=1" "langchain-core>=0.3" python-dotenv
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

# Matches the citation form the system prompt asks for: [203-071.pdf, Seite 12].
CITATION_PATTERN = re.compile(r"\[([^\[\]]+?)\s*,\s*Seite\s*(\d+)\s*\]", re.IGNORECASE)

# The model marks its own refusals, rather than the evaluation guessing from free text.
#
# Detecting refusals by matching phrases does not work, and that is measured rather than
# assumed: on 14 unanswerable questions the system refused all 14, and a list of seven
# refusal phrases recognised 2. Extending it to eleven recognised 8 - the misses were
# "diese Frage" instead of "die Frage", "Ihre Frage", and an inserted subject. Every
# addition creates new variants, because German word order has no fixed form for this.
#
# One marker the model emits itself is exact, needs no second model to judge it, and
# also lets a refusal suppress its own source list - which string matching could not do.
REFUSAL_TOKEN = "[KEINE_ANTWORT]"
REFUSAL_PATTERN = re.compile(r"^\s*\[\s*KEINE_ANTWORT\s*\]\s*", re.IGNORECASE)

# Instructs the model to answer only from the retrieved context and to cite document and
# page for every claim. The document matters: across fifteen documents "page 43" alone
# identifies nothing, and a regulation is only useful if the reader can go and check the
# exact clause it came from.
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
        retriever: Retriever,
        openai_api_key: str | None = None,
        model: str = ANSWER_MODEL,
        temperature: float = 0.0,
        top_k: int = 5,
    ):
        """
        retriever: Any Retriever implementation - dense, hybrid, or re-ranked. The engine
            never touches the vector store directly, so swapping the retrieval strategy
            leaves answer generation untouched and keeps the comparison honest: only one
            part of the system changes between measurements.
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key found. Pass openai_api_key= or set OPENAI_API_KEY in your .env file."
            )

        self.retriever = retriever
        self.top_k = top_k
        # Display only. The model keeps citing filenames, because that is what appears
        # in the context and what CITATION_PATTERN parses; titles are attached
        # afterwards, where a reader sees them. Keeping the two apart means a change to
        # how documents are named cannot break citation parsing.
        self.titles = load_titles()

        self.llm = ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        # LangChain Expression Language (LCEL) pipe - same idea as the notebook's chains,
        # just assembled from smaller pieces instead of RetrievalQA.from_chain_type().
        self.chain = self.prompt | self.llm | StrOutputParser()

    def retrieve(self, query: str) -> list[dict]:
        """Fetch the top_k most relevant text/table chunks for the query."""
        return self.retriever.retrieve(query, k=self.top_k)

    def build_context(self, chunks: list[dict]) -> str:
        """
        Label each passage with document and page, in exactly the form the model is asked
        to cite it back. Anything the model cannot see here, it cannot cite correctly.
        """
        blocks = []
        for chunk in chunks:
            meta = chunk["metadata"]
            label = f"[{meta.get('source_file', '?')}, Seite {meta.get('page_number', '?')}]"
            blocks.append(f"{label}\n{chunk['text']}")
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _parse_citations(answer: str) -> set[tuple[str, int]]:
        """
        Collect the (document, page) pairs the answer actually cites.
        Tolerates a missing .pdf extension and spacing variants around the comma.
        """
        found = set()
        for name, page in CITATION_PATTERN.findall(answer):
            name = name.strip()
            if not name.lower().endswith(".pdf"):
                name = f"{name}.pdf"
            found.add((name, int(page)))
        return found

    @staticmethod
    def _split_refusal(answer: str) -> tuple[bool, str]:
        """
        Separate the refusal marker from the text shown to a reader.
        returns: (refused, answer without the marker)
        """
        if REFUSAL_PATTERN.match(answer):
            return True, REFUSAL_PATTERN.sub("", answer, count=1).strip()
        return False, answer

    def build_sources(self, chunks: list[dict], answer: str | None = None) -> list[dict]:
        """
        Sources to display beneath an answer.
        chunks: The retrieved passages
        answer: The generated answer; when given, only the passages it actually cites are
            returned

        Listing every retrieved passage would overstate the evidence: a one-sentence
        answer citing a single clause would appear to rest on five documents, and an
        answer that refuses would appear to rest on five documents while saying it has
        nothing. So the list is exactly what the answer cites — and when the answer cites
        nothing, it is empty. There is deliberately no fallback to the retrieved set:
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
        Run the full retrieve -> stuff -> generate pipeline for one question.
        returns: {"answer": str, "abstained": bool, "sources": list[dict], "chunks": list[dict]}
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

        # A refusal cites nothing, whatever the model wrote. Measured behaviour: asked
        # about a topic outside the corpus, the model refused and appended five
        # citations - which would render as "here is the answer, backed by five
        # documents". The guarantee this system makes is that a listed source supports
        # the answer, so it is enforced here rather than left to the model.
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

    # Assumes build_index.py has already been run, so the vector store is populated.
    _, chroma_dir = config.corpus_paths("full")
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