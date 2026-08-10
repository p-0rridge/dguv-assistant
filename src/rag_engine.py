# pip install "langchain-openai>=1" "langchain-core>=0.3" python-dotenv
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from data_preprocessing import MultiModalPreprocessor

load_dotenv(find_dotenv())

# Chosen deliberately for this MVP: grounded question-answering over already-retrieved
# context is a comparatively easy task for an LLM (mostly reading comprehension plus
# citation, not open-ended reasoning), so a lightweight, inexpensive model is enough -
# no need for a frontier-tier model here. Check OpenAI's current model list before
# running this, since exact model ids get renamed/replaced over time; gpt-4.1-mini is a
# solid, well-established fallback if the id below is no longer valid.
DEFAULT_MODEL = "gpt-5.4-mini"

# Instructs the model to answer only from the retrieved context and to cite the page
# number of every claim - required here because a norm/spec is only useful to the reader
# if they can go verify the exact clause it came from.
SYSTEM_PROMPT = """Du bist ein Assistent, der Fragen ausschließlich anhand des \
bereitgestellten Kontexts aus technischen Normdokumenten beantwortet.

Regeln:
- Nutze ausschließlich die Informationen aus dem Kontext. Wenn die Antwort nicht im \
Kontext enthalten ist, sage das explizit - erfinde nichts.
- Belege jede inhaltliche Aussage mit der Seitenzahl der Quelle in eckigen Klammern, \
z. B. [Seite 4].
- Antworte auf Deutsch, klar und knapp.

Kontext:
{context}
"""


class RAGEngine:

    def __init__(
        self,
        preprocessor: MultiModalPreprocessor,
        openai_api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        top_k: int = 5,
    ):
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key found. Pass openai_api_key= or set OPENAI_API_KEY in your .env file."
            )

        self.preprocessor = preprocessor
        self.top_k = top_k

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
        return self.preprocessor.search_text(query, k=self.top_k)

    def build_context(self, chunks: list[dict]) -> str:
        blocks = []
        for chunk in chunks:
            page = chunk["metadata"].get("page_number", "?")
            blocks.append(f"[Seite {page}]\n{chunk['text']}")
        return "\n\n---\n\n".join(blocks)

    def build_sources(self, chunks: list[dict]) -> list[dict]:
        """Deduplicated, sorted list of sources actually retrieved, for display in a UI."""
        seen = {}
        for chunk in chunks:
            meta = chunk["metadata"]
            key = (meta.get("source_file"), meta.get("page_number"))
            seen[key] = {"source_file": meta.get("source_file"), "page_number": meta.get("page_number")}
        return sorted(seen.values(), key=lambda s: (s["source_file"] or "", s["page_number"] or 0))

    def answer(self, question: str) -> dict:
        """
        Run the full retrieve -> stuff -> generate pipeline for one question.
        returns: {"answer": str, "sources": list[dict], "chunks": list[dict]}
        """
        chunks = self.retrieve(question)
        if not chunks:
            return {
                "answer": "Dazu habe ich keine relevanten Informationen in den Dokumenten gefunden.",
                "sources": [],
                "chunks": [],
            }

        context = self.build_context(chunks)
        answer_text = self.chain.invoke({"context": context, "question": question})

        return {
            "answer": answer_text,
            "sources": self.build_sources(chunks),
            "chunks": chunks,
        }


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    PERSIST_DIR = BASE_DIR / "chroma_db"

    # Assumes data_preprocessing.py has already been run once against the mock PDF,
    # so PERSIST_DIR already contains the indexed chunks.
    preprocessor = MultiModalPreprocessor(persist_dir=PERSIST_DIR)
    engine = RAGEngine(preprocessor=preprocessor)

    question = "Wie wird der Isolationswiderstand von Rumpfsegmenten geprüft?"
    result = engine.answer(question)

    print("Frage:", question)
    print("\nAntwort:\n", result["answer"])
    print("\nQuellen:")
    for src in result["sources"]:
        print(f"  - {src['source_file']}, Seite {src['page_number']}")