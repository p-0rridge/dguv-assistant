# Electrician Standards Agentic RAG

A high-precision Retrieval-Augmented Generation (RAG) system designed to answer complex queries regarding professional electrical engineering standards, safety regulations, and technical installation guidelines.

## Core Objective

In electrical engineering, inaccurate information can pose severe safety risks. The primary goal of this project is clear: **Minimize room for hallucinations as much as possible.**

To ensure maximum factual accuracy, precision, and traceability, the architecture incorporates the following key components:

* **Multimodal Embeddings:** Ingestion and joint vector space representation of full texts, circuit diagrams, data tables, and technical illustrations for high-precision retrieval.
* **Hybrid Search (BM25 + Dense Vectors):** Combination of exact keyword matching (for specific standard designations or technical metrics like 30 mA) with semantic vector search.
* **Re-Ranker (Cross-Encoder):** Secondary evaluation step for retrieved text and image chunks prior to passing them to the synthesis LLM, ensuring only the most relevant context reaches the prompt.
* **Grounded Generation & Strict Citations:** The synthesis model is strictly constrained to generate responses solely based on provided source chunks and to cite every statement with exact references.

## Document Ingestion & Extraction Engine: PyMuPDF

We chose **PyMuPDF** over heavy extractors like `UnstructuredPDFLoader` for the data extraction pipeline:

* **Lightning-Fast & Lightweight:** Built on the native C-based MuPDF library, significantly reducing CPU/RAM overhead and processing time.
* **Native Markdown Table Extraction:** Leverages built-in table detection (`page.find_tables()`) to convert complex technical tables directly into Markdown, preserving structural integrity for the LLM.
* **Layout Control & Deduplication:** Bounding-box matching avoids redundant text extraction by skipping narrative blocks overlapping with detected tables.
* **Zero External Dependencies:** Runs pure Python (`pip install pymupdf`) without requiring system-level packages like `poppler-utils` or `tesseract-ocr`, streamlining Docker and Cloud deployments.

## Tech Stack & Architecture

* **Parsing & Extraction:** PyMuPDF
* **Retrieval:** Elasticsearch (Hybrid Search: Dense Vector + BM25)
* **Processing:** Multimodal Chunking & Layout Analysis
* **Quality Gate:** Re-Ranker (Cross-Encoder) for context optimization
* **Interface:** Streamlit Web-UI