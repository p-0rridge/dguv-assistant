# VDE Standards Agentic RAG

A high-precision Retrieval-Augmented Generation (RAG) system designed to answer complex queries regarding VDE electrical standards and technical installations.

## Core Objective

In electrical engineering, inaccurate information can pose severe safety risks. The primary goal of this project is clear: **Minimize room for hallucinations as much as possible.**

To ensure maximum factual accuracy, precision, and traceability, the architecture incorporates the following key components:

* **Multimodal Embeddings:** Ingestion and joint vector space representation of full texts, circuit diagrams, data tables, and technical illustrations for high-precision retrieval.
* **Hybrid Search (BM25 + Dense Vectors):** Combination of exact keyword matching (for standard numbers such as DIN VDE 0100-410 or technical metrics like 30 mA) with semantic vector search.
* **Re-Ranker (Cross-Encoder):** Secondary evaluation step for retrieved text and image chunks prior to passing them to the synthesis LLM, ensuring only the most relevant context reaches the prompt.
* **Grounded Generation & Strict Citations:** The synthesis model is strictly constrained to generate responses solely based on provided source chunks and to cite every statement with exact references.

## Tech Stack & Architecture

* **Retrieval:** Elasticsearch (Hybrid Search: Dense Vector + BM25)
* **Processing:** Multimodal Chunking & Layout Analysis
* **Quality Gate:** Re-Ranker (Cross-Encoder) for context optimization
* **Interface:** Streamlit Web-UI