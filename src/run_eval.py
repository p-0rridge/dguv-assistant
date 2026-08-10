"""
Entry point for evaluation runs.

Usage:
    python src/run_eval.py                            # retrieval metrics, MVP baseline
    python src/run_eval.py --config mvp_baseline
    python src/run_eval.py --abstention               # also measure refusal on unanswerable questions

Retrieval evaluation runs entirely offline - the embedding model and the vector store
are local, so a run costs nothing and can be repeated freely. Only --abstention calls
the answer model, which is why it is opt-in rather than part of every run.
"""
import argparse
import json

import config
from data_preprocessing import MultiModalPreprocessor
from evaluator import RetrievalEvaluator
from retriever import build_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a retrieval variant against the gold set.")
    parser.add_argument("--config", default="mvp_baseline", help="named variant from config.py")
    parser.add_argument("--corpus", choices=["full", "dev"], default="full")
    parser.add_argument("--abstention", action="store_true",
                        help="also measure refusal on unanswerable questions (uses the answer model)")
    args = parser.parse_args()

    variant = config.get_config(args.config)
    _, chroma_dir = config.corpus_paths(args.corpus)

    preprocessor = MultiModalPreprocessor(persist_dir=chroma_dir)
    retriever = build_retriever(variant, preprocessor)
    evaluator = RetrievalEvaluator(retriever, variant)

    goldset = evaluator.load_goldset(config.GOLDSET_FILE)
    print(f"Evaluating '{variant.name}' on {len(goldset)} gold entries "
          f"({2 * len(goldset)} questions)...")

    results = evaluator.evaluate(goldset)

    if args.abstention:
        if not config.UNANSWERABLE_FILE.exists():
            print(f"\nSkipping abstention: {config.UNANSWERABLE_FILE} not found.")
        else:
            from rag_engine import RAGEngine

            with config.UNANSWERABLE_FILE.open(encoding="utf-8") as handle:
                unanswerable = json.load(handle)
            print(f"\nMeasuring abstention on {len(unanswerable)} unanswerable questions...")
            engine = RAGEngine(retriever=retriever, top_k=variant.top_k)
            results["abstention"] = evaluator.evaluate_abstention(engine, unanswerable)

    evaluator.report(results)

    if "abstention" in results:
        abstention = results["abstention"]
        print(f"\n## Abstention\n")
        print(f"Overall: {abstention['abstention_rate']:.2f} "
              f"({abstention['questions']} questions)")
        for category, rate in abstention["by_category"].items():
            print(f"  {category:<24} {rate:.2f}")

    evaluator.save(results, config.RESULTS_DIR / f"{variant.name}.json")


if __name__ == "__main__":
    main()