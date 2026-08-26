"""
Why this file? Runs one retrieval variant against the gold set and reports what it found.

The two question forms are scored separately and never averaged together, because the
difference between them is the subject of the experiment: a purely semantic retriever
copes with paraphrase and struggles with exact designations.

One retrieval per question, at candidate_k depth. Recall at the smaller top_k is read
off the same list rather than fetched again.
"""
import json
import statistics
import time
from pathlib import Path

from config import RetrievalConfig
from metrics import hits_from_results, mean, ndcg_at_k, recall_at_k, reciprocal_rank
from retriever import Retriever

QUESTION_VARIANTS = {
    "colloquial": "question_colloquial",
    "precise": "question_precise",
}


class RetrievalEvaluator:

    def __init__(self, retriever: Retriever, config: RetrievalConfig):
        self.retriever = retriever
        self.config = config # Frozen, so a result file traces back to exact settings

    @staticmethod
    def load_goldset(path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run goldset_builder.py first.")
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def evaluate_variant(self, goldset: list[dict], variant: str) -> dict:
        """Score one question form across the whole gold set."""
        field = QUESTION_VARIANTS[variant]
        top_k, candidate_k = self.config.top_k, self.config.candidate_k

        recall_top, recall_candidate, ranks, ndcgs, latencies = [], [], [], [], []
        per_document: dict[str, list[float]] = {}
        misses = []
        # Per-question outcomes, so two runs can be compared question by question. An
        # averaged recall cannot say whether a difference exceeds chance; a paired test
        # needs to know *which* questions moved ----> compare_runs
        per_question = []

        for position, entry in enumerate(goldset, start=1):
            if position % 5 == 0 or position == 1:
                print(f"  {variant} {position}/{len(goldset)}", flush=True)
            question = entry[field]
            started = time.perf_counter()
            results = self.retriever.retrieve(question, k=candidate_k)
            latencies.append((time.perf_counter() - started) * 1000)

            hits = hits_from_results(results, entry)
            hit_at_top = recall_at_k(hits, top_k)

            recall_top.append(hit_at_top)
            recall_candidate.append(recall_at_k(hits, candidate_k))
            ranks.append(reciprocal_rank(hits))
            ndcgs.append(ndcg_at_k(hits, top_k))
            per_document.setdefault(entry["source_file"], []).append(hit_at_top)

            rank = next((r for r, hit in enumerate(hits, start=1) if hit), None)
            per_question.append({
                "id": entry["id"],
                f"hit_at_{top_k}": bool(hit_at_top),
                f"hit_at_{candidate_k}": any(hits),
                "rank": rank,
            })

            if not any(hits):
                misses.append({
                    "id": entry["id"],
                    "question": question,
                    "expected": f"{entry['source_file']} p.{entry['page_number']}",
                    "retrieved": [
                        f"{r['metadata'].get('source_file')} p.{r['metadata'].get('page_number')}"
                        for r in results[:3]
                    ],
                })

        return {
            "questions": len(goldset),
            f"recall_at_{top_k}": mean(recall_top),
            f"recall_at_{candidate_k}": mean(recall_candidate),
            "mrr": mean(ranks),
            f"ndcg_at_{top_k}": mean(ndcgs),
            "latency_ms_median": statistics.median(latencies) if latencies else 0.0,
            "per_document_recall": {
                source: mean(values) for source, values in sorted(per_document.items())
            },
            "per_question": per_question,
            "misses": misses,
        }

    def evaluate(self, goldset: list[dict]) -> dict:
        """Score every question form, as one result record for this config."""
        return {
            "config": {
                "name": self.config.name,
                "top_k": self.config.top_k,
                "candidate_k": self.config.candidate_k,
                "use_bm25": self.config.use_bm25,
                "use_reranker": self.config.use_reranker,
            },
            "retriever": self.retriever.name,
            "goldset_size": len(goldset),
            "variants": {
                variant: self.evaluate_variant(goldset, variant)
                for variant in QUESTION_VARIANTS
            },
        }

    @staticmethod
    def evaluate_abstention(engine, questions: list[dict]) -> dict:
        """
        How often the system declines questions the corpus cannot answer.

        The metric that matters most for a system whose selling point is refusing to
        invent - and one no retrieval metric can express, because retrieval always
        returns its k nearest neighbours whether or not any of them is relevant.

        Counts the engine's own refusal flag rather than searching the text for refusal
        phrases, which was tried and could not be made to work. A model that ignores the
        instruction counts as having answered; answered_anyway keeps every such case.
        """
        by_category: dict[str, list[float]] = {}
        answered_anyway = []

        for item in questions:
            result = engine.answer(item["question"])
            abstained = result["abstained"]
            by_category.setdefault(item.get("category", "uncategorised"), []).append(float(abstained))
            if not abstained:
                answered_anyway.append({"question": item["question"], "answer": result["answer"][:300]})

        all_scores = [score for scores in by_category.values() for score in scores]
        return {
            "questions": len(questions),
            "abstention_rate": mean(all_scores),
            "by_category": {name: mean(scores) for name, scores in sorted(by_category.items())},
            "answered_anyway": answered_anyway,
        }

    def report(self, results: dict) -> None:
        """Print the result as a markdown table, ready to paste into a write-up."""
        top_k, candidate_k = self.config.top_k, self.config.candidate_k
        print(f"\n## {results['config']['name']}  ({results['goldset_size']} gold entries)\n")
        header = (
            f"| Question form | Recall@{top_k} | Recall@{candidate_k} | MRR | "
            f"nDCG@{top_k} | Median latency |"
        )
        print(header)
        print("|" + "---|" * 6)
        for variant, scores in results["variants"].items():
            print(
                f"| {variant} | {scores[f'recall_at_{top_k}']:.3f} | "
                f"{scores[f'recall_at_{candidate_k}']:.3f} | {scores['mrr']:.3f} | "
                f"{scores[f'ndcg_at_{top_k}']:.3f} | {scores['latency_ms_median']:.0f} ms |"
            )

        print(f"\n### Recall@{top_k} per document\n")
        print("| Document | colloquial | precise |")
        print("|" + "---|" * 3)
        colloquial = results["variants"]["colloquial"]["per_document_recall"]
        precise = results["variants"]["precise"]["per_document_recall"]
        for source in sorted(colloquial):
            print(f"| {source[:40]} | {colloquial[source]:.2f} | {precise.get(source, 0):.2f} |")

        total_misses = sum(len(v["misses"]) for v in results["variants"].values())
        print(f"\n{total_misses} questions found nothing at all - listed in the result file.")

    @staticmethod
    def save(results: dict, path: Path) -> None:
        """Write the record to disk so it stays comparable across days."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"Saved to {path}")
