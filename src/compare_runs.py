"""
Compares two evaluation runs question by question.

An averaged recall cannot say whether a difference is larger than chance. With 64
questions per form, +7.8 points is five questions, and the confidence interval around a
single rate that size is wider than the difference. Since both variants answer the same
questions, the comparison is paired - which allows a far more sensitive test than
comparing two rates: only the questions where the two disagree carry information.

McNemar's exact test, computed from the binomial distribution rather than imported, so
the arithmetic is visible and the project keeps one dependency fewer.

Usage:
    python src/compare_runs.py mvp_baseline hybrid
"""
import argparse
import json
from math import comb

import config


def load(name: str) -> dict:
    path = config.RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run run_eval.py --config {name} first.")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def hits_by_id(variant: dict, key: str) -> dict[str, bool]:
    """Map gold id to outcome. Fails loudly if the run predates per-question logging."""
    if "per_question" not in variant:
        raise KeyError(
            "This result file has no per_question data - it was produced before the "
            "evaluator recorded it. Re-run that config."
        )
    return {entry["id"]: entry[key] for entry in variant["per_question"]}


def mcnemar(before: dict[str, bool], after: dict[str, bool]) -> dict:
    """
    Paired comparison of two binary outcomes over the same questions.

    Returns the two discordant counts and the two-sided exact p-value. Questions both
    variants get right, or both get wrong, carry no information about which is better
    and are excluded - that exclusion is what makes the test sensitive at this sample
    size.
    """
    shared = sorted(set(before) & set(after))
    fixed = sum(1 for i in shared if not before[i] and after[i])
    broken = sum(1 for i in shared if before[i] and not after[i])

    discordant = fixed + broken
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(comb(discordant, i) for i in range(min(fixed, broken) + 1))
        p_value = min(1.0, 2 * tail / 2 ** discordant)

    return {
        "questions": len(shared),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "p_value": p_value,
    }


def report(name_a: str, name_b: str) -> None:
    a, b = load(name_a), load(name_b)
    print(f"\n{name_a}  ->  {name_b}\n")

    for form in ("colloquial", "precise"):
        variant_a, variant_b = a["variants"][form], b["variants"][form]
        print(f"  {form}")
        for key in ("hit_at_5", "hit_at_20"):
            result = mcnemar(hits_by_id(variant_a, key), hits_by_id(variant_b, key))
            # A result is only worth calling an improvement if it would be unlikely to
            # arise from a coin flip. 0.05 is convention, not a law of nature, and with
            # 64 questions a real but modest effect can easily sit above it.
            verdict = (
                "distinguishable from chance" if result["p_value"] < 0.05
                else "not distinguishable from chance at this sample size"
            )
            print(
                f"    {key:<10} fixed {result['fixed']:>2}, broke {result['broken']:>2}, "
                f"net {result['net']:>+3}  p = {result['p_value']:.3f}  - {verdict}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation runs, paired.")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    report(args.baseline, args.candidate)


if __name__ == "__main__":
    main()
