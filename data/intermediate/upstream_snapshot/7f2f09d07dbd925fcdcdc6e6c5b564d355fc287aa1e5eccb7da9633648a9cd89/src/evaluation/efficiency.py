"""Aggregation helpers for paired computational-efficiency benchmarks."""

from __future__ import annotations

from statistics import median, quantiles
from typing import Any, Iterable


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    """Return compact descriptive statistics without implying a large-sample CI."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot summarize an empty measurement series")
    if len(ordered) == 1:
        q1 = q3 = ordered[0]
    else:
        q1, _, q3 = quantiles(ordered, n=4, method="inclusive")
    return {
        "n": len(ordered),
        "min": ordered[0],
        "q1": q1,
        "median": median(ordered),
        "q3": q3,
        "max": ordered[-1],
    }


def paired_comparison(
    runs: list[dict[str, Any]], metric: str, *, fixed: str = "fixed_matched", structure: str = "structure"
) -> dict[str, Any]:
    """Compare structure and fixed inside each adjacent AB/BA benchmark pair."""
    by_pair: dict[int, dict[str, float]] = {}
    for run in runs:
        value = run.get("measurements", {}).get(metric)
        if value is None:
            raise ValueError(f"Run has no measurement {metric!r}")
        by_pair.setdefault(int(run["pair_id"]), {})[str(run["strategy"])] = float(value)

    rows: list[dict[str, float | int]] = []
    for pair_id in sorted(by_pair):
        pair = by_pair[pair_id]
        if set(pair) != {fixed, structure}:
            raise ValueError(f"Pair {pair_id} is incomplete for metric {metric}")
        fixed_value = pair[fixed]
        structure_value = pair[structure]
        if fixed_value == 0:
            raise ValueError(f"Pair {pair_id} has a zero fixed value for {metric}")
        rows.append(
            {
                "pair_id": pair_id,
                "fixed": fixed_value,
                "structure": structure_value,
                "difference": structure_value - fixed_value,
                "ratio": structure_value / fixed_value,
                "overhead_percent": (structure_value / fixed_value - 1.0) * 100.0,
            }
        )
    return {
        "metric": metric,
        "pairs": rows,
        "fixed": distribution(row["fixed"] for row in rows),
        "structure": distribution(row["structure"] for row in rows),
        "paired_difference": distribution(row["difference"] for row in rows),
        "paired_ratio": distribution(row["ratio"] for row in rows),
        "paired_overhead_percent": distribution(row["overhead_percent"] for row in rows),
    }


def pareto_status(
    *, quality_difference: float, cost_comparisons: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Apply strict observed-median Pareto rules across declared cost dimensions."""
    median_differences = {
        metric: float(comparison["paired_difference"]["median"])
        for metric, comparison in cost_comparisons.items()
    }
    structure_not_worse = {
        metric: difference <= 0.0 for metric, difference in median_differences.items()
    }
    fixed_not_worse = {
        metric: difference >= 0.0 for metric, difference in median_differences.items()
    }
    structure_dominates = quality_difference > 0 and all(structure_not_worse.values())
    fixed_dominates = quality_difference < 0 and all(fixed_not_worse.values())
    return {
        "basis": "observed paired medians; no significance claim for cost measurements",
        "cost_median_differences_structure_minus_fixed": median_differences,
        "structure_not_worse_by_cost": structure_not_worse,
        "structure_globally_dominates": structure_dominates,
        "fixed_globally_dominates": fixed_dominates,
        "both_potentially_pareto_optimal": not structure_dominates and not fixed_dominates,
    }
