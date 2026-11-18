"""Cost-aware placement evaluation with sourced storage prices.

Each test row is one document at one cutoff with a 24-hour decision window.
Every policy assigns HOT/WARM/COLD (S3 Standard / Standard-IA / Glacier
Instant Retrieval, see ``configs/pricing_aws_s3_us_east_1.yaml``), and the
realized cost of that window is charged from the *actual* future accesses:

* storage for one day on the billable size (128 KB minimum for IA/GIR),
* per-access GET and per-GB retrieval fees,
* tier-change fees (lifecycle transition into IA/GIR; copy + retrieval to
  return to Standard),
* early-deletion charges when a document leaves IA/GIR before its minimum
  storage duration (measured between that document's consecutive rows).

Policy limit: HOT may hold at most ``hot_capacity_fraction`` of the cutoff's
billable bytes; overflow is demoted to WARM in ascending policy-score order.
Decisions are myopic (storage + expected access cost); transition and
minimum-duration charges are scored but not anticipated by any policy.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

DEFAULT_PRICING = Path("configs/pricing_aws_s3_us_east_1.yaml")
TIERS = ("HOT", "WARM", "COLD")
DAYS_PER_MONTH = 365.25 / 12
GB = 1024 ** 3
PUT_PER_1000 = 0.005  # S3 Standard PUT/COPY list price, used to restore to HOT


def load_pricing(path: Path = DEFAULT_PRICING) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _billable(size, tier_price):
    return np.maximum(size, tier_price["min_billable_bytes"]) / GB


def _unit_costs(sizes, pricing, retrieval_multiplier):
    """Per-row daily storage cost and per-access cost for each tier, shape (n, 3)."""
    storage = np.stack([
        _billable(sizes, pricing["tiers"][t]) * pricing["tiers"][t]["storage_per_gb_month"]
        / DAYS_PER_MONTH for t in TIERS], axis=1)
    access = np.stack([
        pricing["tiers"][t]["get_per_1000"] / 1000
        + retrieval_multiplier * pricing["tiers"][t]["retrieval_per_gb"] * sizes / GB
        for t in TIERS], axis=1)
    return storage, access


def cost_minimizing_tiers(expected_counts, storage, access):
    return np.argmin(storage + np.asarray(expected_counts)[:, None] * access, axis=1)


def _apply_capacity(tiers, scores, sizes, cutoffs, fraction):
    if fraction >= 1.0:
        return tiers
    tiers = tiers.copy()
    for cutoff in np.unique(cutoffs):
        rows = np.flatnonzero(cutoffs == cutoff)
        budget = fraction * sizes[rows].sum()
        hot = rows[tiers[rows] == 0]
        hot = hot[np.argsort(-scores[hot], kind="stable")]
        used = np.cumsum(sizes[hot])
        tiers[hot[used > budget]] = 1
    return tiers


def realized_cost(tiers, counts, sizes, file_ids, cutoff_days, pricing,
                  retrieval_multiplier=1.0, charge_tier_changes=True):
    """Total realized cost and its components for one placement.

    ``charge_tier_changes=False`` drops transition and minimum-duration charges,
    approximating tiers without per-move fees (e.g. on-prem capacity tiers).
    """
    storage, access = _unit_costs(sizes, pricing, retrieval_multiplier)
    n = np.arange(len(tiers))
    parts = {"storage": float(storage[n, tiers].sum()),
             "access": float((counts * access[n, tiers]).sum()),
             "transition": 0.0, "early_deletion": 0.0}
    by_doc = defaultdict(list)
    for row, file_id in enumerate(file_ids):
        by_doc[file_id].append(row)
    for rows in (by_doc.values() if charge_tier_changes else ()):
        rows.sort(key=lambda r: cutoff_days[r])
        previous, previous_day = 0, None  # new documents land in Standard
        for row in rows:
            tier = int(tiers[row])
            if tier != previous:
                if tier == 0:
                    parts["transition"] += (PUT_PER_1000 / 1000 + retrieval_multiplier
                                            * pricing["tiers"][TIERS[previous]]
                                            ["retrieval_per_gb"] * sizes[row] / GB)
                else:
                    parts["transition"] += pricing["tiers"][TIERS[tier]][
                        "transition_in_per_1000"] / 1000
                if previous != 0 and previous_day is not None:
                    held = cutoff_days[row] - previous_day
                    remaining = pricing["tiers"][TIERS[previous]]["min_storage_days"] - held
                    if remaining > 0:
                        parts["early_deletion"] += remaining * storage[row, previous]
            if tier != previous or previous_day is None:
                previous_day = cutoff_days[row]
            previous = tier
    parts["total"] = sum(parts.values())
    served = np.bincount(tiers, weights=counts, minlength=3)
    parts["accesses_served_share"] = {
        TIERS[i]: float(served[i] / counts.sum()) if counts.sum() else 0.0
        for i in range(3)}
    parts["placement_share"] = {TIERS[i]: float(np.mean(tiers == i)) for i in range(3)}
    return parts


def _baseline_policies(features):
    last = np.asarray([f.get("workload_last_access_hours", 24 * 30) for f in features])
    week = np.asarray([f.get("workload_access_count_7d", 0.0) for f in features])
    day = np.asarray([f.get("workload_access_count_24h", 0.0) for f in features])
    age = np.asarray([f.get("created_age_days", 0.0) for f in features])
    return {
        "LRU_recency": (np.where(last <= 24, 0, np.where(last <= 24 * 7, 1, 2)), -last),
        "LFU_7d_frequency": (np.where(week >= 2, 0, np.where(week >= 1, 1, 2)), week),
        "age_based": (np.where(age < 7, 0, np.where(age < 30, 1, 2)), -age),
    }, day


def evaluate_policies(samples, sizes, model_counts, pricing=None,
                      size_overrides=(None, 1 << 20, 10 << 20),
                      retrieval_multipliers=(0.5, 1.0, 2.0),
                      hot_capacity_fractions=(1.0, 0.25, 0.10),
                      fee_models=("s3_full", "storage_and_access_only"),
                      headline=(1 << 20, 1.0, 1.0, "s3_full"), bootstrap_reps=500,
                      seed=42):
    """Score static, heuristic, model and oracle placements over a sensitivity grid.

    ``samples``: test rows with ``file_id``, ``cutoff`` (ISO), ``future_count`` and
    ``features``; ``sizes``: actual bytes per row; ``model_counts``: run name ->
    predicted next-24h access counts per row.
    """
    pricing = pricing or load_pricing()
    counts = np.asarray([row["future_count"] for row in samples], dtype=np.float64)
    file_ids = [row["file_id"] for row in samples]
    cutoffs = np.asarray([row["cutoff"] for row in samples])
    day_index = {value: i for i, value in enumerate(sorted(set(cutoffs)))}
    # Cutoffs are whole days apart; convert to day offsets for duration rules.
    first = datetime.fromisoformat(min(cutoffs))
    cutoff_days = np.asarray([(datetime.fromisoformat(c) - first).days for c in cutoffs])
    heuristics, past_day = _baseline_policies([row["features"] for row in samples])
    actual_sizes = np.asarray(sizes, dtype=np.float64)

    def placements(row_sizes, multiplier):
        storage, access = _unit_costs(row_sizes, pricing, multiplier)
        result = {name: (np.full(len(samples), i), np.zeros(len(samples)))
                  for i, name in enumerate(("static_all_HOT", "static_all_WARM",
                                            "static_all_COLD"))}
        result.update(heuristics)
        result["P_persistence"] = (cost_minimizing_tiers(past_day, storage, access),
                                   past_day)
        for run, predicted in model_counts.items():
            predicted = np.asarray(predicted, dtype=np.float64)
            result[run] = (cost_minimizing_tiers(predicted, storage, access), predicted)
        result["oracle_actual_counts"] = (cost_minimizing_tiers(counts, storage, access),
                                          counts)
        return result

    grid = []
    headline_tiers = None
    for override in size_overrides:
        row_sizes = actual_sizes if override is None else np.full(len(samples), float(override))
        for multiplier in retrieval_multipliers:
            policies = placements(row_sizes, multiplier)
            for fraction, fee_model in ((f, m) for f in hot_capacity_fractions
                                        for m in fee_models):
                cell = {"object_size_bytes": "actual" if override is None else int(override),
                        "retrieval_fee_multiplier": multiplier,
                        "hot_capacity_fraction": fraction, "fee_model": fee_model,
                        "policies": {}}
                limited = {}
                for name, (tiers, scores) in policies.items():
                    tiers = _apply_capacity(np.asarray(tiers, dtype=np.int64),
                                            np.asarray(scores, dtype=np.float64),
                                            row_sizes, cutoffs, fraction)
                    limited[name] = tiers
                    cell["policies"][name] = realized_cost(
                        tiers, counts, row_sizes, file_ids, cutoff_days, pricing,
                        multiplier, fee_model == "s3_full")
                best_static = min(cell["policies"][p]["total"] for p in
                                  ("static_all_HOT", "static_all_WARM", "static_all_COLD"))
                for result in cell["policies"].values():
                    result["saving_vs_best_static"] = (
                        1 - result["total"] / best_static if best_static else 0.0)
                grid.append(cell)
                if (override, multiplier, fraction, fee_model) == headline:
                    headline_tiers = (limited, row_sizes, multiplier,
                                      fee_model == "s3_full")

    report = {"pricing_source": pricing.get("source_url"),
              "pricing_checked": str(pricing.get("checked")),
              "rows": len(samples), "documents": len(set(file_ids)),
              "cutoffs": len(day_index),
              "decision_rule": "myopic argmin(storage_day + predicted_count * access_cost)",
              "grid": grid}
    if headline_tiers is not None:
        report["headline"] = _bootstrap_costs(
            headline_tiers, counts, file_ids, cutoff_days, pricing,
            list(model_counts), bootstrap_reps, seed)
        report["headline"]["setting"] = {
            "object_size_bytes": headline[0], "retrieval_fee_multiplier": headline[1],
            "hot_capacity_fraction": headline[2], "fee_model": headline[3]}
    return report


def _bootstrap_costs(headline, counts, file_ids, cutoff_days, pricing, runs, reps, seed):
    """Document-clustered bootstrap of total cost per policy and model differences."""
    limited, row_sizes, multiplier, charge = headline
    rows_by_doc = defaultdict(list)
    for row, file_id in enumerate(file_ids):
        rows_by_doc[file_id].append(row)
    clusters = list(rows_by_doc.values())
    rng = np.random.default_rng(seed)
    # Cost decomposes over documents, so precompute each policy's per-document cost.
    per_doc = {}
    for name, tiers in limited.items():
        per_doc[name] = np.asarray([
            realized_cost(tiers[rows], counts[rows], row_sizes[rows],
                          [file_ids[r] for r in rows], cutoff_days[rows], pricing,
                          multiplier, charge)["total"] for rows in clusters])
    draws = rng.integers(0, len(clusters), (reps, len(clusters)))
    totals = {name: per_doc[name][draws].sum(axis=1) for name in per_doc}
    scale = 1e6 / len(counts)  # USD per million document-days
    reference = [name for name in ("LRU_recency", "LFU_7d_frequency", "age_based",
                                   "P_persistence", "static_all_HOT")]
    summary = {name: {"usd_per_million_doc_days": float(per_doc[name].sum() * scale),
                      "ci95": [float(np.percentile(totals[name], 2.5) * scale),
                               float(np.percentile(totals[name], 97.5) * scale)]}
               for name in per_doc}
    pairs = [(run, other) for run in runs for other in reference]
    pairs += [("C_combined", "A_metadata_workload"),
              ("D_combined_retrieval", "C_combined"),
              ("D_combined_retrieval", "A_metadata_workload")]
    differences = {}
    for left, right in pairs:
        if left not in totals or right not in totals:
            continue
        relative = totals[left] / totals[right] - 1
        differences[f"{left} vs {right}"] = {
            "relative_cost_change": float(per_doc[left].sum() / per_doc[right].sum() - 1),
            "ci95": [float(np.percentile(relative, 2.5)),
                     float(np.percentile(relative, 97.5))],
            "share_of_resamples_left_cheaper": float(np.mean(relative < 0)),
        }
    return {"method": f"document-clustered bootstrap, {reps} resamples",
            "policies": summary, "relative_differences": differences}


def _demo():
    """Self-check: oracle never costs more than any policy; capacity is respected."""
    pricing = load_pricing()
    rng = np.random.default_rng(0)
    samples = []
    for i in range(300):
        future = int(rng.poisson(0.4))
        samples.append({"file_id": f"d{i % 120}", "cutoff": f"2001-0{1 + i // 120}-01T00:00:00+00:00",
                        "future_count": future,
                        "features": {"workload_access_count_24h": float(rng.poisson(0.4)),
                                     "workload_access_count_7d": float(rng.poisson(2)),
                                     "workload_last_access_hours": float(rng.uniform(0, 720)),
                                     "created_age_days": float(rng.uniform(0, 60))}})
    noisy = np.asarray([s["future_count"] for s in samples]) + rng.normal(0, 0.3, 300)
    report = evaluate_policies(samples, np.full(300, 50_000.0),
                               {"noisy_model": np.maximum(noisy, 0)},
                               size_overrides=(1 << 20,), retrieval_multipliers=(1.0,),
                               hot_capacity_fractions=(1.0, 0.1), bootstrap_reps=50,
                               headline=(1 << 20, 1.0, 1.0, "storage_and_access_only"))
    for cell in report["grid"]:
        oracle = cell["policies"]["oracle_actual_counts"]
        if cell["fee_model"] == "storage_and_access_only" and cell["hot_capacity_fraction"] == 1:
            assert all(oracle["total"] <= r["total"] + 1e-12
                       for r in cell["policies"].values())
        # Oracle is optimal for the myopic objective (storage + access).
        for name, result in cell["policies"].items():
            assert oracle["storage"] + oracle["access"] <= \
                result["storage"] + result["access"] + 1e-12, name
        if cell["hot_capacity_fraction"] < 1:
            for result in cell["policies"].values():
                assert result["placement_share"]["HOT"] <= 0.1 + 1e-9
    print("cost_evaluation demo OK:",
          {k: round(v["usd_per_million_doc_days"], 4)
           for k, v in report["headline"]["policies"].items()})


if __name__ == "__main__":
    _demo()
