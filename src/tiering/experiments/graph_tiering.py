"""Recency vs group prefetch (random, co-access learned, optional semantic).

Each policy replays the same evaluation window from the first test cutoff.
``recency`` promotes an accessed file to HOT and demotes idle files; the graph
variants also pre-promote the file's top-k same-ACL-group neighbors:

- ``random_k``: random same-group neighbors (the control).
- ``coaccess_k``: neighbors that usually get read within ``lag`` after this
  file, learned online from the trace (no lookahead); ``_c<conf>`` variants
  only prefetch links at least that confident (admission threshold).
- ``coaccess_fb_k``: co-access, topped up from random same-group neighbors
  for files with too few learned links (``_c30``: learned links need 0.30).
- ``coaccess_be_k``: co-access with break-even admission in I/O-ms (see
  ``GraphPrefetchPolicy``): a prefetch must expect to save more read time
  than a wasted move pair would cost.
- ``graph_k`` (``--semantic`` only): TF-IDF text-similar neighbors. Off by
  default: building it dominates time and memory and it tied random on
  Wikipedia.

Sweeping the HOT idle timeout and HOT capacity traces latency vs storage and
migration cost for each, so policies are compared at matched cost. Each run
also reports ``usage_units_io_ms`` = total read latency + migration cost, both
already in ms of device time; storage cost is reported separately.

    python -m src.tiering.experiments.graph_tiering --wikipedia
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

import numpy as np

from src.tiering.coaccess import CoAccessGraph
from src.tiering.experiments.product_ablation import TIER_CONFIGS
from src.tiering.ingestion.production_synth import generate_production_bundle
from src.tiering.migration.controller import MigrationController
from src.tiering.semantic.graph import GraphPrefetchPolicy, SemanticGraph
from src.tiering.semantic.tfidf_extractor import TFIDFExtractor
from src.tiering.simulation.simulator import SimPySimulator


def run_graph_tiering(bundle, hot_idle_hours=(6, 24, 72, 168, 336), k=5,
                      min_similarity=0.3, window_days=28, seed=0,
                      capacity_fractions=(0.02, 0.05, 0.10, 0.20), capped_idle_hours=72,
                      semantic=False, lag_hours=1.0, confidences=(0.1, 0.3)):
    evaluation_start = bundle.start + timedelta(days=bundle.test_days[0])
    evaluation_end = evaluation_start + timedelta(days=window_days)
    files = [f for f in bundle.files if f.created_at < evaluation_end]
    traces = [e for e in bundle.traces if e.timestamp < evaluation_end]
    # Control: same ACL pools, random neighbors. Separates "prefetch files the
    # same people can see" from "prefetch the right ones".
    rng = np.random.default_rng(seed)
    random_graph = SemanticGraph(
        files, {f.file_id: rng.normal(size=8) for f in files}, min_similarity=-1.0)
    groups = {f.file_id: f.metadata.get("access_groups", []) for f in files}

    def coaccess(confidence=0.0, fallback=None):  # fresh per run: it learns online
        return lambda: CoAccessGraph(groups, lag_hours=lag_hours,
                                     min_confidence=confidence, fallback=fallback)

    variants = {"recency": (lambda: None, 0), f"random_k{k}": (lambda: random_graph, k),
                f"coaccess_k{k}": (coaccess(), k),
                f"coaccess_fb_k{k}": (coaccess(fallback=random_graph), k),
                f"coaccess_fb_k{k}_c30": (coaccess(0.3, fallback=random_graph), k),
                f"coaccess_be_k{k}": (coaccess(), k)}
    for confidence in confidences:
        variants[f"coaccess_k{k}_c{round(confidence * 100):02d}"] = (coaccess(confidence), k)
    break_even = {name for name in variants if "_be_" in name}
    graph_info = {"random": random_graph.stats(), "lag_hours": lag_hours,
                  "coaccess_rule": "same access_group; B read within lag after A in >=2 "
                                   "of A's reads; confidence above B's base rate"}
    if semantic:
        # Vocabulary from text visible at the evaluation start only; no access data.
        tfidf = TFIDFExtractor({"max_features": 256, "ngram_range": (1, 2)})
        tfidf.fit([f.content for f in files if f.created_at <= evaluation_start])
        vectors = {f.file_id: tfidf.transform([f.content])[0] for f in files}
        graph = SemanticGraph(files, vectors, min_similarity=min_similarity)
        variants[f"graph_k{k}"] = (lambda: graph, k)
        graph_info["semantic"] = {**graph.stats(), "min_similarity": min_similarity}

    def simulate(policy):
        result = SimPySimulator(
            config={"decision_interval_hours": 1.0,
                    "evaluation_start_time": evaluation_start,
                    "access_policy": policy},
            tiers=TIER_CONFIGS,
            migration_controller=MigrationController({"tiers": TIER_CONFIGS}),
        ).run(files, traces)
        out = {**result.metrics.model_dump(mode="json"),
               **(policy.report() if policy is not None else {})}
        out["usage_units_io_ms"] = (out["avg_access_latency_ms"] * out["total_accesses"]
                                    + out["total_migration_cost"])
        if policy is not None and policy.graph is not None:
            out["graph_stats"] = policy.graph.stats()
        return out

    sizes = {f.file_id: f.size_bytes for f in files}
    total_bytes = sum(sizes.values())
    runs = {"all_cold": simulate(None)}
    for idle in hot_idle_hours:  # uncapped: idle timeout is the only cost lever
        for name, (make, kk) in variants.items():
            runs[f"{name}_idle{idle}h"] = simulate(GraphPrefetchPolicy(
                make(), k=kk, hot_idle_hours=idle, sizes=sizes,
                tier_configs=TIER_CONFIGS if name in break_even else None))
    capped = {}
    for fraction in capacity_fractions:  # HOT capped at a share of stored bytes
        cap = int(total_bytes * fraction)
        for name, (make, kk) in variants.items():
            capped[f"{name}_cap{round(fraction * 100)}pct"] = simulate(GraphPrefetchPolicy(
                make(), k=kk, hot_idle_hours=capped_idle_hours,
                hot_capacity_bytes=cap, sizes=sizes,
                tier_configs=TIER_CONFIGS if name in break_even else None))
    return {"dataset": bundle.label, "stats": bundle.stats,
            "evaluation_window": [evaluation_start.isoformat(), evaluation_end.isoformat()],
            "graph": {**graph_info, "k": k,
                      "neighbors_filtered_to_files_created_by_access_time": True},
            "total_bytes": total_bytes, "capped_idle_hours": capped_idle_hours,
            "runs": runs, "capped_runs": capped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, nargs="+", default=[0.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/metrics"))
    parser.add_argument("--wikipedia", action="store_true",
                        help="use the fetched Wikipedia store (wikipedia_store) instead")
    parser.add_argument("--start", default="2026-11-08", help="--wikipedia: fetch start")
    parser.add_argument("--days", type=int, default=8, help="--wikipedia: fetched days")
    parser.add_argument("--target-events", type=int, default=55_000,
                        help="--wikipedia: thin pageviews to about this many events")
    parser.add_argument("--semantic", action="store_true",
                        help="also run the TF-IDF semantic graph (slow, memory-heavy)")
    parser.add_argument("--lag-hours", type=float, default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.wikipedia:
        from datetime import datetime, timezone
        from src.tiering.ingestion.wikipedia_store import load_wikipedia_bundle
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        jobs = [(load_wikipedia_bundle(start, args.days, target_events=args.target_events,
                                       with_text=args.semantic), args.days - 1)]
    else:
        jobs = [(generate_production_bundle(coupling=c, seed=args.seed), 28)
                for c in args.coupling]
    for bundle, window_days in jobs:
        report = run_graph_tiering(bundle, k=args.k, window_days=window_days,
                                   semantic=args.semantic, lag_hours=args.lag_hours)
        # separate file: keeps the earlier semantic-graph result intact
        path = args.output / f"graph_tiering_{bundle.suffix}_coaccess.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(path)
        for name, run in {**report["runs"], **report["capped_runs"]}.items():
            print(f"  {name:24s} hit {run['cache_hit_rate']:.3f}  "
                  f"first-touch-hot {run.get('first_touch_hot_rate', 0):.3f}  "
                  f"lat {run['avg_access_latency_ms']:9.0f}ms  "
                  f"storage {run['total_storage_cost']:.3f}  "
                  f"usage {run['usage_units_io_ms'] / 1e6:7.2f}M  "
                  f"migr {run['total_migrations']:5d}  "
                  f"prefetch-prec {run.get('prefetch_precision', 0):.2f}  "
                  f"evict {run.get('evictions', 0):5d}  "
                  f"peakHOT {run.get('peak_hot_bytes', 0) / 1e9:6.1f}GB")


if __name__ == "__main__":
    main()
