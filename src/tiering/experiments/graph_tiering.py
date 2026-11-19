"""Recency tiering vs semantic-graph group tiering on the same trace.

Each policy replays the same 28-day evaluation window from the first test
cutoff (same window as product_ablation's SimPy run). ``recency`` promotes an
accessed file to HOT and demotes idle files; ``graph`` additionally
pre-promotes the file's top-k ACL-sharing semantic neighbors. Sweeping the HOT
idle timeout traces a hit-rate vs storage-cost curve for each, so the graph is
judged at matched cost, not just on latency.

    python -m src.tiering.experiments.graph_tiering --coupling 0.0 0.5 0.8
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

import numpy as np

from src.tiering.experiments.product_ablation import TIER_CONFIGS
from src.tiering.ingestion.production_synth import generate_production_bundle
from src.tiering.migration.controller import MigrationController
from src.tiering.semantic.graph import GraphPrefetchPolicy, SemanticGraph
from src.tiering.semantic.tfidf_extractor import TFIDFExtractor
from src.tiering.simulation.simulator import SimPySimulator


def run_graph_tiering(bundle, hot_idle_hours=(6, 24, 72, 168, 336), k=5,
                      min_similarity=0.3, window_days=28, seed=0):
    evaluation_start = bundle.start + timedelta(days=bundle.test_days[0])
    evaluation_end = evaluation_start + timedelta(days=window_days)
    files = [f for f in bundle.files if f.created_at < evaluation_end]
    traces = [e for e in bundle.traces if e.timestamp < evaluation_end]
    # Vocabulary from text visible at the evaluation start only; no access data.
    tfidf = TFIDFExtractor({"max_features": 256, "ngram_range": (1, 2)})
    tfidf.fit([f.content for f in files if f.created_at <= evaluation_start])
    vectors = {f.file_id: tfidf.transform([f.content])[0] for f in files}
    graph = SemanticGraph(files, vectors, min_similarity=min_similarity)
    # Control: same ACL pools, random neighbors. Separates "prefetch files the
    # same people can see" from "prefetch semantically similar files".
    rng = np.random.default_rng(seed)
    random_graph = SemanticGraph(
        files, {f.file_id: rng.normal(size=8) for f in files}, min_similarity=-1.0)

    def simulate(policy):
        result = SimPySimulator(
            config={"decision_interval_hours": 1.0,
                    "evaluation_start_time": evaluation_start,
                    "access_policy": policy},
            tiers=TIER_CONFIGS,
            migration_controller=MigrationController({"tiers": TIER_CONFIGS}),
        ).run(files, traces)
        return {**result.metrics.model_dump(mode="json"),
                **(policy.report() if policy is not None else {})}

    runs = {"all_cold": simulate(None)}
    for idle in hot_idle_hours:
        runs[f"recency_idle{idle}h"] = simulate(
            GraphPrefetchPolicy(None, k=0, hot_idle_hours=idle))
        runs[f"graph_k{k}_idle{idle}h"] = simulate(
            GraphPrefetchPolicy(graph, k=k, hot_idle_hours=idle))
        runs[f"random_k{k}_idle{idle}h"] = simulate(
            GraphPrefetchPolicy(random_graph, k=k, hot_idle_hours=idle))
    return {"dataset": bundle.label, "stats": bundle.stats,
            "evaluation_window": [evaluation_start.isoformat(), evaluation_end.isoformat()],
            "graph": {**graph.stats(), "k": k, "min_similarity": min_similarity,
                      "edge_rule": "shared access_group and TF-IDF cosine >= min_similarity; "
                                   "neighbors filtered to files created by access time"},
            "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, nargs="+", default=[0.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/metrics"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for coupling in args.coupling:
        bundle = generate_production_bundle(coupling=coupling, seed=args.seed)
        report = run_graph_tiering(bundle, k=args.k)
        path = args.output / f"graph_tiering_{bundle.suffix}.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(path)
        for name, run in report["runs"].items():
            print(f"  {name:24s} hit {run['cache_hit_rate']:.3f}  "
                  f"first-touch-hot {run.get('first_touch_hot_rate', 0):.3f}  "
                  f"lat {run['avg_access_latency_ms']:9.0f}ms  "
                  f"storage {run['total_storage_cost']:.3f}  "
                  f"migr {run['total_migrations']:5d}  "
                  f"prefetch-prec {run.get('prefetch_precision', 0):.2f}")


if __name__ == "__main__":
    main()
