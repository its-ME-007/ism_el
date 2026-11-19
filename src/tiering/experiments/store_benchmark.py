"""End-to-end benchmark of TieredStore on a replayed production-shaped trace.

Files that exist before the window are put at their creation time (with
hourly ticks), then the window's puts (new files) and gets (trace accesses)
are replayed in time order through the real on-disk store. Payload bytes are
scaled down by ``scale`` so the benchmark fits on a laptop disk; latency is
modelled from each file's real size and the tier that served it, using the
simulator's TIER_CONFIGS. Reports wall-clock op rates, HOT hit rate, first-touch
hits, modelled latency, peak HOT bytes against the cap, and an on-disk
consistency check.

    python -m src.tiering.experiments.store_benchmark --coupling 0.8
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from src.tiering.experiments.product_ablation import TIER_CONFIGS
from src.tiering.ingestion.production_synth import generate_production_bundle
from src.tiering.ingestion.schemas import Tier
from src.tiering.store import TieredStore


def _latency_ms(tier: Tier, size_bytes: int) -> float:
    cfg = TIER_CONFIGS[tier]
    return cfg["latency_ms"] + size_bytes * 8 / (cfg["throughput_mbps"] * 1e6) * 1000


def replay(bundle, root: Path, k: int, cap_fraction: float | None,
           window_days: int = 7, scale: int = 10_000) -> dict:
    start = bundle.start + timedelta(days=bundle.test_days[0])
    end = start + timedelta(days=window_days)
    files = sorted((f for f in bundle.files if f.created_at < end), key=lambda f: f.created_at)
    payload = {f.file_id: max(1, f.size_bytes // scale) for f in files}
    cap = int(sum(payload.values()) * cap_fraction) if cap_fraction else None
    store = TieredStore(root, k=k, hot_capacity_bytes=cap)
    by_id = {f.file_id: f for f in files}

    ops = [(f.created_at, 0, "put", f.file_id) for f in files]
    ops += [(e.timestamp, 1, "get", e.file_id) for e in bundle.traces
            if start <= e.timestamp < end and e.file_id in by_id]
    ops.sort()
    timing = {"put": [0, 0.0], "get": [0, 0.0], "tick": [0, 0.0]}
    served, first_seen, peak_hot = [], set(), 0
    next_tick = None
    for when, _, op, file_id in ops:
        if next_tick is None:
            next_tick = when + timedelta(hours=1)
        while when >= next_tick:
            t = time.perf_counter()
            store.tick(next_tick)
            timing["tick"][0] += 1
            timing["tick"][1] += time.perf_counter() - t
            next_tick += timedelta(hours=1)
        file = by_id[file_id]
        t = time.perf_counter()
        if op == "put":
            store.put(file_id, b"\0" * payload[file_id], groups=file.metadata["access_groups"],
                      text=file.content, now=when)
        else:
            tier = store.tier_of(file_id)
            store.get(file_id, now=when)
            served.append((tier, file.size_bytes, file_id not in first_seen))
            first_seen.add(file_id)
        timing[op][0] += 1
        timing[op][1] += time.perf_counter() - t
        if len(served) % 200 == 0:
            peak_hot = max(peak_hot, store.status()["tiers"]["HOT"]["bytes"])
    peak_hot = max(peak_hot, store.status()["tiers"]["HOT"]["bytes"])

    first = [s for s in served if s[2]]
    return {
        "k": k, "hot_capacity_fraction": cap_fraction, "hot_capacity_bytes": cap,
        "files": len(files), "gets": len(served), "puts": timing["put"][0],
        "ops_per_second": {op: round(n / secs, 1) if secs else None
                           for op, (n, secs) in timing.items()},
        "mean_ms_per_op": {op: round(secs / n * 1000, 2) if n else None
                           for op, (n, secs) in timing.items()},
        "hot_hit_rate": float(np.mean([t == Tier.HOT for t, _, _ in served])),
        "first_touch_hot_rate": float(np.mean([t == Tier.HOT for t, _, _ in first]))
        if first else 0.0,
        "modelled_mean_latency_ms": float(np.mean([_latency_ms(t, s) for t, s, _ in served])),
        "peak_hot_bytes": peak_hot, "cap_respected": cap is None or peak_hot <= cap,
        "misplaced_after_replay": store.verify(),
        "final_tiers": {t: v["files"] for t, v in store.status()["tiers"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/metrics"))
    args = parser.parse_args()
    bundle = generate_production_bundle(coupling=args.coupling, seed=args.seed)
    results = {}
    for name, k, cap in (("recency", 0, None), ("graph_k5", 5, None),
                         ("recency_cap5pct", 0, 0.05), ("graph_k5_cap5pct", 5, 0.05)):
        root = Path(tempfile.mkdtemp(prefix=f"store-{name}-"))
        try:
            results[name] = replay(bundle, root, k, cap, args.window_days)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        r = results[name]
        print(f"{name:18s} hit {r['hot_hit_rate']:.3f} first-touch {r['first_touch_hot_rate']:.3f} "
              f"lat {r['modelled_mean_latency_ms']:8.0f}ms  get {r['mean_ms_per_op']['get']}ms "
              f"put {r['mean_ms_per_op']['put']}ms tick {r['mean_ms_per_op']['tick']}ms "
              f"cap-ok {r['cap_respected']} misplaced {len(r['misplaced_after_replay'])}",
              flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"store_benchmark_{bundle.suffix}.json"
    path.write_text(json.dumps({"dataset": bundle.label, "window_days": args.window_days,
                                "runs": results}, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
