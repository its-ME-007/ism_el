"""Paper figures for the co-access prefetch results.

Reads results/metrics/graph_tiering_wikipedia_coaccess.json and writes PDF (for LaTeX)
and 300-dpi PNG files to results/figures/.

    python -m src.tiering.experiments.plot_coaccess
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results/metrics/graph_tiering_wikipedia_coaccess.json"
OUT = ROOT / "results/figures"

CAPS = [2, 5, 10, 20]
POLICIES = [  # key, label, colour, marker
    ("recency", "Recency (no prefetch)", "#7a8499", "o"),
    ("random_k5", "Random same-group (control)", "#d4762c", "s"),
    ("coaccess_k5", "Co-access", "#8a63d2", "^"),
    ("coaccess_fb_k5", "Co-access + fallback", "#1f6feb", "D"),
    ("coaccess_be_k5", "Co-access + break-even", "#1a9b6c", "v"),
]
METRICS = [  # field, y label, scale, file stem
    ("avg_access_latency_ms", "Average read latency (ms)", 1, "latency_vs_cap"),
    ("total_migrations", "Migrations (thousands)", 1e3, "migrations_vs_cap"),
    ("usage_units_io_ms", "Usage (M device-ms)", 1e6, "usage_vs_cap"),
]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def main():
    runs = json.loads(SRC.read_text())["capped_runs"]
    OUT.mkdir(parents=True, exist_ok=True)
    get = lambda p, c, f: runs[f"{p}_cap{c}pct"][f]  # noqa: E731

    for field, ylabel, scale, stem in METRICS:
        fig, ax = plt.subplots(figsize=(3.4, 2.5))  # one IEEE column
        for key, label, colour, marker in POLICIES:
            ax.plot(CAPS, [get(key, c, field) / scale for c in CAPS], marker=marker,
                    color=colour, label=label, lw=1.4, ms=4)
        ax.set_xscale("log")
        ax.set_xticks(CAPS, [f"{c}%" for c in CAPS])
        ax.minorticks_off()
        ax.set_xlabel("HOT tier cap (% of bytes)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, frameon=False)
        save(fig, stem)

    # Trade-off at the 5% cap: latency against migrations.
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for key, label, colour, marker in POLICIES:
        ax.scatter(get(key, 5, "total_migrations") / 1e3, get(key, 5, "avg_access_latency_ms"),
                   color=colour, marker=marker, s=36, label=label, zorder=3)
    ax.set_xlabel("Migrations (thousands)")
    ax.set_ylabel("Average read latency (ms)")
    ax.set_title("5% HOT cap (lower-left is better)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    save(fig, "tradeoff_cap5")

    # Prefetch precision per policy at each cap.
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    pre = POLICIES[1:]
    w = 0.8 / len(pre)
    for i, (key, label, colour, _) in enumerate(pre):
        ax.bar([j + (i - (len(pre) - 1) / 2) * w for j in range(len(CAPS))],
               [get(key, c, "prefetch_precision") for c in CAPS], w, color=colour, label=label)
    ax.set_xticks(range(len(CAPS)), [f"{c}%" for c in CAPS])
    ax.set_xlabel("HOT tier cap (% of bytes)")
    ax.set_ylabel("Prefetch precision")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "precision_vs_cap")

    print(f"wrote {len(list(OUT.glob('*.pdf')))} figures to {OUT}")


if __name__ == "__main__":
    main()
