"""Markdown/HTML reports and optional plots for experiment result dictionaries."""

from html import escape
from pathlib import Path
import json


class ExperimentReporter:
    @staticmethod
    def write_markdown(results, path):
        from .framework import ExperimentFramework
        return ExperimentFramework.write_markdown_report(results, path)

    @staticmethod
    def write_html(results, path):
        markdown_lines = ["# Experiment Results", ""]
        for index, result in enumerate(results, 1):
            markdown_lines.extend((f"## Run {index}: {result.get('experiment_type', 'unknown')}",
                                   "", "```json", json.dumps(result, indent=2, default=str),
                                   "```", ""))
        content = "\n".join(markdown_lines)
        html = ("<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Experiment Results</title></head><body><pre>" +
                escape(content) + "</pre></body></html>")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        return target

    @staticmethod
    def plot_comparison(results, metric, path):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Plots require the visualization extra: pip install -e '.[visualization]'") from exc
        labels, values = [], []
        for result in results:
            experiment = result.get("experiment_type", "run")
            detail = result.get("result", result.get("workload_ml", result))
            if metric in detail and isinstance(detail[metric], (int, float)):
                labels.append(experiment)
                values.append(float(detail[metric]))
        if not values:
            raise ValueError(f"No numeric '{metric}' values found in results")
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
        ax.bar(labels, values, color="#3274a1")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Experiment comparison: {metric}")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150)
        plt.close(fig)
        return target
