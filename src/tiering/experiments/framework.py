"""Configuration-driven orchestration and reporting for tiering experiments."""

from datetime import datetime, timezone
import json
from pathlib import Path

import yaml
from concurrent.futures import ProcessPoolExecutor

from .runners import RUNNERS


def _run_experiment_spec(spec):
    output_dir, experiment = spec
    experiment = dict(experiment)
    experiment_type = experiment.pop("type")
    return ExperimentFramework(output_dir).run(experiment_type, **experiment)


class ExperimentFramework:
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)

    def run(self, experiment_type, **parameters):
        try:
            runner_cls = RUNNERS[experiment_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported experiment type: {experiment_type}") from exc
        result = runner_cls().run(**parameters)
        result["run_metadata"] = {
            "experiment_type": experiment_type,
            "seed": parameters.get("seed"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return result

    def run_yaml(self, config_path):
        config_path = Path(config_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        experiments = config.get("experiments")
        if not isinstance(experiments, list):
            raise ValueError("Experiment YAML must contain an 'experiments' list")
        if config.get("parallel", False) and len(experiments) > 1:
            workers = config.get("max_workers")
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_run_experiment_spec,
                                        [(str(self.output_dir), item) for item in experiments]))
        else:
            results = [_run_experiment_spec((str(self.output_dir), item))
                       for item in experiments]
        output = config.get("output", {}).get("results_file")
        if output:
            self.save_results(results, output)
        return results

    def save_results(self, results, path=None):
        target = Path(path) if path else self.output_dir / "experiment_results.json"
        if not target.is_absolute():
            target = self.output_dir / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        return target

    @staticmethod
    def write_markdown_report(results, path):
        rows = ["# Experiment Summary", "", "| Experiment | Key result |", "|---|---|"]
        for result in results:
            kind = result.get("experiment_type", "unknown")
            if "baselines" in result:
                summary = f"{len(result['baselines'])} baseline(s) compared"
            elif "migration_count" in result:
                summary = f"{result['migration_count']} migration(s) accepted"
            else:
                detail = result.get("result", result.get("workload_ml", {}))
                summary = f"accuracy={detail.get('accuracy', 'n/a')}, macro F1={detail.get('macro_f1', 'n/a')}"
            rows.append(f"| {kind} | {summary} |")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return target
