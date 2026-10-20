"""Command line entry points for generating reproducible sample inputs."""

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from src.tiering.ingestion.dataset_generator import DatasetGenerator
from src.tiering.ingestion.workload_generator import WorkloadGenerator


def generate(config_path, output_dir, num_files=None, duration_hours=None):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    seed = int(config.get("seed", 42))
    dataset_config = config.get("dataset", {})
    workload_config = config.get("workload", {})
    count = int(num_files or dataset_config.get("num_files", 100))
    duration = float(duration_hours or workload_config.get("duration_hours", 24))
    output_dir = Path(output_dir)
    dataset = DatasetGenerator(dataset_config, seed=seed)
    files = dataset.generate(count)
    dataset_path = output_dir / "files.jsonl"
    dataset.to_jsonl(dataset_path)

    distribution = workload_config.get("pattern_distribution", {})
    trace_config = {
        "hot_ratio": distribution.get("hot", 0.3),
        "warm_ratio": distribution.get("warm", 0.4),
        "cold_ratio": distribution.get("cold", 0.2),
        "seasonal_ratio": distribution.get("seasonal", 0.1),
        "hot_frequency": workload_config.get("hot_base_frequency", 10.0),
        "warm_frequency": workload_config.get("warm_base_frequency", 1.0),
        "cold_frequency": workload_config.get("cold_base_frequency", 0.1),
        "seasonal_period_hours": workload_config.get("seasonal_period_hours", 24),
        "seasonal_amplitude": workload_config.get("seasonal_amplitude_factor", 2.0),
    }
    traces = WorkloadGenerator(trace_config, seed=seed).generate_traces(
        files, duration_hours=duration, start_time=datetime(2024, 1, 1))
    trace_path = output_dir / "access_traces.jsonl"
    WorkloadGenerator(trace_config, seed=seed).to_jsonl(traces, trace_path)
    return {"files": str(dataset_path), "traces": str(trace_path),
            "file_count": len(files), "event_count": len(traces), "seed": seed}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tiering")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate", help="generate synthetic files and access traces")
    generate_parser.add_argument("--config", default="configs/default.yaml")
    generate_parser.add_argument("--output-dir", default="data/generated")
    generate_parser.add_argument("--num-files", type=int)
    generate_parser.add_argument("--duration-hours", type=float)
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate(args.config, args.output_dir, args.num_files, args.duration_hours)
        print(f"Wrote {result['file_count']} files and {result['event_count']} access events.")
        print(f"Dataset: {result['files']}\nTraces: {result['traces']}\nSeed: {result['seed']}")


if __name__ == "__main__":
    main()
