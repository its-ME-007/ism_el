"""Command line entry points for generating reproducible sample inputs."""

import argparse
from datetime import datetime
import json
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

    fetch = subparsers.add_parser("fetch-enron", help="parse Enron mailboxes to JSONL")
    fetch.add_argument("--source", default="data/raw/enron_mail.tar.gz")
    fetch.add_argument("--output", default="data/processed/enron_sample.jsonl")
    fetch.add_argument("--owners", default="", help="comma-separated mailbox names")
    fetch.add_argument("--max-emails", type=int, default=0)

    ablate = subparsers.add_parser("ablate", help="train and score A/B/C/D")
    ablate.add_argument("--enron", help="parsed Enron JSONL; omit for synthetic")
    ablate.add_argument("--proxy", choices=["thread", "participant"], default="thread")
    ablate.add_argument("--horizon-days", type=int, default=14)
    ablate.add_argument("--bootstrap-reps", type=int, default=1000)

    predict = subparsers.add_parser("predict", help="recommend a tier for one document")
    predict.add_argument("input", type=Path)
    predict.add_argument("--run", choices=["C_combined", "D_combined_retrieval"],
                         default="C_combined")
    predict.add_argument("--model", type=Path, help="bundle path; default: active version")
    predict.add_argument("--registry", type=Path, default=Path("results/models"))
    predict.add_argument("--log", type=Path, default=Path("results/logs/predictions.jsonl"))
    predict.add_argument("--no-log", action="store_true")

    models = subparsers.add_parser("models", help="manage versioned model bundles")
    models.add_argument("action", choices=["list", "promote", "rollback"])
    models.add_argument("bundle", nargs="?", type=Path)
    models.add_argument("--note", default="")
    models.add_argument("--registry", type=Path, default=Path("results/models"))

    watch = subparsers.add_parser("monitor", help="summarize served predictions; flag drift")
    watch.add_argument("--log", type=Path, default=Path("results/logs/predictions.jsonl"))
    watch.add_argument("--registry", type=Path, default=Path("results/models"))
    watch.add_argument("--last", type=int, default=1000)
    watch.add_argument("--threshold", type=float, default=0.2)
    watch.add_argument("--min-predictions", type=int, default=50)

    store = subparsers.add_parser("store", help="semantic tiered file store")
    store.add_argument("--root", type=Path, default=Path("data/store"))
    store.add_argument("--hot-capacity-gb", type=float, help="persisted HOT cap")
    store.add_argument("--k", type=int, help="persisted: neighbors prefetched per access")
    for tier in ("hot", "warm", "cold"):
        store.add_argument(f"--{tier}-dir", help=f"persisted: directory for {tier.upper()}")
    actions = store.add_subparsers(dest="action", required=True)
    put = actions.add_parser("put", help="store a file (counts as an access)")
    put.add_argument("path", type=Path)
    put.add_argument("--id", help="file id; default: file name")
    put.add_argument("--groups", default="", help="comma-separated ACL groups")
    get = actions.add_parser("get", help="read a file (promotes it and its neighbors)")
    get.add_argument("file_id")
    get.add_argument("--out", type=Path, help="write bytes here; default: stdout")
    actions.add_parser("tick", help="demote idle files")
    actions.add_parser("status", help="files and bytes per tier")
    actions.add_parser("verify", help="check bytes sit in the indexed tier")

    args = parser.parse_args(argv)
    if args.command == "store":
        import sys
        from src.tiering.store import TieredStore
        tier_dirs = {name.upper(): getattr(args, f"{name}_dir")
                     for name in ("hot", "warm", "cold") if getattr(args, f"{name}_dir")}
        opened = TieredStore(
            args.root, k=args.k, tier_dirs=tier_dirs or None,
            hot_capacity_bytes=(int(args.hot_capacity_gb * 1024 ** 3)
                                if args.hot_capacity_gb is not None else None))
        if args.action == "put":
            groups = [g.strip() for g in args.groups.split(",") if g.strip()]
            tier = opened.put(args.id or args.path.name, args.path.read_bytes(), groups)
            print(f"{args.id or args.path.name}: {tier.value}")
        elif args.action == "get":
            data = opened.get(args.file_id)
            if args.out:
                args.out.write_bytes(data)
            else:
                sys.stdout.buffer.write(data)
        elif args.action == "tick":
            print(f"{opened.tick()} files moved")
        elif args.action == "verify":
            bad = opened.verify()
            print(json.dumps({"ok": not bad, "misplaced": bad}, indent=2))
        else:
            print(json.dumps(opened.status(), indent=2))
        return
    if args.command == "generate":
        result = generate(args.config, args.output_dir, args.num_files, args.duration_hours)
        print(f"Wrote {result['file_count']} files and {result['event_count']} access events.")
        print(f"Dataset: {result['files']}\nTraces: {result['traces']}\nSeed: {result['seed']}")
    elif args.command == "fetch-enron":
        from src.tiering.ingestion.fetch_enron import fetch
        owners = {name.strip() for name in args.owners.split(",") if name.strip()} or None
        fetch(args.source, Path(args.output), args.max_emails, owners=owners)
    elif args.command == "ablate":
        from src.tiering.experiments import product_ablation as ablation
        if args.enron:
            report = ablation.run_enron_ablation(
                jsonl=Path(args.enron), horizon_days=args.horizon_days,
                proxy=args.proxy, bootstrap_reps=args.bootstrap_reps)
        else:
            report = ablation.run_product_ablation(bootstrap_reps=args.bootstrap_reps)
        print(f"Model: {report['model_path']}")
    elif args.command == "predict":
        from src.tiering.prediction import registry
        from src.tiering.prediction.mvp import predict_document
        if args.model:
            model_id, model_path = f"file:{args.model}", args.model
        else:
            model_id, model_path = registry.active_model(args.registry)
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        result = predict_document(payload, model_path, args.run)
        result["model_version"] = model_id
        if not args.no_log:
            registry.log_prediction(result, model_id, args.log)
        print(json.dumps(result, indent=2))
    elif args.command == "models":
        from src.tiering.prediction import registry
        if args.action == "promote":
            if not args.bundle:
                parser.error("models promote needs a bundle path")
            print(json.dumps(registry.promote(args.bundle, args.registry, args.note), indent=2))
        elif args.action == "rollback":
            print(json.dumps(registry.rollback(args.registry), indent=2))
        else:
            print(json.dumps(registry.load_registry(args.registry), indent=2))
    elif args.command == "monitor":
        import joblib
        from src.tiering.prediction import registry
        share = None
        try:
            _, model_path = registry.active_model(args.registry)
            bundle = joblib.load(model_path)
            share = (bundle.get("served_tier_share_reference")
                     or bundle.get("training_tier_share"))
        except RuntimeError:
            pass
        print(json.dumps(registry.monitor(args.log, share, args.last, args.threshold,
                                          args.min_predictions),
                         indent=2))


if __name__ == "__main__":
    main()
