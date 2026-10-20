from datetime import datetime, timedelta

import numpy as np
import pytest

from src.tiering.ingestion.schemas import AccessEvent, Category, FileDocument, OperationType, Tier
from src.tiering.models.feature_fusion import FeatureFusion
from src.tiering.migration.controller import MigrationController
from src.tiering.semantic.knowledge_base import KnowledgeBase
from src.tiering.semantic.tfidf_extractor import TFIDFExtractor
from src.tiering.simulation.simulator import SimPySimulator
from src.tiering.semantic.rag import RAGContextExtractor
from src.tiering.metrics import MetricsCollector
from src.tiering.validation.leakage import DataLeakageError, DataLeakageValidator
from src.tiering.experiments.framework import ExperimentFramework
from src.tiering.experiments.reporting import ExperimentReporter
from src.tiering.experiments.runners import ConventionalBaselineExperiment
from src.tiering.baselines.lru import LRUBaseline


def test_tfidf_handles_empty_and_nonempty_documents():
    extractor = TFIDFExtractor({"max_features": 16})
    matrix = extractor.fit_transform(["finance quarterly report", "", "legal case record"])
    assert matrix.shape[0] == 3
    assert matrix.dtype == np.float32
    assert np.all(np.isfinite(matrix))


def test_numpy_knowledge_base_search_and_persistence(tmp_path):
    kb = KnowledgeBase({"dimension": 3})
    kb.add(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), ["a", "b"])
    scores, ids = kb.search(np.array([0.9, 0.1, 0], dtype=np.float32), 1)
    assert ids == [["a"]]
    assert scores.shape == (1, 1)
    kb.save(tmp_path / "kb")
    restored = KnowledgeBase.load(tmp_path / "kb")
    assert restored.search(np.array([0.9, 0.1, 0]), 2)[1] == [["a", "b"]]


def test_feature_fusion_ablation_and_missing_features():
    fusion = FeatureFusion({"imputation_strategy": "zeros"})
    fused = fusion.fit_transform(workload=np.array([[1, 2], [3, 4]]), semantic=np.array([[5], [6]]))
    assert fused.shape == (2, 3)
    fusion.set_ablation(["semantic"])
    assert fusion.fuse(workload=np.array([[1, 2]]), semantic=np.array([[5]] )).shape == (1, 2)


def test_migration_hysteresis_and_cooldown():
    controller = MigrationController({"confidence_threshold": 0.7, "hysteresis_count": 2,
                                      "cooldown_hours": 6})
    now = datetime(2024, 1, 1)
    first = controller.evaluate("f", Tier.COLD, Tier.HOT, 0.9, now)
    second = controller.evaluate("f", Tier.COLD, Tier.HOT, 0.9, now + timedelta(hours=1))
    assert first.recommended_tier == Tier.COLD
    assert second.recommended_tier == Tier.HOT
    assert controller.last_evaluation_migrated
    held = controller.evaluate("f", Tier.HOT, Tier.COLD, 0.99, now + timedelta(hours=2))
    assert held.recommended_tier == Tier.HOT


def test_simulator_collects_access_and_tier_metrics():
    file = FileDocument(file_id="f", filename="f.txt", category=Category.FINANCE,
                        content="report", size_bytes=1024, created_at=datetime(2024, 1, 1),
                        modified_at=datetime(2024, 1, 1))
    trace = [AccessEvent(timestamp=datetime(2024, 1, 1), file_id="f",
                         operation=OperationType.READ, size_bytes=1024)]
    tiers = {Tier.COLD: {"latency_ms": 20, "throughput_mbps": 100,
                         "cost_per_gb_month": 0.01}}
    result = SimPySimulator({"decision_interval_hours": 1}, tiers).run([file], trace)
    assert result.metrics.total_accesses == 1
    assert result.metrics.avg_access_latency_ms > 20


def test_rag_deterministic_context_uses_similar_file_category():
    class StubEmbedder:
        def encode(self, files):
            return np.array([[1.0, 0.0] for _ in files], dtype=np.float32)

    first = FileDocument(file_id="a", filename="a.txt", category=Category.FINANCE,
                         content="budget", size_bytes=10, created_at=datetime(2024, 1, 1),
                         modified_at=datetime(2024, 1, 1))
    second = FileDocument(file_id="b", filename="b.txt", category=Category.FINANCE,
                          content="ledger", size_bytes=10, created_at=datetime(2024, 1, 1),
                          modified_at=datetime(2024, 1, 1))
    kb = KnowledgeBase({"dimension": 2})
    rag = RAGContextExtractor(kb, {"top_k": 2}, StubEmbedder())
    rag.add_documents([first, second])
    context = rag.extract(first)
    assert context.file_id == "a"
    assert context.related_files == ["b"]
    assert context.category_confidence == 1.0


def test_leakage_validator_rejects_future_features_and_overlapping_split():
    now = datetime(2024, 1, 1)
    with pytest.raises(DataLeakageError):
        DataLeakageValidator.validate_feature_timestamps([now + timedelta(seconds=1)], now)
    with pytest.raises(DataLeakageError):
        DataLeakageValidator.validate_temporal_split([now], [now])


def test_metrics_collector_computes_accuracy_latency_and_writes_json(tmp_path):
    metrics = MetricsCollector()
    assert metrics.classification([0, 1], [0, 0])["accuracy"] == 0.5
    assert metrics.latency([5.0, 15.0])["p95_ms"] == 14.5
    output = metrics.save({"run": "smoke"}, tmp_path / "metrics.json")
    assert output.exists()


def test_experiment_framework_runs_migration_stress_and_writes_report(tmp_path):
    framework = ExperimentFramework(tmp_path)
    now = datetime(2024, 1, 1)
    predictions = [
        {"file_id": "f", "current_tier": "COLD", "predicted_tier": "HOT",
         "probability": 0.99, "timestamp": now + timedelta(hours=index)}
        for index in range(3)
    ]
    result = framework.run("migration_policy_stress_test", predictions=predictions)
    assert result["migration_count"] == 1
    output = framework.save_results([result], "summary.json")
    report = framework.write_markdown_report([result], tmp_path / "summary.md")
    assert output.exists() and report.exists()
    html = ExperimentReporter.write_html([result], tmp_path / "summary.html")
    assert html.exists()


def test_conventional_baseline_runner_processes_traces():
    now = datetime(2024, 1, 1)
    file = FileDocument(file_id="f", filename="f.txt", category=Category.FINANCE,
                        content="report", size_bytes=1024, created_at=now, modified_at=now)
    trace = [AccessEvent(timestamp=now, file_id="f", operation=OperationType.READ,
                         size_bytes=1024)]
    baseline = LRUBaseline({Tier.HOT: 1.0, Tier.WARM: 5.0, Tier.COLD: 100.0})
    result = ConventionalBaselineExperiment().run([file], trace, {"LRU": baseline})
    assert result["baselines"]["LRU"]["tier_occupancy"]["HOT"] > 0
