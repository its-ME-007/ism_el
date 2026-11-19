from datetime import datetime, timedelta, timezone

import numpy as np

from src.tiering.ingestion.schemas import AccessEvent, FileDocument, Tier
from src.tiering.semantic.graph import GraphPrefetchPolicy, SemanticGraph
from src.tiering.simulation.simulator import SimPySimulator

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
TIERS = {Tier.HOT: {"latency_ms": 1.0, "cost_per_gb_month": 0.02, "throughput_mbps": 1000.0},
         Tier.WARM: {"latency_ms": 10.0, "cost_per_gb_month": 0.01, "throughput_mbps": 100.0},
         Tier.COLD: {"latency_ms": 100.0, "cost_per_gb_month": 0.001, "throughput_mbps": 10.0}}


def _file(file_id, groups, created_hours=0):
    when = T0 + timedelta(hours=created_hours)
    return FileDocument(file_id=file_id, filename=file_id, category="Finance",
                        content=file_id, size_bytes=1000, created_at=when,
                        modified_at=when, metadata={"access_groups": groups})


def test_graph_respects_acl_similarity_and_time():
    files = [_file("a", ["p1"]), _file("b", ["p1"]), _file("c", ["p1"], created_hours=48),
             _file("d", ["p2"]), _file("e", ["p1"])]
    vectors = {"a": np.array([1.0, 0.0]), "b": np.array([0.9, 0.1]),
               "c": np.array([1.0, 0.01]), "d": np.array([1.0, 0.0]),
               "e": np.array([0.0, 1.0])}
    graph = SemanticGraph(files, vectors, min_similarity=0.3)
    # d is identical but shares no group; e is in the group but dissimilar.
    assert [n for n, _ in graph.edges["a"]] == ["c", "b"]
    # c does not exist yet at T0 + 1h, so only b is returned.
    assert [n for n, _ in graph.neighbors("a", T0 + timedelta(hours=1), k=5)] == ["b"]
    assert [n for n, _ in graph.neighbors("a", T0 + timedelta(hours=49), k=1)] == ["c"]


def test_policy_prefetches_neighbors_then_demotes_idle_files():
    files = [_file("a", ["p1"]), _file("b", ["p1"])]
    graph = SemanticGraph(files, {"a": np.array([1.0]), "b": np.array([1.0])})
    policy = GraphPrefetchPolicy(graph, k=5, hot_idle_hours=1, warm_idle_hours=2)
    tiers = {"a": Tier.COLD, "b": Tier.COLD}
    assert policy.on_access("a", T0, tiers) == [("a", Tier.HOT), ("b", Tier.HOT)]
    tiers.update(a=Tier.HOT, b=Tier.HOT)
    assert sorted(policy.on_tick(T0 + timedelta(hours=1.5), tiers)) == [
        ("a", Tier.WARM), ("b", Tier.WARM)]
    tiers.update(a=Tier.WARM, b=Tier.WARM)
    assert sorted(policy.on_tick(T0 + timedelta(hours=3), tiers)) == [
        ("a", Tier.COLD), ("b", Tier.COLD)]
    assert GraphPrefetchPolicy(None, k=0).on_access("a", T0, tiers) == [("a", Tier.HOT)]


def test_simulator_serves_prefetched_first_touch_from_hot():
    files = [_file("a", ["p1"]), _file("b", ["p1"])]
    graph = SemanticGraph(files, {"a": np.array([1.0]), "b": np.array([1.0])})
    events = [AccessEvent(timestamp=T0 + timedelta(hours=h), file_id=f, operation="read")
              for h, f in ((1, "a"), (1.5, "b"))]

    def run(policy):
        result = SimPySimulator(config={"decision_interval_hours": 1.0,
                                        "evaluation_start_time": T0,
                                        "access_policy": policy},
                                tiers=TIERS).run(files, events)
        return result.metrics, policy

    recency_metrics, _ = run(GraphPrefetchPolicy(None, k=0))
    graph_metrics, graph_policy = run(GraphPrefetchPolicy(graph, k=5))
    assert recency_metrics.cache_hit_rate == 0.0  # both first touches hit COLD
    assert graph_metrics.cache_hit_rate == 0.5    # b was prefetched by a's access
    assert graph_policy.report()["prefetch_precision"] == 1.0
