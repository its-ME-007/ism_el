"""Checks for the real-data validity fixes: thread proxy, text visibility,
age eligibility, vectorized retrieval parity, strict cutoffs, and bootstrap."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from src.tiering.experiments.product_ablation import (
    _make_samples, _matrix, bootstrap_comparison,
)
from src.tiering.ingestion.enron_adapter import load_enron_bundle, normalize_subject
from src.tiering.ingestion.schemas import AccessEvent
from src.tiering.semantic.retrieval_context import (
    RetrievalIndex, retrieval_context_features,
)
from src.tiering.workload.feature_extractor import WorkloadFeatureExtractor

UTC = timezone.utc
BASE = datetime(2001, 3, 1, tzinfo=UTC)


def _email(mid, day, subject, participants):
    return {"message_id": mid, "date": (BASE + timedelta(days=day)).isoformat(),
            "subject": subject, "body": f"body of {mid} about {subject}",
            "participants": participants, "owner": "a"}


def test_normalize_subject_strips_reply_and_forward_prefixes():
    assert normalize_subject("RE: Fw: re[2]:  Gas  Deal") == "gas deal"
    assert normalize_subject("Gas deal") == "gas deal"
    assert normalize_subject("") == ""


def test_thread_proxy_counts_replies_not_unrelated_mail_from_same_people():
    records = [
        _email("m1", 0, "Gas deal", ["a@x.com", "b@x.com"]),
        _email("m2", 1, "RE: Gas deal", ["a@x.com", "b@x.com"]),   # reply
        _email("m3", 2, "Lunch", ["a@x.com", "b@x.com"]),          # unrelated
        _email("m4", 3, "RE: Gas deal", ["z@x.com"]),              # no shared person
    ]
    thread = load_enron_bundle(records, horizon_days=30)
    counts = defaultdict(int)
    for event in thread.traces:
        counts[event.file_id] += 1
    assert counts["m1"] == 1  # only the reply
    participant = load_enron_bundle(records, horizon_days=30, proxy="participant")
    legacy = sum(event.file_id == "m1" for event in participant.traces)
    assert legacy == 2  # reply + unrelated lunch mail: the owner-activity artifact


def test_misdated_messages_are_dropped():
    records = [_email("ok", 0, "a", ["a@x.com"]),
               {**_email("old", 0, "a", ["a@x.com"]), "date": "1979-12-31T00:00:00+00:00"},
               {**_email("new", 0, "a", ["a@x.com"]), "date": "2044-01-01T00:00:00+00:00"}]
    bundle = load_enron_bundle(records, horizon_days=30)
    assert [file.file_id for file in bundle.files] == ["ok"]


def test_weekly_splits_do_not_overlap_label_windows():
    records = [_email(f"m{i}", i, f"s{i % 7}", ["a@x.com"]) for i in range(200)]
    bundle = load_enron_bundle(records, horizon_days=14)
    assert bundle.train_days[-1] < bundle.validation_days[0]
    assert bundle.validation_days[-1] < bundle.test_days[0]
    assert len(bundle.train_days) > 3 and bundle.max_sample_age_days == 14


def _doc(file_id, created, groups, modified=None):
    return SimpleNamespace(file_id=file_id, created_at=created,
                           modified_at=modified or created, size_bytes=10,
                           content=file_id, category="operations",
                           metadata={"access_groups": groups})


def test_max_age_drops_documents_whose_target_is_fixed_by_construction():
    files = [_doc("young", BASE, ["g"]), _doc("old", BASE - timedelta(days=40), ["g"])]
    samples = _make_samples(files, defaultdict(list), [1], {"young", "old"},
                            BASE, WorkloadFeatureExtractor(normalize=False),
                            max_age_days=14)
    assert [sample["file_id"] for sample in samples] == ["young"]


def test_text_visible_only_when_content_version_exists_at_cutoff():
    files = [_doc("seen", BASE, ["g"]),
             _doc("edited_later", BASE, ["g"], modified=BASE + timedelta(days=5))]
    samples = _make_samples(files, defaultdict(list), [1], {"seen", "edited_later"},
                            BASE, WorkloadFeatureExtractor(normalize=False))
    vectors = {"seen": np.ones(3, dtype=np.float32),
               "edited_later": np.ones(3, dtype=np.float32)}
    rows = {sample["file_id"]: row for sample, row in
            zip(samples, _matrix(samples, [], vectors))}
    assert rows["seen"].tolist() == [1.0, 1.0, 1.0]
    assert rows["edited_later"].tolist() == [0.0, 0.0, 0.0]


def test_retrieval_index_matches_reference_and_excludes_events_at_cutoff():
    rng = np.random.default_rng(3)
    documents, vectors, events = [], {}, defaultdict(list)
    for i in range(60):
        created = BASE + timedelta(hours=int(rng.integers(0, 24 * 20)))
        groups = [f"g{int(g)}" for g in rng.choice(5, size=int(rng.integers(0, 3)),
                                                    replace=False)]
        documents.append(_doc(f"d{i:02d}", created, groups))
        vector = rng.integers(0, 3, size=8).astype(np.float32)  # ties likely
        vectors[f"d{i:02d}"] = vector
        for _ in range(int(rng.integers(0, 6))):
            events[f"d{i:02d}"].append(AccessEvent(
                timestamp=created + timedelta(hours=int(rng.integers(1, 24 * 10))),
                file_id=f"d{i:02d}", operation="read"))
    cutoff = BASE + timedelta(days=12)
    # One neighbor event exactly at the cutoff must not count.
    events["d00"].append(AccessEvent(timestamp=cutoff, file_id="d00", operation="read"))
    index = RetrievalIndex(documents, vectors, events)
    for query in documents:
        for when in (cutoff, cutoff + timedelta(days=5)):
            expected = retrieval_context_features(query, documents, vectors, events, when)
            actual = index.features(query, when)
            assert actual.keys() == expected.keys()
            for key in expected:
                assert abs(actual[key] - expected[key]) < 1e-5, (query.file_id, key)


def test_bootstrap_identical_predictions_have_zero_difference():
    samples = [{"file_id": f"f{i % 10}", "future_count": i % 3,
                "future_access": int(i % 3 > 0), "tier": i % 3} for i in range(40)]
    access = np.linspace(0, 1, 40)
    predictions = {"count": np.asarray([s["future_count"] for s in samples], float),
                   "access": access, "tier": np.asarray([s["tier"] for s in samples]),
                   "tier_probabilities": None}
    result = bootstrap_comparison(samples, {"X": predictions, "Y": dict(predictions)},
                                  pairs=[("X", "Y")], reps=50)
    assert result["documents"] == 10
    assert result["runs"]["X"]["tier_macro_f1"]["point"] == 1.0
    for metric in result["paired_differences"]["X - Y"].values():
        assert metric["mean"] == 0.0
