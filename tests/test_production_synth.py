from collections import defaultdict

import numpy as np
import pytest

from src.tiering.ingestion.production_synth import generate_production_bundle


def _within_project_corr(bundle, days):
    daily = defaultdict(lambda: np.zeros(days))
    for event in bundle.traces:
        daily[event.file_id][(event.timestamp - bundle.start).days] += 1
    by_project = defaultdict(list)
    for file in bundle.files:
        by_project[file.metadata["project"]].append(daily[file.file_id])
    means = []
    for rows in by_project.values():
        matrix = np.asarray(rows)
        matrix = matrix[matrix.std(axis=1) > 0]
        if len(matrix) > 2:
            corr = np.corrcoef(matrix)
            means.append(corr[np.triu_indices(len(matrix), 1)].mean())
    return float(np.mean(means))


def test_deterministic_and_coupling_drives_project_correlation():
    small = dict(n_projects=12, days=60, seed=3)
    a = generate_production_bundle(coupling=0.8, **small)
    b = generate_production_bundle(coupling=0.8, **small)
    assert [e.timestamp for e in a.traces] == [e.timestamp for e in b.traces]
    assert a.train_days and a.validation_days and a.test_days
    assert a.validation_days[0] > a.train_days[-1] and a.test_days[0] > a.validation_days[-1]
    assert all(e.timestamp >= next(f.created_at for f in a.files if f.file_id == e.file_id)
               for e in a.traces[:500])
    control = generate_production_bundle(coupling=0.0, **small)
    assert _within_project_corr(a, 60) > _within_project_corr(control, 60) + 0.15
    assert a.suffix == "prodsynth_c080" and control.suffix == "prodsynth_c000"


def test_rejects_bad_coupling():
    with pytest.raises(ValueError):
        generate_production_bundle(coupling=1.5)
