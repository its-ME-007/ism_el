"""Production-shaped synthetic file store for the A/B/C/D ablation.

Models a shared enterprise file store rather than a mailbox:

- files belong to projects; each project has a lifecycle (ramp-up, active
  phase, decay) and occasional reactivation bursts (audit, release,
  quarter-end) that revive its older files;
- per-file popularity is heavy-tailed, demand decays with file age, and
  access follows a working-hours/weekday cycle;
- sizes are MB-to-GB by file type, so tier cost actually matters;
- content mixes project topic words, department wording and filler, and ACL
  groups are project + department (+ occasional all-staff).

``coupling`` is the share of each file's demand that follows its project's
shared schedule; the remainder follows an independent per-file schedule drawn
from the same distribution. ``coupling=0`` is the negative control: content and
ACLs still reflect the project, but carry no demand signal. Sweep it to find
how strong a content-demand link must be before C or D beat A; a gain here is
conditional on the planted link and is not evidence about real stores.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.tiering.ingestion.enron_adapter import EnronBundle, _split_days
from src.tiering.ingestion.schemas import AccessEvent, Category, FileDocument

DEPARTMENTS = {
    "IT": [Category.ENGINEERING, Category.SECURITY, Category.SYSTEM],
    "Finance": [Category.FINANCE, Category.LEGAL],
    "HR": [Category.HR],
    "Operations": [Category.OPERATIONS, Category.MARKETING, Category.RESEARCH],
}
DEPT_WORDS = {
    "IT": "server deployment incident patch network config release backup".split(),
    "Finance": "invoice ledger budget forecast audit revenue expense quarter".split(),
    "HR": "hiring payroll benefits onboarding review policy leave training".split(),
    "Operations": "vendor shipment inventory schedule campaign launch supplier report".split(),
}
FILLER = ("the and for with this that from will have are was our team please "
          "update draft final notes meeting action item status next week plan "
          "review attached version summary details owner date issue open").split()
FILE_TYPES = {  # type: (probability, median bytes, log-sigma)
    "document": (0.60, 2e6, 1.0),
    "database": (0.10, 5e8, 0.8),
    "archive": (0.15, 2e8, 1.0),
    "log": (0.15, 5e7, 1.0),
}
SYLLABLES = "ka lo mi nu ve ra to si pe zu dor fen gal hix jor kel mur nax".split()


def _schedule(rng, t_days, start, length, days, burst_rate=1.5, burst_height=4.0):
    """Lifecycle curve and burst curve on the hourly grid ``t_days``."""
    end = start + length
    ramp = np.clip((t_days - start) / 3.0, 0.0, 1.0)
    life = np.where(t_days < end, ramp, np.exp(-(t_days - end) / 7.0))
    life[t_days < start] = 0.0
    bursts = np.zeros_like(t_days)
    for _ in range(rng.poisson(burst_rate * days / 120.0)):
        at = rng.uniform(max(start, 0.0), days)
        bursts[(t_days >= at) & (t_days < at + rng.uniform(1.0, 3.0))] += burst_height
    return life, bursts


def generate_production_bundle(n_projects: int = 50, mean_files_per_project: int = 36,
                               days: int = 120, coupling: float = 0.8,
                               events_per_file_day: float = 1.0,
                               seed: int = 42) -> EnronBundle:
    if not 0.0 <= coupling <= 1.0:
        raise ValueError("coupling must be in [0, 1]")
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 6, tzinfo=timezone.utc)  # a Monday
    hours = days * 24
    t_days = np.arange(hours) / 24.0
    hour_of_day, weekday = np.arange(hours) % 24, (np.arange(hours) // 24) % 7
    cycle = (np.where((hour_of_day >= 8) & (hour_of_day < 18), 1.0, 0.15)
             * np.where(weekday < 5, 1.0, 0.2))
    types = list(FILE_TYPES)
    type_p = [FILE_TYPES[name][0] for name in types]

    files, rates = [], []
    for p in range(n_projects):
        dept = str(rng.choice(list(DEPARTMENTS)))
        topic = ["".join(rng.choice(SYLLABLES, 3)) for _ in range(3)]
        p_start, p_len = rng.uniform(-20, days - 15), rng.uniform(15, 60)
        p_life, p_bursts = _schedule(rng, t_days, p_start, p_len, days)
        confidentiality = str(rng.choice(["public", "internal", "confidential", "restricted"],
                                     p=[0.1, 0.5, 0.3, 0.1]))
        for i in range(max(1, rng.poisson(mean_files_per_project))):
            created = rng.uniform(max(p_start, -20.0), min(p_start + p_len, days - 1.0))
            # 10% of files are cross-project (a shared spec, a copied template).
            words = (topic if rng.random() > 0.1 else
                     ["".join(rng.choice(SYLLABLES, 3)) for _ in range(3)])
            content = " ".join(np.concatenate([
                rng.choice(words, 20), rng.choice(DEPT_WORDS[dept], 15),
                rng.choice(FILLER, 25)]))
            file_type = str(rng.choice(types, p=type_p))
            _, median, sigma = FILE_TYPES[file_type]
            groups = [f"proj-{p}", f"dept-{dept}"] + (["all-staff"] if rng.random() < 0.1 else [])
            when = start + timedelta(days=float(created))
            files.append(FileDocument(
                file_id=f"p{p:03d}-f{i:03d}", filename=f"{topic[0]}_{file_type}_{i}",
                category=DEPARTMENTS[dept][rng.integers(len(DEPARTMENTS[dept]))], content=content,
                size_bytes=int(rng.lognormal(np.log(median), sigma)),
                created_at=when, modified_at=when,
                metadata={"access_groups": groups, "file_type": file_type,
                          "department": dept, "confidentiality": confidentiality,
                          "indexed": bool(rng.random() < 0.5),
                          "compressed": file_type == "archive", "project": p},
            ))
            age = t_days - created
            decay = np.where(age >= 0, np.exp(-np.maximum(age, 0) / 10.0) + 0.05, 0.0)
            # Independent schedule from the same distribution, starting at creation.
            f_life, f_bursts = _schedule(rng, t_days, created, rng.uniform(15, 60), days)
            shared = p_life * decay + p_bursts * (age >= 0)
            own = f_life * decay + f_bursts * (age >= 0)
            popularity = rng.lognormal(0.0, 1.0)
            rates.append(popularity * (coupling * shared + (1 - coupling) * own) * cycle)

    rates = np.asarray(rates)
    rates *= events_per_file_day * len(files) * days / rates.sum()
    counts = rng.poisson(rates)
    traces = []
    for f_index, h_index in zip(*np.nonzero(counts)):
        for _ in range(counts[f_index, h_index]):
            traces.append(AccessEvent(
                timestamp=start + timedelta(hours=float(h_index + rng.random())),
                file_id=files[f_index].file_id,
                operation="write" if rng.random() < 0.1 else "read"))
    traces.sort(key=lambda event: event.timestamp)

    train_days, validation_days, test_days = _split_days(files, start, step_days=3)
    stats = {"generator": "production_synth", "coupling": coupling, "seed": seed,
             "projects": n_projects, "documents": len(files),
             "access_events_total": len(traces), "span_days": days,
             "distinct_access_groups": len({g for f in files
                                            for g in f.metadata["access_groups"]})}
    tag = f"prodsynth_c{round(coupling * 100):03d}"
    return EnronBundle(
        files, traces, start, train_days, validation_days, test_days, None, stats,
        suffix=tag,
        label=f"Production-shaped synthetic file store (coupling={coupling})",
        note=("Synthetic production-shaped store. Demand follows project lifecycles and "
              f"bursts with coupling={coupling}; content and ACLs reflect the project. "
              "Any C/D gain is conditional on that planted link, not evidence about "
              "real stores. coupling=0 is the negative control."),
    )
