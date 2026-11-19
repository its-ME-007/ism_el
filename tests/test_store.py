from datetime import datetime, timedelta, timezone

import pytest

from src.tiering.cli import main
from src.tiering.ingestion.schemas import Tier
from src.tiering.store import TieredStore

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
LEDGER = b"quarterly ledger invoice budget forecast for project falcon"


def test_store_put_get_prefetch_tick_and_reopen(tmp_path):
    store = TieredStore(tmp_path, k=5, hot_idle_hours=1, warm_idle_hours=2)
    store.put("a.txt", LEDGER, groups=["finance"], now=T0)
    store.put("b.txt", LEDGER + b" annex", groups=["finance"], now=T0)
    store.put("c.txt", b"holiday party menu cake", groups=["hr"], now=T0)
    store.tick(T0 + timedelta(hours=10))  # everything idle -> demoted
    assert {store.tier_of(f) for f in ("a.txt", "b.txt", "c.txt")} == {Tier.WARM}
    store.tick(T0 + timedelta(hours=20))
    assert store.tier_of("a.txt") == Tier.COLD

    assert store.get("a.txt", now=T0 + timedelta(hours=21)) == LEDGER
    assert store.tier_of("a.txt") == Tier.HOT
    assert store.tier_of("b.txt") == Tier.HOT    # similar + same ACL group: prefetched
    assert store.tier_of("c.txt") == Tier.COLD   # other group: never a neighbor
    assert (tmp_path / "hot" / "b.txt").exists() and store.verify() == []

    reopened = TieredStore(tmp_path)  # index + last-touch survive a restart
    assert reopened.tier_of("b.txt") == Tier.HOT and reopened.get("b.txt") == LEDGER + b" annex"
    assert len((tmp_path / "access_log.jsonl").read_text().splitlines()) == 5


def test_store_capacity_rejects_bad_ids_and_repairs_index(tmp_path):
    store = TieredStore(tmp_path, k=0, hot_capacity_bytes=100)
    store.put("one", b"x" * 80, now=T0)
    store.put("two", b"y" * 80, now=T0 + timedelta(hours=1))
    assert store.tier_of("two") == Tier.HOT and store.tier_of("one") == Tier.WARM
    assert store.status()["tiers"]["HOT"]["bytes"] <= 100
    for bad in ("../escape", "a/b", "", ".hidden"):
        with pytest.raises(ValueError):
            store.put(bad, b"z")
    # Simulate a crash after a move but before the index save.
    (tmp_path / "warm" / "one").rename(tmp_path / "cold" / "one")
    assert store.get("one") == b"x" * 80 and store.verify() == []


def test_cli_store_round_trip(tmp_path, capsys):
    source = tmp_path / "report.txt"
    source.write_bytes(LEDGER)
    root = str(tmp_path / "store")
    main(["store", "--root", root, "--hot-capacity-gb", "1", "put", str(source),
          "--groups", "finance"])
    assert capsys.readouterr().out.strip() == "report.txt: HOT"
    out = tmp_path / "copy.txt"
    main(["store", "--root", root, "get", "report.txt", "--out", str(out)])
    assert out.read_bytes() == LEDGER
    main(["store", "--root", root, "verify"])
    assert '"ok": true' in capsys.readouterr().out
