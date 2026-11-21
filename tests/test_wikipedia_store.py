from datetime import datetime, timezone

import pytest

from src.tiering.ingestion import wikipedia_store as ws


def test_plain_strips_nested_templates_refs_and_links():
    text = ("{{Infobox|a={{nested|x}}}}'''Mars''' is a [[planet|world]]"
            "<ref name=a>cite</ref> near [[Earth]].<ref name=b/> == Orbit ==")
    assert ws._plain(text) == "Mars is a world near Earth. Orbit"


def test_load_refuses_unfetched_hours(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "CACHE", tmp_path)
    (tmp_path / "articles.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="24 hours not fetched"):
        ws.load_wikipedia_bundle(datetime(2026, 11, 8, tzinfo=timezone.utc), 1)
