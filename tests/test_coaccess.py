from datetime import datetime, timedelta, timezone

from src.tiering.coaccess import CoAccessGraph

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)


def _graph(**kw):
    return CoAccessGraph({"a": ["g"], "b": ["g"], "c": ["g"], "x": ["other"]}, **kw)


def test_learns_directed_lagged_links_within_groups_only():
    g = _graph()
    for day in range(5):  # a, then b 10 min later, then x; c reads far away
        t = T0 + day * 24 * H
        g.observe("a", t)
        g.observe("b", t + H / 6)
        g.observe("x", t + H / 3)
        g.observe("c", t + 5 * H)
    now = T0 + 5 * 24 * H
    assert [n for n, _ in g.neighbors("a", now, 5)] == ["b"]   # c is outside the lag
    assert g.neighbors("b", now, 5) == []                      # direction: a -> b only
    assert g.neighbors("x", now, 5) == []                      # x shares no group
    assert abs(g.confidence("a", "b") - 5 / 6) < 1e-9


def test_support_confidence_threshold_and_fallback():
    g = _graph(min_support=2)
    g.observe("a", T0)
    g.observe("b", T0 + H / 6)
    assert g.neighbors("a", T0 + 10 * H, 5) == []              # one follow < min_support
    for day in range(1, 4):
        g.observe("a", T0 + day * 24 * H)
        g.observe("b", T0 + day * 24 * H + H / 6)
    late = T0 + 5 * 24 * H
    assert [n for n, _ in g.neighbors("a", late, 5)] == ["b"]
    g.min_confidence = 0.9                                     # confidence 4/5 = 0.8
    assert g.neighbors("a", late, 5) == []

    class Pool:
        def neighbors(self, file_id, now, k):
            return [("a", 0.1), ("b", 0.1), ("c", 0.1)][:k]
    fb = _graph(fallback=Pool())
    assert [n for n, _ in fb.neighbors("a", late, 2)] == ["b", "c"]  # unread: all fallback, never self
