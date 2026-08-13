"""Live Polyscope latency table: snapshot + row formatting."""

import time

from geodude_teleoperation.viz.live_state import (
    LiveVizState,
    latency_table_rows,
)


def test_latency_table_rows_warn_when_loop_or_query_exceed_target():
    rows = latency_table_rows(
        {
            "target_dt_ms": 20.0,
            "loop_ms": 45.0,
            "query_ms": 80.0,
            "relock_age_ms": 150.0,
            "relock_reason": "hold",
            "sm_age_ms": 12.0,
            "publish_mono": 100.0,
        },
        now_mono=100.05,
    )
    by_name = {name: (val, warn) for name, val, warn in rows}
    assert by_name["target dt"][0] == "20.0"
    assert by_name["loop"][1] is True
    assert by_name["query (WoS)"][1] is True
    assert "150.0" in by_name["relock age"][0]
    assert "hold" in by_name["relock age"][0]
    assert by_name["spacemouse age"][0] == "12.0"
    assert by_name["viz age"][0] == "50.0"
    assert by_name["viz age"][1] is False


def test_latency_table_rows_ok_when_under_target():
    rows = latency_table_rows(
        {
            "target_dt_ms": 20.0,
            "loop_ms": 8.0,
            "query_ms": 3.0,
            "relock_age_ms": 40.0,
            "relock_reason": "dist",
            "sm_age_ms": None,
            "publish_mono": None,
        },
        now_mono=1.0,
    )
    by_name = {name: (val, warn) for name, val, warn in rows}
    assert by_name["loop"][1] is False
    assert by_name["query (WoS)"][1] is False
    assert by_name["spacemouse age"][0] == "n/a"
    assert by_name["viz age"][0] == "n/a"


def test_live_viz_publish_and_snapshot_latency():
    live = LiveVizState()
    assert live.latency_snapshot() == {}
    t0 = time.monotonic()
    live.publish_latency(
        loop_ms=12.5,
        query_ms=4.0,
        relock_age_ms=90.0,
        relock_reason="rate_limit",
        sm_age_ms=3.0,
        target_dt_ms=20.0,
        publish_mono=t0,
    )
    snap = live.latency_snapshot()
    assert snap["loop_ms"] == 12.5
    assert snap["query_ms"] == 4.0
    assert snap["relock_reason"] == "rate_limit"
    assert snap["sm_age_ms"] == 3.0
    assert snap["target_dt_ms"] == 20.0
    assert snap["publish_mono"] == t0


def test_draw_latency_imgui_second_window():
    from geodude_teleoperation.viz.field_viz import draw_latency_imgui

    calls = []

    class _Imgui:
        def Begin(self, title, *_a, **_k):
            calls.append(("Begin", title))
            return True

        def End(self):
            calls.append(("End",))

        def TextUnformatted(self, s):
            calls.append(("Text", s))

        def BeginTable(self, *_a, **_k):
            return False

    live = LiveVizState()
    live.show_latency_table = True
    draw_latency_imgui(_Imgui(), live)
    assert ("Begin", "Latency") in calls
    assert any(c[0] == "Text" and "waiting" in str(c[1]) for c in calls)
    assert ("End",) in calls

    live.publish_latency(
        loop_ms=30.0,
        query_ms=5.0,
        relock_age_ms=10.0,
        relock_reason="hold",
        sm_age_ms=2.0,
        target_dt_ms=20.0,
        publish_mono=time.monotonic(),
    )
    calls.clear()
    draw_latency_imgui(_Imgui(), live)
    texts = [c[1] for c in calls if c[0] == "Text"]
    assert any("loop" in str(t) for t in texts)
    assert any("!" in str(t) for t in texts)
