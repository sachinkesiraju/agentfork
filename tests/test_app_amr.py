"""The vendored map-reduce bookkeeping: selection arithmetic and tsv format."""

import pytest

from agentfork.app import amr


def _results(tmp_path):
    return amr.Results(tmp_path / "results.tsv")


def test_log_roundtrip_and_header(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "baseline")
    r.log(0, "base", "-", 1.02, "ran", "baseline second")
    rows = r.rows()
    assert r.path.read_text().splitlines()[0] == amr.HEADER
    assert [row.commit for row in rows] == ["base", "base"]
    assert rows[0].score == 1.0


def test_crash_and_note_rows_carry_no_score(tmp_path):
    r = _results(tmp_path)
    r.log(1, "c1", "base", None, "crash", "eval exited 1")
    r.log(1, "stall-1", "-", None, "note", "LAW: wider models OOM")
    rows = r.rows()
    assert [row.status for row in rows] == ["crash", "note"]
    assert all(row.score is None for row in rows)


def test_log_rejects_ran_without_score(tmp_path):
    with pytest.raises(amr.AmrError):
        _results(tmp_path).log(1, "c1", "base", None, "ran", "no score")


def test_survives_only_by_beating_own_parent(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "baseline")
    r.log(1, "win", "base", 0.80, "ran", "real win")
    r.log(1, "noise", "base", 0.99, "ran", "inside the margin")
    red = r.reduce(1, margin=0.05, beam=2)
    assert [s.commit for s in red.kept] == ["win"]
    assert red.beat_parent == 1
    assert [s.commit for s in red.discarded] == ["noise"]
    assert red.frontier == ["win"]
    assert not red.stall


def test_beam_width_truncates_and_orders_by_score(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "baseline")
    for name, score in (("a", 0.5), ("b", 0.7), ("c", 0.6)):
        r.log(1, name, "base", score, "ran", name)
    red = r.reduce(1, margin=0.01, beam=2)
    assert [s.commit for s in red.kept] == ["a", "c"]
    assert [s.commit for s in red.dropped] == ["b"]


def test_maximize_flips_comparison(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 0.5, "ran", "baseline")
    r.log(1, "up", "base", 0.9, "ran", "better when maximizing")
    r.log(1, "down", "base", 0.1, "ran", "worse when maximizing")
    red = r.reduce(1, margin=0.01, beam=2, minimize=False)
    assert [s.commit for s in red.kept] == ["up"]


def test_stall_keeps_parent_frontier(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "baseline")
    r.log(1, "flat", "base", 1.001, "ran", "nothing")
    red = r.reduce(1, margin=0.05, beam=2)
    assert red.stall
    assert red.kept == []
    assert red.frontier == ["base"]
    assert "STALL" in red.render()


def test_cost_guard_rejects_regression(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "baseline", costs={"seconds": 100.0})
    r.log(1, "slow", "base", 0.5, "ran", "great but slow",
          costs={"seconds": 200.0})
    red = r.reduce(1, margin=0.01, beam=2, guards={"seconds": 0.10})
    assert red.kept == []
    assert red.cost_failures[0][0] == "slow"
    assert "seconds" in red.cost_failures[0][1]


def test_cost_guard_allows_within_tolerance(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "baseline", costs={"seconds": 100.0})
    r.log(1, "ok", "base", 0.5, "ran", "slightly slower",
          costs={"seconds": 105.0})
    red = r.reduce(1, margin=0.01, beam=2, guards={"seconds": 0.10})
    assert [s.commit for s in red.kept] == ["ok"]


def test_disjoint_regions_produce_fuse_candidate(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "baseline")
    r.log(1, "a", "base", 0.8, "ran", "data change", regions={"data"})
    r.log(1, "b", "base", 0.7, "ran", "model change", regions={"model"})
    red = r.reduce(1, margin=0.01, beam=2)
    assert red.fuse and red.fuse[0][:2] == ("b", "a")
    assert "FUSE_CANDIDATE" in red.render()


def test_overlapping_regions_do_not_fuse(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "baseline")
    r.log(1, "a", "base", 0.8, "ran", "lr", regions={"optimizer"})
    r.log(1, "b", "base", 0.7, "ran", "wd", regions={"optimizer"})
    assert r.reduce(1, margin=0.01, beam=2).fuse == []


def test_node_best_uses_best_repeat_of_a_node(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "baseline")
    r.log(0, "base", "-", 0.9, "ran", "baseline again")
    r.log(1, "c", "base", 0.88, "ran", "beats the worse baseline only")
    red = r.reduce(1, margin=0.0, beam=2)
    assert [s.commit for s in red.kept] == ["c"]
    assert red.kept[0].delta == pytest.approx(-0.02)


def test_reduce_without_candidates_errors(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "baseline")
    with pytest.raises(amr.AmrError):
        r.reduce(1, margin=0.01)


def test_unscored_parent_warns_and_skips(tmp_path):
    r = _results(tmp_path)
    r.log(1, "orphan", "ghost", 0.1, "ran", "parent never scored")
    red = r.reduce(1, margin=0.01)
    assert red.kept == []
    assert any("ghost" in w for w in red.warnings)


def test_tree_render_includes_notes(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "baseline")
    r.log(1, "child", "base", 0.5, "ran", "a change")
    r.log(1, "stall-1", "-", None, "note", "LAW: no more of that")
    text = r.tree().render()
    assert "base" in text and "child" in text
    assert "notes/laws:" in text and "LAW: no more of that" in text


def test_old_six_column_file_is_upgraded(tmp_path):
    path = tmp_path / "results.tsv"
    path.write_text(amr.OLD_HEADER + "\n0\tbase\t-\t1.0\tran\tbaseline\n")
    r = amr.Results(path)
    r.log(1, "c", "base", 0.5, "ran", "new row")
    assert path.read_text().splitlines()[0] == amr.HEADER
    rows = r.rows()
    assert len(rows) == 2 and rows[0].commit == "base"


def test_bad_header_is_rejected(tmp_path):
    path = tmp_path / "results.tsv"
    path.write_text("nonsense\n")
    with pytest.raises(amr.AmrError):
        amr.Results(path).rows()


def test_tabs_and_newlines_cannot_corrupt_a_row(tmp_path):
    r = _results(tmp_path)
    r.log(0, "base", "-", 1.0, "ran", "desc\twith\ttabs\nand newline")
    assert len(r.path.read_text().strip().splitlines()) == 2
    assert r.rows()[0].desc == "desc with tabs and newline"


def test_cli_log_reduce_tree(tmp_path, capsys):
    tsv = str(tmp_path / "results.tsv")
    assert amr.main(["--tsv", tsv, "log", "0", "base", "-", "1.0", "ran", "b"]) == 0
    assert amr.main(["--tsv", tsv, "log", "1", "c", "base", "0.5", "ran", "win"]) == 0
    capsys.readouterr()
    assert amr.main(["--tsv", tsv, "reduce", "--gen", "1", "--margin", "0.01"]) == 0
    out = capsys.readouterr().out
    assert "KEEP\tc" in out and "FRONTIER: c" in out
    assert amr.main(["--tsv", tsv, "tree"]) == 0
    assert "base" in capsys.readouterr().out


def test_cli_rejects_bad_score(tmp_path, capsys):
    tsv = str(tmp_path / "results.tsv")
    assert amr.main(["--tsv", tsv, "log", "0", "b", "-", "abc", "ran", "x"]) == 1
    assert "score must be" in capsys.readouterr().err


def test_parse_cost_and_guard_formats():
    assert amr.parse_cost_list(["a=1,b=2.5"]) == {"a": 1.0, "b": 2.5}
    assert amr.parse_guards(["seconds:+0.1"]) == {"seconds": 0.1}
    with pytest.raises(amr.AmrError):
        amr.parse_cost_list(["a"])
    with pytest.raises(amr.AmrError):
        amr.parse_guards(["seconds=0.1"])
