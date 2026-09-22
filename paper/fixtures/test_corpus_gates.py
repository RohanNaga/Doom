"""The two gate tools the launch protocol runs before anything is scored.

`check_action_alignment.audit_sidecar` re-read `m["buttons"][i]` and `m["action"][i]` from the NPZ on
every sampled row. NPZ access materialises the whole column each time, so auditing one ~5,035-row
episode re-created about 1.9 GB of `<U19` arrays to compare 0.38 MB of data; over 2,000 training
episodes that is ~3.85 TB of pointless allocation. It also compared only `buttons`, while `tic`,
`deaths` and `map_id` decide the join, the respawn boundary and the arena a score belongs to.

`make_dense_eval_splits.build` published whatever was in the directory, so a shard still running, or
a directory holding the wrong id range, produced an apparently valid evaluation split.

    python -m pytest paper/fixtures/test_corpus_gates.py -q
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

pytest.importorskip("pyarrow")

import check_action_alignment as caa  # noqa: E402
import make_dense_eval_splits  # noqa: E402
from pertic_fixtures import held_actions, write_pertic_episode  # noqa: E402
from test_canonical_table import FORWARD, STUCK, write_raw  # noqa: E402


# ---------------------------------------------------------------------------------------
# the sidecar audit
# ---------------------------------------------------------------------------------------

def _pair(tmp_path, rows=40, eps=(0,), buttons=None, deaths=None, map_id=3, actions=None):
    """A raw directory and a per-tic latent directory that agree, as a healthy corpus does."""
    raw, lat = tmp_path / "raw", tmp_path / "lat"
    os.makedirs(raw, exist_ok=True)
    act = np.zeros(rows, dtype=np.int64) if actions is None else np.asarray(actions)
    btn = np.array([FORWARD] * rows) if buttons is None else np.asarray(buttons)
    dth = np.zeros(rows, dtype=np.int64) if deaths is None else np.asarray(deaths)
    for ep in eps:
        write_raw(os.path.join(raw, f"ep_{ep:05d}.parquet"), act, list(btn), deaths=dth)
        write_pertic_episode(str(lat), ep, act, buttons=btn, deaths=dth,
                             map_ids=np.full(rows, map_id, dtype=np.int64))
    return str(raw), str(lat)


class _CountingNpz:
    """Wraps an NpzFile and counts every column materialisation."""

    def __init__(self, inner):
        self.inner, self.gets = inner, []

    @property
    def files(self):
        return self.inner.files

    def __getitem__(self, k):
        self.gets.append(k)
        return self.inner[k]

    def close(self):
        self.inner.close()


def _count_column_reads(monkeypatch, latents_dir, parquet_dir, rows):
    loaded = []
    real = np.load

    def spy(path, *a, **k):
        obj = real(path, *a, **k)
        if str(path).endswith("_meta.npz"):
            obj = _CountingNpz(obj)
            loaded.append(obj)
        return obj

    monkeypatch.setattr(np, "load", spy)
    r = caa.audit_sidecar(latents_dir, parquet_dir, episodes=1, rows=rows, seed=0)
    monkeypatch.undo()
    return r, [g for o in loaded for g in o.gets]


def test_every_sidecar_column_is_materialised_once_per_episode(tmp_path, monkeypatch):
    """The defect, measured: the number of column reads used to scale with the ROW count."""
    raw, lat = _pair(tmp_path, rows=40)
    few, gets_few = _count_column_reads(monkeypatch, lat, raw, rows=4)
    many, gets_many = _count_column_reads(monkeypatch, lat, raw, rows=40)
    assert few["mismatches"] == 0 and many["rows_checked"] == 40
    assert gets_few == gets_many, (gets_few, gets_many)
    assert len(gets_many) == 5, gets_many          # buttons, tic, deaths, map_id, action
    assert sorted(gets_many) == ["action", "buttons", "deaths", "map_id", "tic"]


def test_a_healthy_corpus_audits_clean(tmp_path):
    raw, lat = _pair(tmp_path, rows=40, eps=(0, 1))
    r = caa.audit_sidecar(lat, raw, episodes=2, rows=40, seed=0)
    assert r["ok"] and r["mismatches"] == 0 and r["rows_checked"] == 80
    assert r["columns"] == ["buttons", "tic", "deaths", "map_id"]


def test_an_empty_audit_is_not_a_pass(tmp_path):
    """`ok` used to be `mismatches == 0`, which is true of an audit that checked nothing."""
    raw, lat = _pair(tmp_path, rows=40)
    r = caa.audit_sidecar(lat, raw, episodes=1, rows=0, seed=0)
    assert r["rows_checked"] == 0 and not r["ok"]


@pytest.mark.parametrize("column,other", [("deaths", 4), ("map_id", 9)])
def test_a_sidecar_column_that_disagrees_with_the_recording_is_caught(tmp_path, column, other):
    """Only `buttons` was compared, so a sidecar whose `deaths` or `map_id` came from elsewhere
    passed the audit and then silently decided respawn boundaries or arena attribution."""
    raw, lat = _pair(tmp_path, rows=40)
    p = os.path.join(lat, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols[column] = np.full(40, other, dtype=np.int64)
    np.savez(p, **cols)
    r = caa.audit_sidecar(lat, raw, episodes=1, rows=40, seed=0)
    assert not r["ok"] and r["mismatches"] == 40
    assert any(column in s for s in r["problems"]), r["problems"]


def test_a_button_mismatch_is_still_caught(tmp_path):
    raw, lat = _pair(tmp_path, rows=40)
    p = os.path.join(lat, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["buttons"] = np.array([STUCK] * 40)
    np.savez(p, **cols)
    r = caa.audit_sidecar(lat, raw, episodes=1, rows=40, seed=0)
    assert not r["ok"] and any("buttons" in s for s in r["problems"])


def test_a_tic_in_the_sidecar_and_not_the_recording_is_caught(tmp_path):
    raw, lat = _pair(tmp_path, rows=40)
    p = os.path.join(lat, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["tic"] = cols["tic"] + 1000
    np.savez(p, **cols)
    r = caa.audit_sidecar(lat, raw, episodes=1, rows=40, seed=0)
    assert not r["ok"] and "not in" in r["problems"][0]


def test_a_duplicate_raw_tic_is_reported_because_the_join_is_ambiguous(tmp_path):
    raw, lat = _pair(tmp_path, rows=40)
    tics = np.arange(40, dtype=np.int64)
    tics[7] = 6
    write_raw(os.path.join(raw, "ep_00000.parquet"), np.zeros(40, dtype=np.int64),
              [FORWARD] * 40, tic=tics)
    r = caa.audit_sidecar(lat, raw, episodes=1, rows=40, seed=0)
    assert not r["ok"] and any("twice" in s for s in r["problems"])


def test_the_override_rows_are_counted_and_the_table_can_be_supplied(tmp_path):
    """The audit prints how many sampled rows are anti-stuck overrides, because those are exactly
    the rows on which conditioning on the action id would be wrong."""
    rows = 40
    btn = np.array([FORWARD] * rows)
    btn[:6] = STUCK
    raw, lat = _pair(tmp_path, rows=rows, buttons=btn)
    r = caa.audit_sidecar(lat, raw, episodes=1, rows=rows, seed=0, canonical={0: FORWARD})
    assert r["ok"] and r["canonical_table"]
    assert r["anti_stuck_override_rows"] == 6
    assert abs(r["override_fraction"] - 6 / rows) < 1e-9
    assert caa.audit_sidecar(lat, raw, episodes=1, rows=rows, seed=0)["canonical_table"] is False


def test_the_audit_runs_as_a_command_on_its_own(tmp_path):
    raw, lat = _pair(tmp_path, rows=40)
    table = tmp_path / "canon.json"
    table.write_text(json.dumps({"0": FORWARD}))
    argv = ["--latents-dir", lat, "--audit-parquet-dir", raw, "--audit-only",
            "--episodes", "1", "--audit-rows", "40", "--canonical", str(table)]
    assert caa.main(caa.build_parser().parse_args(argv)) == caa.EXIT_ALIGNED
    p = os.path.join(lat, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["map_id"] = np.full(40, 8, dtype=np.int64)
    np.savez(p, **cols)
    assert caa.main(caa.build_parser().parse_args(argv)) == caa.EXIT_MISALIGNED


def test_audit_only_needs_both_directories(tmp_path):
    raw, lat = _pair(tmp_path, rows=40)
    with pytest.raises(SystemExit, match="needs --latents-dir"):
        caa.main(caa.build_parser().parse_args(["--latents-dir", lat, "--audit-only"]))


# ---------------------------------------------------------------------------------------
# the evaluation split publishes only a complete, valid corpus
# ---------------------------------------------------------------------------------------

def _eval_corpus(d, eps, rows=20):
    for ep in eps:
        write_pertic_episode(str(d), ep, held_actions([1] * (rows // 4)))
    return str(d)


def test_a_complete_corpus_publishes(tmp_path):
    d = _eval_corpus(tmp_path / "val", [6000, 6001, 6002])
    path, split = make_dense_eval_splits.build(d, expect_ids=[6000, 6001, 6002])
    assert split[make_dense_eval_splits.SUBSET] == [6000, 6001, 6002]
    assert split["meta"]["expected_ids"] == "6000:6003"
    with open(path) as f:
        assert json.load(f)["meta"]["num_episodes"] == 3


def test_a_still_running_shard_cannot_publish_a_partial_split(tmp_path):
    """The defect: `build` named whatever was there, so a split file written halfway through an
    encode became the definition of the held-out set."""
    d = _eval_corpus(tmp_path / "val", [6000, 6001])
    with pytest.raises(SystemExit, match="not validly encoded"):
        make_dense_eval_splits.build(d, expect_ids=list(range(6000, 6004)))
    assert not os.path.exists(make_dense_eval_splits.split_path(d)), "a split file was left behind"


def test_the_wrong_id_range_cannot_publish(tmp_path):
    d = _eval_corpus(tmp_path / "test", [7000, 7001])
    with pytest.raises(SystemExit, match="present but not expected"):
        make_dense_eval_splits.build(d, expect_ids=[7000])


def test_an_orphan_latent_is_reported_not_ignored(tmp_path):
    """`encode_episode` renames the latents into place before writing the sidecar, so an interrupted
    encode leaves a latent with no sidecar; `list_latent_episodes` skips it in silence."""
    d = _eval_corpus(tmp_path / "val", [6000, 6001])
    os.remove(os.path.join(d, "ep_06001_meta.npz"))
    report = make_dense_eval_splits.validate(d, [6000, 6001])
    assert report["orphans"] == [6001] and not report["ok"]
    assert any("no sidecar" in s for s in report["problems"])
    with pytest.raises(SystemExit, match="interrupted encode"):
        make_dense_eval_splits.build(d, expect_ids=[6000, 6001])


def test_a_sidecar_shorter_than_the_latents_is_refused(tmp_path):
    d = _eval_corpus(tmp_path / "val", [6000])
    p = os.path.join(d, "ep_06000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k][:8] for k in z.files}
    np.savez(p, **cols)
    with pytest.raises(SystemExit, match="rows of"):
        make_dense_eval_splits.build(d, expect_ids=[6000])


def test_an_un_normalised_buttons_column_is_refused_and_names_the_repair(tmp_path):
    """A `<U2503` column is a 40 MB sidecar of Arnold's raw request strings; the fix needs no encode."""
    d = _eval_corpus(tmp_path / "val", [6000])
    p = os.path.join(d, "ep_06000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["buttons"] = np.array(["100000000" + "0" * 2493 + "1"] * len(cols["tic"]))
    np.savez(p, **cols)
    report = make_dense_eval_splits.validate(d, [6000])
    assert not report["ok"] and report["invalid"] == [6000]
    assert any("normalize-sidecars" in s for s in report["problems"])
    with pytest.raises(SystemExit, match="not validly encoded"):
        make_dense_eval_splits.build(d, expect_ids=[6000])


def test_a_buttons_column_at_the_executed_width_passes(tmp_path):
    d = _eval_corpus(tmp_path / "val", [6000])
    p = os.path.join(d, "ep_06000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["buttons"] = np.array(["1".ljust(19, "0")] * len(cols["tic"]), dtype="<U19")
    np.savez(p, **cols)
    assert make_dense_eval_splits.validate(d, [6000])["ok"]


def test_a_missing_sidecar_column_is_refused(tmp_path):
    d = _eval_corpus(tmp_path / "val", [6000])
    p = os.path.join(d, "ep_06000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files if k != "deaths"}
    np.savez(p, **cols)
    with pytest.raises(SystemExit, match="missing columns"):
        make_dense_eval_splits.build(d, expect_ids=[6000])


def test_a_non_finite_latent_is_refused(tmp_path):
    d = _eval_corpus(tmp_path / "val", [6000])
    p = os.path.join(d, "ep_06000_latents.npy")
    lat = np.load(p)
    lat[:] = np.nan
    np.save(p, lat)
    with pytest.raises(SystemExit, match="non-finite"):
        make_dense_eval_splits.build(d, expect_ids=[6000])


def test_the_channel_count_can_be_asserted(tmp_path):
    d = _eval_corpus(tmp_path / "val", [6000])
    with pytest.raises(SystemExit, match="latent shape"):
        make_dense_eval_splits.build(d, expect_ids=[6000], latent_channels=16)
    path, _ = make_dense_eval_splits.build(d, expect_ids=[6000], latent_channels=4)
    assert os.path.isfile(path)


def test_the_temporary_file_is_unique_per_process(tmp_path):
    """Two shards finalising one evaluation root shared `<path>.tmp` and could rename each other's
    half-written file into place."""
    src = open(os.path.join(REPO, "make_dense_eval_splits.py")).read()
    assert 'f"{path}.tmp.{os.getpid()}"' in src, src[src.find("tmp ="):][:80]


def test_check_only_reports_and_writes_nothing(tmp_path, capsys):
    d = _eval_corpus(tmp_path / "val", [6000, 6001])
    argv = ["--latents-dir", d, "--expect-ids", "6000:6003", "--check-only"]
    assert make_dense_eval_splits.main(make_dense_eval_splits.build_parser().parse_args(argv)) == 1
    assert json.loads(capsys.readouterr().out)["missing"] == [6002]
    assert not os.path.exists(make_dense_eval_splits.split_path(d))
    ok = ["--latents-dir", d, "--expect-ids", "6000:6002", "--check-only"]
    assert make_dense_eval_splits.main(make_dense_eval_splits.build_parser().parse_args(ok)) == 0


def test_the_launcher_passes_the_expected_ids(tmp_path):
    import subprocess
    e = {**os.environ, "DRY": "0", "DOOM_ROOT": str(tmp_path)}
    src = open(os.path.join(REPO, "scripts", "spiderman", "encode_nexttic.sh")).read()
    line = [ln for ln in src.splitlines() if "make_dense_eval_splits.py" in ln and "$PY" in ln]
    assert line and '--expect-ids "$4"' in line[0], line
    assert subprocess.run(["bash", "-n", os.path.join(REPO, "scripts", "spiderman",
                                                      "encode_nexttic.sh")], env=e).returncode == 0
