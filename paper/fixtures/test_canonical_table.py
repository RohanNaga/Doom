"""One canonical control table, built by streaming, shared by every shard and every corpus.

`encode_parquet.py` used to scan the `action` and `buttons` columns of EVERY episode in `--in-dir`
before it even looked at `--canonical`, so passing the shared table skipped nothing: each of six
shards materialised the whole 8,000-episode corpus as `<U19` arrays plus Python strings, roughly
10 GB per process, while the encode it was meant to serve had not started. The table also decides
which rows are verified decisions, so two corpora derived from two tables have decision masks that
cannot be compared and an IDM that scores different transitions.

Each test reproduces the specific way the code was wrong, not only the fixed behaviour.

    python -m pytest paper/fixtures/test_canonical_table.py -q
"""
import glob
import json
import os
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

pytest.importorskip("pyarrow")

import encode_parquet  # noqa: E402
import transitions  # noqa: E402
from pertic_fixtures import write_pertic_episode  # noqa: E402
from transitions import EPISODE_META_KEY, canonical_table  # noqa: E402

SCRIPTS = os.path.join(REPO, "scripts", "spiderman")

FORWARD = "100000000"
ATTACK = "000000100"
STUCK = "100100010"      # the anti-stuck override: never canonical for any action


# ---------------------------------------------------------------------------------------
# raw-recording fixtures: the columns the canonical table and the mask repair read
# ---------------------------------------------------------------------------------------

def write_raw(path, action, buttons, deaths=None, tic=None, stored=1):
    """One `ep_XXXXX.parquet` in the recorder's schema, without the frame column.

    Nothing under test reads `frame`: the canonical table reads `action` and `buttons`, and the mask
    repair reads those plus `deaths` and `tic`. Leaving the PNGs out is what lets a memory test use
    thousands of rows per episode.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(action)
    tic = np.arange(n, dtype=np.int64) if tic is None else np.asarray(tic)
    deaths = np.zeros(n, dtype=np.int64) if deaths is None else np.asarray(deaths)
    t = pa.table({
        "episode_id": pa.array([0] * n, pa.int32()), "map_id": pa.array([3] * n, pa.int8()),
        "tic": pa.array(list(map(int, tic)), pa.int32()),
        "action": pa.array(list(map(int, action)), pa.int16()),
        "buttons": pa.array(list(buttons), pa.string()),
        "deaths": pa.array(list(map(int, deaths)), pa.int16()),
    })
    prov = {"corpus_id": "test"}
    if stored > 1:
        prov["stored_tic_stride"] = stored
    t = t.replace_schema_metadata({EPISODE_META_KEY: json.dumps(prov, sort_keys=True).encode()})
    pq.write_table(t, path, compression=None)
    return t


def raw_corpus(out_dir, n_eps, rows=64, control=FORWARD, first_ep=0):
    """`n_eps` episodes of `rows` rows each, every row requesting action 0 with `control`."""
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(first_ep, first_ep + n_eps):
        write_raw(os.path.join(out_dir, f"ep_{ep:05d}.parquet"),
                  np.zeros(rows, dtype=np.int64), [control] * rows)
    return str(out_dir)


def enc_args(in_dir, out_dir, *extra):
    return encode_parquet.build_parser().parse_args(
        ["--in-dir", str(in_dir), "--out-dir", str(out_dir), "--every-tic", *extra])


# ---------------------------------------------------------------------------------------
# a supplied --canonical must cost ZERO parquet column reads
# ---------------------------------------------------------------------------------------

def test_a_supplied_canonical_table_reads_no_parquet_column(tmp_path, monkeypatch):
    """The load sat AFTER the scan, so `--canonical` skipped nothing at all: every shard still paid
    for the whole 8,000-episode corpus and only then threw its own table away."""
    in_dir = raw_corpus(tmp_path / "raw", 5)
    table = tmp_path / "canonical_controls.json"
    table.write_text(json.dumps({"0": FORWARD, "1": ATTACK}))
    import pyarrow.parquet as pq
    calls = []

    def boom(path, **kw):
        calls.append((path, kw.get("columns")))
        raise AssertionError(f"read {kw.get('columns')} of {path} although a canonical table was supplied")

    monkeypatch.setattr(pq, "read_table", boom)
    args = enc_args(in_dir, tmp_path / "out", "--canonical", str(table))
    got, where = encode_parquet.resolve_canonical(args, encode_parquet.select_paths(in_dir))
    assert calls == [], calls
    assert got == {0: FORWARD, 1: ATTACK}
    assert "supplied" in where


def test_the_whole_corpus_concatenation_is_gone_from_the_source():
    src = open(os.path.join(REPO, "encode_parquet.py")).read()
    assert "acts.append" not in src and "btns.append" not in src, \
        "the per-episode arrays are retained again"
    assert "np.concatenate(acts)" not in src, "the whole-corpus concatenation is back"


# ---------------------------------------------------------------------------------------
# with no table supplied, the scan streams and covers only the chosen episodes
# ---------------------------------------------------------------------------------------

def test_the_table_is_folded_in_one_episode_at_a_time(tmp_path, monkeypatch):
    """The row-count proxy for peak memory: the counters must be updated once per episode with that
    episode's rows, never once with the corpus's."""
    in_dir = raw_corpus(tmp_path / "raw", 6, rows=500)
    seen = []
    real = transitions.count_controls

    def spy(counters, action, buttons):
        seen.append(len(action))
        return real(counters, action, buttons)

    monkeypatch.setattr(transitions, "count_controls", spy)
    encode_parquet.build_canonical(sorted(glob.glob(os.path.join(in_dir, "ep_*.parquet"))))
    assert seen == [500] * 6, seen


def _peak(fn):
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_the_scan_peak_does_not_grow_with_the_number_of_episodes(tmp_path):
    """The regression the outage needs: peak Python allocation is one episode's rows, so a 12-episode
    corpus costs what a 2-episode one costs. The old form concatenated every episode first, which at
    40M rows is ~6 GB of `<U19` arrays plus ~2.7 GB of Python strings per process."""
    rows = 4000
    small = raw_corpus(tmp_path / "small", 2, rows=rows)
    big = raw_corpus(tmp_path / "big", 12, rows=rows)

    def run(d):
        return lambda: encode_parquet.build_canonical(sorted(glob.glob(os.path.join(d, "ep_*.parquet"))))

    run(small)()          # warm the pyarrow import and its one-off allocations
    peak_small = _peak(run(small))
    peak_big = _peak(run(big))
    assert peak_big < 1.6 * peak_small, (peak_small, peak_big)
    # and an absolute bound: one episode's strings, not six times as many
    assert peak_big < 6 * rows * 100, peak_big


def test_the_streamed_table_is_the_one_shot_table(tmp_path):
    in_dir = tmp_path / "raw"
    os.makedirs(in_dir)
    acts = [np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])]
    btns = [[FORWARD, STUCK, ATTACK, ATTACK], [FORWARD, ATTACK, ATTACK, STUCK]]
    for ep, (a, b) in enumerate(zip(acts, btns)):
        write_raw(os.path.join(in_dir, f"ep_{ep:05d}.parquet"), a, b)
    streamed = encode_parquet.build_canonical(sorted(glob.glob(os.path.join(in_dir, "ep_*.parquet"))))
    one_shot = canonical_table(np.concatenate(acts), np.array([x for b in btns for x in b]))
    assert streamed == one_shot == {0: FORWARD, 1: ATTACK}


def test_without_a_table_the_scan_covers_only_this_run_s_episodes(tmp_path):
    """The scan used to glob `--in-dir` itself, so the unseen corpus's 60 selected episodes still
    derived their table from all 3,000 recordings in that directory."""
    in_dir = tmp_path / "raw"
    raw_corpus(in_dir, 2, rows=8, control=FORWARD, first_ep=0)
    raw_corpus(in_dir, 2, rows=8, control=STUCK, first_ep=2)
    args = enc_args(in_dir, tmp_path / "out", "--episode-ids", "2:4")
    got, where = encode_parquet.resolve_canonical(args, encode_parquet.select_paths(in_dir, "2:4"))
    assert got == {0: STUCK}, "the table came from episodes this run does not encode"
    assert "this run's 2 episodes" in where


def test_canonical_ids_name_the_episodes_the_table_comes_from(tmp_path):
    in_dir = tmp_path / "raw"
    raw_corpus(in_dir, 2, rows=8, control=FORWARD, first_ep=0)
    raw_corpus(in_dir, 2, rows=8, control=STUCK, first_ep=2)
    args = enc_args(in_dir, tmp_path / "out", "--episode-ids", "2:4", "--canonical-ids", "0:2")
    got, where = encode_parquet.resolve_canonical(args, encode_parquet.select_paths(in_dir, "2:4"))
    assert got == {0: FORWARD}
    assert "--canonical-ids 0:2" in where


def test_canonical_ids_that_select_nothing_are_refused(tmp_path):
    in_dir = raw_corpus(tmp_path / "raw", 2, rows=8)
    args = enc_args(in_dir, tmp_path / "out", "--canonical-ids", "900:901")
    with pytest.raises(SystemExit, match="selects no episode"):
        encode_parquet.resolve_canonical(args, encode_parquet.select_paths(in_dir))


# ---------------------------------------------------------------------------------------
# --canonical-only: build the one shared table, load no autoencoder
# ---------------------------------------------------------------------------------------

def test_canonical_only_writes_the_table_and_never_builds_a_vae(tmp_path, monkeypatch):
    in_dir = raw_corpus(tmp_path / "raw", 3, rows=16)
    out = tmp_path / "out"
    monkeypatch.setattr(encode_parquet, "build_vae",
                        lambda *a, **k: pytest.fail("--canonical-only loaded an autoencoder"))
    encode_parquet.main(enc_args(in_dir, out, "--canonical-only", "--canonical-ids", "0:3"))
    path = out / encode_parquet.CANONICAL_FILE
    assert path.is_file()
    assert encode_parquet.load_canonical(str(path)) == {0: FORWARD}


def test_the_table_written_by_canonical_only_is_the_one_a_shard_loads(tmp_path, monkeypatch):
    in_dir = raw_corpus(tmp_path / "raw", 3, rows=16)
    out = tmp_path / "out"
    monkeypatch.setattr(encode_parquet, "build_vae", lambda *a, **k: pytest.fail("no VAE here"))
    encode_parquet.main(enc_args(in_dir, out, "--canonical-only"))
    shard = enc_args(in_dir, tmp_path / "shard", "--canonical", str(out / encode_parquet.CANONICAL_FILE))
    got, _ = encode_parquet.resolve_canonical(shard, [])
    assert got == {0: FORWARD}


# ---------------------------------------------------------------------------------------
# --rebuild-sidecar-masks: repair is_decision/chain_id without re-encoding a latent
# ---------------------------------------------------------------------------------------

def _paired_corpus(tmp_path, rows=32, control=FORWARD, sidecar_decisions=None):
    """A raw directory and a latent directory describing the same rows, as the server holds them."""
    raw, lat = tmp_path / "raw", tmp_path / "lat"
    os.makedirs(raw, exist_ok=True)
    actions = np.zeros(rows, dtype=np.int64)
    write_raw(os.path.join(raw, "ep_00000.parquet"), actions, [control] * rows)
    write_pertic_episode(str(lat), 0, actions, buttons=np.array([control] * rows),
                         map_ids=np.full(rows, 3, dtype=np.int64),
                         decisions=np.zeros(rows, dtype=bool) if sidecar_decisions is None
                         else sidecar_decisions)
    return str(raw), str(lat)


def test_the_masks_are_recomputed_from_the_raw_metadata(tmp_path):
    """The latents already on the server were written with per-shard tables, so their decision masks
    disagree; this recovers the shared table's masks with no GPU and no re-encode."""
    raw, lat = _paired_corpus(tmp_path)
    latent_before = open(os.path.join(lat, "ep_00000_latents.npy"), "rb").read()
    with np.load(os.path.join(lat, "ep_00000_meta.npz")) as z:
        assert not z["is_decision"].any(), "the fixture must start from the wrong-table mask"
    r = encode_parquet.rebuild_sidecar_masks(
        sorted(glob.glob(os.path.join(raw, "ep_*.parquet"))), lat, {0: FORWARD}, stride=4)
    assert r == {"episodes": 1, "changed": 1, "unchanged": 0, "missing": [], "refused": []}
    with np.load(os.path.join(lat, "ep_00000_meta.npz")) as z:
        dec, chain = z["is_decision"], z["chain_id"]
        assert z["action"].shape == (32,) and z["buttons"].shape == (32,), "a column was dropped"
    want_rows, want_chain = transitions.decision_rows(
        np.zeros(32, dtype=np.int64), np.array([FORWARD] * 32), np.zeros(32, dtype=np.int64), 4, {0: FORWARD}, 1)
    assert np.flatnonzero(dec).tolist() == want_rows.tolist()
    assert dec.sum() > 0, "the repair produced no decisions at all"
    assert chain[want_rows].tolist() == want_chain.tolist()
    assert (chain[~dec] == -1).all()
    assert open(os.path.join(lat, "ep_00000_latents.npy"), "rb").read() == latent_before, \
        "the latents were rewritten"


def test_a_repair_that_changes_nothing_says_so(tmp_path):
    raw, lat = _paired_corpus(tmp_path)
    paths = sorted(glob.glob(os.path.join(raw, "ep_*.parquet")))
    encode_parquet.rebuild_sidecar_masks(paths, lat, {0: FORWARD}, stride=4)
    again = encode_parquet.rebuild_sidecar_masks(paths, lat, {0: FORWARD}, stride=4)
    assert again["unchanged"] == 1 and again["changed"] == 0


def test_a_dry_run_reports_without_writing(tmp_path):
    raw, lat = _paired_corpus(tmp_path)
    before = open(os.path.join(lat, "ep_00000_meta.npz"), "rb").read()
    r = encode_parquet.rebuild_sidecar_masks(
        sorted(glob.glob(os.path.join(raw, "ep_*.parquet"))), lat, {0: FORWARD}, stride=4, dry_run=True)
    assert r["changed"] == 1
    assert open(os.path.join(lat, "ep_00000_meta.npz"), "rb").read() == before


def test_a_missing_sidecar_is_reported_not_invented(tmp_path):
    raw, lat = _paired_corpus(tmp_path)
    write_raw(os.path.join(raw, "ep_00009.parquet"), np.zeros(8, dtype=np.int64), [FORWARD] * 8)
    r = encode_parquet.rebuild_sidecar_masks(
        sorted(glob.glob(os.path.join(raw, "ep_*.parquet"))), lat, {0: FORWARD}, stride=4)
    assert r["missing"] == ["ep_00009"] and r["episodes"] == 1


def test_a_sidecar_whose_rows_are_not_the_raw_rows_is_refused(tmp_path):
    raw, lat = _paired_corpus(tmp_path)
    p = os.path.join(lat, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k][:20] for k in z.files}
    np.savez(p, **cols)
    r = encode_parquet.rebuild_sidecar_masks(
        sorted(glob.glob(os.path.join(raw, "ep_*.parquet"))), lat, {0: FORWARD}, stride=4)
    assert r["refused"] and "20 sidecar rows vs 32 raw rows" in r["refused"][0]


def test_a_sidecar_whose_tics_differ_is_refused(tmp_path):
    raw, lat = _paired_corpus(tmp_path)
    p = os.path.join(lat, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["tic"] = cols["tic"] + 5
    np.savez(p, **cols)
    r = encode_parquet.rebuild_sidecar_masks(
        sorted(glob.glob(os.path.join(raw, "ep_*.parquet"))), lat, {0: FORWARD}, stride=4)
    assert r["refused"] and "tics differ" in r["refused"][0]


def test_a_decision_only_recording_cannot_be_repaired_this_way(tmp_path):
    raw, lat = _paired_corpus(tmp_path)
    write_raw(os.path.join(raw, "ep_00000.parquet"), np.zeros(32, dtype=np.int64),
              [FORWARD] * 32, stored=4)
    r = encode_parquet.rebuild_sidecar_masks(
        sorted(glob.glob(os.path.join(raw, "ep_*.parquet"))), lat, {0: FORWARD}, stride=4)
    assert r["refused"] and "stored_tic_stride 4" in r["refused"][0]


@pytest.mark.parametrize("column,value", [("action", 7), ("deaths", 3)])
def test_a_sidecar_column_the_repair_depends_on_must_match_the_recording(tmp_path, column, value):
    """The repair recomputes the masks from RAW action/buttons/deaths while copying those same
    columns through from the OLD sidecar, so a sidecar that disagrees with the recording would come
    out self-inconsistent: masks from one recording beside controls from another."""
    raw, lat = _paired_corpus(tmp_path)
    p = os.path.join(lat, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols[column] = np.full(32, value, dtype=np.int64)
    np.savez(p, **cols)
    r = encode_parquet.rebuild_sidecar_masks(
        sorted(glob.glob(os.path.join(raw, "ep_*.parquet"))), lat, {0: FORWARD}, stride=4)
    assert r["changed"] == 0 and r["refused"], r
    assert column in r["refused"][0] and "self" not in r["refused"][0]
    with np.load(p) as z:
        assert not z["is_decision"].any(), "the corrupt sidecar was repaired anyway"


def test_a_sidecar_whose_buttons_differ_from_the_recording_is_refused(tmp_path):
    raw, lat = _paired_corpus(tmp_path)
    p = os.path.join(lat, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["buttons"] = np.array([STUCK] * 32)
    np.savez(p, **cols)
    r = encode_parquet.rebuild_sidecar_masks(
        sorted(glob.glob(os.path.join(raw, "ep_*.parquet"))), lat, {0: FORWARD}, stride=4)
    assert r["refused"] and "buttons" in r["refused"][0] and "ep_00000" in r["refused"][0]


def test_a_sidecar_missing_a_depended_on_column_is_refused(tmp_path):
    raw, lat = _paired_corpus(tmp_path)
    p = os.path.join(lat, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files if k != "deaths"}
    np.savez(p, **cols)
    r = encode_parquet.rebuild_sidecar_masks(
        sorted(glob.glob(os.path.join(raw, "ep_*.parquet"))), lat, {0: FORWARD}, stride=4)
    assert r["refused"] and "deaths" in r["refused"][0]


def test_an_agreeing_sidecar_is_still_repaired(tmp_path):
    raw, lat = _paired_corpus(tmp_path)
    r = encode_parquet.rebuild_sidecar_masks(
        sorted(glob.glob(os.path.join(raw, "ep_*.parquet"))), lat, {0: FORWARD}, stride=4)
    assert r["refused"] == [] and r["changed"] == 1


def test_the_repair_mode_is_reachable_from_the_command_line(tmp_path):
    raw, lat = _paired_corpus(tmp_path)
    table = tmp_path / "canon.json"
    table.write_text(json.dumps({"0": FORWARD}))
    encode_parquet.main(enc_args(raw, lat, "--rebuild-sidecar-masks", "--canonical", str(table)))
    with np.load(os.path.join(lat, "ep_00000_meta.npz")) as z:
        assert z["is_decision"].any()


# ---------------------------------------------------------------------------------------
# the table is written atomically, once, and never by a mode that promises to write nothing
# ---------------------------------------------------------------------------------------

def test_the_table_is_renamed_into_place(tmp_path, monkeypatch):
    """A reader that opens the file while a writer is inside `json.dump` sees a truncated table.
    Two shards auto-building into one output directory do exactly that."""
    out = tmp_path / "out"
    out.mkdir()
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: seen.append((a, b)) or real(a, b))
    path = encode_parquet.write_canonical({0: FORWARD}, str(out))
    assert seen and seen[0][1] == path, seen
    assert ".tmp" in seen[0][0]
    assert encode_parquet.load_canonical(path) == {0: FORWARD}
    assert [p for p in os.listdir(out) if ".tmp" in p] == []


def test_a_failed_write_leaves_no_partial_table(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(json, "dump", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected")))
    with pytest.raises(RuntimeError, match="injected"):
        encode_parquet.write_canonical({0: FORWARD}, str(out))
    assert os.listdir(out) == [], os.listdir(out)


def test_a_dry_run_writes_no_table(tmp_path):
    """`--rebuild-sidecar-masks --dry-run` promises to change no file, and wrote one anyway."""
    raw, lat = _paired_corpus(tmp_path)
    table = tmp_path / "canon.json"
    table.write_text(json.dumps({"0": FORWARD}))
    before = sorted(os.listdir(lat))
    encode_parquet.main(enc_args(raw, lat, "--rebuild-sidecar-masks", "--dry-run",
                                 "--canonical", str(table)))
    assert sorted(os.listdir(lat)) == before, "the dry run wrote a file"
    assert encode_parquet.write_canonical({0: FORWARD}, str(lat), dry_run=True) is None


def test_the_shared_table_is_not_rewritten_from_itself(tmp_path):
    """A shard given the table that already lives in its own output directory must leave it alone:
    rewriting it is at best a no-op and at worst the truncation above, on the one file every other
    shard is reading."""
    out = tmp_path / "out"
    out.mkdir()
    path = encode_parquet.write_canonical({0: FORWARD}, str(out))
    stamp = os.stat(path).st_ino, os.path.getsize(path)
    assert encode_parquet.write_canonical({0: ATTACK}, str(out), supplied=path) is None
    assert encode_parquet.load_canonical(path) == {0: FORWARD}
    assert (os.stat(path).st_ino, os.path.getsize(path)) == stamp


def test_a_table_supplied_from_elsewhere_is_still_recorded(tmp_path):
    src = tmp_path / "shared" / encode_parquet.CANONICAL_FILE
    src.parent.mkdir()
    src.write_text(json.dumps({"0": FORWARD}))
    out = tmp_path / "out"
    out.mkdir()
    path = encode_parquet.write_canonical({0: FORWARD}, str(out), supplied=str(src))
    assert path == os.path.join(str(out), encode_parquet.CANONICAL_FILE)
    assert encode_parquet.load_canonical(path) == {0: FORWARD}


# ---------------------------------------------------------------------------------------
# the launcher builds ONE table and hands the same file to every shard and corpus
# ---------------------------------------------------------------------------------------

def _dry(**env):
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": "/d", **env}
    r = subprocess.run(["bash", os.path.join(SCRIPTS, "encode_nexttic.sh")],
                       capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.mark.parametrize("vae", ["sd15", "sd35"])
def test_the_launcher_builds_the_table_once_from_the_training_ids(vae):
    out = _dry(CORPUS="all", VAE=vae)
    line = [ln for ln in out.splitlines() if ln.startswith("DRY canonical")]
    assert len(line) == 1, out
    assert "--canonical-only" in line[0] and "--canonical-ids 0:2000" in line[0]
    assert "raw_arnold_dense/arenas " in line[0] + " ", "the table must come from the training source"


@pytest.mark.parametrize("vae", ["sd15", "sd35"])
def test_every_corpus_is_given_the_same_canonical_file(vae):
    out = _dry(CORPUS="all", VAE=vae)
    corpora = [ln for ln in out.splitlines() if ln.startswith("DRY ") and " canonical " not in ln]
    assert len(corpora) == 4, out
    paths = set()
    for ln in corpora:
        tok = ln.split()
        assert "--canonical" in tok, ln
        paths.add(tok[tok.index("--canonical") + 1])
    assert len(paths) == 1, paths


def test_the_two_latent_spaces_share_one_table():
    """The table is a property of the RECORDING, not of the autoencoder, so the 4-channel and the
    SD 3.5 corpora must not each derive one: their decision masks would not be comparable."""
    def table_of(vae):
        for ln in _dry(CORPUS="train", VAE=vae).splitlines():
            tok = ln.split()
            if "--canonical" in tok:
                return tok[tok.index("--canonical") + 1]
        raise AssertionError("no --canonical in the train command")
    assert table_of("sd15") == table_of("sd35")
    assert "_sd35" not in table_of("sd35")
