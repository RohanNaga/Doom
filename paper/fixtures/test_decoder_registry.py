"""A decoder records the episodes it was tuned on, and a score names the decoder that made it.

The 2026-09-22 review (H4): the tuned decoders were trained on the 17-map corpus, whose training maps
include arenas 6 to 8, and the planned dense MSE tune streamed row groups from every file in
`raw_arnold_dense/arenas`, validation and test episodes included, while recording
`split_subset: "train"`. These tests pin that

  * the streamed tune draws only from the ids it is given, refuses a dense segment without ids or
    with ids outside that segment's train range, and writes the exact ids it used to
    `provenance.json` beside the decoder;
  * the decoders tuned before that are registered as `provenance: unknown, corpus: all maps`;
  * `decoder_provenance.py` refuses such a decoder, or one trained on a val, test or unseen-segment
    episode or an unseen map, for an unseen-map claim, and `after_nexttic.sh` will not score the
    unseen row with it unless the weaker claim is asked for by name;
  * the evaluators record the decoder's name, identity and provenance next to every number.

    python -m pytest paper/fixtures/test_decoder_registry.py -q
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

pytest.importorskip("pyarrow")

import decoder_provenance as dp  # noqa: E402
from finetune_decoder import sample_row_groups, stream_episode_ids  # noqa: E402
from test_decoder_stream import corpus, tune  # noqa: E402

DECODER_MSE = os.path.join(REPO, "scripts", "spiderman", "decoder_mse.sh")


# ---------------------------------------------------------------------------------------
# the streamed tune draws from the ids it is given, and records them
# ---------------------------------------------------------------------------------------

def test_without_ids_the_row_group_sample_is_unchanged(tmp_path):
    d = corpus(str(tmp_path / "dense"), range(4))
    assert sample_row_groups(d, 600, seed=3) == sample_row_groups(d, 600, seed=3, episode_ids=None)


def test_the_row_groups_come_only_from_the_given_ids(tmp_path):
    d = corpus(str(tmp_path / "dense"), range(6))
    groups, _, _ = sample_row_groups(d, 10**9, seed=0, episode_ids=[1, 4])
    assert {os.path.basename(p) for p, _ in groups} == {"ep_00001.parquet", "ep_00004.parquet"}


def test_a_dense_segment_needs_training_ids(tmp_path):
    arenas = str(tmp_path / "arenas")
    os.makedirs(arenas)
    with pytest.raises(SystemExit, match="--stream-ids"):
        stream_episode_ids(arenas, "")
    assert stream_episode_ids(arenas, "0:6000") == list(range(6000))
    with pytest.raises(SystemExit, match="overlap"):
        stream_episode_ids(arenas, "5990:6010")          # reaches into val 6000:7000
    with pytest.raises(SystemExit, match="overlap"):
        stream_episode_ids(arenas, "7000:7001")          # test


def test_the_unseen_segment_is_never_a_tuning_corpus(tmp_path):
    a678 = str(tmp_path / "arenas_678")
    os.makedirs(a678)
    with pytest.raises(SystemExit, match="unseen"):
        stream_episode_ids(a678, "0:10")


def test_a_non_dense_directory_keeps_the_old_behaviour(tmp_path):
    assert stream_episode_ids(str(tmp_path / "dense"), "") is None
    assert stream_episode_ids(str(tmp_path / "dense"), "1:3") == [1, 2]


def test_a_streamed_tune_writes_the_exact_ids_it_used(tmp_path, monkeypatch):
    dense = corpus(str(tmp_path / "dense"), range(5))
    m = tune(tmp_path, monkeypatch, stream_dir=dense, stream_frames=300, max_steps=3, stream_ids="1:4")
    prov = json.load(open(os.path.join(str(tmp_path / "run"), "provenance.json")))
    assert prov["provenance"] == "recorded"
    streamed = [t for t in prov["train_episodes"] if t["dir"] == dense]
    assert streamed and set(streamed[0]["ids"]) <= {1, 2, 3} and streamed[0]["ids"]
    assert streamed[0]["ids"] == sorted(streamed[0]["ids"])
    assert m["provenance"]["train_episodes"] == prov["train_episodes"]
    # the decoder directory carries its own copy, so a copied decoder keeps its record
    inner = json.load(open(os.path.join(str(tmp_path / "run"), "vae", "provenance.json")))
    assert inner["train_episodes"] == prov["train_episodes"]


def test_a_cached_tune_records_the_split_train_ids(tmp_path, monkeypatch):
    tune(tmp_path, monkeypatch)
    prov = json.load(open(os.path.join(str(tmp_path / "run"), "provenance.json")))
    cached = prov["train_episodes"][0]
    assert cached["ids"] == [0] and cached["maps"] == [3]       # test_decision_only writes map 3


def test_the_dense_decoder_tune_draws_from_training_ids_only():
    text = open(DECODER_MSE).read()
    assert "TRAIN_IDS=${TRAIN_IDS:-0:6000}" in text
    assert '--stream-ids "$TRAIN_IDS"' in text


# ---------------------------------------------------------------------------------------
# the registry and the unseen-map check
# ---------------------------------------------------------------------------------------

def test_the_old_tuned_decoders_are_registered_as_unknown_all_maps():
    reg = dp.load_registry()
    for name in ("vae_decoder_arnold_lpips", "vae_decoder_sd35_lpips"):
        assert reg[name]["provenance"] == "unknown" and reg[name]["corpus"] == "all maps"


def test_a_registered_decoder_is_named_and_refused_for_unseen_claims(tmp_path):
    d = tmp_path / "vae_decoder_arnold_lpips" / "vae"
    d.mkdir(parents=True)
    (d / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    rec = dp.describe(str(d))
    assert rec["name"] == "vae_decoder_arnold_lpips"
    assert rec["provenance"] == "unknown" and rec["corpus"] == "all maps"
    assert rec["identity"].startswith("sha256:")
    assert rec["unseen_map_claim"] == "refused"


def test_the_stock_decoders_may_back_an_unseen_claim():
    for name in ("", "stock", "stabilityai/sd-vae-ft-mse", "stabilityai/stable-diffusion-3.5-medium"):
        rec = dp.describe(name)
        assert rec["provenance"] == "public" and rec["unseen_map_claim"] == "ok", name


@pytest.mark.parametrize("episodes,ok", [
    ([{"segment": "arenas", "ids": list(range(10)), "maps": [2, 3, 4, 5]}], True),
    ([{"segment": "arenas", "ids": [5, 6000], "maps": [2, 3]}], False),             # a val episode
    ([{"segment": "arenas", "ids": [7050], "maps": [4]}], False),                   # a test episode
    ([{"segment": "arenas_678", "ids": [100], "maps": [6]}], False),                # the unseen segment
    ([{"segment": None, "dir": "/raw_arnold", "ids": [3], "maps": [2, 7]}], False),  # an unseen map
    ([], False),                                                                    # nothing recorded
])
def test_the_unseen_claim_check(episodes, ok):
    prov = {"provenance": "recorded", "train_episodes": episodes}
    assert (dp.unseen_claim_problems(prov) == []) is ok


def test_the_check_cli_exit_codes(tmp_path):
    good = tmp_path / "good" / "vae"
    good.mkdir(parents=True)
    (good / "diffusion_pytorch_model.safetensors").write_bytes(b"w")
    (good.parent / "provenance.json").write_text(json.dumps(
        {"provenance": "recorded", "train_episodes": [{"segment": "arenas", "ids": [0, 1], "maps": [2, 3]}]}))
    bad = tmp_path / "vae_decoder_sd35_lpips" / "vae"
    bad.mkdir(parents=True)
    (bad / "diffusion_pytorch_model.safetensors").write_bytes(b"w")
    run = [sys.executable, os.path.join(REPO, "decoder_provenance.py"), "check"]
    assert subprocess.run(run + [str(good)], capture_output=True).returncode == 0
    r = subprocess.run(run + [str(bad)], capture_output=True, text=True)
    assert r.returncode == 1 and "unknown" in r.stdout


# ---------------------------------------------------------------------------------------
# every score names its decoder
# ---------------------------------------------------------------------------------------

def test_eval_tf_records_the_decoder_beside_its_numbers():
    import eval_tf
    a = eval_tf.build_parser().parse_args(["--ckpt", "x", "--backbone", "unet", "--latents-dir", "l",
                                           "--split", "s", "--out-dir", "o"])
    rec = eval_tf.decoder_record(a)
    assert rec["name"] == "stabilityai/sd-vae-ft-mse" and rec["provenance"] == "public"
    assert 'summary["decoder"] = decoder_record(args)' in open(os.path.join(REPO, "eval_tf.py")).read()


def test_the_rollout_score_and_the_gate_score_record_the_decoder_too():
    assert 'out["decoder"] = decoder_record(args)' in open(os.path.join(REPO, "rollout_eval.py")).read()
    assert '"decoder_provenance": {n: describe(' in open(os.path.join(REPO, "vae_gate_score.py")).read()


def _sealed(tmp_path, **env):
    """Validation selection, then the unseen row, with the tuned (unknown-provenance) decoder."""
    from test_after_nexttic_stages import AFTER, STUB_PY
    from test_eval_stages import root_with
    import rollout_eval
    root, r = root_with(tmp_path)
    d = root / "vae_decoder_arnold_lpips" / "vae"
    d.mkdir(parents=True)
    (d / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    py = tmp_path / "stubpy"
    py.write_text(STUB_PY)
    py.chmod(0o755)
    e = {**os.environ, "DOOM_ROOT": str(root), "PY": str(py), "PY_SD35": str(py),
         "STUB_LOG": str(tmp_path / "stub.log"), "STUB_PICK": str(r / "0300000.pt"),
         "STUB_SCORE_FILE": rollout_eval.SCORE_FILE, "STUB_REAL_PY": sys.executable, "STUB_REPO": REPO,
         "NUM_WINDOWS": "8", "SELECT_WINDOWS": "4"}
    sel = subprocess.run(["bash", AFTER, "0", "unet", "--select"], capture_output=True, text=True, env=e)
    assert sel.returncode == 0, sel.stderr
    p = subprocess.run(["bash", AFTER, "0", "unet"], capture_output=True, text=True,
                       env={**e, "CORPORA": "arenas_678", **env})
    return p, r


def test_the_unseen_row_is_refused_with_a_decoder_that_saw_the_unseen_maps(tmp_path):
    p, r = _sealed(tmp_path)
    assert p.returncode != 0 and "unseen-map claim" in p.stderr
    assert not (r / "eval_tf_arenas_678" / "metrics.json").exists()
    assert "arenas_678 done" not in (r / "test_scored_at").read_text() if (r / "test_scored_at").exists() else True


def test_the_weaker_claim_can_be_asked_for_and_is_recorded(tmp_path):
    p, r = _sealed(tmp_path, UNSEEN_CLAIM="dynamics-only")
    assert p.returncode == 0, p.stderr
    text = (r / "eval_tf_arenas_678" / "decoder_provenance.txt").read_text()
    assert "unseen_claim=dynamics-only" in text
    assert "decoder=vae_decoder_arnold_lpips" in text and "provenance=unknown" in text
