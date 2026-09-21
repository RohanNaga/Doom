"""A resume must continue THIS experiment, not one that shares its step count.

`RESUME_IDENTITY` covered the architecture and the objective but not the optimization: a local
execution of the old function accepted `--clip 999` and `--ema-decay 0.9` silently. The corpus was
compared by directory BASENAME only, so `/other/corpus/arenas` passed and a validation directory
holding the same ids but different latents passed as well.

    python -m pytest paper/fixtures/test_resume_preflight.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

os.environ.setdefault("ACCELERATE_USE_CPU", "1")

import train_wm  # noqa: E402
from pertic_fixtures import held_actions, write_pertic_episode  # noqa: E402


def _ck(**over):
    base = {"backbone": "pixart", "latent_channels": 4, "context_frames": 32, "tic_stride": 1,
            "action_history": 32, "control_bits": 19, "objective": "v", "noise_buckets": 10,
            "noise_aug_max": 0.7, "global_batch": 32, "lr": 5e-5, "warm_start": "pixart",
            "phase_buckets": 5, "phase_conditioning": False, "latents_dir": "/a/arenas",
            "optim": "adamw", "wd": 0.0, "warmup": 2000, "clip": 1.0, "ema_every": 8,
            "ema_decay": 0.9999, "seed": 0, "skip_grad_norm": 5.0, "skip_grad_after": 3000}
    base.update(over)
    return {"args": base, "model": {}, "step": 10}


def _args_for(**over):
    argv = ["--backbone", "pixart", "--tic-stride", "1", "--action-history", "32",
            "--context-frames", "32", "--latent-channels", "4", "--objective", "v",
            "--noise-buckets", "10", "--noise-aug-max", "0.7", "--global-batch", "32",
            "--lr", "5e-5", "--warm-start", "pixart", "--latents-dir", "/b/arenas",
            "--optim", "adamw", "--wd", "0", "--warmup", "2000", "--clip", "1.0",
            "--ema-every", "8", "--ema-decay", "0.9999", "--seed", "0",
            "--skip-grad-norm", "5", "--skip-grad-after", "3000", "--resume", "x.pt"]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    a = train_wm.build_parser().parse_args(argv)
    a.control_bits = 19
    return a


# ---------------------------------------------------------------------------------------
# the optimization half of the recipe
# ---------------------------------------------------------------------------------------

def test_the_matching_recipe_is_still_allowed():
    assert train_wm.check_resume_identity(_ck(), _args_for(), {}) == {}


@pytest.mark.parametrize("key,value", [
    ("clip", 999.0),                 # accepted silently before: the old set had no clip
    ("ema_decay", 0.9),
    ("ema_every", 1),
    ("warmup", 500),
    ("wd", 0.01),
    ("optim", "adamw8bit"),
    ("seed", 7),
    ("skip_grad_norm", 0.0),
    ("skip_grad_after", 0),
])
def test_every_optimization_setting_is_part_of_the_identity(key, value):
    with pytest.raises(SystemExit) as e:
        train_wm.check_resume_identity(_ck(), _args_for(**{key: value}), {})
    assert key in str(e.value)


def test_the_learning_rate_is_still_the_one_overridable_key():
    with pytest.raises(SystemExit, match="lr"):
        train_wm.check_resume_identity(_ck(), _args_for(lr="2.5e-5"), {})
    assert "lr" in train_wm.check_resume_identity(_ck(), _args_for(lr="2.5e-5"), {"lr": 2.5e-5})
    with pytest.raises(SystemExit, match="not allowed"):
        train_wm.parse_resume_override(["clip=999"])
    with pytest.raises(SystemExit, match="not allowed"):
        train_wm.parse_resume_override(["ema_decay=0.9"])


def test_a_checkpoint_from_before_these_keys_still_resumes():
    """A key absent from the checkpoint's args is skipped, so the finished rows stay resumable."""
    assert train_wm.check_resume_identity({"args": {"backbone": "pixart"}}, _args_for(), {}) == {}


# ---------------------------------------------------------------------------------------
# the corpus manifest: episode ids AND contents
# ---------------------------------------------------------------------------------------

def _corpus(d, eps, rows=20, fill=1):
    for ep in eps:
        write_pertic_episode(str(d), ep, held_actions([fill] * (rows // 4)))
    return str(d)


def _ns(latents_dir, val_dir=""):
    a = train_wm.build_parser().parse_args(["--backbone", "dit", "--latents-dir", str(latents_dir)]
                                           + (["--val-latents-dir", str(val_dir)] if val_dir else []))
    return a


def test_the_manifest_names_the_episodes_and_fingerprints_them(tmp_path):
    d = _corpus(tmp_path / "lat", [0, 1, 2])
    m = train_wm.corpus_manifest(d, [2, 0])
    assert m["episodes"] == [0, 2], "the ids are sorted, so the list is order-independent"
    assert m["dir"] == "lat"
    assert len(m["fingerprint"]) == 32
    assert train_wm.corpus_manifest(d, [0, 2]) == m, "the fingerprint is not reproducible"


def test_the_fingerprint_survives_a_byte_for_byte_copy(tmp_path):
    """An `rsync -a` copy or a fresh clone must keep the fingerprint; filesystem mtime must not
    enter it, or a resume from a copied corpus would be refused for no reason."""
    import shutil
    a = _corpus(tmp_path / "a", [0, 1])
    b = str(tmp_path / "b")
    shutil.copytree(a, b)
    for name in os.listdir(b):
        os.utime(os.path.join(b, name), (1, 1))      # a copy that did not preserve times
    assert train_wm.corpus_manifest(a, [0, 1])["fingerprint"] == \
        train_wm.corpus_manifest(b, [0, 1])["fingerprint"]


def test_a_different_corpus_of_the_same_ids_has_a_different_fingerprint(tmp_path):
    """The defect: `/other/corpus/arenas` holding the same ids passed the basename check, and a
    same-shape corpus encoded by another autoencoder fails no later assertion either."""
    a = _corpus(tmp_path / "a", [0, 1], fill=1)
    b = _corpus(tmp_path / "b", [0, 1], fill=5)      # same ids, same shapes, different contents
    ma, mb = train_wm.corpus_manifest(a, [0, 1]), train_wm.corpus_manifest(b, [0, 1])
    assert ma["episodes"] == mb["episodes"]
    assert ma["fingerprint"] != mb["fingerprint"]


def test_a_repaired_sidecar_changes_the_fingerprint(tmp_path):
    d = _corpus(tmp_path / "lat", [0])
    before = train_wm.corpus_manifest(d, [0])["fingerprint"]
    p = os.path.join(d, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["is_decision"] = ~cols["is_decision"]
    np.savez(p, **cols)
    assert train_wm.corpus_manifest(d, [0])["fingerprint"] != before


def test_a_truncated_latent_changes_the_fingerprint(tmp_path):
    d = _corpus(tmp_path / "lat", [0])
    before = train_wm.corpus_manifest(d, [0])["fingerprint"]
    p = os.path.join(d, "ep_00000_latents.npy")
    data = open(p, "rb").read()
    with open(p, "wb") as f:
        f.write(data[:-4096])
    assert train_wm.corpus_manifest(d, [0])["fingerprint"] != before


def test_a_resume_onto_another_corpus_is_refused(tmp_path):
    a = _corpus(tmp_path / "a", [0, 1], fill=1)
    b = _corpus(tmp_path / "b", [0, 1], fill=5)
    old = train_wm.corpus_identity(_ns(a), [0, 1], [0, 1])
    now = train_wm.corpus_identity(_ns(b), [0, 1], [0, 1])
    with pytest.raises(SystemExit, match="train corpus contents"):
        train_wm.check_corpus_identity({train_wm.CORPUS_KEY: old}, now)


def test_a_resume_onto_another_validation_corpus_is_refused(tmp_path):
    t = _corpus(tmp_path / "t", [0, 1])
    v1 = _corpus(tmp_path / "v1", [6], fill=1)
    v2 = _corpus(tmp_path / "v2", [6], fill=9)
    old = train_wm.corpus_identity(_ns(t, v1), [0, 1], [6])
    now = train_wm.corpus_identity(_ns(t, v2), [0, 1], [6])
    with pytest.raises(SystemExit, match="val corpus contents"):
        train_wm.check_corpus_identity({train_wm.CORPUS_KEY: old}, now)


def test_a_resume_on_a_grown_corpus_is_refused(tmp_path):
    d = _corpus(tmp_path / "lat", [0, 1, 2])
    old = train_wm.corpus_identity(_ns(d), [0, 1], [2])
    now = train_wm.corpus_identity(_ns(d), [0, 1, 2], [2])
    with pytest.raises(SystemExit, match="train episode ids"):
        train_wm.check_corpus_identity({train_wm.CORPUS_KEY: old}, now)


def test_the_same_corpus_at_another_mount_is_allowed(tmp_path):
    """The basename check exists because the corpus is legitimately mounted elsewhere; the manifest
    must agree with that and reject only a change of CONTENTS."""
    import shutil
    a = _corpus(tmp_path / "machine1" / "arenas", [0, 1])
    b = str(tmp_path / "machine2" / "arenas")
    os.makedirs(os.path.dirname(b), exist_ok=True)
    shutil.copytree(a, b)
    old = train_wm.corpus_identity(_ns(a), [0, 1], [0, 1])
    now = train_wm.corpus_identity(_ns(b), [0, 1], [0, 1])
    assert train_wm.check_corpus_identity({train_wm.CORPUS_KEY: old}, now) == old


def test_a_checkpoint_without_a_manifest_says_so_and_continues(tmp_path, capsys):
    d = _corpus(tmp_path / "lat", [0, 1])
    now = train_wm.corpus_identity(_ns(d), [0, 1], [0, 1])
    assert train_wm.check_corpus_identity({"step": 5}, now) == {}
    assert "no corpus manifest" in capsys.readouterr().out


def test_every_checkpoint_writer_carries_the_manifest():
    src = open(os.path.join(REPO, "train_wm.py")).read()
    assert src.count("CORPUS_KEY: corpus") == 3, "best.pt, the snapshot and the recovery checkpoint"


# ---------------------------------------------------------------------------------------
# what a resume does NOT restore, written down where a reader will find it
# ---------------------------------------------------------------------------------------

def test_the_resume_help_states_that_rng_and_sampler_position_are_not_restored():
    help_text = [a for a in train_wm.build_parser()._actions if a.dest == "resume"][0].help
    for phrase in ("never restored", "sampler's position", "fresh permutation"):
        assert phrase in help_text, help_text


def test_the_saved_rng_is_written_but_never_read_back():
    src = open(os.path.join(REPO, "train_wm.py")).read()
    assert '"rng"' in src, "the RNG record is gone"
    assert 'ck["rng"]' not in src and 'set_rng_state' not in src, \
        "the RNG is restored now, which changes what a resume is"
