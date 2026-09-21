"""Regression tests for the five defects Astra found in `df9a4b4` (second pass, Sep 21 2026).

As in `test_nexttic_defects.py`, each test reproduces the specific way the code was wrong rather
than only asserting the fixed behaviour. Two of these needed a REAL run of the encoder launcher
against a throwaway data root with a stub encoder, because the bug was in bash scoping and could
not be seen through the DRY path at all.

    python -m pytest paper/fixtures/test_nexttic_defects2.py -q
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

os.environ.setdefault("ACCELERATE_USE_CPU", "1")

from pertic_fixtures import held_actions, write_pertic_episode  # noqa: E402

import eval_tf  # noqa: E402
import make_dense_eval_splits  # noqa: E402
import train_wm  # noqa: E402

SCRIPTS = os.path.join(REPO, "scripts", "spiderman")


def _dry(script, args, **env):
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": "/tmp/nexttic-defects2-root", **env}
    r = subprocess.run(["bash", os.path.join(SCRIPTS, script), *args],
                       capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    return r.stdout


STUB_ENCODER = '''import os, sys
import numpy as np
out = sys.argv[sys.argv.index("--out-dir") + 1]
lo = int(sys.argv[sys.argv.index("--episode-ids") + 1].split(":")[0])
os.makedirs(out, exist_ok=True)
T = 8
np.save(os.path.join(out, "ep_%05d_latents.npy" % lo), np.zeros((T, 4, 32, 40), dtype=np.float16))
np.savez(os.path.join(out, "ep_%05d_meta.npz" % lo),
         action=np.zeros(T, np.int64), buttons=np.array(["100000000"] * T),
         tic=np.arange(T, dtype=np.int64), deaths=np.zeros(T, np.int64),
         map_id=np.full(T, 2, np.int64), episode_id=np.full(T, lo, np.int64),
         is_decision=(np.arange(T) % 4 == 0), chain_id=np.zeros(T, np.int64))
'''


def _real_run(tmp_path, corpus, python=None):
    """Run the encoder launcher FOR REAL against a throwaway root, with a stub encoder.

    Not DRY: the bug in (1) was bash variable scoping in the lock cleanup, which the DRY path
    returns before ever reaching.
    """
    root = tmp_path / "root"
    for seg in ("arenas", "arenas_678"):
        (root / "raw_arnold_dense" / seg).mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "encode_parquet.py").write_text(STUB_ENCODER)
    with open(os.path.join(REPO, "make_dense_eval_splits.py")) as f:
        (repo / "make_dense_eval_splits.py").write_text(f.read())
    env = {**os.environ, "DOOM_ROOT": str(root), "CORPUS": corpus, "REPO": str(repo),
           "PY": python or sys.executable, "PYTHONPATH": REPO}
    return subprocess.run(["bash", os.path.join(SCRIPTS, "encode_nexttic.sh")],
                          capture_output=True, text=True, env=env), root


# ---------------------------------------------------------------------------------------
# (1) the RETURN trap that killed the shell after a SUCCESSFUL encode
# ---------------------------------------------------------------------------------------

def test_a_successful_encode_visits_every_corpus_and_exits_zero(tmp_path):
    """The RETURN trap referenced a `local` LOCK and was never cleared, so it fired again when the
    CALLER returned with LOCK out of scope; under `set -u` that killed the shell with exit 127
    after a successful encode, before ENCODE_NEXTTIC_DONE and before the other corpora."""
    r, root = _real_run(tmp_path, "evals")
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}"
    assert "ENCODE_NEXTTIC_DONE" in r.stdout, r.stdout
    eval_root = root / "latents_arnold_dense_pertic_eval"
    for corpus in ("val", "test", "arenas_678"):
        assert (eval_root / corpus).is_dir(), f"{corpus} was never encoded"
        assert list((eval_root / corpus).glob("ep_*_latents.npy")), f"{corpus} produced no latents"
        assert not list((eval_root / corpus).glob(".lock*")), f"{corpus} left its lock behind"


def test_a_failing_encoder_still_reports_and_releases_the_lock(tmp_path):
    r, root = _real_run(tmp_path, "val", python="/bin/false")
    assert r.returncode != 0
    assert "ENCODE_NEXTTIC_FAILED" in r.stderr
    assert "ENCODE_NEXTTIC_DONE" not in r.stdout
    assert not list((root / "latents_arnold_dense_pertic_eval" / "val").glob(".lock*")), \
        "the lock survived a failed encode"


def test_the_lock_cleanup_is_not_a_return_trap():
    src = open(os.path.join(SCRIPTS, "encode_nexttic.sh")).read()
    code = [ln for ln in src.splitlines() if ln.strip().startswith("trap ")]
    assert code, "no trap at all"
    assert not any("RETURN" in ln for ln in code), \
        f"a RETURN trap fires again when the caller returns: {code}"
    assert any(ln.strip() == "trap 'rmdir \"$LOCK\" 2>/dev/null' EXIT" for ln in code), code


# ---------------------------------------------------------------------------------------
# (2) one canonical split location, shared by the writer and the reader
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", ["", "_sd35"])
@pytest.mark.parametrize("corpus", ["val", "test", "arenas_678"])
def test_the_split_the_encoder_writes_is_the_one_the_evaluator_reads(corpus, suffix):
    """The encoder named the file after its internal tag and put it INSIDE the corpus directory
    (`.../arenas_678/split_unseen.json`) while after_nexttic.sh read `.../split_arenas_678.json`."""
    root = f"/d/latents_arnold_dense_pertic_eval{suffix}"
    written = make_dense_eval_splits.split_path(f"{root}/{corpus}")
    assert written == f"{root}/split_{corpus}.json"
    out = _dry("after_nexttic.sh", ["3", "sd35" if suffix else "unet"])
    read = [tok for ln in out.splitlines() for tok in ln.split()
            if os.path.basename(tok) == f"split_{corpus}.json"]
    assert read, f"the evaluation script never reads split_{corpus}.json:\n{out}"
    for tok in read:
        assert os.path.basename(os.path.dirname(tok)) == os.path.basename(root), tok


def test_the_split_name_comes_from_the_directory_not_a_tag(tmp_path):
    d = str(tmp_path / "arenas_678")
    for ep in (0, 1):
        write_pertic_episode(d, ep, held_actions([0, 1]))
    path, _ = make_dense_eval_splits.build(d)      # no --name, as the launcher now calls it
    assert os.path.basename(path) == "split_arenas_678.json"
    assert os.path.dirname(path) == str(tmp_path)


def test_the_real_encoder_run_writes_the_split_where_the_evaluator_looks(tmp_path):
    r, root = _real_run(tmp_path, "evals")
    assert r.returncode == 0, r.stderr
    eval_root = root / "latents_arnold_dense_pertic_eval"
    for corpus in ("val", "test", "arenas_678"):
        p = eval_root / f"split_{corpus}.json"
        assert p.is_file(), f"no split for {corpus} at {p}"
        with open(p) as f:
            assert make_dense_eval_splits.SUBSET in json.load(f)


def test_the_launcher_does_not_pass_its_own_name_or_path():
    src = open(os.path.join(SCRIPTS, "encode_nexttic.sh")).read()
    line = [ln for ln in src.splitlines() if "make_dense_eval_splits.py" in ln and "$PY" in ln]
    assert line and "--name" not in line[0] and "--out" not in line[0], line


# ---------------------------------------------------------------------------------------
# (3) a resume must not change what the experiment is
# ---------------------------------------------------------------------------------------

def _ck(**over):
    base = dict(backbone="pixart", latent_channels=4, context_frames=32, tic_stride=1,
                action_history=32, control_bits=19, objective="v", noise_buckets=10,
                noise_aug_max=0.7, global_batch=32, lr=5e-5, warm_start="pixart",
                phase_buckets=5, phase_conditioning=False, latents_dir="/a/arenas")
    base.update(over)
    return {"args": base, "model": {}, "step": 10}


def _args_for(**over):
    argv = ["--backbone", "pixart", "--tic-stride", "1", "--action-history", "32",
            "--context-frames", "32", "--latent-channels", "4", "--objective", "v",
            "--noise-buckets", "10", "--noise-aug-max", "0.7", "--global-batch", "32",
            "--lr", "5e-5", "--warm-start", "pixart", "--latents-dir", "/b/arenas",
            "--resume", "x.pt"]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    a = train_wm.build_parser().parse_args(argv)
    a.control_bits = 19
    return a


def test_a_resume_under_a_different_objective_is_refused():
    """A v-prediction checkpoint resumed under --objective eps trained as eps and saved an
    eps-labelled checkpoint whose first N steps were velocity."""
    with pytest.raises(SystemExit, match="different recipe"):
        train_wm.check_resume_identity(_ck(), _args_for(objective="eps"), {})


@pytest.mark.parametrize("key,value", [("context_frames", 64), ("tic_stride", 4),
                                       ("noise_aug_max", 0.0), ("global_batch", 64),
                                       ("action_history", 0), ("noise_buckets", 4)])
def test_every_recipe_identity_arg_is_checked(key, value):
    with pytest.raises(SystemExit) as e:
        train_wm.check_resume_identity(_ck(), _args_for(**{key: value}), {})
    assert key in str(e.value)


def test_a_resume_with_the_same_recipe_is_allowed():
    """The latents directory is compared by basename, so the same corpus at a different mount is fine."""
    assert train_wm.check_resume_identity(_ck(), _args_for(), {}) == {}


def test_a_different_corpus_is_refused():
    with pytest.raises(SystemExit, match="latents_dir"):
        train_wm.check_resume_identity(_ck(latents_dir="/a/arenas_678"), _args_for(), {})


def test_the_learning_rate_can_be_changed_only_by_asking():
    """The existing LR-override behaviour stays, but it now has to be named."""
    with pytest.raises(SystemExit, match="lr"):
        train_wm.check_resume_identity(_ck(), _args_for(lr="2.5e-5"), {})
    allowed = train_wm.check_resume_identity(_ck(), _args_for(lr="2.5e-5"), {"lr": 2.5e-5})
    assert "lr" in allowed


def test_only_the_learning_rate_may_be_overridden():
    with pytest.raises(SystemExit, match="not allowed"):
        train_wm.parse_resume_override(["objective=eps"])
    with pytest.raises(SystemExit, match="key=value"):
        train_wm.parse_resume_override(["lr"])
    assert train_wm.parse_resume_override(["lr=1e-5"]) == {"lr": 1e-5}


def test_a_checkpoint_from_before_these_flags_resumes_without_complaint():
    assert train_wm.check_resume_identity({"args": {"backbone": "pixart"}}, _args_for(), {}) == {}
    assert train_wm.check_resume_identity({}, _args_for(), {}) == {}


def test_the_episode_pin_travels_in_the_checkpoint(tmp_path):
    """`pin_episodes` read the destination results dir, so a resume into a NEW directory found no
    record there and re-resolved the list against whatever was encoded by then."""
    d = str(tmp_path / "lat")
    for ep in range(4):
        write_pertic_episode(d, ep, held_actions([0, 1] * 5))
    fresh = str(tmp_path / "elsewhere")
    os.makedirs(fresh)
    a = train_wm.build_parser().parse_args(
        ["--backbone", "dit", "--episode-ids", "0:3", "--val-episode-ids", "3:4",
         "--latents-dir", d, "--results-dir", fresh, "--resume", "x.pt"])
    ck = {"episodes": {"episodes": [0, 1], "val_episodes": [3], "num_episodes": 2}}
    assert train_wm.pin_episodes(a, [0, 1, 2], [3], checkpoint=ck) == ([0, 1], [3])
    # with no record anywhere, the freshly resolved lists are used
    assert train_wm.pin_episodes(a, [0, 1, 2], [3], checkpoint={}) == ([0, 1, 2], [3])


def test_the_checkpoint_pin_beats_a_stale_results_directory(tmp_path):
    d = str(tmp_path / "lat2")
    for ep in range(4):
        write_pertic_episode(d, ep, held_actions([0, 1] * 5))
    out = str(tmp_path / "run")
    os.makedirs(out)
    json.dump({"episodes": [0, 1, 2], "val_episodes": [3]},
              open(os.path.join(out, train_wm.EPISODES_FILE), "w"))
    a = train_wm.build_parser().parse_args(
        ["--backbone", "dit", "--episode-ids", "0:3", "--val-episode-ids", "3:4",
         "--latents-dir", d, "--results-dir", out, "--resume", "x.pt"])
    ck = {"episodes": {"episodes": [0, 1], "val_episodes": [3]}}
    assert train_wm.pin_episodes(a, [0, 1, 2], [3], checkpoint=ck) == ([0, 1], [3])


def test_the_episode_record_is_what_goes_in_both_places():
    a = train_wm.build_parser().parse_args(["--backbone", "dit"])
    rec = train_wm.episode_record(a, [1, 2], [3])
    assert rec["episodes"] == [1, 2] and rec["val_episodes"] == [3] and rec["num_episodes"] == 2


def test_every_checkpoint_writer_carries_the_pin():
    src = open(os.path.join(REPO, "train_wm.py")).read()
    assert src.count('"episodes": pin') == 3, "best.pt, the snapshot and the recovery checkpoint"


# ---------------------------------------------------------------------------------------
# (4) the noise key must not collide, and eta > 0 must be paired too
# ---------------------------------------------------------------------------------------

def test_the_old_arithmetic_noise_key_collided():
    """(ep 11, start 0, step 97) and (ep 11, start 1, step 0) both gave 87,206."""
    def old(seed, ep, s, h):
        return seed * 1_000_003 + ep * 7919 + s * 97 + h
    assert old(0, 11, 0, 97) == old(0, 11, 1, 0) == 87206
    assert eval_tf.window_seed("init", 0, 11, 0, 97) != eval_tf.window_seed("init", 0, 11, 1, 0)


def test_the_hashed_key_separates_every_component():
    base = ("init", 0, 11, 3, 2)
    seen = {eval_tf.window_seed(*base)}
    for i in range(1, 5):
        for delta in (1, 7, 1000):
            parts = list(base)
            parts[i] += delta
            seen.add(eval_tf.window_seed(*parts))
    assert len(seen) == 1 + 4 * 3, "two different windows share a seed"
    assert all(0 <= s < 2 ** 63 for s in seen)


def test_the_purpose_tag_separates_the_initial_and_stochastic_streams():
    assert eval_tf.window_seed("init", 1, 2) != eval_tf.window_seed("eta", 1, 2)
    a = eval_tf.window_noise((1, 4, 4, 5), [(1, 2)], purpose="init")
    b = eval_tf.window_noise((1, 4, 4, 5), [(1, 2)], purpose="eta")
    assert not torch.allclose(a, b)


def test_the_eta_noise_is_per_window_and_per_step():
    fn = eval_tf.eta_noise_fn((2, 4, 4, 5), [(0, 1, 0), (0, 2, 0)], "cpu")
    s0, s1 = fn(0), fn(1)
    assert not torch.allclose(s0, s1), "the same noise at every sampler step"
    assert not torch.allclose(s0[0], s0[1]), "the same noise for both windows"
    assert torch.equal(fn(0), s0), "not reproducible"


def test_the_sampler_takes_a_noise_callback_and_defaults_to_today():
    from diffusion_v import VDiffusion
    d = VDiffusion(num_steps=20)

    def model_fn(xt, t):
        return torch.zeros_like(xt)
    noise = torch.zeros(1, 4, 4, 5)
    calls = []
    d.ddim_sample(model_fn, noise.shape, steps=3, eta=1.0, noise=noise,
                  noise_fn=lambda i: (calls.append(i), torch.zeros(1, 4, 4, 5))[1])
    assert calls == [0, 1], "the callback was not used at every stochastic step"
    torch.manual_seed(0)
    a = d.ddim_sample(model_fn, noise.shape, steps=3, eta=1.0, noise=noise)
    torch.manual_seed(0)
    b = d.ddim_sample(model_fn, noise.shape, steps=3, eta=1.0, noise=noise)
    assert torch.equal(a, b), "the default path changed"


def test_an_eta_one_comparison_is_paired_across_batch_sizes():
    """What the callback buys: the same window gets the same stochastic noise in any batch."""
    from diffusion_v import VDiffusion
    d = VDiffusion(num_steps=20)

    def model_fn(xt, t):
        return torch.zeros_like(xt)
    keys = [(0, 5, 0), (0, 9, 0)]
    big = d.ddim_sample(model_fn, (2, 4, 4, 5), steps=3, eta=1.0,
                        noise=eval_tf.window_noise((2, 4, 4, 5), keys),
                        noise_fn=eval_tf.eta_noise_fn((2, 4, 4, 5), keys, "cpu"))
    one = d.ddim_sample(model_fn, (1, 4, 4, 5), steps=3, eta=1.0,
                        noise=eval_tf.window_noise((1, 4, 4, 5), keys[1:]),
                        noise_fn=eval_tf.eta_noise_fn((1, 4, 4, 5), keys[1:], "cpu"))
    assert torch.allclose(big[1], one[0], atol=1e-6)


def test_the_rollout_uses_the_hashed_key_and_the_callback():
    src = open(os.path.join(REPO, "rollout_eval.py")).read()
    assert "1_000_003" not in src, "the colliding arithmetic key is still there"
    assert "eta_noise_fn" in src and "noise_fn=nfn" in src


# ---------------------------------------------------------------------------------------
# (5) the IDM needs a verified chain, not just four tics
# ---------------------------------------------------------------------------------------

def _longest_verified_run(k, chain):
    """The rule as `rollout_eval.judged` applies it, reproduced here on plain arrays."""
    if len(k) <= 1:
        return k
    step_ok = np.diff(k) == 4
    if chain is not None:
        step_ok = step_ok & (chain[k[1:]] == chain[k[:-1]]) & (chain[k[:-1]] >= 0)
    runs, start = [], 0
    for i, ok in enumerate(list(step_ok) + [False]):
        if not ok:
            runs.append((start, i + 1)); start = i + 1
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return k[a:b]


def test_a_four_tic_interval_across_a_chain_boundary_is_dropped():
    """An override invalidates the transitions around it, so the two accepted decisions either side
    can land four tics apart in DIFFERENT chains; four tics alone would accept that pair."""
    k = np.array([3, 7, 11, 15, 19])
    chain = np.full(24, -1, dtype=np.int64)
    chain[[3, 7, 11]] = 0
    chain[[15, 19]] = 1                    # a new chain after an invalidated stretch
    assert _longest_verified_run(k, chain).tolist() == [3, 7, 11]
    # four tics alone would have kept the whole thing
    assert _longest_verified_run(k, None).tolist() == k.tolist()


def test_an_unverified_stretch_is_never_accepted():
    k = np.array([3, 7, 11])
    chain = np.full(24, -1, dtype=np.int64)       # no verified chain at all
    assert len(_longest_verified_run(k, chain)) == 1


def test_the_eight_tic_gap_is_still_dropped():
    k = np.array([3, 7, 11, 19, 23])
    chain = np.zeros(24, dtype=np.int64)
    assert _longest_verified_run(k, chain).tolist() == [3, 7, 11]


def test_the_rollout_stores_and_uses_the_chain_ids():
    src = open(os.path.join(REPO, "rollout_eval.py")).read()
    assert 'extra["chain"]' in src and 'extra["seed_chain"]' in src
    assert "chain[k[:-1]] >= 0" in src
    assert "different chains" in src, "the reason has to be written down next to the rule"
