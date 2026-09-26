"""The MSE decoder tune's launcher (`scripts/spiderman/decoder_mse.sh`), run against stub interpreters.

Astra's review of 2026-09-26 (section 5) found that the launcher never named the autoencoder it
tuned, so an SD 3.5 tune would silently have run on sd-vae-ft-mse. These tests pin that `SPACE`
picks the autoencoder, its asserted latent contract, its interpreter and its output directory, and
that nothing else about the recipe moves with it.

Every run here points `DOOM_ROOT`, `REPO` and `HOME` at the test's own directory, and both default
interpreters (`~/miniconda3/envs/doom/bin/python`, `~/wanenc/bin/python`) are stubs that log their
argv, so the space switch is observed through the interpreter the launcher actually chose.

    python -m pytest paper/fixtures/test_decoder_mse_launcher.py -q
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
DECODER_MSE = os.path.join(REPO, "scripts", "spiderman", "decoder_mse.sh")

STUB = '''#!/usr/bin/env python3
"""Stands in for finetune_decoder.py and vae_gate_score.py: logs its argv, then exits as told.

STUB_FAIL names the script that exits 3. A tune that succeeds leaves <out-dir>/vae and one hourly
vae_h1 with a weight file each, as the real one does, unless STUB_NO_WEIGHTS=1.
"""
import os, sys
script = os.path.basename(sys.argv[1])
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(sys.argv[0] + " " + " ".join(sys.argv[1:]) + "\\n")
if os.environ.get("STUB_FAIL") == script:
    sys.exit(3)
if script == "finetune_decoder.py" and os.environ.get("STUB_NO_WEIGHTS") != "1":
    out = sys.argv[sys.argv.index("--out-dir") + 1]
    for d in ("vae", "vae_h1"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
        open(os.path.join(out, d, "diffusion_pytorch_model.safetensors"), "wb").close()
'''

SD1_PY = "miniconda3/envs/doom/bin/python"
SD35_PY = "wanenc/bin/python"


def launch(tmp_path, *args, **env):
    """Run the launcher with a throwaway data root, checkout and home. Returns (proc, calls, root)."""
    home, root, checkout = tmp_path / "home", tmp_path / "root", tmp_path / "repo"
    for rel in (SD1_PY, SD35_PY):
        stub = home / rel
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(STUB)
        stub.chmod(0o755)
    root.mkdir(parents=True, exist_ok=True)
    checkout.mkdir(exist_ok=True)
    log = tmp_path / "stub.log"
    if log.exists():
        log.unlink()
    e = {k: v for k, v in os.environ.items() if k not in ("PY", "SPACE", "OUT", "FIT", "CURVE", "DRY")}
    e.update(HOME=str(home), DOOM_ROOT=str(root), REPO=str(checkout), STUB_LOG=str(log), **env)
    proc = subprocess.run(["bash", DECODER_MSE, *(args or ("0", "8", "100"))],
                          capture_output=True, text=True, env=e, timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls, root


def calls_to(calls, script):
    return [c.split() for c in calls if c.split()[1] == script]


def flag(argv, name):
    """The value after `name` in an argv, or None."""
    return argv[argv.index(name) + 1] if name in argv else None


def has(argv, *seq):
    """Does `seq` occur as consecutive tokens of `argv`?"""
    n = len(seq)
    return any(tuple(argv[i:i + n]) == seq for i in range(len(argv) - n + 1))


# ---------------------------------------------------------------------------------------
# SPACE picks the autoencoder, its contract, its interpreter and its output
# ---------------------------------------------------------------------------------------

def test_sd1_is_the_default_and_names_sd_vae_ft_mse(tmp_path):
    p, calls, root = launch(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    [tune] = calls_to(calls, "finetune_decoder.py")
    assert tune[0] == str(tmp_path / "home" / SD1_PY)
    assert has(tune, "--vae-id", "stabilityai/sd-vae-ft-mse")
    assert has(tune, "--latent-channels", "4") and has(tune, "--scaling-factor", "0.18215")
    # sd-vae-ft-mse has no shift and is not a pipeline repo; it loads from the default cache, where
    # every earlier SD 1.x tune found it
    assert "--shift-factor" not in tune and "--vae-subfolder" not in tune and "--cache-dir" not in tune
    assert flag(tune, "--out-dir") == str(root / "vae_decoder_sd1x_mse")


def test_sd35_tunes_its_own_autoencoder_under_its_own_interpreter(tmp_path):
    p, calls, root = launch(tmp_path, SPACE="sd35")
    assert p.returncode == 0, p.stdout + p.stderr
    [tune] = calls_to(calls, "finetune_decoder.py")
    assert tune[0] == str(tmp_path / "home" / SD35_PY)
    assert has(tune, "--vae-id", "stabilityai/stable-diffusion-3.5-medium", "--vae-subfolder", "vae")
    assert has(tune, "--latent-channels", "16")
    assert has(tune, "--scaling-factor", "1.5305") and has(tune, "--shift-factor", "0.0609")
    assert has(tune, "--cache-dir", str(root / "hf" / "hub"))
    assert flag(tune, "--out-dir") == str(root / "vae_decoder_sd35_mse")
    assert all(c.split()[0] == str(tmp_path / "home" / SD35_PY) for c in calls)


def test_an_unknown_space_is_refused_before_anything_runs(tmp_path):
    p, calls, root = launch(tmp_path, SPACE="sd3")
    assert p.returncode == 2 and "SPACE" in p.stderr
    assert calls == [] and list(root.iterdir()) == []


def test_the_recipe_does_not_move_with_the_space(tmp_path):
    """Only the autoencoder's identity differs between the two tunes; the stream and the schedule do not."""
    _, sd1, _ = launch(tmp_path / "a")
    _, sd35, _ = launch(tmp_path / "b", SPACE="sd35")
    [a] = calls_to(sd1, "finetune_decoder.py")
    [b] = calls_to(sd35, "finetune_decoder.py")
    space_flags = {"--vae-id", "--vae-subfolder", "--latent-channels", "--scaling-factor", "--shift-factor",
                   "--cache-dir", "--out-dir"}

    def recipe(argv, root):
        out, i = [], 2
        while i < len(argv):
            if argv[i] in space_flags:
                i += 2
                continue
            out.append(argv[i].replace(root, "<root>"))
            i += 1
        return out
    assert recipe(a, str(tmp_path / "a" / "root")) == recipe(b, str(tmp_path / "b" / "root"))
    for argv in (a, b):
        assert has(argv, "--stream-ids", "0:2000") and has(argv, "--stream-frames", "400000")
        assert has(argv, "--stream-episodes", "2000") and has(argv, "--lpips-weight", "0")
        assert has(argv, "--max-hours", "4.0", "--ckpt-every-hours", "1")
        assert has(argv, "--batch-size", "8", "--accum", "1") and has(argv, "--max-steps", "100")


def test_the_two_spaces_never_share_a_fit_directory(tmp_path):
    _, a, root_a = launch(tmp_path / "a", FIT="1")
    _, b, root_b = launch(tmp_path / "b", FIT="1", SPACE="sd35")
    fa = flag(calls_to(a, "finetune_decoder.py")[0], "--out-dir")
    fb = flag(calls_to(b, "finetune_decoder.py")[0], "--out-dir")
    assert os.path.relpath(fa, root_a) != os.path.relpath(fb, root_b)


def test_dry_prints_the_commands_and_touches_nothing(tmp_path):
    p, calls, root = launch(tmp_path, DRY="1", SPACE="sd35")
    assert p.returncode == 0, p.stderr
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("DRY ")]
    assert any("finetune_decoder.py" in ln and "--vae-id stabilityai/stable-diffusion-3.5-medium" in ln
               for ln in lines), p.stdout
    assert any("vae_gate_score.py" in ln for ln in lines), p.stdout
    assert calls == [] and list(root.iterdir()) == []


# ---------------------------------------------------------------------------------------
# the validation frames are held-out episodes of the training arenas
# ---------------------------------------------------------------------------------------

def test_validation_is_on_the_held_out_dense_episodes(tmp_path):
    for space in ("sd1", "sd35"):
        p, calls, root = launch(tmp_path / space, SPACE=space)
        assert p.returncode == 0, p.stdout + p.stderr
        [tune] = calls_to(calls, "finetune_decoder.py")
        assert has(tune, "--val-dir", f"{root}/raw_arnold_dense/arenas", "--val-ids", "6000:6100")
        assert has(tune, "--val-frames", "2000") and has(tune, "--stride", "1")
        # nothing of the 17-map corpus reaches the tune any more
        assert "--in-dir" not in tune and "--split" not in tune
        assert not any(a.startswith(f"{root}/raw_arnold/") or a == f"{root}/raw_arnold" or "split_arnold" in a
                       for a in tune)


def test_finetune_decoder_accepts_exactly_the_launchers_flags(tmp_path):
    import finetune_decoder
    for space in ("sd1", "sd35"):
        _, calls, root = launch(tmp_path / space, SPACE=space)
        [tune] = calls_to(calls, "finetune_decoder.py")
        a = finetune_decoder.build_parser().parse_args(tune[2:])
        assert a.val_dir == f"{root}/raw_arnold_dense/arenas" and a.val_ids == "6000:6100"
        assert a.stream_dir == a.val_dir and a.stream_ids == "0:2000"
        assert a.in_dir == "" and a.split == ""
        # the ids the launcher names pass the checks the tune makes before it reads a frame
        assert finetune_decoder.validation_episode_ids(a.val_dir, a.val_ids, a.stream_dir, list(range(2000))) \
            == list(range(6000, 6100))
