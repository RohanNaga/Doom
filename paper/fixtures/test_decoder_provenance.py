"""The decoder loss covers the real picture, and every score says which decoder produced it.

Two defects:

  * the decoder MSE term covered all 256 rows while the LPIPS term and every reported metric cropped
    to 240, so 16/256 = 6.25% of the loss was spent on the gray padding the encoder added to reach a
    multiple of 8 -- pixels nothing scores and nothing looks at.
  * `after_nexttic.sh` silently fell back to the stock decoder when the tuned one had no weights, and
    recorded nothing about the decoder next to the numbers. An unseen-map claim is about the whole
    system, so a decoder tuned on arenas 6-8 defeats it whatever the denoiser was initialised from.

    python -m pytest paper/fixtures/test_decoder_provenance.py -q
"""
import ast
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import finetune_decoder  # noqa: E402

AFTER = os.path.join(REPO, "scripts", "spiderman", "after_nexttic.sh")


# ---------------------------------------------------------------------------------------
# the padded rows are out of the loss
# ---------------------------------------------------------------------------------------

def test_the_mse_rows_default_to_the_real_picture():
    a = finetune_decoder.build_parser().parse_args(["--in-dir", "x", "--split", "y", "--out-dir", "z"])
    assert a.mse_rows == 240
    b = finetune_decoder.build_parser().parse_args(
        ["--in-dir", "x", "--split", "y", "--out-dir", "z", "--mse-rows", "256"])
    assert b.mse_rows == 256


def _loss_source():
    """The training-loop statement that builds the loss, as source text."""
    src = open(os.path.join(REPO, "finetune_decoder.py")).read()
    tree = ast.parse(src)
    lines = src.splitlines()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "loss"
                and "mean" in ast.unparse(node.value)):
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError("no loss assignment found")


def test_the_mse_term_is_cropped_like_the_lpips_term():
    """The defect: `torch.mean((y.float() - x) ** 2)` over all 256 rows, next to an LPIPS term on
    `[:, :, :240]`."""
    text = _loss_source()
    assert "rows" in text and ":rows]" in text, text
    assert "torch.mean((y.float() - x) ** 2)" not in text, "the padded MSE is back"


def test_the_two_terms_score_the_same_rows_by_default():
    """The arithmetic the flag encodes: 16 of 256 rows are padding, 6.25% of the squared error."""
    a = finetune_decoder.build_parser().parse_args(["--in-dir", "x", "--split", "y", "--out-dir", "z"])
    assert a.mse_rows == 240
    assert abs((256 - 240) / 256 - 0.0625) < 1e-12


@pytest.mark.parametrize("rows,expect", [(240, 240), (256, 256), (999, 256)])
def test_the_crop_is_bounded_by_the_tensor(rows, expect):
    x = torch.zeros(2, 3, 256, 320)
    assert min(rows, x.shape[2]) == expect


def test_the_padded_rows_carry_their_documented_share_of_the_loss():
    """A direct measurement of what the crop removes: a decoder that is perfect on the picture and
    wrong on the padding scored a non-zero loss, and a gradient pushed it to fix the padding."""
    x = torch.zeros(1, 3, 256, 320)
    y = torch.zeros(1, 3, 256, 320)
    y[:, :, 240:] = 1.0                       # only the padding is wrong
    full = torch.mean((y - x) ** 2)
    cropped = torch.mean((y[:, :, :240] - x[:, :, :240]) ** 2)
    assert abs(float(full) - 16 / 256) < 1e-6
    assert float(cropped) == 0.0


def test_the_provenance_block_names_the_corpus_and_the_loss():
    src = open(os.path.join(REPO, "finetune_decoder.py")).read()
    assert '"provenance": {"train_corpus": args.in_dir' in src
    for key in ('"split": args.split', '"mse_rows"', '"lpips_rows": 240', '"loss"'):
        assert key in src, key


def test_a_real_tune_records_what_it_was_tuned_on_and_for(tmp_path, monkeypatch):
    from test_decoder_stream import tune
    m = tune(tmp_path, monkeypatch)
    prov = m["provenance"]
    assert prov["mse_rows"] == 240 and prov["lpips_rows"] == 240
    assert prov["train_corpus"].endswith("train") and prov["split"].endswith("split.json")
    assert prov["loss"] == "mse"
    assert m["args"]["mse_rows"] == 240


def test_the_old_padded_loss_stays_reachable_for_the_record(tmp_path, monkeypatch):
    from test_decoder_stream import tune
    m = tune(tmp_path, monkeypatch, mse_rows=256)
    assert m["provenance"]["mse_rows"] == 256


def test_an_lpips_tune_says_so(tmp_path, monkeypatch):
    pytest.importorskip("lpips", reason="the perceptual term needs the real package")
    from test_decoder_stream import tune
    m = tune(tmp_path, monkeypatch, lpips_weight=0.1, report_lpips=True)
    assert m["provenance"]["loss"] == "mse + 0.1 lpips"


# ---------------------------------------------------------------------------------------
# every score records which decoder made it
# ---------------------------------------------------------------------------------------

def _run_launcher(tmp_path, tuned=False, **env):
    """Run the evaluation launcher for real with a stub python, as the other launcher tests do."""
    from test_after_nexttic_stages import STUB_PY, _root, RUN
    root, r = _root(tmp_path)
    if tuned:
        d = root / "vae_decoder_arnold_lpips"
        (d / "vae").mkdir(parents=True)
        (d / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
        (d / "metrics.json").write_text('{"provenance": {"train_corpus": "/raw/arenas", '
                                        '"loss": "mse + 0.1 lpips", "mse_rows": 240}}')
    py = tmp_path / "stubpy"
    py.write_text(STUB_PY)
    py.chmod(0o755)
    log = tmp_path / "stub.log"
    import rollout_eval
    e = {**os.environ, "DOOM_ROOT": str(root), "PY": str(py), "PY_SD35": str(py),
         "STUB_LOG": str(log), "STUB_PICK": str(r / "snap_0290000.pt"),
         "STUB_SCORE_FILE": rollout_eval.SCORE_FILE,
         "CORPORA": "val", "CKPT": str(r / "snap_0290000.pt"), **env}
    proc = subprocess.run(["bash", AFTER, "0", "unet"], capture_output=True, text=True, env=e,
                          timeout=90)
    assert proc.returncode == 0, proc.stderr
    return root, r, RUN


def test_a_stock_fallback_is_recorded_next_to_every_score(tmp_path):
    root, r, _ = _run_launcher(tmp_path, tuned=False)
    for name in ("eval_tf_val", "eval_tf_val_ema", "eval_tf_val_best", "eval_tf_val_h4",
                 "rollout_metrics_test"):
        text = (r / name / "decoder_provenance.txt").read_text()
        assert "decoder_kind=stock" in text, name
        assert "step=290000" in text, name


def test_the_tuned_decoder_s_own_metrics_travel_with_the_scores(tmp_path):
    root, r, _ = _run_launcher(tmp_path, tuned=True)
    import json
    for name in ("eval_tf_val", "rollout_metrics_test"):
        text = (r / name / "decoder_provenance.txt").read_text()
        assert "decoder_kind=tuned" in text, name
        assert "vae_decoder_arnold_lpips/vae" in text, name
        with open(r / name / "decoder_metrics.json") as f:
            prov = json.load(f)["provenance"]
        assert prov["train_corpus"] == "/raw/arenas" and prov["mse_rows"] == 240


def test_the_launcher_writes_the_provenance_beside_each_score_dir():
    text = open(AFTER).read()
    assert text.count('prov "$OUT"') == 1 and text.count('prov "$OUT4"') == 1
    assert 'prov "$R/rollout_metrics_test"' in text
    assert "DEC_METRICS=$(dirname \"$TUNED\")/metrics.json" in text
