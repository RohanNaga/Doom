"""Table and curve for the MSE-only decoder tune: reconstruction ceiling against presentations.

Reads the `metrics.json` `vae_gate_score.py` wrote for the stock decoder, the incumbent LPIPS-tuned
one and every hourly checkpoint of the new run, plus the tune's own `metrics.json` for how many
presentations each of those hours bought. The point of the curve is whether the ceiling is still
climbing at the four-hour budget, because that decides whether more decoder hours are worth buying
at all.

    python make_curve.py --gate metrics.json --tune tune_metrics.json
"""
import argparse
import json
import os
import re

SETS = ("dev", "seen", "unseen", "unseen2")
STOCK_LABEL = "stock_sd"
INCUMBENT = "tuned_sd_lpips"


def merge_gates(gates):
    """One gate result from several scoring passes over the same frames and baseline.

    The ceilings are scored in two passes so that losing the second one to the card's hand-back
    deadline costs only the hourly curve, not the headline. The passes share the baseline decoder,
    so later passes must not overwrite the earlier reading of a decoder both of them scored.
    """
    if not gates:
        raise SystemExit("no gate metrics given")
    out = {"decoders": {}, "metrics": {}, "paired": {}, "baseline": gates[0].get("baseline", "")}
    for g in gates:
        for n, path in g.get("decoders", {}).items():
            out["decoders"].setdefault(n, path)
        for s, m in g.get("metrics", {}).items():
            into = out["metrics"].setdefault(s, {})
            for k, v in m.items():
                into.setdefault(k, v)
        for s, p in g.get("paired", {}).items():
            into = out["paired"].setdefault(s, {})
            for k, v in p.items():
                into.setdefault(k, v)
    return out


def hour_of(name):
    """The hour a checkpoint name encodes: mse_h3 -> 3, mse_final -> the end of the run."""
    m = re.fullmatch(r"mse_h(\d+)", name)
    if m:
        return int(m.group(1))
    return None if name != "mse_final" else 10**6


def presentations(tune, hour):
    """Presentations at an hourly checkpoint, from the tune's own history."""
    if tune is None:
        return None
    if hour == 10**6:
        return tune.get("presentations")
    eff = tune.get("effective_batch")
    for e in tune.get("history", []):
        if e.get("hour_ckpt") == hour:
            return None if eff is None else e["step"] * eff
    return None


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def table(gate, tune):
    names = list(gate["decoders"])
    lines = ["# Reconstruction ceiling under an MSE-only decoder tune on the dense corpus", "",
             "Every decoder scored on the identical frames: the tune development set and the target",
             "frames of the 2,048 windows `eval_tf.py` reports on each corpus. `stock_sd` is",
             "sd-vae-ft-mse untouched, `tuned_sd_lpips` the incumbent 50k-frame MSE + 0.1 LPIPS tune,",
             "`mse_h<N>` the new run after N hours and `mse_final` at the budget.", ""]
    for s in SETS:
        if s not in gate["metrics"]:
            continue
        m = gate["metrics"][s]
        lines += [f"### {s} ({m.get('episodes', '--')} episodes)", "",
                  "| decoder | presentations | PSNR | +/- | LPIPS | HUD PSNR | dPSNR vs incumbent | 95% CI |",
                  "|---|---|---|---|---|---|---|---|"]
        for n in names:
            if n not in m:
                continue
            p = gate.get("paired", {}).get(s, {}).get(n, {}).get("psnr")
            pres = presentations(tune, hour_of(n))
            ci = "--" if p is None else "[%.3f, %.3f]" % tuple(p["ci95"])
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                n, "--" if pres is None else pres,
                fmt(m[n]["psnr"]["mean"]), fmt(m[n]["psnr"]["sem"], 4),
                fmt(m[n]["lpips"]["mean"], 4), fmt(m[n]["hud_psnr"]["mean"]),
                fmt(None if p is None else p["delta"]), ci))
        lines.append("")
    return "\n".join(lines)


def curve(gate, tune, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hours = sorted((hour_of(n), n) for n in gate["decoders"] if hour_of(n) is not None)
    if not hours:
        print("no hourly checkpoints to plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for s in SETS:
        if s not in gate["metrics"]:
            continue
        m = gate["metrics"][s]
        xs = [presentations(tune, h) or 0 for h, n in hours if n in m]
        line, = axes[0].plot(xs, [m[n]["psnr"]["mean"] for h, n in hours if n in m], "o-", lw=1.6, label=s)
        axes[1].plot(xs, [m[n]["lpips"]["mean"] for h, n in hours if n in m], "o-", lw=1.6, label=s,
                     color=line.get_color())
        for ax, key in ((axes[0], "psnr"), (axes[1], "lpips")):
            for ref, style in ((INCUMBENT, "--"), (STOCK_LABEL, ":")):
                if ref in m:
                    ax.axhline(m[ref][key]["mean"], ls=style, lw=1.0, alpha=.6, color=line.get_color())
    axes[0].set_ylabel("ceiling PSNR (dB)"); axes[1].set_ylabel("ceiling LPIPS")
    for ax in axes:
        ax.set_xlabel("presentations"); ax.grid(alpha=.3); ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Decoder reconstruction ceiling against presentations (dashed: incumbent, dotted: stock)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--gate", action="append", default=[],
                   help="a scoring pass's metrics.json; repeat for the headline and the curve pass")
    p.add_argument("--tune", default=os.path.join(here, "tune_metrics.json"))
    p.add_argument("--out-table", default=os.path.join(here, "TABLE.md"))
    p.add_argument("--out-plot", default=os.path.join(here, "ceiling_vs_presentations.png"))
    a = p.parse_args()
    paths = a.gate or [os.path.join(here, "metrics.json"),
                       os.path.join(here, os.pardir, "e2-decoder-curve", "metrics.json")]
    gate = merge_gates([json.load(open(p)) for p in paths if os.path.exists(p)])
    tune = json.load(open(a.tune)) if os.path.exists(a.tune) else None
    t = table(gate, tune)
    open(a.out_table, "w").write(t + "\n")
    print(t)
    curve(gate, tune, a.out_plot)
