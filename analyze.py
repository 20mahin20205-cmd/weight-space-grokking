"""Analyze pilot runs: curves + Fourier circuit sanity check.

Usage: python analyze.py runs/pilot_wd1.0_s0 [runs/...]
Outputs: <run>/curves.png, <run>/fourier.png, printed Fourier summary.
"""
import glob, json, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run):
    with open(f"{run}/metrics.json") as f:
        m = json.load(f)
    meta = json.load(open(f"{run}/meta.json"))
    return m, meta


def fourier_power(run, p):
    """Fourier power spectrum of the embedding matrix rows (Nanda-style sanity)."""
    files = sorted(glob.glob(f"{run}/snapshots/step*.pt"))
    if not files:
        return None
    sd = torch.load(files[-1], weights_only=False)
    emb = sd["embed.weight"].float().numpy()[:p]  # (p, d_model), rows = token i
    # FFT along the token axis for each channel
    F = np.abs(np.fft.rfft(emb, axis=0)) ** 2
    power_per_freq = F.sum(axis=1)
    return power_per_freq / power_per_freq.sum(), files[-1]


def main(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in runs:
        m, meta = load(run)
        steps = [x["step"] for x in m]
        ax = axes[0]
        ax.plot(steps, [x["train_acc"] for x in m], label=f"{run} train", ls="-")
        ax.plot(steps, [x["test_acc"] for x in m], label=f"{run} test", ls="--")
        ax2 = axes[1]
        ax2.plot(steps, [np.log10(max(x["test_loss"], 1e-6)) for x in m], ls="--")
        ax2.plot(steps, [np.log10(max(x["train_loss"], 1e-6)) for x in m], ls="-")
        print(f"{run}: T_mem={meta['T_mem']} T_grok={meta['T_grok']} "
              f"delay={meta['delay']} wall={meta['wall_seconds']:.0f}s")
        fp = fourier_power(run, meta["p"])
        if fp:
            power, f_last = fp
            top = np.argsort(power[1:])[-5:] + 1
            print(f"   {f_last}: top freqs {top}, top-4 energy share "
                  f"{np.sort(power[1:])[-4:].sum()/power[1:].sum():.2f}")
            figf, axf = plt.subplots(figsize=(6, 3))
            axf.bar(range(1, len(power)), power[1:])
            axf.set_xlabel("frequency (token-space FFT of embeddings)")
            axf.set_title(f"Fourier power, last snapshot ({run})")
            figf.tight_layout(); figf.savefig(f"{run}/fourier.png", dpi=140)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=7)
    axes[0].set_title("Accuracy")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("log10 loss"); axes[1].set_title("Loss")
    fig.tight_layout(); fig.savefig("runs/curves.png", dpi=140)
    print("wrote runs/curves.png")


if __name__ == "__main__":
    main(sys.argv[1:])
