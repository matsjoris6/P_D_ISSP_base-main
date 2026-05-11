#!/usr/bin/env python3
"""
test_doa_filter.py

Test verschillende DOA-smoothing filters op reverberant LMA data.
Score: side-accuracy = % van de hops dat geschatte linker hoek > 90° EN
       geschatte rechter hoek < 90°.
Rapporteert de beste filter over alle pairs.
"""

import sys
import os
import numpy as np
import scipy.linalg
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT  = "data/phase3_audioData/audiodata_batch_1/reverberant"
NUM_PAIRS  = 15
FS         = 16000
L          = 1024
HOP        = L // 2       # 512 samples = 32 ms per hop
BETA       = 0.85
Q          = 2            # aantal sprekers
PEAK_THRESHOLD = -12.0    # dB drempel voor geldige DOA-piek

TIE_THRESHOLD_PCT = 1.0   # gelijkspeldrempel voor ranking


# ── helpers ──────────────────────────────────────────────────────────────────

def build_music_lut(npz_path, L):
    d     = np.load(npz_path)
    rirs  = d["rirs"]
    theta = d["thetas"]
    num_bins = L // 2 + 1
    M        = rirs.shape[1]
    A_lut    = np.zeros((num_bins, M, len(theta)), dtype=complex)
    for i in range(len(theta)):
        H = np.fft.rfft(rirs[:, :, i], n=L, axis=0)
        for k in range(num_bins):
            h = H[k, :].reshape(M, 1)
            A1 = h[0, 0]
            h  = h / (A1 if np.abs(A1) > 1e-12 else A1 + 1e-12)
            A_lut[k, :, i] = h.flatten()
    return A_lut, theta


def estimate_doa(lma, lut_angles, A_lut):
    """
    Streaming MUSIC DOA-schatting per hop.
    Returns two (N_hops,) arrays: angles_left, angles_right.
    """
    num_bins   = L // 2 + 1
    M          = lma.shape[1]
    win        = np.sqrt(signal.windows.hann(L, sym=False))
    Ryy        = np.zeros((num_bins, M, M), dtype=complex)
    buf        = np.zeros((L, M))
    valid_k    = np.arange(1, L // 2)
    left_mask  = lut_angles > 90.0
    right_mask = lut_angles <= 90.0

    n_hops = max(0, (lma.shape[0] - L) // HOP + 1)
    al = np.zeros(n_hops)
    ar = np.zeros(n_hops)
    last_l = lut_angles[left_mask][len(lut_angles[left_mask]) // 2]
    last_r = lut_angles[right_mask][len(lut_angles[right_mask]) // 2]

    for i in range(n_hops):
        hop     = lma[i * HOP: i * HOP + HOP]
        buf     = np.roll(buf, -HOP, axis=0)
        buf[-HOP:] = hop

        frame_fft = np.fft.rfft(buf * win[:, None], n=L, axis=0)
        Y         = frame_fft[valid_k, :, None]
        Ryy[valid_k] = BETA * Ryy[valid_k] + (1 - BETA) * (Y @ Y.conj().transpose(0, 2, 1))

        _, evecs = np.linalg.eigh(Ryy[valid_k])
        En       = evecs[:, :, :M - Q]
        EnHA     = En.conj().transpose(0, 2, 1) @ A_lut[valid_k]
        spec_db  = 10 * np.log10(
            np.exp(np.mean(np.log(np.clip(1.0 / np.sum(np.abs(EnHA) ** 2, axis=1), 1e-10, None)), axis=0))
        )
        spec_db -= spec_db.max()

        bli = np.argmax(np.where(left_mask,  spec_db, -np.inf))
        bri = np.argmax(np.where(right_mask, spec_db, -np.inf))
        if spec_db[bli] > PEAK_THRESHOLD:
            last_l = lut_angles[bli]
        if spec_db[bri] > PEAK_THRESHOLD:
            last_r = lut_angles[bri]
        al[i] = last_l
        ar[i] = last_r

    return al, ar


def gt_to_hops(gt_npz, n_hops):
    """Vertaal sample-level GT naar hop-level arrays."""
    gt    = np.load(gt_npz)
    dur_l = np.diff(np.insert(gt["endSamples_l"], 0, 0))
    dur_r = np.diff(np.insert(gt["endSamples_r"], 0, 0))
    gl    = np.concatenate([np.repeat(a, n) for a, n in zip(gt["angles_l"], dur_l)])
    gr    = np.concatenate([np.repeat(a, n) for a, n in zip(gt["angles_r"], dur_r)])
    idx   = np.minimum(np.arange(n_hops) * HOP + HOP // 2, len(gl) - 1)
    return gl[idx], gr[idx]


# ── filters ───────────────────────────────────────────────────────────────────

def ema_filter(seq, alpha):
    out    = np.empty_like(seq)
    out[0] = seq[0]
    for i in range(1, len(seq)):
        out[i] = alpha * seq[i] + (1 - alpha) * out[i - 1]
    return out


def apply_strategy(raw_l, raw_r, name):
    if name == "Raw":
        return raw_l.copy(), raw_r.copy()
    if name.startswith("Mediaan"):
        N = int(name.split("N=")[1])
        return median_filter(raw_l, size=N, mode="nearest"), \
               median_filter(raw_r, size=N, mode="nearest")
    if name.startswith("EMA"):
        a = float(name.split("α=")[1])
        return ema_filter(raw_l, a), ema_filter(raw_r, a)
    raise ValueError(name)


STRATEGIES = [
    "Raw",
    "Mediaan N=3",
    "Mediaan N=7",
    "Mediaan N=15",
    "Mediaan N=31",
    "Mediaan N=63",
    "EMA α=0.2",
    "EMA α=0.4",
    "EMA α=0.6",
    "EMA α=0.8",
]


def side_accuracy(el, er):
    """% van de hops dat geschatte linker hoek > 90° EN rechter hoek ≤ 90°."""
    return np.mean((el > 90.0) & (er <= 90.0)) * 100


# ── hoofd ─────────────────────────────────────────────────────────────────────

def main():
    lut_npz  = os.path.join(DATA_ROOT, "lma_16kHz_200ms.npz")
    A_lut, lut_angles = build_music_lut(lut_npz, L)

    pair_results = []  # lijst van (pair_no, {strategy: acc})

    for pair_no in range(1, NUM_PAIRS + 1):
        pair_dir = os.path.join(DATA_ROOT, f"pair{pair_no}")
        wav_path = os.path.join(pair_dir, "mixture_LMA.wav")
        gt_path  = os.path.join(pair_dir, "gt.npz")

        if not os.path.exists(wav_path) or not os.path.exists(gt_path):
            print(f"Pair {pair_no}: data ontbreekt, overgeslagen.")
            continue

        print(f"\nPaar {pair_no}/{NUM_PAIRS} — DOA berekenen ...", flush=True)
        _, lma = wavfile.read(wav_path)
        lma    = lma.astype(np.float32)

        raw_l, raw_r = estimate_doa(lma, lut_angles, A_lut)
        n_hops       = len(raw_l)

        gt_l, gt_r = gt_to_hops(gt_path, n_hops)

        accs = {}
        best_name = None
        best_acc  = -1
        for name in STRATEGIES:
            el, er    = apply_strategy(raw_l, raw_r, name)
            acc       = side_accuracy(el, er)
            accs[name] = acc
            marker = ""
            if acc > best_acc:
                best_acc  = acc
                best_name = name
                marker = "  ←"
            print(f"  {name:20s}  acc={acc:.1f}%{marker}")

        pair_results.append((pair_no, accs))
        print(f"  → best voor paar {pair_no}: {best_name}  ({best_acc:.1f}%)")

    # ── aggregaat ──────────────────────────────────────────────────────────────
    if not pair_results:
        print("Geen resultaten — controleer DATA_ROOT.")
        return

    print("\n" + "=" * 60)
    print("TOTAALOVERZICHT")
    print("=" * 60)
    print(f"{'Strategie':<22}  {'Gem.acc':>8}  {'Med.acc':>8}")
    print("-" * 42)

    summary = {}
    for name in STRATEGIES:
        vals = [r[name] for _, r in pair_results]
        summary[name] = {"mean": np.mean(vals), "median": np.median(vals), "vals": vals}

    sorted_strats = sorted(summary.items(), key=lambda x: (-x[1]["mean"], -x[1]["median"]))

    for name, s in sorted_strats:
        print(f"  {name:<20}  {s['mean']:>7.1f}%  {s['median']:>7.1f}%")

    # Winnaar: hoogste gemiddelde, mediaan als tiebreak
    winner_name, winner_s = sorted_strats[0]
    runner_name, runner_s = sorted_strats[1]
    gap = winner_s["mean"] - runner_s["mean"]

    if gap <= TIE_THRESHOLD_PCT:
        # gelijkspel → kies op mediaan
        if runner_s["median"] > winner_s["median"]:
            winner_name, winner_s = runner_name, runner_s

    print("\n" + "=" * 60)
    print(f"  EINDWINNAAR: '{winner_name}'")
    print(f"  Gemiddelde side-accuracy : {winner_s['mean']:.1f}%")
    print(f"  Mediaan side-accuracy    : {winner_s['median']:.1f}%")
    print("=" * 60)

    # ── plot ──────────────────────────────────────────────────────────────────
    means   = [summary[n]["mean"]   for n in STRATEGIES]
    medians = [summary[n]["median"] for n in STRATEGIES]
    x       = np.arange(len(STRATEGIES))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x - 0.2, means,   0.35, label="Gemiddeld", color="steelblue")
    ax.bar(x + 0.2, medians, 0.35, label="Mediaan",   color="coral")

    winner_idx = STRATEGIES.index(winner_name)
    bars[winner_idx].set_color("gold")
    bars[winner_idx].set_edgecolor("black")
    bars[winner_idx].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES, rotation=30, ha="right")
    ax.set_ylabel("Side-accuracy (%)")
    ax.set_title("DOA filter vergelijking — reverberant (side-accuracy)")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.axhline(summary["Raw"]["mean"], color="grey", linestyle="--", linewidth=0.8, label="Raw baseline")
    plt.tight_layout()
    out = "doa_filter_vergelijking.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot opgeslagen als: {out}")

    # ── advies voor processor.py ──────────────────────────────────────────────
    print(f"\n>>> Beste filter: {winner_name}")
    print(">>> Pas dit aan in processor.py in de methode processing_microarray,")
    print("    na de MUSIC-berekening (last_angle_left / last_angle_right).")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
