#!/usr/bin/env python3
"""
test_doa_filter.py

Test verschillende DOA-smoothing filters op reverberant LMA data.
Score: side-accuracy = % van de hops dat geschatte linker hoek > 90° EN
       geschatte rechter hoek < 90°.
Rapporteert de beste filter over alle pairs.
"""

# Beste filter: Mediaan N=63 , EN EENVOUDIGST

import sys
import os
import numpy as np
import scipy.linalg
from scipy import signal
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Zorg dat we vanuit deze directory werken, ongeacht waar het script gerund wordt
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import SCENARIO, RIR_PATH as _LUT_NPZ_PATH
DATA_ROOT  = f"data/phase3_audioData/audiodata_batch_1/{SCENARIO}"
NUM_PAIRS  = 15
FS         = 16000
L          = 1024
HOP        = L // 2       # 512 samples = 32 ms per hop
BETA       = 0.85
Q          = 2            # aantal sprekers
PEAK_THRESHOLD = -12.0    # dB drempel voor geldige DOA-piek

TIE_THRESHOLD_DEG = 1.0   # gelijkspeldrempel voor ranking (in graden MAE)


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


def causal_median_filter(seq, N):
    """
    Causaal mediaan filter: enkel de laatste N waarden (geen toekomst).
    """
    out = np.empty(len(seq), dtype=float)
    for i in range(len(seq)):
        out[i] = np.median(seq[max(0, i - N + 1):i + 1])
    return out


def causal_mode_filter(seq, N):
    """
    Causaal modus-filter op de (discrete) LUT-hoeken.
    Kiest de meest voorkomende waarde in het venster — robuuster dan mediaan
    als uitschieters op dezelfde foute hoek clusteren.
    """
    out = np.empty(len(seq), dtype=float)
    for i in range(len(seq)):
        window = seq[max(0, i - N + 1):i + 1]
        vals, counts = np.unique(window, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def causal_percentile_filter(seq_l, seq_r, N, pct_l=75, pct_r=25):
    """
    Directional percentiel-filter:
      • Linkse spreker (hoek > 90°): gebruik pct_l-de percentiel → bias naar hogere hoek.
      • Rechtse spreker (hoek < 90°): gebruik pct_r-de percentiel → bias naar lagere hoek.
    Hierdoor worden uitschieters naar de verkeerde kant extra onderdrukt.
    """
    out_l = np.empty(len(seq_l), dtype=float)
    out_r = np.empty(len(seq_r), dtype=float)
    for i in range(len(seq_l)):
        wl = seq_l[max(0, i - N + 1):i + 1]
        wr = seq_r[max(0, i - N + 1):i + 1]
        out_l[i] = np.percentile(wl, pct_l)
        out_r[i] = np.percentile(wr, pct_r)
    return out_l, out_r


def apply_strategy(raw_l, raw_r, name):
    if name == "Raw":
        return raw_l.copy(), raw_r.copy()
    if name.startswith("Mediaan"):
        N = int(name.split("N=")[1])
        return causal_median_filter(raw_l, N), causal_median_filter(raw_r, N)
    if name.startswith("EMA α="):
        a = float(name.split("α=")[1])
        return ema_filter(raw_l, a), ema_filter(raw_r, a)
    if name.startswith("Mode"):
        N = int(name.split("N=")[1])
        return causal_mode_filter(raw_l, N), causal_mode_filter(raw_r, N)
    if name.startswith("Med") and "+EMA" in name:
        # bijv. "Med31+EMA0.30"
        parts = name.replace("Med", "").split("+EMA")
        N, a  = int(parts[0]), float(parts[1])
        ml    = causal_median_filter(raw_l, N)
        mr    = causal_median_filter(raw_r, N)
        return ema_filter(ml, a), ema_filter(mr, a)
    if name.startswith("Pct"):
        # bijv. "Pct75/25 N=63"  → pct_l=75, pct_r=25
        pct_part, n_part = name.split(" N=")
        pcts = pct_part.replace("Pct", "").split("/")
        pct_l, pct_r = int(pcts[0]), int(pcts[1])
        N = int(n_part)
        return causal_percentile_filter(raw_l, raw_r, N, pct_l, pct_r)
    raise ValueError(f"Onbekende strategie: {name}")


STRATEGIES = [
    # ── Baseline ────────────────────────────────────────────────
    "Raw",

    # ── Causaal mediaan-filter ───────────────────────────────────
    # N = venstergrootte in hops  (1 hop = 32 ms @ 16kHz/512)
    "Mediaan N=3",
    "Mediaan N=7",
    "Mediaan N=15",
    "Mediaan N=31",
    "Mediaan N=63",    # huidig beste
    "Mediaan N=127",   # 2× huidig beste
    "Mediaan N=255",   # 4× huidig beste
    "Mediaan N=511",   # 8× huidig beste  (~16 s aanlooptijd)

    # ── EMA ─────────────────────────────────────────────────────
    # Kleine α = traag/stabiel, grote α = snel/reactief
    "EMA α=0.05",
    "EMA α=0.10",
    "EMA α=0.20",
    "EMA α=0.30",
    "EMA α=0.40",
    "EMA α=0.60",
    "EMA α=0.80",

    # ── Mode (modus) op discrete LUT-hoeken ─────────────────────
    # Kiest de meest-voorkomende hoek in het venster.
    # Beter dan mediaan als uitschieters altijd op dezelfde foute hoek vallen.
    "Mode N=31",
    "Mode N=63",
    "Mode N=127",
    "Mode N=255",

    # ── Cascaded: Mediaan → EMA ──────────────────────────────────
    # Mediaan verwijdert uitschieters; daarna EMA voor extra gladheid.
    "Med31+EMA0.30",
    "Med63+EMA0.20",
    "Med63+EMA0.30",
    "Med127+EMA0.20",

    # ── Directional percentiel-filter ───────────────────────────
    # Linker spreker (>90°): pak hoge percentiel → bias naar grotere hoek.
    # Rechter spreker (<90°): pak lage percentiel → bias naar kleinere hoek.
    # Onderdrukt uitschieters die naar de verkeerde kant springen.
    "Pct75/25 N=31",
    "Pct75/25 N=63",
    "Pct75/25 N=127",
    "Pct80/20 N=63",
    "Pct80/20 N=127",
]


def angle_mae(el, er, gt_l, gt_r):
    """
    Gemiddelde absolute fout (MAE) in graden tussen geschatte en werkelijke hoeken.
    Combineert links én rechts: lagere waarde = betere filter.
    """
    n = min(len(el), len(gt_l), len(er), len(gt_r))
    mae_l = np.mean(np.abs(el[:n] - gt_l[:n]))
    mae_r = np.mean(np.abs(er[:n] - gt_r[:n]))
    return (mae_l + mae_r) / 2.0


# ── hoofd ─────────────────────────────────────────────────────────────────────

def main():
    lut_npz  = _LUT_NPZ_PATH   # uit config.py — zelfde RIR als processor.py
    A_lut, lut_angles = build_music_lut(lut_npz, L)

    pair_results = []  # lijst van (pair_no, {strategy: mae})

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

        maes = {}
        best_name = None
        best_mae  = np.inf
        for name in STRATEGIES:
            el, er  = apply_strategy(raw_l, raw_r, name)
            mae     = angle_mae(el, er, gt_l, gt_r)
            maes[name] = mae
            marker = ""
            if mae < best_mae:
                best_mae  = mae
                best_name = name
                marker = "  ←"
            print(f"  {name:20s}  MAE={mae:5.1f}°{marker}")

        pair_results.append((pair_no, maes))
        print(f"  → best voor paar {pair_no}: {best_name}  (MAE={best_mae:.1f}°)")

    # ── aggregaat ──────────────────────────────────────────────────────────────
    if not pair_results:
        print("Geen resultaten — controleer DATA_ROOT.")
        return

    print("\n" + "=" * 60)
    print("TOTAALOVERZICHT  (lager MAE = beter)")
    print("=" * 60)
    print(f"{'Strategie':<22}  {'Gem.MAE':>9}  {'Med.MAE':>9}")
    print("-" * 44)

    summary = {}
    for name in STRATEGIES:
        vals = [r[name] for _, r in pair_results]
        summary[name] = {"mean": np.mean(vals), "median": np.median(vals), "vals": vals}

    # Laagste gemiddelde MAE wint; mediaan als tiebreak
    sorted_strats = sorted(summary.items(), key=lambda x: (x[1]["mean"], x[1]["median"]))

    for name, s in sorted_strats:
        print(f"  {name:<20}  {s['mean']:>8.1f}°  {s['median']:>8.1f}°")

    # Winnaar: laagste gemiddelde, mediaan als tiebreak
    winner_name, winner_s = sorted_strats[0]
    runner_name, runner_s = sorted_strats[1]
    gap = runner_s["mean"] - winner_s["mean"]

    if gap <= TIE_THRESHOLD_DEG:
        # gelijkspel → kies op mediaan
        if runner_s["median"] < winner_s["median"]:
            winner_name, winner_s = runner_name, runner_s

    print("\n" + "=" * 60)
    print(f"  EINDWINNAAR: '{winner_name}'")
    print(f"  Gemiddelde MAE : {winner_s['mean']:.1f}°")
    print(f"  Mediaan MAE    : {winner_s['median']:.1f}°")
    print("=" * 60)

    # ── plot (horizontale barplot — overzichtelijker bij veel strategieën) ────
    # Sorteer op gemiddelde MAE voor leesbaarheid
    sorted_names  = [n for n, _ in sorted_strats]
    sorted_means  = [summary[n]["mean"]   for n in sorted_names]
    sorted_meds   = [summary[n]["median"] for n in sorted_names]

    fig, ax = plt.subplots(figsize=(11, max(6, len(STRATEGIES) * 0.38)))
    y = np.arange(len(sorted_names))

    bars = ax.barh(y - 0.18, sorted_means, 0.32, label="Gemiddeld MAE", color="steelblue")
    ax.barh(y + 0.18, sorted_meds,  0.32, label="Mediaan MAE",   color="coral", alpha=0.85)

    winner_idx = sorted_names.index(winner_name)
    bars[winner_idx].set_color("gold")
    bars[winner_idx].set_edgecolor("black")
    bars[winner_idx].set_linewidth(1.5)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.invert_yaxis()   # beste bovenaan
    ax.set_xlabel("Hoekfout MAE (°)")
    ax.set_title("DOA filter vergelijking — reverberant (MAE t.o.v. ground truth)\n"
                 f"Winnaar: {winner_name}  (gem. {winner_s['mean']:.1f}°)")
    ax.axvline(summary["Raw"]["mean"], color="grey", linestyle="--", linewidth=0.8, label="Raw baseline")
    ax.legend(fontsize=8)
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
