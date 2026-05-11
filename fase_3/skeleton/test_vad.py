#!/usr/bin/env python3
"""
test_vad.py

Test verschillende VAD-parametercombinaties (alpha_up, alpha_down, vad_threshold)
op de reverberant LMA gt-signalen.

Ground truth: anechoische gt-wav, percentile-gebaseerde spraak/stilte.
Score: F1-score op spraakdetectie.

Test op alle 15 pairs × 2 kanten (links/rechts) = 30 signalen.
"""

import os
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuratie ─────────────────────────────────────────────────────────────
ANEC_ROOT = "data/phase3_audioData/audiodata_batch_1/anechoic"
REV_ROOT  = "data/phase3_audioData/audiodata_batch_1/reverberant"

NUM_PAIRS = 15
FS        = 16000
HOP       = 512                  # zelfde als processor.py (L=1024, hop=L/2)

# Percentile-drempel voor anechoic-GT: hops met RMS onder deze percentile = stilte
GT_SILENCE_PERCENTILE = 30       # bottom 30% = stilte (default)

# Parameter sweep
ALPHA_UP_GRID    = [0.90, 0.95, 0.98, 0.99]
ALPHA_DOWN_GRID  = [0.5, 0.7, 0.8, 0.9]
THRESHOLD_GRID   = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

OUT_PLOT = "vad_filter_vergelijking.png"


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_hop_rms(signal_1d):
    """RMS per hop van een 1D audiosignaal."""
    n_hops = len(signal_1d) // HOP
    s = signal_1d[: n_hops * HOP].reshape(n_hops, HOP).astype(np.float64)
    return np.sqrt(np.mean(s ** 2, axis=1))


def derive_gt_labels(anec_rms, percentile=GT_SILENCE_PERCENTILE):
    """Binaire spraak/stilte labels: True = spraak (RMS > percentile-drempel)."""
    threshold = np.percentile(anec_rms, percentile)
    return anec_rms > threshold


def run_vad(rev_rms, alpha_up, alpha_down, vad_threshold):
    """
    Run de exacte VAD-logica uit processor.py op een sequentie van RMS-waarden.
    Returns: binary array, True = spraak gedetecteerd.
    """
    noise_floor = None
    vad_out     = np.zeros(len(rev_rms), dtype=bool)
    for i, rms in enumerate(rev_rms):
        if noise_floor is None:
            noise_floor = rms
        elif rms < noise_floor:
            noise_floor = alpha_down * noise_floor + (1 - alpha_down) * rms
        else:
            noise_floor = alpha_up * noise_floor + (1 - alpha_up) * rms
        vad_out[i] = rms > vad_threshold * noise_floor
    return vad_out


def f1_score(pred, gt):
    """F1-score voor binaire klassen (spraak = positieve klasse)."""
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_pair_signals(pair_no):
    """
    Laad anechoic+reverberant left/right gt-wavs voor dit paar.
    Returns: (anec_l, rev_l, anec_r, rev_r) als 1D arrays (channel 0).
    Returns None als bestanden ontbreken.
    """
    anec_dir = os.path.join(ANEC_ROOT, f"pair{pair_no}")
    rev_dir  = os.path.join(REV_ROOT,  f"pair{pair_no}")

    files = {
        "anec_l": os.path.join(anec_dir, "leftSpeaker_LMA.wav"),
        "rev_l":  os.path.join(rev_dir,  "leftSpeaker_LMA.wav"),
        "anec_r": os.path.join(anec_dir, "rightSpeaker_LMA.wav"),
        "rev_r":  os.path.join(rev_dir,  "rightSpeaker_LMA.wav"),
    }
    if any(not os.path.exists(p) for p in files.values()):
        return None

    out = {}
    for k, p in files.items():
        _, sig = wavfile.read(p)
        # gebruik channel 0 (zelfde als processor.py: audio_buffer_gt0[:, 0])
        if sig.ndim > 1:
            sig = sig[:, 0]
        out[k] = sig.astype(np.float64)
    return out["anec_l"], out["rev_l"], out["anec_r"], out["rev_r"]


# ── Hoofd ─────────────────────────────────────────────────────────────────────

def main():
    # 1) Verzamel per pair × kant: (anec_rms, rev_rms, gt_labels)
    signals = []  # list of (label, gt, rev_rms)
    for pair_no in range(1, NUM_PAIRS + 1):
        loaded = load_pair_signals(pair_no)
        if loaded is None:
            print(f"Pair {pair_no}: data ontbreekt, overgeslagen.")
            continue
        anec_l, rev_l, anec_r, rev_r = loaded

        # truncate beide naar dezelfde lengte (vaak iets verschillend door reverb-tails)
        n_l = min(len(anec_l), len(rev_l))
        n_r = min(len(anec_r), len(rev_r))

        anec_l_rms = compute_hop_rms(anec_l[:n_l])
        rev_l_rms  = compute_hop_rms(rev_l[:n_l])
        anec_r_rms = compute_hop_rms(anec_r[:n_r])
        rev_r_rms  = compute_hop_rms(rev_r[:n_r])

        gt_l = derive_gt_labels(anec_l_rms)
        gt_r = derive_gt_labels(anec_r_rms)

        # Truncate naar gelijke hop-aantal
        n_hops_l = min(len(rev_l_rms), len(gt_l))
        n_hops_r = min(len(rev_r_rms), len(gt_r))

        signals.append((f"pair{pair_no}-L", gt_l[:n_hops_l], rev_l_rms[:n_hops_l]))
        signals.append((f"pair{pair_no}-R", gt_r[:n_hops_r], rev_r_rms[:n_hops_r]))

        print(f"Pair {pair_no}: L spraak={gt_l.mean()*100:.1f}%, R spraak={gt_r.mean()*100:.1f}%  "
              f"(L hops={n_hops_l}, R hops={n_hops_r})")

    if not signals:
        print("Geen data gevonden — controleer paden.")
        return

    print(f"\n{len(signals)} signalen geladen ({NUM_PAIRS} pairs × 2 kanten).")
    print(f"Parameter sweep: {len(ALPHA_UP_GRID)} × {len(ALPHA_DOWN_GRID)} × {len(THRESHOLD_GRID)} = "
          f"{len(ALPHA_UP_GRID)*len(ALPHA_DOWN_GRID)*len(THRESHOLD_GRID)} combinaties.")
    print(f"\nLogica: spraak = RMS > vad_threshold · noise_floor")
    print(f"        noise_floor stijgt langzaam (alpha_up), daalt snel (alpha_down)")
    print(f"\nLopen...")

    # 2) Parameter sweep
    results = []  # list of (au, ad, th, f1_mean, f1_median, f1_per_signal)
    for au in ALPHA_UP_GRID:
        for ad in ALPHA_DOWN_GRID:
            for th in THRESHOLD_GRID:
                f1s = []
                for _, gt, rev_rms in signals:
                    pred = run_vad(rev_rms, au, ad, th)
                    f1s.append(f1_score(pred, gt))
                results.append((au, ad, th, np.mean(f1s), np.median(f1s), f1s))

    # 3) Ranking
    results.sort(key=lambda r: (-r[3], -r[4]))  # gem F1 desc, mediaan als tiebreak

    # 4) Print tabel (top-10 + bottom-3)
    print("\n" + "=" * 76)
    print("TOP 10 — beste parameter-combinaties (gerangschikt op gem. F1)")
    print("=" * 76)
    print(f"{'rank':>4}  {'α_up':>6}  {'α_down':>7}  {'thresh':>7}  {'gem F1':>8}  {'med F1':>8}")
    print("-" * 76)
    for i, (au, ad, th, mean_f1, med_f1, _) in enumerate(results[:10], 1):
        marker = "  ←" if i == 1 else ""
        print(f"{i:>4d}  {au:>6.2f}  {ad:>7.2f}  {th:>7.2f}  {mean_f1*100:>7.1f}%  {med_f1*100:>7.1f}%{marker}")

    print("\n" + "=" * 76)
    print("BOTTOM 3 — slechtste combinaties (ter contrast)")
    print("=" * 76)
    print(f"{'rank':>4}  {'α_up':>6}  {'α_down':>7}  {'thresh':>7}  {'gem F1':>8}  {'med F1':>8}")
    print("-" * 76)
    for i, (au, ad, th, mean_f1, med_f1, _) in enumerate(results[-3:], len(results) - 2):
        print(f"{i:>4d}  {au:>6.2f}  {ad:>7.2f}  {th:>7.2f}  {mean_f1*100:>7.1f}%  {med_f1*100:>7.1f}%")

    # 5) Winnaar
    win_au, win_ad, win_th, win_mean, win_med, _ = results[0]

    print("\n" + "=" * 76)
    print(f"  EINDWINNAAR")
    print(f"  alpha_up      = {win_au}")
    print(f"  alpha_down    = {win_ad}")
    print(f"  vad_threshold = {win_th}")
    print(f"  Gemiddelde F1 : {win_mean*100:.1f}%")
    print(f"  Mediaan F1    : {win_med*100:.1f}%")
    print("=" * 76)

    # 6) Plot heatmap voor de optimale threshold
    print(f"\nPlot maken voor optimale threshold ({win_th})...")
    heat = np.zeros((len(ALPHA_UP_GRID), len(ALPHA_DOWN_GRID)))
    for au, ad, th, mean_f1, _, _ in results:
        if th == win_th:
            i_au = ALPHA_UP_GRID.index(au)
            i_ad = ALPHA_DOWN_GRID.index(ad)
            heat[i_au, i_ad] = mean_f1 * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heat, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(ALPHA_DOWN_GRID)))
    ax.set_xticklabels([f"{a}" for a in ALPHA_DOWN_GRID])
    ax.set_yticks(range(len(ALPHA_UP_GRID)))
    ax.set_yticklabels([f"{a}" for a in ALPHA_UP_GRID])
    ax.set_xlabel("alpha_down (snelle daling)")
    ax.set_ylabel("alpha_up (langzame stijging)")
    ax.set_title(f"VAD F1-score (%) bij threshold={win_th}")

    # annoteer cellen
    for i in range(len(ALPHA_UP_GRID)):
        for j in range(len(ALPHA_DOWN_GRID)):
            ax.text(j, i, f"{heat[i,j]:.1f}", ha="center", va="center",
                    color="white" if heat[i,j] < heat.max() * 0.7 else "black",
                    fontsize=9)

    # markeer winnaar
    win_i = ALPHA_UP_GRID.index(win_au)
    win_j = ALPHA_DOWN_GRID.index(win_ad)
    ax.add_patch(plt.Rectangle((win_j - 0.5, win_i - 0.5), 1, 1,
                                fill=False, edgecolor="red", linewidth=3))

    plt.colorbar(im, ax=ax, label="Gem. F1 (%)")
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"Plot opgeslagen als: {OUT_PLOT}")

    # 7) Advies voor processor.py
    print("\n>>> Implementeer in processor.py (rond regels 230-236):")
    print(f"    self.alpha_up      = {win_au}")
    print(f"    self.alpha_down    = {win_ad}")
    print(f"    self.vad_threshold = {win_th}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
