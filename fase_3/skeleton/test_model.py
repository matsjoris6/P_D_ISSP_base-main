"""
test_model.py  —  vergelijk alle 4 modellen op ALLE proefpersonen

Geteste configuraties
─────────────────────
  dilated_5s   hop=1s, hop=2s
  hybrid_3s    hop=1s
  hybrid_5s    hop=1s, hop=2s
  hybrid_10s   hop=1s, hop=2s, hop=5s

Efficiëntie
────────────
  • Enveloppen worden één keer berekend per audiopaar (cache).
  • Per model + subject: batch-inferentie van ALLE vensters bij hop=1s
    → daarna subsampling voor grotere hops (geen extra model-calls nodig).

Output
──────
  • Console-tabel (raw + EMA α=0.3, gesorteerd op raw gem. accuracy)
  • model_vergelijking.png  (staafdiagram)

Gebruik
───────
  python test_model.py                   # volledige run (~30-90 min)
  python test_model.py --max_sec 60      # snelle test: eerste 60s per opname
"""

import os, sys, glob, warnings, time, argparse
warnings.filterwarnings("ignore")
import logging
logging.getLogger("brian2").setLevel(logging.ERROR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from math import gcd
from collections import defaultdict

import brian2
brian2.prefs.codegen.target = "cython"
from brian2 import Hz, kHz
from brian2hears import Sound, erbspace, Gammatone, Filterbank

import tensorflow as tf

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from config import MODELS

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATIE
# ════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--max_sec", type=int, default=None,
                    help="Max seconden per opname te verwerken (None = alles)")
args = parser.parse_args()

DATA_DIR    = "data/data_phase3"
STIMULI_DIR = "data/data_phase3/stimuli"
TARGET_FS   = 64
EEG_FS_IN   = 128
AUDIO_FS_IN = 48000
MAX_SEC     = args.max_sec
EMA_ALPHA   = 0.3          # voor de EMA-kolom in het rapport

OUTPUT_PNG  = "model_vergelijking.png"

# Model × hop combinaties om te testen
# hybrid_10s hop=1s weggelaten: te traag voor real-time én conceptueel niet zinvol
# hybrid_3s  hop=0.5s toegevoegd: snel model kan vaker updaten
TEST_CONFIGS = [
    ("dilated_5s",  1),
    ("dilated_5s",  2),
    ("hybrid_3s",   0.5),
    ("hybrid_3s",   1),
    ("hybrid_5s",   1),
    ("hybrid_5s",   2),
    ("hybrid_10s",  2),
    ("hybrid_10s",  5),
]

# Welk wav-bestand is LINKS voor elk audiopaar
LEFTRIGHT_MAPPING = {
    "audiobook_1_part2.wav":     "left",   "audiobook_2_2_part2.wav":   "right",
    "podcast_3_part2.wav":       "left",   "podcast_4_part2.wav":       "right",
    "audiobook_8_2_part2.wav":   "left",   "audiobook_8_1_part2.wav":   "right",
    "audiobook_9_1_part2.wav":   "left",   "audiobook_9_2_part2.wav":   "right",
    "audiobook_10_1_part2.wav":  "left",   "audiobook_10_2_part2.wav":  "right",
    "audiobook_11_2_part2.wav":  "left",   "audiobook_11_1_part2.wav":  "right",
    "podcast_22_part2.wav":      "left",   "podcast_21_part2.wav":      "right",
    "podcast_24_part2.wav":      "left",   "podcast_25_part2.wav":      "right",
    "podcast_30_part2.wav":      "left",   "podcast_31_part2.wav":      "right",
    "audiobook_14_2_part2.wav":  "left",   "podcast_32_part2.wav":      "right",
    "podcast_33_part2.wav":      "left",   "audiobook_14_1_part2.wav":  "right",
    "podcast_34_part2.wav":      "right",
    "podcast_36_part2.wav":      "left",   "podcast_35_part2.wav":      "right",
    "podcast_37_part2.wav":      "right",
}

# ════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (zelfde als processor.py)
# ════════════════════════════════════════════════════════════════════════════

class _EnvFilterbank(Filterbank):
    def __init__(self, source):
        super().__init__(source)
        self.nchannels = 1
    def buffer_apply(self, inp):
        return np.sum(np.abs(inp) ** 0.6, axis=1, keepdims=True)


def compute_audio_envelope(audio, sr_in=AUDIO_FS_IN, sr_out=TARGET_FS,
                            lowcut=1.0, highcut=32.0):
    brian2.start_scope()
    sound = Sound(audio.reshape(-1, 1).astype(np.float32), samplerate=sr_in * Hz)
    cf    = erbspace(50 * Hz, 5 * kHz, 28)
    env   = _EnvFilterbank(Gammatone(sound, cf)).process().flatten()
    sos   = signal.butter(4, [lowcut, highcut], btype="bandpass", fs=sr_in, output="sos")
    env   = signal.sosfiltfilt(sos, env)
    g     = gcd(int(sr_in), sr_out)
    return signal.resample_poly(env, sr_out // g, int(sr_in) // g)


def preprocess_eeg(eeg, fs_in=EEG_FS_IN, fs_out=TARGET_FS,
                   lowcut=1.0, highcut=32.0):
    sos = signal.butter(4, [lowcut, highcut], btype="bandpass", fs=fs_in, output="sos")
    eeg_f = signal.sosfiltfilt(sos, eeg, axis=0)
    g = gcd(int(fs_in), fs_out)
    return signal.resample_poly(eeg_f, fs_out // g, int(fs_in) // g, axis=0)


# ════════════════════════════════════════════════════════════════════════════
#  FILTERS  (post-processing op ruwe kansen)
# ════════════════════════════════════════════════════════════════════════════

def apply_ema(probs, alpha=EMA_ALPHA):
    """EMA smoothing + drempel 0.5."""
    if len(probs) == 0:
        return np.array([], dtype=int)
    e = float(probs[0])
    out = []
    for p in probs:
        e = alpha * float(p) + (1 - alpha) * e
        out.append(1 if e >= 0.5 else 0)
    return np.array(out, dtype=int)


def apply_raw(probs):
    """Geen filter: directe drempel 0.5."""
    return (np.asarray(probs) >= 0.5).astype(int)


# ════════════════════════════════════════════════════════════════════════════
#  INFERENTIE PER SUBJECT  (batch op hop=1s, daarna subsampling)
# ════════════════════════════════════════════════════════════════════════════

def run_subject_batch(eeg_proc, env_left_full, env_right_full,
                      gt_ds, model, win_samples, hop_sec):
    """
    Batch-inferentie voor één subject bij een gegeven hop_sec (mag float, bv. 0.5).
    Geeft (raw_probs, gt_arr) terug, of None als er te weinig vensters zijn.

    Convention model output:
      pred[0,0] ≈ 1  →  attending LEFT   (gt_left = 1)
      pred[0,0] ≈ 0  →  attending RIGHT  (gt_left = 0)
    """
    step  = int(round(hop_sec * TARGET_FS))   # samples per hop (bv. 0.5s → 32)
    n_env = min(len(env_left_full), len(env_right_full), len(eeg_proc))
    n_win = max(0, (n_env - win_samples) // step + 1)

    if n_win < 2:
        return None

    eeg_b   = np.zeros((n_win, win_samples, eeg_proc.shape[1]), dtype=np.float32)
    env_l_b = np.zeros((n_win, win_samples, 1),                 dtype=np.float32)
    env_r_b = np.zeros((n_win, win_samples, 1),                 dtype=np.float32)
    gt_arr  = np.zeros(n_win, dtype=int)

    valid = 0
    for w in range(n_win):
        s = w * step
        e = s + win_samples
        if e > n_env:
            break
        eeg_b[valid]         = eeg_proc[s:e]
        env_l_b[valid, :, 0] = env_left_full[s:e]
        env_r_b[valid, :, 0] = env_right_full[s:e]
        gt_arr[valid]        = int(round(float(np.mean(gt_ds[s:e]))))
        valid += 1

    if valid < 2:
        return None

    preds = model([eeg_b[:valid], env_l_b[:valid], env_r_b[:valid]], training=False)
    probs = preds[:, 0].numpy().astype(np.float32)
    return probs, gt_arr[:valid]


# ════════════════════════════════════════════════════════════════════════════
#  HOOFDPROGRAMMA
# ════════════════════════════════════════════════════════════════════════════

def main():
    W = 70
    print(f"\n{'='*W}")
    print(f"  MODEL VERGELIJKING  —  test_model.py")
    if MAX_SEC:
        print(f"  ⚠  MAX_SEC={MAX_SEC}s  (niet de volledige opname)")
    print(f"{'='*W}\n")

    all_eeg_files = sorted(glob.glob(os.path.join(DATA_DIR, "sub-*", "*.npz")))
    print(f"  {len(all_eeg_files)} proefpersoon-bestanden gevonden.")

    # Groepeer per audiopaar zodat envelops slechts 1× berekend worden
    pair_groups = defaultdict(list)
    for f in all_eeg_files:
        npz = np.load(f)
        key = (str(npz["stimulus_0"]), str(npz["stimulus_1"]))
        pair_groups[key].append(f)
    print(f"  {len(pair_groups)} unieke audioparen.\n")

    # Bepaal welke modellen en hops we nodig hebben
    model_to_hops = defaultdict(set)
    for model_name, hop_sec in TEST_CONFIGS:
        model_to_hops[model_name].add(hop_sec)

    # Resultaten: {(model_name, hop_sec): {"raw": [acc,...], "ema": [acc,...]}}
    results = {cfg: {"raw": [], "ema": []} for cfg in TEST_CONFIGS}

    # ── Loop over modellen ────────────────────────────────────────────────────
    for model_name, test_hops in model_to_hops.items():
        m_cfg      = MODELS[model_name]
        win_samples = m_cfg["eeg_window_samples"]
        print(f"\n{'─'*W}")
        print(f"  Model: {model_name}  —  {m_cfg['description']}")
        print(f"  Venster: {m_cfg['window_sec']}s  ({win_samples} samples @ {TARGET_FS} Hz)")
        print(f"  Te testen hops: {sorted(test_hops)}s")
        print(f"{'─'*W}")

        model = tf.keras.models.load_model(m_cfg["model_path"])
        print(f"  Model geladen: {m_cfg['model_path']}")

        # Envelop cache voor dit model-batch
        env_cache = {}   # (stim0, stim1) → (env_left_full, env_right_full)

        n_total = sum(len(fs) for fs in pair_groups.values())
        n_done  = 0

        for (stim0, stim1), eeg_files in sorted(pair_groups.items()):
            # ── Enveloppen laden / berekenen ──────────────────────────────────
            if (stim0, stim1) not in env_cache:
                path0 = os.path.join(STIMULI_DIR, stim0)
                path1 = os.path.join(STIMULI_DIR, stim1)
                if not os.path.exists(path0) or not os.path.exists(path1):
                    print(f"  [SKIP] Stimuli niet gevonden: {stim0} / {stim1}")
                    n_done += len(eeg_files)
                    continue
                t_env = time.time()
                _, aud0 = wavfile.read(path0)
                _, aud1 = wavfile.read(path1)
                if MAX_SEC:
                    aud0 = aud0[:MAX_SEC * AUDIO_FS_IN]
                    aud1 = aud1[:MAX_SEC * AUDIO_FS_IN]
                env0 = compute_audio_envelope(aud0.astype(np.float32))
                env1 = compute_audio_envelope(aud1.astype(np.float32))

                # links / rechts toewijzen
                left0 = LEFTRIGHT_MAPPING.get(stim0, "left")
                if left0 == "left":
                    env_left_full, env_right_full = env0, env1
                else:
                    env_left_full, env_right_full = env1, env0

                env_cache[(stim0, stim1)] = (env_left_full, env_right_full)
                print(f"  Envelop {stim0[:30]:<30} berekend in {time.time()-t_env:.0f}s")
            else:
                env_left_full, env_right_full = env_cache[(stim0, stim1)]

            # ── EEG-bestanden voor dit audiopaar ──────────────────────────────
            for eeg_file in sorted(eeg_files):
                n_done += 1
                npz    = np.load(eeg_file)
                stim0_ = str(npz["stimulus_0"])
                raw_gt = npz["attended_speaker"].astype(int)

                total_eeg = npz["eeg"].shape[0]
                if MAX_SEC:
                    total_eeg = min(total_eeg, MAX_SEC * EEG_FS_IN)

                # EEG preprocessing: één keer per subject, hergebruikt voor alle hops
                eeg_proc = preprocess_eeg(npz["eeg"][:total_eeg].astype(np.float64))

                # GT in "links-geattendeerd"-conventie
                left0_    = LEFTRIGHT_MAPPING.get(stim0_, "left")
                swap      = (left0_ != "left")
                gt_raw    = raw_gt[:total_eeg]
                gt_left   = gt_raw if swap else 1 - gt_raw
                ratio     = EEG_FS_IN // TARGET_FS
                gt_ds_len = int(total_eeg * TARGET_FS / EEG_FS_IN)
                gt_ds = np.array([
                    round(float(np.mean(gt_left[i*ratio:(i+1)*ratio])))
                    for i in range(gt_ds_len)
                ])

                subj = os.path.basename(eeg_file).split("_")[0]
                hop_line = []

                # Batch inferentie per hop (aparte aanroep per hop_sec)
                for hop_sec in sorted(test_hops):
                    res = run_subject_batch(
                        eeg_proc, env_left_full, env_right_full,
                        gt_ds, model, win_samples, hop_sec
                    )
                    if res is None:
                        continue
                    probs_hop, gt_hop = res
                    raw_acc = np.mean(apply_raw(probs_hop) == gt_hop) * 100
                    ema_acc = np.mean(apply_ema(probs_hop) == gt_hop) * 100
                    results[(model_name, hop_sec)]["raw"].append(raw_acc)
                    results[(model_name, hop_sec)]["ema"].append(ema_acc)
                    hop_line.append(f"hop={hop_sec}s raw={raw_acc:.1f}% ema={ema_acc:.1f}%")

                print(f"  [{n_done:>3}/{n_total}]  {subj:<10}  " + "  ".join(hop_line))

        del model   # vrij GPU geheugen voor volgend model
        tf.keras.backend.clear_session()

    # ════════════════════════════════════════════════════════════════════════
    #  RAPPORT
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*W}")
    print(f"  EINDRESULTATEN")
    print(f"{'='*W}")
    print(f"  {'Configuratie':<22}  {'N':>4}  "
          f"{'Raw gem':>8}  {'Raw med':>8}  {'EMA gem':>8}  {'EMA med':>8}")
    print(f"  {'─'*66}")

    rows = []
    for (model_name, hop_sec) in TEST_CONFIGS:
        raw_list = results[(model_name, hop_sec)]["raw"]
        ema_list = results[(model_name, hop_sec)]["ema"]
        if not raw_list:
            continue
        raw_arr = np.array(raw_list)
        ema_arr = np.array(ema_list)
        rows.append({
            "label":    f"{model_name}  hop={hop_sec}s",
            "n":        len(raw_arr),
            "raw_gem":  np.mean(raw_arr),
            "raw_med":  np.median(raw_arr),
            "ema_gem":  np.mean(ema_arr),
            "ema_med":  np.median(ema_arr),
        })

    # Sorteer op raw gem (hoogste eerst)
    rows.sort(key=lambda r: r["raw_gem"], reverse=True)

    for i, r in enumerate(rows):
        marker = "  ← BESTE" if i == 0 else ""
        print(f"  {r['label']:<22}  {r['n']:>4}  "
              f"{r['raw_gem']:>7.1f}%  {r['raw_med']:>7.1f}%  "
              f"{r['ema_gem']:>7.1f}%  {r['ema_med']:>7.1f}%{marker}")

    print(f"{'='*W}")
    if rows:
        best_raw = rows[0]
        best_ema = max(rows, key=lambda r: r["ema_gem"])
        print(f"\n  WINNAAR (raw gem accuracy) : {best_raw['label']}  "
              f"→  {best_raw['raw_gem']:.1f}%")
        print(f"  WINNAAR (EMA gem accuracy) : {best_ema['label']}  "
              f"→  {best_ema['ema_gem']:.1f}%")
    print(f"{'='*W}\n")

    # ════════════════════════════════════════════════════════════════════════
    #  PLOT
    # ════════════════════════════════════════════════════════════════════════
    if not rows:
        print("Geen resultaten om te plotten.")
        return

    labels   = [r["label"] for r in rows]
    raw_gems = [r["raw_gem"] for r in rows]
    ema_gems = [r["ema_gem"] for r in rows]
    raw_meds = [r["raw_med"] for r in rows]
    ema_meds = [r["ema_med"] for r in rows]

    x     = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(12, len(labels)*1.6), 6))

    b1 = ax.bar(x - 1.5*width, raw_gems, width, label="Raw gem.",  color="#4a90d9", alpha=0.9)
    b2 = ax.bar(x - 0.5*width, raw_meds, width, label="Raw med.",  color="#4a90d9", alpha=0.55, hatch="//")
    b3 = ax.bar(x + 0.5*width, ema_gems, width, label="EMA gem.",  color="#e07b39", alpha=0.9)
    b4 = ax.bar(x + 1.5*width, ema_meds, width, label="EMA med.",  color="#e07b39", alpha=0.55, hatch="//")

    ax.axhline(50,  color="red",  lw=0.9, linestyle="--", label="kansniveau (50%)")
    ax.axhline(100, color="gray", lw=0.4, linestyle=":")

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 105)
    ax.set_title(
        f"Model vergelijking  —  {rows[0]['n']} proefpersonen"
        + (f"  (eerste {MAX_SEC}s)" if MAX_SEC else ""),
        fontsize=13
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight")
    print(f"  Plot opgeslagen als: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
