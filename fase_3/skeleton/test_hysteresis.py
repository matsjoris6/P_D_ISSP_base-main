"""
Vergelijk hysteresis-strategieën voor AAD over ALLE beschikbare proefpersonen.

Gebruik:
  python test_hysteresis.py                        # alle proefpersonen, aggregaat
  python test_hysteresis.py --pair_no 1 --subject_no 2   # detail-plot voor sub-002/pair1

Audioenveloppen worden één keer berekend per audiopaar (cache).
Rangschikking: accuracy is de enige maatstaf; switches worden alleen vermeld.
"""
#!!!!!!
# EINDWINNAAR: 'EMA0.6+Schmitt0.60' VOOR NON REVERB!
# er is geen reverb voor de AAD. Reverb is zuiver een beamformer-probleem ALS CLEAN AUDIO AAN AAD WORDT GEGEVEN
# Doa wordt uiteraard wel beïnvloed door reverb, maar dat is een ander verhaal. 
#!!!!!!

#!!!!!!
# EEG ──→ AAD ←── Clean audio envelopes
#        ↑
#       Model getraind op clean audio
#!!!!!!

#!!!!!!
# EEG ──→ AAD ←── GSC output envelopes
#        ↑
#        Model getraind op clean audio
#        ⚠️ MISMATCH!
#!!!!!!
# => Eigenlijk niet de bedoeling met gsc, maar is extratje



import argparse
import os, sys, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from math import gcd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.getLogger("brian2").setLevel(logging.ERROR)
import brian2
brian2.prefs.codegen.target = "cython"
from brian2 import Hz, kHz
from brian2hears import Sound, erbspace, Gammatone, Filterbank

import tensorflow as tf

# ── configuratie ─────────────────────────────────────────────────────────────
DATA_DIR    = "data/data_phase3"
STIMULI_DIR = "data/data_phase3/stimuli"
MODEL_PATH  = "models/generic_dilated_alle_proefpersonen_beste_pieter_3laag_5sec_VERVOLG.keras"

WINDOW_SEC  = 5
STEP_SEC    = 2
MAX_SEC     = None   # None = volledige opname; bv. 60 voor snelle test

EEG_FS_IN   = 128
AUDIO_FS_IN = 48000
TARGET_FS   = 64
WIN_SAMPLES = WINDOW_SEC * TARGET_FS   # 320

OUTPUT_PNG        = "hysteresis_vergelijking_alle.png"
OUTPUT_DETAIL_PNG = "hysteresis_detail_sub{subject_no:03d}_pair{pair_no}.png"
TIE_THRESHOLD_PCT = 1.0   # strategieën binnen dit % van de beste acc → tiebreak op switches

# Welk wav-bestand is LINKS voor elk audiopaar (uit issp_data.py)
LEFTRIGHT_MAPPING = {
    "audiobook_1_part2.wav":     "left",   "audiobook_2_2_part2.wav":  "right",
    "podcast_3_part2.wav":       "left",   "podcast_4_part2.wav":      "right",
    "audiobook_8_2_part2.wav":   "left",   "audiobook_8_1_part2.wav":  "right",
    "audiobook_9_1_part2.wav":   "left",   "audiobook_9_2_part2.wav":  "right",
    "audiobook_10_1_part2.wav":  "left",   "audiobook_10_2_part2.wav": "right",
    "audiobook_11_2_part2.wav":  "left",   "audiobook_11_1_part2.wav": "right",
    "podcast_22_part2.wav":      "left",   "podcast_21_part2.wav":     "right",
    "podcast_24_part2.wav":      "left",   "podcast_25_part2.wav":     "right",
    "podcast_30_part2.wav":      "left",   "podcast_31_part2.wav":     "right",
    "audiobook_14_2_part2.wav":  "left",   "podcast_32_part2.wav":     "right",
    "podcast_33_part2.wav":      "left",   "audiobook_14_1_part2.wav": "right",
    "audiobook_1_part2.wav":     "left",   "podcast_34_part2.wav":     "right",
    "podcast_36_part2.wav":      "left",   "podcast_35_part2.wav":     "right",
    "podcast_37_part2.wav":      "right",
}
# ─────────────────────────────────────────────────────────────────────────────


# ── preprocessing (zelfde als processor.py) ──────────────────────────────────

class _EnvFilterbank(Filterbank):
    def __init__(self, source):
        super().__init__(source)
        self.nchannels = 1

    def buffer_apply(self, inp):
        return np.sum(np.abs(inp) ** 0.6, axis=1, keepdims=True)


def compute_audio_envelope(audio, sr_in=AUDIO_FS_IN, sr_out=TARGET_FS,
                            lowcut=1.0, highcut=32.0):
    sound = Sound(audio.reshape(-1, 1).astype(np.float32),
                  samplerate=sr_in * Hz)
    cf = erbspace(50 * Hz, 5 * kHz, 28)
    env = _EnvFilterbank(Gammatone(sound, cf)).process().flatten()
    sos = signal.butter(4, [lowcut, highcut], btype="bandpass",
                        fs=sr_in, output="sos")
    env = signal.sosfiltfilt(sos, env)
    g = gcd(int(sr_in), sr_out)
    return signal.resample_poly(env, sr_out // g, int(sr_in) // g)


def preprocess_eeg(eeg, fs_in=EEG_FS_IN, fs_out=TARGET_FS,
                   lowcut=1.0, highcut=32.0):
    sos = signal.butter(4, [lowcut, highcut], btype="bandpass",
                        fs=fs_in, output="sos")
    eeg_f = signal.sosfiltfilt(sos, eeg, axis=0)
    g = gcd(int(fs_in), fs_out)
    return signal.resample_poly(eeg_f, fs_out // g, int(fs_in) // g, axis=0)


# ── strategieën ──────────────────────────────────────────────────────────────

def make_strategies():
    def no_filter(probs):
        return [round(p) for p in probs]

    def schmitt(thresh_high):
        def _f(probs):
            thresh_low = 1.0 - thresh_high
            state = round(probs[0])
            out = []
            for p in probs:
                if state == 1 and p < thresh_low:
                    state = 0
                elif state == 0 and p > thresh_high:
                    state = 1
                out.append(state)
            return out
        return _f

    def majority(n):
        def _f(probs):
            out, hist = [], []
            for p in probs:
                hist.append(round(p))
                if len(hist) > n:
                    hist.pop(0)
                out.append(1 if sum(hist) > len(hist) / 2 else 0)
            return out
        return _f

    def ema(alpha):
        def _f(probs):
            e = probs[0]
            out = []
            for p in probs:
                e = alpha * p + (1 - alpha) * e
                out.append(round(e))
            return out
        return _f

    def ema_then_schmitt(alpha, thresh_high):
        def _f(probs):
            e = probs[0]
            smoothed = []
            for p in probs:
                e = alpha * p + (1 - alpha) * e
                smoothed.append(e)
            thresh_low = 1.0 - thresh_high
            state = round(smoothed[0])
            out = []
            for s in smoothed:
                if state == 1 and s < thresh_low:
                    state = 0
                elif state == 0 and s > thresh_high:
                    state = 1
                out.append(state)
            return out
        return _f

    return {
        "Geen filter":          no_filter,
        "Schmitt 0.55":         schmitt(0.55),
        "Schmitt 0.60":         schmitt(0.60),
        "Schmitt 0.65":         schmitt(0.65),
        "Schmitt 0.70":         schmitt(0.70),
        "Schmitt 0.75":         schmitt(0.75),
        "Meerderheid N=3":      majority(3),
        "Meerderheid N=5":      majority(5),
        "EMA α=0.6":            ema(0.6),
        "EMA α=0.4":            ema(0.4),
        "EMA α=0.2":            ema(0.2),
        "EMA0.6+Schmitt0.60":   ema_then_schmitt(0.6, 0.60),
        "EMA0.4+Schmitt0.60":   ema_then_schmitt(0.4, 0.60),
        "EMA0.4+Schmitt0.65":   ema_then_schmitt(0.4, 0.65),
    }


def evaluate(decisions, labels):
    dec = np.array(decisions)
    lab = np.array(labels)
    return np.mean(dec == lab) * 100, int(np.sum(np.abs(np.diff(dec))))


# ── per-proefpersoon pipeline ─────────────────────────────────────────────────

def run_subject(eeg_file, env_left_full, env_right_full, model, strategies,
                return_timeseries=False):
    """
    Verwerkt één proefpersoon.
    Als return_timeseries=True: geeft ook (raw_probs, window_gt, decisions_dict) terug.
    """
    npz = np.load(eeg_file)
    stim0  = str(npz["stimulus_0"])
    raw_gt = npz["attended_speaker"].astype(int)

    total_eeg = npz["eeg"].shape[0]
    if MAX_SEC is not None:
        total_eeg = min(total_eeg, MAX_SEC * EEG_FS_IN)

    eeg_proc = preprocess_eeg(npz["eeg"][:total_eeg].astype(np.float64))

    # GT: 1 = links geattendeerd (intern)
    left0 = LEFTRIGHT_MAPPING.get(stim0, "unknown")
    swap  = (left0 != "left")
    gt_raw  = raw_gt[:total_eeg]
    gt_left = gt_raw if swap else 1 - gt_raw

    ratio    = EEG_FS_IN // TARGET_FS
    gt_ds_len = int(total_eeg * TARGET_FS / EEG_FS_IN)
    gt_ds = np.array([round(float(np.mean(gt_left[i*ratio:(i+1)*ratio])))
                      for i in range(gt_ds_len)])

    step  = STEP_SEC * TARGET_FS
    n_win = (len(eeg_proc) - WIN_SAMPLES) // step + 1
    n_env = min(len(env_left_full), len(env_right_full))

    raw_probs, window_gt = [], []
    for w in range(n_win):
        s, e = w * step, w * step + WIN_SAMPLES
        if e > len(eeg_proc) or e > n_env:
            break
        eeg_w   = eeg_proc[s:e][np.newaxis].astype(np.float32)
        env_l_w = env_left_full[s:e][np.newaxis, :, np.newaxis].astype(np.float32)
        env_r_w = env_right_full[s:e][np.newaxis, :, np.newaxis].astype(np.float32)
        pred = model([eeg_w, env_l_w, env_r_w], training=False)
        raw_probs.append(float(pred[0, 0]))
        window_gt.append(int(round(float(np.mean(gt_ds[s:e])))))

    if len(raw_probs) < 2:
        return (None, None) if return_timeseries else None

    raw_probs = np.array(raw_probs)
    window_gt = np.array(window_gt)

    results = {name: evaluate(fn(raw_probs), window_gt)
               for name, fn in strategies.items()}

    if return_timeseries:
        decisions = {name: fn(raw_probs) for name, fn in strategies.items()}
        return results, (raw_probs, window_gt, decisions)
    return results


# ── detail-plot voor één proefpersoon ─────────────────────────────────────────

def plot_subject_detail(pair_no, subject_no, raw_probs, window_gt, decisions,
                        results, strategies):
    """
    Twee-paneel plot vergelijkbaar met de GUI:
      Boven : raw probability (serverconventie: 0≈links) + ground truth
      Onder  : nauwkeurigheid per strategie (staafdiagram) met switches als label
    """
    n_win = len(raw_probs)
    times = np.array([w * STEP_SEC + WINDOW_SEC / 2 for w in range(n_win)])

    # Serverconventie: GUI zendt 1 - intern_pred  en gt 0=links
    server_prob = 1.0 - raw_probs          # matcht GUI "probability"-lijn
    server_gt   = 1 - window_gt            # 0=links, 1=rechts  (matcht GUI "attended speaker gt")

    # Rangschikking (zelfde als aggregate: accuracy primair, switches als tiebreak)
    best_acc_subj = max(results[n][0] for n in results)
    def rank_key(name):
        acc, sw = results[name]
        if best_acc_subj - acc < TIE_THRESHOLD_PCT:
            return (1, acc, -sw)
        return (0, acc, -sw)
    winner = max(results, key=rank_key)

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Hysteresis detail — pair {pair_no}, subject {subject_no:03d}\n"
        f"Winnaar: '{winner}'  "
        f"(acc={results[winner][0]:.1f}%,  switches={results[winner][1]})",
        fontsize=12
    )

    # ── Boven: tijdreeks (GUI-vergelijking) ──────────────────────────────────
    ax_ts = fig.add_subplot(3, 1, (1, 2))

    ax_ts.plot(times, server_prob, color="crimson", lw=1.5,
               label="Probability (serverconventie, 0≈links)")
    ax_ts.step(times, server_gt, where="mid", color="limegreen", lw=1.5,
               linestyle="--", label="Ground truth (0=links, 1=rechts)")

    # Huidig processor.py filter (Schmitt 0.65) + winnaar tonen
    for strat, color, zorder in [("Schmitt 0.65", "royalblue", 3),
                                  (winner, "gold", 4)]:
        dec = np.array(decisions[strat])
        server_dec = 1 - dec
        acc, sw = results[strat]
        lbl = f"{strat}  [acc={acc:.1f}%  sw={sw}]"
        ax_ts.step(times, server_dec + np.random.uniform(-0.01, 0.01, n_win),
                   where="mid", lw=1.2, alpha=0.85, label=lbl,
                   color=color, zorder=zorder)

    ax_ts.axhline(0.5, color="gray", lw=0.7, linestyle=":")
    ax_ts.set_ylabel("0 = links  /  1 = rechts  (serverconventie)")
    ax_ts.set_xlabel("Tijd (s)")
    ax_ts.set_ylim(-0.15, 1.15)
    ax_ts.legend(fontsize=8, loc="upper right")
    ax_ts.set_title("Tijdreeks — vergelijkbaar met GUI 'probability' en 'attended speaker gt'")

    # ── Onder: staafdiagram nauwkeurigheid per strategie ─────────────────────
    ax_bar = fig.add_subplot(3, 1, 3)

    names  = list(strategies.keys())
    accs   = [results[n][0] for n in names]
    sws    = [results[n][1] for n in names]
    colors = ["gold" if n == winner else
              ("royalblue" if n == "Schmitt 0.65" else "steelblue")
              for n in names]

    bars = ax_bar.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax_bar.axhline(50, color="red", lw=0.8, linestyle="--", label="kansniveau (50%)")
    ax_bar.set_ylabel("Nauwkeurigheid (%)")
    ax_bar.set_title("Score per strategie  (goud = winnaar, blauw = huidig processor.py)")
    ax_bar.legend(fontsize=8)

    for bar, v, sw in zip(bars, accs, sws):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}%\n(sw={sw})",
                    ha="center", va="bottom", fontsize=6.5)

    plt.xticks(rotation=38, ha="right", fontsize=7)
    plt.tight_layout()

    out = OUTPUT_DETAIL_PNG.format(subject_no=subject_no, pair_no=pair_no)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nDetail-plot opgeslagen als: {out}")
    return out


# ── hoofdprogramma ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_no",    type=int, default=None,
                        help="Paar voor detail-plot (bv. 1)")
    parser.add_argument("--subject_no", type=int, default=None,
                        help="Proefpersoon voor detail-plot (bv. 2)")
    args = parser.parse_args()

    detail_mode = (args.pair_no is not None and args.subject_no is not None)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    all_eeg_files = sorted(glob.glob(os.path.join(DATA_DIR, "sub-*", "*.npz")))
    print(f"{len(all_eeg_files)} proefpersoon-bestanden gevonden.\n")

    model      = tf.keras.models.load_model(MODEL_PATH)
    strategies = make_strategies()

    # Groepeer per audiopaar (envelop slechts 1x berekend)
    pair_groups = defaultdict(list)
    for f in all_eeg_files:
        npz = np.load(f)
        key = (str(npz["stimulus_0"]), str(npz["stimulus_1"]))
        pair_groups[key].append(f)

    all_acc = defaultdict(list)
    all_sw  = defaultdict(list)
    n_done  = 0

    detail_result = None   # bewaar detail-data voor extra plot

    for (stim0, stim1), files in sorted(pair_groups.items()):
        path0 = os.path.join(STIMULI_DIR, stim0)
        path1 = os.path.join(STIMULI_DIR, stim1)
        if not os.path.exists(path0) or not os.path.exists(path1):
            print(f"  [SKIP] audio niet gevonden: {stim0} / {stim1}")
            continue

        left0 = LEFTRIGHT_MAPPING.get(stim0, "unknown")
        swap  = (left0 != "left")
        print(f"\nPaar: {stim0}  +  {stim1}")
        print(f"  Links={'stim1' if swap else 'stim0'}  |  {len(files)} proefpersoon(en)")

        fs0, audio0 = wavfile.read(path0)
        fs1, audio1 = wavfile.read(path1)
        if fs0 != AUDIO_FS_IN or fs1 != AUDIO_FS_IN:
            print(f"  [SKIP] onverwachte samplerate: {fs0}/{fs1}")
            continue

        n_audio = audio0.shape[0]
        if MAX_SEC is not None:
            n_audio = min(n_audio, MAX_SEC * AUDIO_FS_IN)

        print(f"  Envelop {stim0} …")
        env0 = compute_audio_envelope(audio0[:n_audio].astype(np.float32))
        print(f"  Envelop {stim1} …")
        env1 = compute_audio_envelope(audio1[:n_audio].astype(np.float32))

        env_left  = env1 if swap else env0
        env_right = env0 if swap else env1

        for eeg_file in files:
            subj    = os.path.basename(os.path.dirname(eeg_file))
            subj_no = int(subj.replace("sub-", ""))

            # Controleer of dit het detail-subject is
            is_detail = detail_mode and (subj_no == args.subject_no)
            # Voor detail-mode: sla pair-check over (vereenvoudigd; pair_no
            # bepaalt welke stimuli gebruikt worden, sub-nummer identificeert het subject)

            print(f"  {subj} …", end=" ", flush=True)
            res, ts = run_subject(eeg_file, env_left, env_right, model, strategies,
                                  return_timeseries=True)
            if res is None:
                print("overgeslagen (te weinig vensters)")
                continue

            for name, (acc, sw) in res.items():
                all_acc[name].append(acc)
                all_sw[name].append(sw)

            # Accuracy primair; tiebreak op switches binnen TIE_THRESHOLD_PCT
            best_acc_here = max(res[n][0] for n in res)
            def rank_local(n):
                acc, sw = res[n]
                if best_acc_here - acc < TIE_THRESHOLD_PCT:
                    return (1, acc, -sw)
                return (0, acc, -sw)
            best = max(res, key=rank_local)
            print(f"best={best}  (acc={res[best][0]:.1f}%,  sw={res[best][1]})")
            n_done += 1

            if is_detail:
                detail_result = (res, ts)
                print(f"    → detail-data bewaard voor sub-{args.subject_no:03d}/pair{args.pair_no}")

    if n_done == 0:
        print("Geen proefpersonen verwerkt.")
        return

    # ── geaggregeerde resultaten ─────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"TOTAALOVERZICHT  ({n_done} proefpersonen)")
    print(f"{'='*68}")
    print(f"{'Strategie':<24}  {'Gem.acc':>8}  {'Med.acc':>8}  {'Gem.sw':>7}")
    print("-" * 58)

    summary = {}
    for name in strategies:
        if name not in all_acc:
            continue
        accs  = np.array(all_acc[name])
        sws   = np.array(all_sw[name])
        summary[name] = (np.mean(accs), np.median(accs), np.mean(sws))
        m_acc, med, m_sw = summary[name]
        print(f"{name:<24}  {m_acc:>7.1f}%  {med:>7.1f}%  {m_sw:>7.1f}")

    # Rangschikking: accuracy primair, tiebreak op minste switches
    best_acc = max(summary[n][0] for n in summary)
    def rank_key(name):
        m_acc, _, m_sw = summary[name]
        if best_acc - m_acc < TIE_THRESHOLD_PCT:
            return (1, m_acc, -m_sw)
        return (0, m_acc, -m_sw)

    winner = max(summary, key=rank_key)
    m_acc, med, m_sw = summary[winner]
    print(f"\n{'='*68}")
    print(f"  EINDWINNAAR: '{winner}'")
    print(f"  Gemiddelde nauwkeurigheid : {m_acc:.1f}%")
    print(f"  Mediaan nauwkeurigheid    : {med:.1f}%")
    print(f"  Gemiddeld aantal switches : {m_sw:.1f}  (alleen ter info)")
    print(f"  (gelijkspeldrempel: {TIE_THRESHOLD_PCT}%)")
    print(f"{'='*68}\n")

    # ── aggregaat staafdiagram ───────────────────────────────────────────────
    names  = list(summary.keys())
    accs   = [summary[n][0] for n in names]
    sws    = [summary[n][2] for n in names]
    colors = ["gold" if n == winner else "steelblue" for n in names]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    bars1 = ax1.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Gemiddelde nauwkeurigheid (%)")
    ax1.set_title(f"Hysteresis-vergelijking — {n_done} proefpersonen  |  winnaar = '{winner}'")
    ax1.axhline(50, color="red", lw=0.8, linestyle="--", label="kansniveau (50%)")
    ax1.legend(fontsize=8)
    for bar, v in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

    bars2 = ax2.bar(names, sws, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Gem. aantal switches  (informatief)")
    ax2.set_xlabel("Strategie")
    for bar, v in zip(bars2, sws):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    plt.xticks(rotation=38, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Aggregaat-plot opgeslagen als: {OUTPUT_PNG}")

    # ── detail-plot voor gevraagd subject ────────────────────────────────────
    if detail_mode:
        if detail_result is None:
            print(f"\n[WARN] sub-{args.subject_no:03d} niet gevonden in de data.")
        else:
            res, (raw_probs, window_gt, decisions) = detail_result
            plot_subject_detail(args.pair_no, args.subject_no,
                                raw_probs, window_gt, decisions,
                                res, strategies)


if __name__ == "__main__":
    main()
