"""
Vergelijk hysteresis-strategieën voor AAD over ALLE beschikbare proefpersonen.

Gebruik:
  python test_hysteresis.py
  python test_hysteresis.py --model hybrid_5s --hop_sec 2
  python test_hysteresis.py --pair_no 1 --subject_no 2

Audioenveloppen worden één keer berekend per audiopaar (cache).
Rangschikking: accuracy is de enige maatstaf; switches worden alleen vermeld.

─── Bekende winnaars ────────────────────────────────────────────────────────
  dilated_5s(of hybrid)  |  hop=1s  |  clean speech  →  EINDWINNAAR: 'EMA α=0.30'
                                              Gem. acc 82.2%  |  Med. acc 83.9%
  hybrid_10s             |  hop=2s  |  clean speech  →  EINDWINNAAR: EMA α=0.5 (+ Schmitt 0.55)

  hybrid_

─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os, sys, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import scipy.linalg

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from math import gcd
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.getLogger("brian2").setLevel(logging.ERROR)
import brian2
brian2.prefs.codegen.target = "cython"
from brian2 import Hz, kHz
from brian2hears import Sound, erbspace, Gammatone, Filterbank

import tensorflow as tf

# ── modelparameters automatisch uit config.py ─────────────────────────────────
from config import ACTIVE_MODEL as CONFIG_ACTIVE_MODEL, MODELS
from config import USE_GSC_AUDIO_FOR_AAD as CONFIG_USE_GSC_AUDIO
MODEL_KEYS = list(MODELS.keys())

DATA_DIR    = "data/data_phase3"
STIMULI_DIR = "data/data_phase3/stimuli"
REVERB_DIR  = "data/phase3_audioData/audiodata_batch_1/reverberant"
RIR_PATH    = "data/phase3_audioData/audiodata_batch_1/reverberant/lma_16kHz_200ms.npz"

MAX_SEC     = None      # None = volledige opname; bv. 60 voor snelle test

EEG_FS_IN   = 128
AUDIO_FS_IN = 48000
GSC_FS      = 16000
TARGET_FS   = 64
TIE_THRESHOLD_PCT = 1.0

# ── links/rechts mapping per stimulus (voor clean speech pad) ─────────────────
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

# ── pair-mapping voor GSC pad (uit issp_data.py) ──────────────────────────────
PAIR_MAPPING = {
    1:  {"left": "audiobook_1_part2.wav",    "right": "audiobook_2_2_part2.wav"},
    2:  {"left": "podcast_3_part2.wav",       "right": "podcast_4_part2.wav"},
    3:  {"left": "audiobook_8_2_part2.wav",   "right": "audiobook_8_1_part2.wav"},
    4:  {"left": "audiobook_9_1_part2.wav",   "right": "audiobook_9_2_part2.wav"},
    5:  {"left": "audiobook_10_1_part2.wav",  "right": "audiobook_10_2_part2.wav"},
    6:  {"left": "audiobook_11_2_part2.wav",  "right": "audiobook_11_1_part2.wav"},
    7:  {"left": "podcast_22_part2.wav",      "right": "podcast_21_part2.wav"},
    8:  {"left": "podcast_24_part2.wav",      "right": "podcast_25_part2.wav"},
    9:  {"left": "podcast_30_part2.wav",      "right": "podcast_31_part2.wav"},
    10: {"left": "audiobook_14_2_part2.wav",  "right": "podcast_32_part2.wav"},
    11: {"left": "podcast_33_part2.wav",      "right": "audiobook_14_1_part2.wav"},
    12: {"left": "audiobook_1_part2.wav",     "right": "podcast_34_part2.wav"},
    13: {"left": "audiobook_14_2_part2.wav",  "right": "podcast_35_part2.wav"},
    14: {"left": "podcast_36_part2.wav",      "right": "audiobook_14_1_part2.wav"},
    15: {"left": "audiobook_1_part2.wav",     "right": "podcast_37_part2.wav"},
}
_STIMSET_TO_PAIR = {
    frozenset({v["left"], v["right"]}): pair_no
    for pair_no, v in PAIR_MAPPING.items()
}


# ════════════════════════════════════════════════════════════════════════════
#  GSC BEAMFORMER (offline — alleen gebruikt als USE_GSC_AUDIO=True)
# ════════════════════════════════════════════════════════════════════════════

_L, _HOP, _BETA, _MU, _M, _Q = 1024, 512, 0.85, 0.01, 5, 2
_DOA_FILTER_N = 63

ACTIVE_MODEL = CONFIG_ACTIVE_MODEL
USE_GSC_AUDIO = CONFIG_USE_GSC_AUDIO
MODEL_PATH = None
WINDOW_SEC = None
HOP_SEC = None
STEP_SEC = None
EEG_WINDOW_SAMPLES = None
WIN_SAMPLES = None
EMA_ALPHA = None
SCHMITT_THRESHOLD = None
SCHMITT_HYSTERESIS = None
OUTPUT_PNG = None
OUTPUT_DETAIL_PNG = None


def _format_hop(hop_sec):
    return str(int(hop_sec)) if float(hop_sec).is_integer() else str(hop_sec).replace(".", "p")


def configure_runtime(model_name, hop_sec=None, use_gsc_audio=None):
    global ACTIVE_MODEL, USE_GSC_AUDIO, MODEL_PATH, WINDOW_SEC, HOP_SEC, STEP_SEC
    global EEG_WINDOW_SAMPLES, WIN_SAMPLES, EMA_ALPHA, SCHMITT_THRESHOLD
    global SCHMITT_HYSTERESIS, OUTPUT_PNG, OUTPUT_DETAIL_PNG

    if model_name not in MODELS:
        raise SystemExit(f"[FOUT] Onbekend model '{model_name}'. Kies uit: {MODEL_KEYS}")

    cfg = MODELS[model_name]
    selected_hop = cfg["hop_sec"] if hop_sec is None else hop_sec
    if selected_hop <= 0:
        raise SystemExit("[FOUT] --hop_sec moet groter zijn dan 0.")
    if selected_hop > cfg["window_sec"]:
        raise SystemExit(
            f"[FOUT] --hop_sec ({selected_hop}) mag niet groter zijn dan het venster "
            f"van model '{model_name}' ({cfg['window_sec']}s)."
        )

    ACTIVE_MODEL = model_name
    USE_GSC_AUDIO = CONFIG_USE_GSC_AUDIO if use_gsc_audio is None else use_gsc_audio
    MODEL_PATH = cfg["model_path"]
    WINDOW_SEC = cfg["window_sec"]
    HOP_SEC = selected_hop
    STEP_SEC = selected_hop
    EEG_WINDOW_SAMPLES = cfg["eeg_window_samples"]
    WIN_SAMPLES = EEG_WINDOW_SAMPLES
    EMA_ALPHA = cfg["ema_alpha"]
    SCHMITT_THRESHOLD = cfg["schmitt_threshold"]
    SCHMITT_HYSTERESIS = cfg["schmitt_hysteresis"]

    hop_tag = _format_hop(STEP_SEC)
    audio_tag = "gsc" if USE_GSC_AUDIO else "clean"
    OUTPUT_PNG = f"hysteresis_vergelijking_{ACTIVE_MODEL}_hop{hop_tag}s_{audio_tag}.png"
    OUTPUT_DETAIL_PNG = f"hysteresis_detail_{{subject_no:03d}}_pair{{pair_no}}_{ACTIVE_MODEL}_hop{hop_tag}s_{audio_tag}.png"


configure_runtime(CONFIG_ACTIVE_MODEL)


def _build_lut(rir, L=_L):
    n_bins = L // 2 + 1
    M = rir.shape[1]
    H = np.fft.rfft(rir, n=L, axis=0)
    W_FAS = np.zeros((n_bins, M), dtype=complex)
    B_mat = np.zeros((n_bins, M - 1, M), dtype=complex)
    for k in range(n_bins):
        h_k = H[k, :].reshape(M, 1)
        A_1 = h_k[0, 0]
        h_k = h_k / (A_1 if np.abs(A_1) > 1e-12 else A_1 + 1e-12)
        denom = (h_k.conj().T @ h_k)[0, 0]
        W_FAS[k, :] = (h_k / denom if np.abs(denom) > 1e-12 else np.ones(M) / M).flatten()
        Z = scipy.linalg.null_space(h_k.conj().T)
        if Z.shape[1] > 0:
            B_mat[k, :, :] = Z.conj().T
    return W_FAS, B_mat


def run_gsc_offline(lma_data, rir_path=RIR_PATH, max_samples=None):
    """
    Offline GSC beamformer op reverberante LMA-opname.
    Geeft (gsc_left, gsc_right) float32 terug bij GSC_FS (16 kHz).
    """
    if max_samples is not None:
        lma_data = lma_data[:max_samples]
    lma_f = lma_data.astype(np.float64)
    N, M  = lma_f.shape
    L, hop, beta, mu, Q = _L, _HOP, _BETA, _MU, _Q
    n_bins = L // 2 + 1

    rir_data   = np.load(rir_path)
    rirs       = rir_data["rirs"]
    lut_angles = np.array(rir_data["thetas"])

    A_lut = np.zeros((n_bins, M, len(lut_angles)), dtype=complex)
    lut   = {}
    for i, angle in enumerate(lut_angles):
        rir = rirs[:, :, i]
        H   = np.fft.rfft(rir, n=L, axis=0)
        for k in range(n_bins):
            h_k = H[k, :].reshape(M, 1)
            A_1 = h_k[0, 0]
            h_k = h_k / (A_1 if np.abs(A_1) > 1e-12 else A_1 + 1e-12)
            A_lut[k, :, i] = h_k.flatten()
        lut[float(angle)] = _build_lut(rir, L=L)

    def _get_lut(doa):
        return lut[float(lut_angles[np.argmin(np.abs(lut_angles - doa))])]

    window    = np.sqrt(signal.windows.hann(L, sym=False))
    audio_buf = np.zeros((L, M))
    Ryy       = np.zeros((n_bins, M, M), dtype=complex)
    w_left    = np.zeros((n_bins, M - 1), dtype=complex)
    w_right   = np.zeros((n_bins, M - 1), dtype=complex)
    ola_left  = np.zeros(L)
    ola_right = np.zeros(L)
    last_l, last_r = 135.0, 45.0
    doa_buf_l = deque(maxlen=_DOA_FILTER_N)
    doa_buf_r = deque(maxlen=_DOA_FILTER_N)
    PEAK_THR  = -12.0
    valid_k   = np.arange(1, L // 2)
    left_mask  = lut_angles > 90
    right_mask = lut_angles <= 90

    gsc_l_out, gsc_r_out = [], []
    for h in range((N - L) // hop + 1):
        s = h * hop
        audio_buf = np.roll(audio_buf, -hop, axis=0)
        audio_buf[-hop:] = lma_f[s:s + hop]

        frame_fft = np.fft.rfft(audio_buf * window[:, np.newaxis], n=L, axis=0)

        Y   = frame_fft[valid_k, :, np.newaxis]
        R_k = Y @ Y.conj().transpose(0, 2, 1)
        Ryy[valid_k] = beta * Ryy[valid_k] + (1 - beta) * R_k
        _, eigvecs = np.linalg.eigh(Ryy[valid_k])
        En       = eigvecs[:, :, :M - Q]
        En_H_A   = En.conj().transpose(0, 2, 1) @ A_lut[valid_k]
        denom    = np.sum(np.abs(En_H_A) ** 2, axis=1)
        p        = 1.0 / np.clip(denom, 1e-10, None)
        log_p    = np.log(np.clip(p, 1e-10, None))
        p_geom   = np.exp(np.mean(log_p, axis=0))
        spec_db  = 10 * np.log10(p_geom / max(np.max(p_geom), 1e-20))

        bl = np.argmax(np.where(left_mask,  spec_db, -np.inf))
        br = np.argmax(np.where(right_mask, spec_db, -np.inf))
        if spec_db[bl] > PEAK_THR: last_l = lut_angles[bl]
        if spec_db[br] > PEAK_THR: last_r = lut_angles[br]

        doa_buf_l.append(last_l)
        doa_buf_r.append(last_r)
        angle_l = float(np.median(doa_buf_l))
        angle_r = float(np.median(doa_buf_r))

        W_L, B_L = _get_lut(angle_l)
        W_R, B_R = _get_lut(angle_r)

        def _gsc(fft, W, B, w):
            y_fas = np.sum(np.conj(W) * fft, axis=1)
            u     = np.einsum('nij,nj->ni', B, fft)
            e     = y_fas - np.sum(np.conj(w) * u, axis=1)
            power = np.real(np.sum(np.conj(u) * u, axis=1))
            w    += mu * u * np.conj(e)[:, np.newaxis] / (power[:, np.newaxis] + 1e-8)
            return e

        out_l = np.fft.irfft(_gsc(frame_fft, W_L, B_L, w_left),  n=L) * window
        out_r = np.fft.irfft(_gsc(frame_fft, W_R, B_R, w_right), n=L) * window

        ola_left  += out_l;  ola_right += out_r
        gsc_l_out.append(ola_left[:hop].copy())
        gsc_r_out.append(ola_right[:hop].copy())
        ola_left  = np.concatenate([ola_left[hop:],  np.zeros(hop)])
        ola_right = np.concatenate([ola_right[hop:], np.zeros(hop)])

    return (np.concatenate(gsc_l_out).astype(np.float32),
            np.concatenate(gsc_r_out).astype(np.float32))


# ════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

class _EnvFilterbank(Filterbank):
    def __init__(self, source):
        super().__init__(source)
        self.nchannels = 1

    def buffer_apply(self, inp):
        return np.sum(np.abs(inp) ** 0.6, axis=1, keepdims=True)


def compute_audio_envelope(audio, sr_in, sr_out=TARGET_FS, lowcut=1.0, highcut=32.0):
    brian2.start_scope()
    sound = Sound(audio.reshape(-1, 1).astype(np.float32), samplerate=sr_in * Hz)
    cf    = erbspace(50 * Hz, 5 * kHz, 28)
    env   = _EnvFilterbank(Gammatone(sound, cf)).process().flatten()
    sos   = signal.butter(4, [lowcut, highcut], btype="bandpass", fs=sr_in, output="sos")
    env   = signal.sosfiltfilt(sos, env)
    g     = gcd(int(sr_in), sr_out)
    return signal.resample_poly(env, sr_out // g, int(sr_in) // g)


def preprocess_eeg(eeg, fs_in=EEG_FS_IN, fs_out=TARGET_FS, lowcut=1.0, highcut=32.0):
    sos   = signal.butter(4, [lowcut, highcut], btype="bandpass", fs=fs_in, output="sos")
    eeg_f = signal.sosfiltfilt(sos, eeg, axis=0)
    g     = gcd(int(fs_in), fs_out)
    return signal.resample_poly(eeg_f, fs_out // g, int(fs_in) // g, axis=0)


def _limit_samples(duration_sec, sample_rate):
    if duration_sec is None:
        return None
    return int(round(duration_sec * sample_rate))


# ════════════════════════════════════════════════════════════════════════════
#  STRATEGIEËN
# ════════════════════════════════════════════════════════════════════════════

def make_strategies():
    def no_filter(probs):
        return [round(p) for p in probs]

    def schmitt(thresh_high):
        def _f(probs):
            thresh_low = 1.0 - thresh_high
            state = round(probs[0])
            out = []
            for p in probs:
                if state == 1 and p < thresh_low:   state = 0
                elif state == 0 and p > thresh_high: state = 1
                out.append(state)
            return out
        return _f

    def majority(n):
        def _f(probs):
            out, hist = [], []
            for p in probs:
                hist.append(round(p))
                if len(hist) > n: hist.pop(0)
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
                if state == 1 and s < thresh_low:   state = 0
                elif state == 0 and s > thresh_high: state = 1
                out.append(state)
            return out
        return _f

    # Optimale EMA-α hangt af van de hop-grootte.
    # Vuistregel: α = 1 - exp(-stap / τ)  met τ ≈ 5.5s
    #   stap=1s → α ≈ 0.17  (dilated_5s / hybrid_3s / hybrid_5s)
    #   stap=5s → α ≈ 0.60  (hybrid_10s)
    if STEP_SEC <= 1:
        return {
            "Geen filter":           no_filter,
            "Schmitt 0.55":          schmitt(0.55),
            "Schmitt 0.60":          schmitt(0.60),
            "Schmitt 0.65":          schmitt(0.65),
            "Meerderheid N=3":       majority(3),
            "Meerderheid N=5":       majority(5),
            "Meerderheid N=9":       majority(9),
            "EMA α=0.10":            ema(0.10),
            "EMA α=0.15":            ema(0.15),
            "EMA α=0.20":            ema(0.20),
            "EMA α=0.30":            ema(0.30),
            "EMA α=0.40":            ema(0.40),
            "EMA α=0.60":            ema(0.60),
            "EMA0.10+Schmitt0.60":   ema_then_schmitt(0.10, 0.60),
            "EMA0.15+Schmitt0.60":   ema_then_schmitt(0.15, 0.60),
            "EMA0.20+Schmitt0.60":   ema_then_schmitt(0.20, 0.60),
            "EMA0.20+Schmitt0.65":   ema_then_schmitt(0.20, 0.65),
            "EMA0.30+Schmitt0.60":   ema_then_schmitt(0.30, 0.60),
            "EMA0.30+Schmitt0.65":   ema_then_schmitt(0.30, 0.65),
            "EMA0.40+Schmitt0.60":   ema_then_schmitt(0.40, 0.60),
            "EMA0.60+Schmitt0.60":   ema_then_schmitt(0.60, 0.60),
        }
    else:
        return {
            "Geen filter":           no_filter,
            "Schmitt 0.55":          schmitt(0.55),
            "Schmitt 0.60":          schmitt(0.60),
            "Schmitt 0.65":          schmitt(0.65),
            "Schmitt 0.70":          schmitt(0.70),
            "Meerderheid N=3":       majority(3),
            "Meerderheid N=5":       majority(5),
            "Meerderheid N=7":       majority(7),
            "EMA α=0.20":            ema(0.20),
            "EMA α=0.30":            ema(0.30),
            "EMA α=0.40":            ema(0.40),
            "EMA α=0.50":            ema(0.50),
            "EMA α=0.60":            ema(0.60),
            "EMA α=0.70":            ema(0.70),
            "EMA α=0.80":            ema(0.80),
            "EMA0.40+Schmitt0.55":   ema_then_schmitt(0.40, 0.55),
            "EMA0.40+Schmitt0.60":   ema_then_schmitt(0.40, 0.60),
            "EMA0.50+Schmitt0.55":   ema_then_schmitt(0.50, 0.55),
            "EMA0.50+Schmitt0.60":   ema_then_schmitt(0.50, 0.60),
            "EMA0.60+Schmitt0.55":   ema_then_schmitt(0.60, 0.55),
            "EMA0.60+Schmitt0.60":   ema_then_schmitt(0.60, 0.60),
            "EMA0.60+Schmitt0.65":   ema_then_schmitt(0.60, 0.65),
            "EMA0.70+Schmitt0.60":   ema_then_schmitt(0.70, 0.60),
            "EMA0.70+Schmitt0.65":   ema_then_schmitt(0.70, 0.65),
            "EMA0.80+Schmitt0.60":   ema_then_schmitt(0.80, 0.60),
        }


def evaluate(decisions, labels):
    dec = np.array(decisions)
    lab = np.array(labels)
    return np.mean(dec == lab) * 100, int(np.sum(np.abs(np.diff(dec))))


# ════════════════════════════════════════════════════════════════════════════
#  PER-PROEFPERSOON PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def run_subject(eeg_file, env_left_full, env_right_full, pair_info_left,
                model, strategies, return_timeseries=False):
    """
    Verwerkt één proefpersoon.
    pair_info_left : naam van de LINKER stimulus voor dit paar (uit PAIR_MAPPING of LEFTRIGHT_MAPPING).
    Als return_timeseries=True: geeft ook (raw_probs, window_gt, decisions_dict) terug.
    """
    npz    = np.load(eeg_file)
    stim0  = str(npz["stimulus_0"])
    raw_gt = npz["attended_speaker"].astype(int)

    total_eeg = npz["eeg"].shape[0]
    if MAX_SEC is not None:
        total_eeg = min(total_eeg, _limit_samples(MAX_SEC, EEG_FS_IN))

    eeg_proc = preprocess_eeg(npz["eeg"][:total_eeg].astype(np.float64))

    # GT: gt_left=1 = attending LEFT
    swap    = (pair_info_left != stim0)
    gt_raw  = raw_gt[:total_eeg]
    gt_left = gt_raw if swap else 1 - gt_raw

    ratio     = EEG_FS_IN // TARGET_FS
    gt_ds_len = int(total_eeg * TARGET_FS / EEG_FS_IN)
    gt_ds = np.array([round(float(np.mean(gt_left[i*ratio:(i+1)*ratio])))
                      for i in range(gt_ds_len)])

    step  = int(round(STEP_SEC * TARGET_FS))
    n_env = min(len(env_left_full), len(env_right_full), gt_ds_len)
    n_win = (n_env - WIN_SAMPLES) // step + 1

    raw_probs, window_gt = [], []
    for w in range(n_win):
        s, e = w * step, w * step + WIN_SAMPLES
        if e > n_env:
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


# ════════════════════════════════════════════════════════════════════════════
#  DETAIL-PLOT
# ════════════════════════════════════════════════════════════════════════════

def plot_subject_detail(pair_no, subject_no, raw_probs, window_gt, decisions,
                        results, strategies):
    n_win = len(raw_probs)
    times = np.array([w * STEP_SEC + WINDOW_SEC / 2 for w in range(n_win)])

    server_prob = 1.0 - raw_probs
    server_gt   = 1 - window_gt

    best_acc_subj = max(results[n][0] for n in results)
    def rank_key(name):
        acc, sw = results[name]
        return (1, acc, -sw) if best_acc_subj - acc < TIE_THRESHOLD_PCT else (0, acc, -sw)
    winner = max(results, key=rank_key)

    # Zoek huidige config-strategie
    def _cfg_name(strategies):
        for name in strategies:
            if (f"EMA{EMA_ALPHA:.2f}+Schmitt{SCHMITT_THRESHOLD:.2f}" in name or
                    f"EMA α={EMA_ALPHA}" in name):
                return name
        return list(strategies.keys())[0]
    current = _cfg_name(decisions)

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Hysteresis detail — pair {pair_no}, subject {subject_no:03d}  "
        f"| model={ACTIVE_MODEL}  audio={'GSC reverberant' if USE_GSC_AUDIO else 'clean speech'}\n"
        f"Winnaar: '{winner}'  (acc={results[winner][0]:.1f}%,  switches={results[winner][1]})",
        fontsize=11
    )

    ax_ts = fig.add_subplot(3, 1, (1, 2))
    ax_ts.plot(times, server_prob, color="crimson", lw=1.5,
               label="Probability (serverconventie, 0≈links)")
    ax_ts.step(times, server_gt, where="mid", color="limegreen", lw=1.5,
               linestyle="--", label="Ground truth (0=links, 1=rechts)")

    for strat, color, zorder in [(current, "royalblue", 3), (winner, "gold", 4)]:
        dec = np.array(decisions[strat])
        acc, sw = results[strat]
        lbl = f"{strat}  [acc={acc:.1f}%  sw={sw}]"
        ax_ts.step(times, (1 - dec) + np.random.uniform(-0.01, 0.01, n_win),
                   where="mid", lw=1.2, alpha=0.85, label=lbl,
                   color=color, zorder=zorder)

    ax_ts.axhline(0.5, color="gray", lw=0.7, linestyle=":")
    ax_ts.set_ylabel("0 = links  /  1 = rechts  (serverconventie)")
    ax_ts.set_xlabel("Tijd (s)")
    ax_ts.set_ylim(-0.15, 1.15)
    ax_ts.legend(fontsize=8, loc="upper right")
    ax_ts.set_title("Tijdreeks — vergelijkbaar met GUI 'probability' en 'attended speaker gt'")

    ax_bar = fig.add_subplot(3, 1, 3)
    names  = list(strategies.keys())
    accs   = [results[n][0] for n in names]
    sws    = [results[n][1] for n in names]
    colors = ["gold" if n == winner else
              ("royalblue" if n == current else "steelblue") for n in names]

    bars = ax_bar.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax_bar.axhline(50, color="red", lw=0.8, linestyle="--", label="kansniveau (50%)")
    ax_bar.set_ylabel("Nauwkeurigheid (%)")
    ax_bar.set_title(
        f"Score per strategie  "
        f"(goud=winnaar  blauw=huidig config.py [{current}]  "
        f"model={ACTIVE_MODEL}  audio={'GSC' if USE_GSC_AUDIO else 'clean'})"
    )
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


# ════════════════════════════════════════════════════════════════════════════
#  HOOFDPROGRAMMA
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Vergelijk hysteresis-strategieen voor AAD.\n\n"
            "Voorbeelden:\n"
            "  python test_hysteresis.py\n"
            "  python test_hysteresis.py --model hybrid_5s --hop_sec 2\n"
            "  python test_hysteresis.py --model hybrid_5s --hop_sec 2 --max_sec 60\n"
            "  python test_hysteresis.py --pair_no 1 --subject_no 2 --model hybrid_10s --hop_sec 5\n"
        ),
    )
    global MAX_SEC
    parser.add_argument("--model", type=str, default=CONFIG_ACTIVE_MODEL, choices=MODEL_KEYS,
                        help="Modelsleutel uit config.py. Standaard: ACTIVE_MODEL uit config.py.")
    parser.add_argument("--hop_sec", type=float, default=None,
                        help="Hop-grootte in seconden. Standaard: hop van het gekozen model.")
    parser.add_argument("--max_sec", type=float, default=None,
                        help="Beperk elke setup tot de eerste N seconden audio/EEG. Standaard: volledige opname.")
    parser.add_argument("--use_gsc_audio", action="store_true", default=CONFIG_USE_GSC_AUDIO,
                        help="Gebruik GSC-output voor AAD i.p.v. clean speech.")
    parser.add_argument("--use_clean_audio", action="store_false", dest="use_gsc_audio",
                        help="Forceer clean speech voor AAD.")
    parser.add_argument("--pair_no",    type=int, default=None)
    parser.add_argument("--subject_no", type=int, default=None)
    args = parser.parse_args()
    detail_mode = (args.pair_no is not None and args.subject_no is not None)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    configure_runtime(args.model, args.hop_sec, args.use_gsc_audio)
    MAX_SEC = args.max_sec

    print(f"[test_hysteresis]  Model={ACTIVE_MODEL}  venster={WINDOW_SEC}s  hop={STEP_SEC}s  "
          f"WIN_SAMPLES={WIN_SAMPLES}  audio={'GSC reverberant' if USE_GSC_AUDIO else 'clean speech'}  "
          f"max_sec={'volledig' if MAX_SEC is None else MAX_SEC}")

    all_eeg_files = sorted(glob.glob(os.path.join(DATA_DIR, "sub-*", "*.npz")))
    print(f"{len(all_eeg_files)} proefpersoon-bestanden gevonden.\n")

    model      = tf.keras.models.load_model(MODEL_PATH)
    strategies = make_strategies()

    # Groepeer per audiodopaar
    pair_groups = defaultdict(list)
    for f in all_eeg_files:
        npz = np.load(f)
        key = frozenset({str(npz["stimulus_0"]), str(npz["stimulus_1"])})
        pair_groups[key].append(f)

    all_acc = defaultdict(list)
    all_sw  = defaultdict(list)
    n_done  = 0
    detail_result = None

    for stim_key, files in sorted(pair_groups.items(), key=lambda x: str(sorted(x[0]))):
        stim_list = sorted(stim_key)   # [stim_a, stim_b]

        # ── Audio laden: clean speech of GSC ─────────────────────────────────
        if USE_GSC_AUDIO:
            # GSC pad: laad mixture_LMA.wav voor het juiste pair
            if stim_key not in _STIMSET_TO_PAIR:
                print(f"\n  [SKIP] Paar niet in PAIR_MAPPING: {stim_list}")
                continue
            pair_no   = _STIMSET_TO_PAIR[stim_key]
            pair_info = PAIR_MAPPING[pair_no]
            lma_path  = os.path.join(REVERB_DIR, f"pair{pair_no}", "mixture_LMA.wav")

            if not os.path.exists(lma_path):
                print(f"\n  [SKIP] mixture_LMA.wav niet gevonden: {lma_path}")
                continue

            print(f"\nPaar: pair{pair_no}  ({pair_info['left']}  +  {pair_info['right']})")
            print(f"  {len(files)} proefpersoon(en)  |  GSC-beamformer draaien …")

            _, lma_data = wavfile.read(lma_path)
            max_samp = _limit_samples(MAX_SEC, GSC_FS)
            gsc_left, gsc_right = run_gsc_offline(lma_data, max_samples=max_samp)
            print(f"  GSC klaar  ({len(gsc_left)/GSC_FS:.0f}s @ {GSC_FS} Hz)  →  envelop …")

            env_left  = compute_audio_envelope(gsc_left,  sr_in=GSC_FS)
            env_right = compute_audio_envelope(gsc_right, sr_in=GSC_FS)
            pair_info_left = pair_info["left"]

        else:
            # Clean speech pad: zelfde als originele test_hysteresis
            stim0, stim1 = stim_list[0], stim_list[1]
            path0 = os.path.join(STIMULI_DIR, stim0)
            path1 = os.path.join(STIMULI_DIR, stim1)
            if not os.path.exists(path0) or not os.path.exists(path1):
                print(f"\n  [SKIP] audio niet gevonden: {stim0} / {stim1}")
                continue

            left0 = LEFTRIGHT_MAPPING.get(stim0, "unknown")
            swap  = (left0 != "left")
            print(f"\nPaar: {stim0}  +  {stim1}")
            print(f"  Links={'stim1' if swap else 'stim0'}  |  {len(files)} proefpersoon(en)")

            _, audio0 = wavfile.read(path0)
            _, audio1 = wavfile.read(path1)
            n_audio = audio0.shape[0]
            if MAX_SEC is not None:
                n_audio = min(n_audio, _limit_samples(MAX_SEC, AUDIO_FS_IN))

            print(f"  Envelop {stim0} …")
            env0 = compute_audio_envelope(audio0[:n_audio].astype(np.float32), sr_in=AUDIO_FS_IN)
            print(f"  Envelop {stim1} …")
            env1 = compute_audio_envelope(audio1[:n_audio].astype(np.float32), sr_in=AUDIO_FS_IN)

            env_left  = env1 if swap else env0
            env_right = env0 if swap else env1

            # Bepaal linker stimulus naam voor GT-swap logica
            pair_info_left = stim1 if swap else stim0

        # ── EEG-bestanden verwerken ───────────────────────────────────────────
        for eeg_file in sorted(files):
            subj    = os.path.basename(os.path.dirname(eeg_file))
            subj_no = int(subj.replace("sub-", ""))
            is_detail = detail_mode and (subj_no == args.subject_no)

            print(f"  {subj} …", end=" ", flush=True)
            res, ts = run_subject(eeg_file, env_left, env_right, pair_info_left,
                                  model, strategies, return_timeseries=True)
            if res is None:
                print("overgeslagen (te weinig vensters)")
                continue

            for name, (acc, sw) in res.items():
                all_acc[name].append(acc)
                all_sw[name].append(sw)

            best_acc_here = max(res[n][0] for n in res)
            def rank_local(n):
                acc, sw = res[n]
                return (1, acc, -sw) if best_acc_here - acc < TIE_THRESHOLD_PCT else (0, acc, -sw)
            best = max(res, key=rank_local)
            print(f"best={best}  (acc={res[best][0]:.1f}%,  sw={res[best][1]})")
            n_done += 1

            if is_detail:
                pno = pair_no if USE_GSC_AUDIO else args.pair_no
                detail_result = (pno, res, ts)
                print(f"    → detail-data bewaard voor sub-{args.subject_no:03d}/pair{pno}")

    if n_done == 0:
        print("Geen proefpersonen verwerkt.")
        return

    # ── geaggregeerde resultaten ──────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"TOTAALOVERZICHT  ({n_done} proefpersonen)  "
          f"model={ACTIVE_MODEL}  audio={'GSC reverberant' if USE_GSC_AUDIO else 'clean speech'}")
    print(f"{'='*68}")
    print(f"{'Strategie':<24}  {'Gem.acc':>8}  {'Med.acc':>8}  {'Gem.sw':>7}")
    print("-" * 58)

    summary = {}
    for name in strategies:
        if name not in all_acc:
            continue
        accs = np.array(all_acc[name])
        sws  = np.array(all_sw[name])
        summary[name] = (np.mean(accs), np.median(accs), np.mean(sws))
        m_acc, med, m_sw = summary[name]
        print(f"{name:<24}  {m_acc:>7.1f}%  {med:>7.1f}%  {m_sw:>7.1f}")

    best_acc = max(summary[n][0] for n in summary)
    def rank_key(name):
        m_acc, _, m_sw = summary[name]
        return (1, m_acc, -m_sw) if best_acc - m_acc < TIE_THRESHOLD_PCT else (0, m_acc, -m_sw)

    winner = max(summary, key=rank_key)
    m_acc, med, m_sw = summary[winner]
    print(f"\n{'='*68}")
    print(f"  EINDWINNAAR: '{winner}'")
    print(f"  Model       : {ACTIVE_MODEL}  |  hop={STEP_SEC}s  |  "
          f"audio={'GSC reverberant' if USE_GSC_AUDIO else 'clean speech'}")
    print(f"  Gem. acc    : {m_acc:.1f}%")
    print(f"  Mediaan acc : {med:.1f}%")
    print(f"  Gem. sw     : {m_sw:.1f}  (alleen ter info)")
    print(f"  (gelijkspeldrempel: {TIE_THRESHOLD_PCT}%)")
    print(f"{'='*68}\n")

    # ── aggregaat staafdiagram ────────────────────────────────────────────────
    names  = list(summary.keys())
    accs   = [summary[n][0] for n in names]
    sws    = [summary[n][2] for n in names]
    colors = ["gold" if n == winner else "steelblue" for n in names]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    bars1 = ax1.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Gemiddelde nauwkeurigheid (%)")
    ax1.set_title(
        f"Hysteresis-vergelijking  |  {n_done} proefpersonen  |  winnaar='{winner}'\n"
        f"model={ACTIVE_MODEL}  hop={STEP_SEC}s  audio={'GSC reverberant' if USE_GSC_AUDIO else 'clean speech'}"
    )
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

    # ── detail-plot ───────────────────────────────────────────────────────────
    if detail_mode:
        if detail_result is None:
            print(f"\n[WARN] sub-{args.subject_no:03d} niet gevonden in de data.")
        else:
            pno, res, (raw_probs, window_gt, decisions) = detail_result
            plot_subject_detail(pno, args.subject_no,
                                raw_probs, window_gt, decisions,
                                res, strategies)


if __name__ == "__main__":
    main()
