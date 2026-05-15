"""
test_model_gsc.py  —  vergelijk alle 4 modellen met reverberant GSC-audio

Identiek aan test_model.py, maar in plaats van clean speech envelopen worden de
gammatone-envelopen berekend op de GSC-output van de beamformer na verwerking
van de reverberante LMA-opnames.

Dit test de realistische conditie: hoe goed werkt het AAD-model als de audio
afkomstig is van de microfoonarray in een reverberante omgeving (T60=200 ms)?

Wat anders is t.o.v. test_model.py:
  • mixture_LMA.wav (5 kanalen, 16 kHz, reverberant) → GSC beamformer
  • GSC-output: gsc_left (linkerbeam) en gsc_right (rechterbeam) bij 16 kHz
  • Gammatone-envelop berekend op 16 kHz signalen (i.p.v. 48 kHz clean speech)
  • RIR van het reverberante scenario geladen voor MUSIC DOA + GSC filters

Wat hetzelfde is:
  • EEG preprocessing en GT-conventie (gt_left=1 = attending LEFT)
  • Modelarchitectuur en inferentie
  • Filters (raw + EMA α=0.3)
  • TEST_CONFIGS, rapportage en plot

Gebruik
───────
  python test_model_gsc.py                   # volledige run (~60-180 min)
  python test_model_gsc.py --max_sec 60      # snelle test: eerste 60s per opname
"""

import os, sys, glob, warnings, time, argparse
warnings.filterwarnings("ignore")
import logging
logging.getLogger("brian2").setLevel(logging.ERROR)

import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from math import gcd
from collections import defaultdict, deque

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
REVERB_DIR  = "data/phase3_audioData/audiodata_batch_1/reverberant"
RIR_PATH    = "data/phase3_audioData/audiodata_batch_1/reverberant/lma_16kHz_200ms.npz"
TARGET_FS   = 64
EEG_FS_IN   = 128
GSC_FS      = 16000        # beamformer output sample rate
MAX_SEC     = args.max_sec
EMA_ALPHA   = 0.3

OUTPUT_PNG  = "model_vergelijking_gsc.png"

# Beamformer parameters (zelfde als Processor in processor.py)
_L    = 1024           # FFT venstergrootte
_HOP  = _L // 2       # 512 samples, 50% overlap
_BETA = 0.85           # Ryy smoothing
_MU   = 0.01           # NLMS stapgrootte
_M    = 5              # aantal LMA microfoons
_Q    = 2              # aantal sprekers

# DOA mediaan filter (zelfde als processor.py defaults: median N=63)
_DOA_FILTER_N = 63

# Model × hop combinaties (zelfde als test_model.py)
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

# Koppeling pair-nummer → {left: stimulus, right: stimulus}
# Overgenomen uit server/issp_data.py
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

# Reverse-mapping: frozenset({stim0, stim1}) → pair_no
_STIMSET_TO_PAIR = {
    frozenset({v["left"], v["right"]}): pair_no
    for pair_no, v in PAIR_MAPPING.items()
}


# ════════════════════════════════════════════════════════════════════════════
#  GSC BEAMFORMER (offline, replica van processor.py)
# ════════════════════════════════════════════════════════════════════════════

def _build_lut(rir, L=_L):
    """FAS + Blocking matrix voor één RIR (zelfde als build_lut_for_target in processor.py)."""
    n_bins = L // 2 + 1
    M = rir.shape[1]
    H_omega = np.fft.rfft(rir, n=L, axis=0)
    W_FAS = np.zeros((n_bins, M), dtype=complex)
    B_mat = np.zeros((n_bins, M - 1, M), dtype=complex)
    for k in range(n_bins):
        h_k = H_omega[k, :].reshape(M, 1)
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

    Parameters
    ----------
    lma_data : (N, 5) int16 of float ndarray
        Vijf-kanaal LMA mengopname (mixture_LMA.wav).
    rir_path : str
        Pad naar het reverberante RIR-bestand (.npz).
    max_samples : int or None
        Verwerk alleen de eerste max_samples samples (voor --max_sec).

    Returns
    -------
    gsc_left  : (K,) float32  — linkerbeam @ 16 kHz
    gsc_right : (K,) float32  — rechterbeam @ 16 kHz
    """
    if max_samples is not None:
        lma_data = lma_data[:max_samples]

    lma_f = lma_data.astype(np.float64)
    N, M  = lma_f.shape
    L, hop, beta, mu, Q = _L, _HOP, _BETA, _MU, _Q
    n_bins = L // 2 + 1

    # ── RIR laden en LUT bouwen ───────────────────────────────────────────────
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

    # ── Beamformer staat ──────────────────────────────────────────────────────
    window          = np.sqrt(signal.windows.hann(L, sym=False))
    audio_buf       = np.zeros((L, M))
    Ryy             = np.zeros((n_bins, M, M), dtype=complex)
    w_left          = np.zeros((n_bins, M - 1), dtype=complex)
    w_right         = np.zeros((n_bins, M - 1), dtype=complex)
    ola_left        = np.zeros(L)
    ola_right       = np.zeros(L)
    last_l          = 135.0   # starthoek links
    last_r          = 45.0    # starthoek rechts
    doa_buf_l       = deque(maxlen=_DOA_FILTER_N)
    doa_buf_r       = deque(maxlen=_DOA_FILTER_N)
    PEAK_THR        = -12.0
    valid_k         = np.arange(1, L // 2)
    left_mask       = lut_angles > 90
    right_mask      = lut_angles <= 90

    gsc_l_out = []
    gsc_r_out = []
    n_hops    = (N - L) // hop + 1

    for h in range(n_hops):
        s = h * hop
        hop_samples = lma_f[s:s + hop]

        # Sliding window
        audio_buf = np.roll(audio_buf, -hop, axis=0)
        audio_buf[-hop:] = hop_samples

        # STFT
        frame_fft = np.fft.rfft(audio_buf * window[:, np.newaxis], n=L, axis=0)

        # MUSIC DOA (zelfde als processor.py)
        Y          = frame_fft[valid_k, :, np.newaxis]
        R_k        = Y @ Y.conj().transpose(0, 2, 1)
        Ryy[valid_k] = beta * Ryy[valid_k] + (1 - beta) * R_k
        _, eigvecs  = np.linalg.eigh(Ryy[valid_k])
        En          = eigvecs[:, :, :M - Q]
        En_H_A      = En.conj().transpose(0, 2, 1) @ A_lut[valid_k]
        denom       = np.sum(np.abs(En_H_A) ** 2, axis=1)
        p           = 1.0 / np.clip(denom, 1e-10, None)
        log_p       = np.log(np.clip(p, 1e-10, None))
        spec_db     = 10 * np.log10(
            np.exp(np.mean(log_p, axis=0)) /
            np.maximum(np.max(np.exp(np.mean(log_p, axis=0))), 1e-20)
        )

        bl = np.argmax(np.where(left_mask,  spec_db, -np.inf))
        br = np.argmax(np.where(right_mask, spec_db, -np.inf))
        if spec_db[bl] > PEAK_THR:
            last_l = lut_angles[bl]
        if spec_db[br] > PEAK_THR:
            last_r = lut_angles[br]

        # Mediaan DOA filter (N=63, causaal)
        doa_buf_l.append(last_l)
        doa_buf_r.append(last_r)
        angle_l = float(np.median(doa_buf_l))
        angle_r = float(np.median(doa_buf_r))

        # FD-GSC
        W_L, B_L = _get_lut(angle_l)
        W_R, B_R = _get_lut(angle_r)

        def _gsc(fft, W, B, w):
            y_fas = np.sum(np.conj(W) * fft, axis=1)
            u     = np.einsum('nij,nj->ni', B, fft)
            y_bm  = np.sum(np.conj(w) * u, axis=1)
            e     = y_fas - y_bm
            power = np.real(np.sum(np.conj(u) * u, axis=1))
            w    += mu * u * np.conj(e)[:, np.newaxis] / (power[:, np.newaxis] + 1e-8)
            return e

        out_l = np.fft.irfft(_gsc(frame_fft, W_L, B_L, w_left),  n=L) * window
        out_r = np.fft.irfft(_gsc(frame_fft, W_R, B_R, w_right), n=L) * window

        ola_left  += out_l
        ola_right += out_r

        gsc_l_out.append(ola_left[:hop].copy())
        gsc_r_out.append(ola_right[:hop].copy())

        ola_left  = np.concatenate([ola_left[hop:],  np.zeros(hop)])
        ola_right = np.concatenate([ola_right[hop:], np.zeros(hop)])

    return (np.concatenate(gsc_l_out).astype(np.float32),
            np.concatenate(gsc_r_out).astype(np.float32))


# ════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (zelfde als processor.py en test_model.py)
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
    sos = signal.butter(4, [lowcut, highcut], btype="bandpass", fs=fs_in, output="sos")
    eeg_f = signal.sosfiltfilt(sos, eeg, axis=0)
    g = gcd(int(fs_in), fs_out)
    return signal.resample_poly(eeg_f, fs_out // g, int(fs_in) // g, axis=0)


# ════════════════════════════════════════════════════════════════════════════
#  FILTERS  (zelfde als test_model.py)
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
#  INFERENTIE PER SUBJECT  (zelfde als test_model.py)
# ════════════════════════════════════════════════════════════════════════════

def run_subject_batch(eeg_proc, env_left_full, env_right_full,
                      gt_ds, model, win_samples, hop_sec):
    step  = int(round(hop_sec * TARGET_FS))
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
    print(f"  MODEL VERGELIJKING (GSC-audio)  —  test_model_gsc.py")
    print(f"  Scenario : reverberant (T60=200ms)")
    print(f"  Audio    : GSC-output beamformer (16 kHz)")
    if MAX_SEC:
        print(f"  ⚠  MAX_SEC={MAX_SEC}s  (niet de volledige opname)")
    print(f"{'='*W}\n")

    all_eeg_files = sorted(glob.glob(os.path.join(DATA_DIR, "sub-*", "*.npz")))
    print(f"  {len(all_eeg_files)} proefpersoon-bestanden gevonden.")

    # Groepeer EEG-bestanden per uniek audiodopaar
    pair_groups = defaultdict(list)
    for f in all_eeg_files:
        npz = np.load(f)
        key = frozenset({str(npz["stimulus_0"]), str(npz["stimulus_1"])})
        pair_groups[key].append(f)
    print(f"  {len(pair_groups)} unieke audioparen.\n")

    # Controleer welke paren in PAIR_MAPPING zitten
    unknown = [k for k in pair_groups if k not in _STIMSET_TO_PAIR]
    if unknown:
        print(f"  [WARN] {len(unknown)} paren niet gevonden in PAIR_MAPPING (overgeslagen).")

    model_to_hops = defaultdict(set)
    for model_name, hop_sec in TEST_CONFIGS:
        model_to_hops[model_name].add(hop_sec)

    results = {cfg: {"raw": [], "ema": []} for cfg in TEST_CONFIGS}

    # ── Loop over modellen ────────────────────────────────────────────────────
    for model_name, test_hops in model_to_hops.items():
        m_cfg       = MODELS[model_name]
        win_samples = m_cfg["eeg_window_samples"]
        print(f"\n{'─'*W}")
        print(f"  Model: {model_name}  —  {m_cfg['description']}")
        print(f"  Venster: {m_cfg['window_sec']}s  ({win_samples} samples @ {TARGET_FS} Hz)")
        print(f"  Te testen hops: {sorted(test_hops)}s")
        print(f"{'─'*W}")

        model = tf.keras.models.load_model(m_cfg["model_path"])
        print(f"  Model geladen: {m_cfg['model_path']}")

        # GSC envelop cache per pair_no (beamformer output is model-onafhankelijk)
        # {pair_no: (env_left_full, env_right_full)}
        gsc_env_cache = {}

        n_total = sum(len(fs) for fs in pair_groups.values())
        n_done  = 0

        for stim_key, eeg_files in sorted(pair_groups.items(), key=lambda x: str(sorted(x[0]))):
            # Zoek pair_no voor dit stimulus-paar
            if stim_key not in _STIMSET_TO_PAIR:
                n_done += len(eeg_files)
                continue
            pair_no = _STIMSET_TO_PAIR[stim_key]
            pair_info = PAIR_MAPPING[pair_no]
            lma_path  = os.path.join(REVERB_DIR, f"pair{pair_no}", "mixture_LMA.wav")

            if not os.path.exists(lma_path):
                print(f"  [SKIP] LMA niet gevonden: {lma_path}")
                n_done += len(eeg_files)
                continue

            # ── GSC + envelop (één keer per pair, hergebruikt voor alle modellen) ──
            if pair_no not in gsc_env_cache:
                t_gsc = time.time()
                _, lma_data = wavfile.read(lma_path)  # (N, 5) int16 @ 16 kHz
                max_samp = MAX_SEC * GSC_FS if MAX_SEC else None

                print(f"  Beamformer pair{pair_no}  "
                      f"({lma_data.shape[0]/GSC_FS:.0f}s)  ... ", end="", flush=True)
                gsc_left, gsc_right = run_gsc_offline(lma_data, max_samples=max_samp)
                print(f"klaar in {time.time()-t_gsc:.0f}s  "
                      f"(output: {len(gsc_left)/GSC_FS:.0f}s @ {GSC_FS} Hz)")

                # Envelop op GSC-output (16 kHz → 64 Hz)
                t_env = time.time()
                env_left_raw  = compute_audio_envelope(gsc_left,  sr_in=GSC_FS)
                env_right_raw = compute_audio_envelope(gsc_right, sr_in=GSC_FS)

                # Links/rechts op basis van PAIR_MAPPING
                # pair_info["left"] = naam van de LINKER stimulus
                # De linkerbeam (gsc_left) is gericht op angle>90° = linker spreker
                # → linkerbeam komt overeen met de linker stimulus
                # De stim_key frozenset zegt niet welke stim0 is, maar dat maakt niet uit
                # want we slaan op als (env_left, env_right) = (linkerbeam, rechterbeam)
                gsc_env_cache[pair_no] = (env_left_raw, env_right_raw)
                print(f"  Envelop pair{pair_no} berekend in {time.time()-t_env:.0f}s")
            else:
                env_left_raw, env_right_raw = gsc_env_cache[pair_no]

            # ── EEG-bestanden voor dit audiodopaar ────────────────────────────
            for eeg_file in sorted(eeg_files):
                n_done += 1
                npz    = np.load(eeg_file)
                stim0  = str(npz["stimulus_0"])
                raw_gt = npz["attended_speaker"].astype(int)

                total_eeg = npz["eeg"].shape[0]
                if MAX_SEC:
                    total_eeg = min(total_eeg, MAX_SEC * EEG_FS_IN)

                # EEG preprocessing
                eeg_proc = preprocess_eeg(npz["eeg"][:total_eeg].astype(np.float64))

                # GT in "links-geattendeerd"-conventie (gt_left=1 = attending LEFT)
                # attended_speaker=0 in npz betekent attending stim0
                left_stim = pair_info["left"]
                swap      = (left_stim != stim0)      # True als stim0 de RECHTER spreker is
                gt_raw    = raw_gt[:total_eeg]
                gt_left   = gt_raw if swap else 1 - gt_raw

                ratio     = EEG_FS_IN // TARGET_FS
                gt_ds_len = int(total_eeg * TARGET_FS / EEG_FS_IN)
                gt_ds = np.array([
                    round(float(np.mean(gt_left[i*ratio:(i+1)*ratio])))
                    for i in range(gt_ds_len)
                ])

                # Trunceer enveloppen tot de EEG-lengte
                env_left  = env_left_raw[:gt_ds_len]
                env_right = env_right_raw[:gt_ds_len]

                subj     = os.path.basename(eeg_file).split("_")[0]
                hop_line = []

                for hop_sec in sorted(test_hops):
                    res = run_subject_batch(
                        eeg_proc, env_left, env_right,
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

                print(f"  [{n_done:>3}/{n_total}]  {subj:<10}  pair{pair_no:<3}  "
                      + "  ".join(hop_line))

        del model
        tf.keras.backend.clear_session()

    # ════════════════════════════════════════════════════════════════════════
    #  RAPPORT
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*W}")
    print(f"  EINDRESULTATEN  (reverberant GSC-audio)")
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
            "label":   f"{model_name}  hop={hop_sec}s",
            "n":       len(raw_arr),
            "raw_gem": np.mean(raw_arr),
            "raw_med": np.median(raw_arr),
            "ema_gem": np.mean(ema_arr),
            "ema_med": np.median(ema_arr),
        })

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
        f"Model vergelijking (GSC-audio, reverberant)  —  {rows[0]['n']} proefpersonen"
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
