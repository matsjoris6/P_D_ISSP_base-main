"""
Final evaluatie over alle 4 configuraties:
  {anechoic, reverberant} x {LMA, HMA}

Per subject/pair worden gemeten:
  - DOA error : gem. abs. fout (links en rechts gecombineerd) over alle hops
  - SIR oracle: gewogen gemiddelde van SIR_left en SIR_right op basis van de
                ground-truth attended speaker. Dit is de SIR van de "final output"
                zoals de opdracht vraagt, met perfecte AAD.
  - SIR avg   : (SIR_left + SIR_right)/2, geen attention-bias, ter referentie

Vereist kleine aanpassing in processor.py:
  - in __init__:
        self.sir_history_left  = []
        self.sir_history_right = []
  - in processing_microarray, direct na de compute_sir(...) berekening (rond r. 465):
        self.sir_history_left.append(sir_left if not np.isnan(sir_left) else 0.0)
        self.sir_history_right.append(sir_right if not np.isnan(sir_right) else 0.0)
"""
import os
import time
import numpy as np
from scipy.io import wavfile

from processor import Processor

# === CONFIG ===
EEG_BASE = "data/phase3_test/eeg_data"
TEST_BASE = "data/phase3_test/audio_data"

CONFIGS = [
    ("anechoic_LMA",     "anechoic/lma_16kHz.npz",            "anechoic",    "LMA", 5),
    #("anechoic_HMA",     "anechoic/hma_16kHz.npz",            "anechoic",    "HMA", 4),
    #("reverberant_LMA",  "reverberant/lma_16kHz_200ms.npz",   "reverberant", "LMA", 5),
    #("reverberant_HMA",  "reverberant/hma_16kHz_200ms.npz",   "reverberant", "HMA", 4),
]

LEFTRIGHT_MAPPING = {
    16: ("audiobook_2_2_part3.wav",  "audiobook_1_part3.wav"),
    17: ("podcast_4_part3.wav",      "podcast_3_part3.wav"),
    18: ("audiobook_8_1_part3.wav",  "audiobook_8_2_part3.wav"),
    19: ("audiobook_9_1_part3.wav",  "audiobook_9_2_part3.wav"),
    20: ("audiobook_10_1_part3.wav", "audiobook_10_2_part3.wav"),
    21: ("audiobook_11_1_part3.wav", "audiobook_11_2_part3.wav"),
    22: ("podcast_22_part3.wav",     "podcast_21_part3.wav"),
    23: ("podcast_24_part3.wav",     "podcast_25_part3.wav"),
    24: ("podcast_30_part3.wav",     "podcast_31_part3.wav"),
    25: ("audiobook_14_2_part3.wav", "podcast_32_part3.wav"),
    28: ("podcast_35_part3.wav",     "audiobook_14_2_part3.wav"),
}
PAIR_LOOKUP = {tuple(sorted([l, r])): p for p, (l, r) in LEFTRIGHT_MAPPING.items()}

TEST_SUBJECTS = [
    "sub-002", "sub-003", "sub-027", "sub-032", "sub-037", "sub-043",
    "sub-047", "sub-049", "sub-057", "sub-063", "sub-072", "sub-079",
]

CHUNK_SIZE = 16000 // 32
EVAL_SECONDS = 60
HOP = 512


def find_pair_for_subject(subject):
    eeg_dir = os.path.join(EEG_BASE, subject)
    if not os.path.exists(eeg_dir):
        return None, None, None
    npz_files = [f for f in os.listdir(eeg_dir) if f.endswith(".npz")]
    if not npz_files:
        return None, None, None
    data = np.load(os.path.join(eeg_dir, npz_files[0]))
    stim_0 = str(data["stimulus_0"])
    stim_1 = str(data["stimulus_1"])
    pair_no = PAIR_LOOKUP.get(tuple(sorted([stim_0, stim_1])))
    attended_speaker = data["attended_speaker"]
    eeg_fs = int(data["fs"])
    return pair_no, attended_speaker, eeg_fs


def test_configuration(config_name, rir_rel_path, audio_subdir, array_type, n_mics):
    print(f"\n{'=' * 90}")
    print(f"CONFIG: {config_name}")
    print(f"{'=' * 90}")

    rir_path = os.path.join(TEST_BASE, rir_rel_path)
    audio_base = os.path.join(TEST_BASE, audio_subdir)

    results = []

    for sub_idx, subject in enumerate(TEST_SUBJECTS, 1):
        pair_no, att_speaker_eeg, eeg_fs = find_pair_for_subject(subject)
        if pair_no is None:
            print(f"[{sub_idx:2d}/{len(TEST_SUBJECTS)}] {subject}: SKIP (geen pair match)")
            continue

        pair_dir = os.path.join(audio_base, f"pair{pair_no}")
        if not os.path.exists(pair_dir):
            print(f"[{sub_idx:2d}/{len(TEST_SUBJECTS)}] {subject} pair{pair_no}: SKIP (geen audio dir)")
            continue

        # Audio
        try:
            fs, mixture = wavfile.read(os.path.join(pair_dir, f"mixture_{array_type}.wav"))
            _, gt0      = wavfile.read(os.path.join(pair_dir, f"leftSpeaker_{array_type}.wav"))
            _, gt1      = wavfile.read(os.path.join(pair_dir, f"rightSpeaker_{array_type}.wav"))
        except Exception as e:
            print(f"[{sub_idx:2d}/{len(TEST_SUBJECTS)}] {subject}: SKIP ({e})")
            continue

        # DOA ground truth
        try:
            gt_npz = np.load(os.path.join(pair_dir, "gt.npz"))
            durations_l = np.diff(np.insert(gt_npz["endSamples_l"], 0, 0))
            durations_r = np.diff(np.insert(gt_npz["endSamples_r"], 0, 0))
            doa_gt_l = np.concatenate([np.repeat(e, n) for e, n in zip(gt_npz["angles_l"], durations_l)])
            doa_gt_r = np.concatenate([np.repeat(e, n) for e, n in zip(gt_npz["angles_r"], durations_r)])
        except Exception as e:
            print(f"[{sub_idx:2d}/{len(TEST_SUBJECTS)}] {subject}: SKIP (geen gt.npz: {e})")
            continue

        # Beperk tot 60s
        n_samples = EVAL_SECONDS * fs
        mixture = mixture[:n_samples]
        gt0     = gt0[:n_samples]
        gt1     = gt1[:n_samples]

        # Attended speaker per 1s venster (= 1 SIR-update)
        # att_speaker is op EEG-rate; we kijken naar de modus per 1s
        att_per_sec = []
        for sec in range(EVAL_SECONDS):
            window_start = sec * eeg_fs
            window_end   = (sec + 1) * eeg_fs
            if window_end > len(att_speaker_eeg):
                break
            window = att_speaker_eeg[window_start:window_end]
            # modus: 1 (links attended) of 0 (rechts attended)
            att_per_sec.append(int(np.round(np.mean(window))))
        att_per_sec = np.array(att_per_sec)

        # Processor opzetten
        proc = Processor(rir_path=rir_path)
        proc.M = n_mics
        proc.audio_buffer       = np.zeros((proc.L, n_mics))
        proc.audio_buffer_gt0   = np.zeros((proc.L, n_mics))
        proc.audio_buffer_gt1   = np.zeros((proc.L, n_mics))
        proc.input_accumulator      = np.zeros((0, n_mics))
        proc.input_accumulator_gt0  = np.zeros((0, n_mics))
        proc.input_accumulator_gt1  = np.zeros((0, n_mics))

        # We hebben self.sir_history_left/right nodig (zie processor-aanpassing in docstring)
        if not hasattr(proc, "sir_history_left"):
            print(f"  FOUT: processor heeft geen sir_history_left attribuut.")
            print(f"  Pas processor.py aan zoals beschreven in de docstring van dit script.")
            return results

        # Streaming run
        n_chunks = len(mixture) // CHUNK_SIZE
        doa_history_l = []
        doa_history_r = []

        t0 = time.time()
        for i in range(n_chunks):
            chunk_mix = mixture[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE, :]
            chunk_g0  = gt0[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE, :]
            chunk_g1  = gt1[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE, :]
            proc.processing_microarray(chunk_mix, chunk_g0, chunk_g1)

            while not proc.data_queue_phase1.empty():
                _, _, doa_l, doa_r, _ = proc.data_queue_phase1.get_nowait()
                doa_history_l.append(doa_l)
                doa_history_r.append(doa_r)
            while not proc.data_queue_phase3.empty():
                proc.data_queue_phase3.get_nowait()

        elapsed = time.time() - t0

        # === DOA error ===
        doa_err_l = []
        doa_err_r = []
        for k in range(len(doa_history_l)):
            sample_idx = int((k + 0.5) * HOP)
            if sample_idx < len(doa_gt_l):
                doa_err_l.append(abs(doa_history_l[k] - doa_gt_l[sample_idx]))
            if sample_idx < len(doa_gt_r):
                doa_err_r.append(abs(doa_history_r[k] - doa_gt_r[sample_idx]))
        doa_err = float(np.mean(doa_err_l + doa_err_r)) if (doa_err_l and doa_err_r) else float("nan")

        # === SIR ===
        # proc.sir_history_left/right hebben één entry per 1s venster (zoals processor-code)
        sir_L = np.array(proc.sir_history_left)
        sir_R = np.array(proc.sir_history_right)

        # Match lengtes met att_per_sec (kleinste wint)
        n_sec = min(len(sir_L), len(sir_R), len(att_per_sec))
        sir_L = sir_L[:n_sec]
        sir_R = sir_R[:n_sec]
        att_v = att_per_sec[:n_sec]

        # Oracle SIR: kies SIR_left als attended=1 (links), anders SIR_right
        sir_oracle_per_sec = np.where(att_v == 1, sir_L, sir_R)
        sir_avg_per_sec    = 0.5 * (sir_L + sir_R)

        # Filter NaN/0.0 out (in case van convergentie-failures of stilte)
        valid_oracle = sir_oracle_per_sec[~np.isnan(sir_oracle_per_sec) & (sir_oracle_per_sec != 0.0)]
        valid_avg    = sir_avg_per_sec   [~np.isnan(sir_avg_per_sec)    & (sir_avg_per_sec    != 0.0)]
        valid_L      = sir_L             [~np.isnan(sir_L)              & (sir_L              != 0.0)]
        valid_R      = sir_R             [~np.isnan(sir_R)              & (sir_R              != 0.0)]

        sir_oracle_mean = float(valid_oracle.mean()) if len(valid_oracle) else float("nan")
        sir_avg_mean    = float(valid_avg.mean())    if len(valid_avg)    else float("nan")
        sir_L_mean      = float(valid_L.mean())      if len(valid_L)      else float("nan")
        sir_R_mean      = float(valid_R.mean())      if len(valid_R)      else float("nan")
        frac_left       = float(np.mean(att_v == 1))

        result = {
            "subject":      subject,
            "pair":         pair_no,
            "doa_err":      doa_err,
            "sir_L":        sir_L_mean,
            "sir_R":        sir_R_mean,
            "sir_avg":      sir_avg_mean,
            "sir_oracle":   sir_oracle_mean,
            "frac_left":    frac_left,
            "n_sec":        n_sec,
            "elapsed":      elapsed,
        }
        results.append(result)

        print(f"[{sub_idx:2d}/{len(TEST_SUBJECTS)}] {subject} pair{pair_no:2d}: "
              f"DOA={doa_err:5.2f}°  SIR L={sir_L_mean:+5.1f} R={sir_R_mean:+5.1f}  "
              f"avg={sir_avg_mean:+5.1f}  oracle={sir_oracle_mean:+5.1f}  "
              f"(fL={frac_left:.2f}, {n_sec}s, {elapsed:.0f}s)")

    return results


# === MAIN ===
all_results = {}
t_total = time.time()

for config_name, rir_path, audio_subdir, array_type, n_mics in CONFIGS:
    all_results[config_name] = test_configuration(config_name, rir_path, audio_subdir, array_type, n_mics)

print(f"\n\nTotale runtime: {(time.time() - t_total)/60:.1f} minuten")

# === SAMENVATTING ===
print("\n" + "=" * 100)
print("EINDSAMENVATTING")
print("=" * 100)
print(f"\n{'Config':>18} | {'N':>3} | {'DOA (°)':>9} | "
      f"{'SIR L (dB)':>11} | {'SIR R (dB)':>11} | "
      f"{'SIR avg (dB)':>13} | {'SIR oracle (dB)':>17}")
print("-" * 100)

for config_name, results in all_results.items():
    if not results:
        print(f"{config_name:>18} | (geen resultaten)")
        continue
    doa  = np.array([r["doa_err"]    for r in results])
    L    = np.array([r["sir_L"]      for r in results])
    R    = np.array([r["sir_R"]      for r in results])
    avg  = np.array([r["sir_avg"]    for r in results])
    orc  = np.array([r["sir_oracle"] for r in results])
    print(f"{config_name:>18} | {len(results):>3d} | "
          f"{np.nanmean(doa):>8.2f}° | "
          f"{np.nanmean(L):>+10.2f} | {np.nanmean(R):>+10.2f} | "
          f"{np.nanmean(avg):>+12.2f} | {np.nanmean(orc):>+16.2f}")

print("\nDOA        = gem. abs. fout L+R over alle hops, gemiddeld over subjects")
print("SIR L/R    = gem. SIR van links/rechts beam, gemiddeld over subjects")
print("SIR avg    = (L+R)/2 per seconde, dan gemiddeld")
print("SIR oracle = SIR van beam die ground-truth attended speaker volgt, per seconde, dan gem.")