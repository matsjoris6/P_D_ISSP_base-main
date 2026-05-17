"""
Test DOA error en SIR over alle 4 configuraties:
  {anechoic, reverberant} x {LMA, HMA}

Per configuratie:
  - DOA error: gemiddelde absolute fout vs ground truth hoeken
  - SIR: gemeten over eerste 60 seconden van beamformer output

Loopt over een subset van subjects om binnen redelijke tijd te blijven.
Output: tabel met de 4 configuraties + per-pair detail.
"""
import os
import time
import numpy as np
from scipy.io import wavfile

from processor import Processor

# === CONFIG ===
EEG_BASE = "data/phase3_test/eeg_data"
TEST_BASE = "data/phase3_test/audio_data"

# Configuraties die we testen
CONFIGS = [
    ("anechoic_LMA",     "anechoic/lma_16kHz.npz",            "anechoic",    "LMA", 5),
    #("anechoic_HMA",     "anechoic/hma_16kHz.npz",            "anechoic",    "HMA", 4),
    ("reverberant_LMA",  "reverberant/lma_16kHz_200ms.npz",   "reverberant", "LMA", 5),
    #("reverberant_HMA",  "reverberant/hma_16kHz_200ms.npz",   "reverberant", "HMA", 4),
]

# Pair mapping (zoals issp_data.py)
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

# Subset van subjects (1-2 per pair voor brede coverage zonder oneindig wachten)
TEST_SUBJECTS = [
    "sub-002",  # pair 16
    "sub-003",  # pair 16
    "sub-027",  # pair 17
    "sub-032",  # pair 18
    "sub-037",  # pair 19
    "sub-043",  # pair 20
    "sub-047",  # pair 21
    "sub-049",  # pair 22
    "sub-057",  # pair 23
    "sub-063",  # pair 24
    "sub-072",  # pair 25
    "sub-079",  # pair 28
]

CHUNK_SIZE = 16000 // 32   # 500 samples per chunk (= 32 chunks per seconde, zoals server)
SIR_SECONDS = 60
DOA_TICK_RATE = 32         # 32 DOA updates per seconde


def find_pair_for_subject(subject):
    eeg_dir = os.path.join(EEG_BASE, subject)
    npz = [f for f in os.listdir(eeg_dir) if f.endswith(".npz")][0]
    data = np.load(os.path.join(eeg_dir, npz))
    stim_0 = str(data["stimulus_0"])
    stim_1 = str(data["stimulus_1"])
    pair_no = PAIR_LOOKUP.get(tuple(sorted([stim_0, stim_1])))
    return pair_no, stim_0, stim_1


def test_configuration(config_name, rir_rel_path, audio_subdir, array_type, n_mics):
    print(f"\n{'=' * 70}")
    print(f"CONFIG: {config_name}")
    print(f"{'=' * 70}")

    rir_path = os.path.join(TEST_BASE, rir_rel_path)
    audio_base = os.path.join(TEST_BASE, audio_subdir)

    results = []

    for sub_idx, subject in enumerate(TEST_SUBJECTS, 1):
        pair_no, stim_0, stim_1 = find_pair_for_subject(subject)
        if pair_no is None:
            print(f"[{sub_idx:2d}/{len(TEST_SUBJECTS)}] {subject}: SKIP (geen pair match)")
            continue

        pair_dir = os.path.join(audio_base, f"pair{pair_no}")
        if not os.path.exists(pair_dir):
            print(f"[{sub_idx:2d}/{len(TEST_SUBJECTS)}] {subject} pair{pair_no}: SKIP (geen audio dir {pair_dir})")
            continue

        # Laad audio
        try:
            fs, mixture = wavfile.read(os.path.join(pair_dir, f"mixture_{array_type}.wav"))
            _, gt0      = wavfile.read(os.path.join(pair_dir, f"leftSpeaker_{array_type}.wav"))
            _, gt1      = wavfile.read(os.path.join(pair_dir, f"rightSpeaker_{array_type}.wav"))
        except Exception as e:
            print(f"[{sub_idx:2d}/{len(TEST_SUBJECTS)}] {subject}: SKIP ({e})")
            continue

        # Laad DOA ground truth
        try:
            gt_npz = np.load(os.path.join(pair_dir, "gt.npz"))
            durations_l = np.diff(np.insert(gt_npz["endSamples_l"], 0, 0))
            durations_r = np.diff(np.insert(gt_npz["endSamples_r"], 0, 0))
            doa_gt_l = np.concatenate([np.repeat(e, n) for e, n in zip(gt_npz["angles_l"], durations_l)])
            doa_gt_r = np.concatenate([np.repeat(e, n) for e, n in zip(gt_npz["angles_r"], durations_r)])
        except Exception as e:
            print(f"[{sub_idx:2d}/{len(TEST_SUBJECTS)}] {subject}: SKIP (geen gt.npz: {e})")
            continue

        # Bouw processor
        proc = Processor(rir_path=rir_path)
        proc.M = n_mics
        # Heeropbouwen van buffers met juiste M
        proc.audio_buffer       = np.zeros((proc.L, n_mics))
        proc.audio_buffer_gt0   = np.zeros((proc.L, n_mics))
        proc.audio_buffer_gt1   = np.zeros((proc.L, n_mics))
        proc.input_accumulator      = np.zeros((0, n_mics))
        proc.input_accumulator_gt0  = np.zeros((0, n_mics))
        proc.input_accumulator_gt1  = np.zeros((0, n_mics))

        # Forceer attended_left=1 voor consistente SIR meting (LEFT beam)
        # Live zou AAD dit bepalen; voor isolatie van de DOA/GSC kwaliteit kiezen we vast
        proc.attended_left = 1

        # Beperk tot 60s data voor snelheid
        n_samples_60s = SIR_SECONDS * fs
        mixture = mixture[:n_samples_60s]
        gt0     = gt0[:n_samples_60s]
        gt1     = gt1[:n_samples_60s]

        # Stream chunk-per-chunk door de processor
        n_chunks = len(mixture) // CHUNK_SIZE
        doa_history_l = []
        doa_history_r = []
        sir_values = []

        t0 = time.time()
        for i in range(n_chunks):
            chunk_mix = mixture[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE, :]
            chunk_g0  = gt0[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE, :]
            chunk_g1  = gt1[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE, :]
            proc.processing_microarray(chunk_mix, chunk_g0, chunk_g1)

            # Trek output uit phase1 queue
            while not proc.data_queue_phase1.empty():
                _, _, doa_l, doa_r, sir = proc.data_queue_phase1.get_nowait()
                doa_history_l.append(doa_l)
                doa_history_r.append(doa_r)
                if sir != 0.0 and not np.isnan(sir):
                    sir_values.append(sir)

        elapsed = time.time() - t0

        # Bereken DOA error
        # DOA wordt elke hop (=32ms) gerapporteerd. Vergelijk met ground truth.
        # ground truth is op audio sample rate; we samplen het op DOA tick rate.
        n_doa_ticks = len(doa_history_l)
        ticks_per_sec = n_doa_ticks / SIR_SECONDS
        samples_per_tick = int(fs / ticks_per_sec)

        doa_err_l = []
        doa_err_r = []
        for k in range(min(n_doa_ticks, len(doa_gt_l) // samples_per_tick)):
            gt_idx = k * samples_per_tick
            if gt_idx < len(doa_gt_l):
                doa_err_l.append(abs(doa_history_l[k] - doa_gt_l[gt_idx]))
            if gt_idx < len(doa_gt_r):
                doa_err_r.append(abs(doa_history_r[k] - doa_gt_r[gt_idx]))

        # SIR statistieken (over 60s data)
        sir_arr = np.array(sir_values) if sir_values else np.array([0.0])

        result = {
            "subject":      subject,
            "pair":         pair_no,
            "doa_err_l":    float(np.mean(doa_err_l)) if doa_err_l else float("nan"),
            "doa_err_r":    float(np.mean(doa_err_r)) if doa_err_r else float("nan"),
            "sir_mean":     float(sir_arr.mean()),
            "sir_median":   float(np.median(sir_arr)),
            "sir_60s":      float(sir_arr.mean()),  # over 60s data
            "elapsed":      elapsed,
        }
        results.append(result)

        print(f"[{sub_idx:2d}/{len(TEST_SUBJECTS)}] {subject} pair{pair_no:2d}: "
              f"DOA_L={result['doa_err_l']:5.2f}°  DOA_R={result['doa_err_r']:5.2f}°  "
              f"SIR_mean={result['sir_mean']:+6.2f}dB  SIR_median={result['sir_median']:+6.2f}dB  "
              f"({elapsed:.1f}s)")

    return results


# === MAIN: loop alle 4 configuraties ===
all_results = {}
t_total = time.time()

for config_name, rir_path, audio_subdir, array_type, n_mics in CONFIGS:
    all_results[config_name] = test_configuration(config_name, rir_path, audio_subdir, array_type, n_mics)

print(f"\n\nTotale runtime: {(time.time() - t_total)/60:.1f} minuten")

# === SAMENVATTINGSTABEL ===
print("\n" + "=" * 90)
print("EINDSAMENVATTING")
print("=" * 90)
print(f"\n{'Config':>18} | {'N':>3} | {'DOA L (°)':>10} | {'DOA R (°)':>10} | "
      f"{'SIR mean (dB)':>13} | {'SIR median (dB)':>15}")
print("-" * 90)

for config_name, results in all_results.items():
    if not results:
        print(f"{config_name:>18} | (geen resultaten)")
        continue
    doa_l = np.array([r["doa_err_l"] for r in results])
    doa_r = np.array([r["doa_err_r"] for r in results])
    sir_m = np.array([r["sir_mean"]  for r in results])
    sir_med = np.array([r["sir_median"] for r in results])
    print(f"{config_name:>18} | {len(results):>3d} | "
          f"{np.nanmean(doa_l):>9.2f} | {np.nanmean(doa_r):>9.2f} | "
          f"{np.nanmean(sir_m):>+12.2f} | {np.nanmean(sir_med):>+14.2f}")

print("\nResultaten per configuratie opgeslagen in geheugen.")