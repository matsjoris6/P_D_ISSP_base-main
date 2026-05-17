"""
Beta optimalisatie voor MUSIC DOA tracking.
Test verschillende beta (forgetting factor) waarden en rapporteert DOA-error statistieken.

Doel: vind de beta die de laagste DOA-error geeft in een tijd-varierend scenario.

Beta = 0.75 → kort geheugen, snelle reactie maar hoge variantie
Beta = 0.99 → lang geheugen, lage variantie maar trage reactie op DOA-switches
"""
import time
import numpy as np
from scipy.io import wavfile
from processor import Processor

# === CONFIG ===
# Standaard reverberant; voor anechoic: BASE_PATH naar 'anechoic' en RIR_PATH naar 'lma_16kHz.npz' (zonder _200ms)
BASE_PATH = "data/phase3_audioData/audiodata_batch_1/anechoic"
RIR_PATH  = "data/phase3_audioData/audiodata_batch_1/anechoic/lma_16kHz.npz"
GT_PATH   = "data/phase3_audioData/audiodata_batch_1/anechoic"  # gt.npz ligt naast de wavs
PAIR = 1
DURATION_SECONDS = 120

# Beta waarden om te testen
BETAS = [0.97]

# === DATA LADEN ===
print(f"Laden data voor pair{PAIR} ({DURATION_SECONDS}s reverberant)...")
fs, lma_audio = wavfile.read(f"{BASE_PATH}/pair{PAIR}/mixture_LMA.wav")
_, lma_gt0   = wavfile.read(f"{BASE_PATH}/pair{PAIR}/leftSpeaker_LMA.wav")
_, lma_gt1   = wavfile.read(f"{BASE_PATH}/pair{PAIR}/rightSpeaker_LMA.wav")

# Ground-truth DOA tracks laden
gt = np.load(f"{GT_PATH}/pair{PAIR}/gt.npz")
durations_l = np.diff(np.insert(gt["endSamples_l"], 0, 0))
durations_r = np.diff(np.insert(gt["endSamples_r"], 0, 0))
doa_gt_left  = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_l"], durations_l)])
doa_gt_right = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_r"], durations_r)])

n_samples = int(DURATION_SECONDS * fs)
lma_audio    = lma_audio[:n_samples]
lma_gt0      = lma_gt0[:n_samples]
lma_gt1      = lma_gt1[:n_samples]
doa_gt_left  = doa_gt_left[:n_samples]
doa_gt_right = doa_gt_right[:n_samples]

chunk_size = fs // 32
n_frames   = n_samples // chunk_size
hop        = 512  # samples per processor hop
print(f"Sample rate: {fs} Hz | {n_frames} chunks van {chunk_size} samples")
print(f"GT DOA samples: L {len(doa_gt_left)} | R {len(doa_gt_right)}\n")

# Detecteer DOA-switch momenten (in audio-samples) om "post-switch" error te kunnen meten
switch_samples_l = np.where(np.diff(doa_gt_left) != 0)[0] + 1
switch_samples_r = np.where(np.diff(doa_gt_right) != 0)[0] + 1
all_switches = np.sort(np.concatenate([switch_samples_l, switch_samples_r]))
print(f"Aantal DOA-switches in {DURATION_SECONDS}s: L={len(switch_samples_l)}, R={len(switch_samples_r)}\n")

# === RESULTATEN PER BETA ===
results = {}

for beta in BETAS:
    print(f"--- Testen beta={beta} ---")

    proc = Processor(rir_path=RIR_PATH)
    proc.beta = beta
    proc.peak_threshold = -np.inf
    doa_est_left  = []   # (sample_index, geschatte hoek)
    doa_est_right = []

    t_start = time.time()
    hops_processed = 0

    for i in range(n_frames):
        chunk     = lma_audio[i * chunk_size : (i + 1) * chunk_size, :]
        chunk_gt0 = lma_gt0  [i * chunk_size : (i + 1) * chunk_size, :]
        chunk_gt1 = lma_gt1  [i * chunk_size : (i + 1) * chunk_size, :]

        proc.processing_microarray(chunk, chunk_gt0, chunk_gt1)

        # Elke output uit de queue komt overeen met één verwerkte hop.
        # Het centrum van die hop ligt op sample (hops_processed + 0.5) * hop.
        while not proc.data_queue_phase1.empty():
            _, _, angle_left, angle_right, _ = proc.data_queue_phase1.get_nowait()
            sample_idx = int((hops_processed + 0.5) * hop)
            doa_est_left.append((sample_idx, angle_left))
            doa_est_right.append((sample_idx, angle_right))
            hops_processed += 1

    elapsed = time.time() - t_start

    # Bereken errors door geschatte DOAs te vergelijken met ground-truth op het overeenkomstige sample.
    errors_l = []
    errors_r = []
    errors_l_postswitch = []   # errors binnen 1s na een switch
    errors_r_postswitch = []
    errors_l_steady     = []   # errors verder dan 1s van enige switch
    errors_r_steady     = []

    for sample_idx, est in doa_est_left:
        if sample_idx >= len(doa_gt_left):
            continue
        true_angle = doa_gt_left[sample_idx]
        err = abs(est - true_angle)
        errors_l.append(err)
        # post-switch detectie
        dist_to_switch = np.min(np.abs(switch_samples_l - sample_idx)) if len(switch_samples_l) > 0 else fs
        if dist_to_switch < fs:  # binnen 1 seconde van een switch
            errors_l_postswitch.append(err)
        else:
            errors_l_steady.append(err)

    for sample_idx, est in doa_est_right:
        if sample_idx >= len(doa_gt_right):
            continue
        true_angle = doa_gt_right[sample_idx]
        err = abs(est - true_angle)
        errors_r.append(err)
        dist_to_switch = np.min(np.abs(switch_samples_r - sample_idx)) if len(switch_samples_r) > 0 else fs
        if dist_to_switch < fs:
            errors_r_postswitch.append(err)
        else:
            errors_r_steady.append(err)

    errors_l = np.array(errors_l)
    errors_r = np.array(errors_r)
    errors_all = np.concatenate([errors_l, errors_r])

    results[beta] = {
        "err_mean":       np.mean(errors_all),
        "err_median":     np.median(errors_all),
        "err_std":        np.std(errors_all),
        "err_l_mean":     np.mean(errors_l),
        "err_r_mean":     np.mean(errors_r),
        "err_steady":     np.mean(np.concatenate([errors_l_steady, errors_r_steady])) if (errors_l_steady or errors_r_steady) else 0.0,
        "err_postswitch": np.mean(np.concatenate([errors_l_postswitch, errors_r_postswitch])) if (errors_l_postswitch or errors_r_postswitch) else 0.0,
        "pct_lt_5deg":    100 * np.mean(errors_all < 5),
        "pct_lt_10deg":   100 * np.mean(errors_all < 10),
        "n":              len(errors_all),
        "elapsed":        elapsed,
    }

    r = results[beta]
    print(f"  DOA error: mean={r['err_mean']:.2f}° | median={r['err_median']:.2f}° | std={r['err_std']:.2f}°")
    print(f"             L={r['err_l_mean']:.2f}° R={r['err_r_mean']:.2f}°")
    print(f"  Steady-state error (>1s van switch): {r['err_steady']:.2f}°")
    print(f"  Post-switch error  (<1s na switch):  {r['err_postswitch']:.2f}°")
    print(f"  Frames <5°:  {r['pct_lt_5deg']:.0f}% | <10°: {r['pct_lt_10deg']:.0f}%")
    print(f"  Verwerkt in {elapsed:.1f}s\n")

# === SAMENVATTINGSTABEL ===
print("=" * 100)
print(f"{'Beta':>5} | {'Gem err':>8} | {'Med err':>8} | {'Std':>6} | "
      f"{'Steady':>7} | {'Post-sw':>8} | {'<5°':>5} | {'<10°':>5}")
print("-" * 100)
for b, r in results.items():
    print(f"{b:>5.2f} | {r['err_mean']:>7.2f}° | {r['err_median']:>7.2f}° | {r['err_std']:>5.2f}° | "
          f"{r['err_steady']:>6.2f}° | {r['err_postswitch']:>7.2f}° | "
          f"{r['pct_lt_5deg']:>4.0f}% | {r['pct_lt_10deg']:>4.0f}%")
print("=" * 100)

# Beste beta op basis van gemiddelde error
best = min(results, key=lambda b: results[b]["err_mean"])
print(f"\nBeste beta op basis van gemiddelde DOA-error: {best}")
print(f"  → Gemiddelde error: {results[best]['err_mean']:.2f}°")
print(f"  → Steady-state:     {results[best]['err_steady']:.2f}°")
print(f"  → Post-switch:      {results[best]['err_postswitch']:.2f}°")