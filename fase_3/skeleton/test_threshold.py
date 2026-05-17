"""
Peak-threshold optimalisatie voor MUSIC DOA tracking.
Test verschillende thresholds (in dB t.o.v. globale spectrum max) en rapporteert DOA-error.

Vraag: voegt de peak-threshold toe? Of geeft 'altijd accepteren' even goede of betere resultaten?

Vereist eerst kleine aanpassing in processor.py:
- in __init__: zelf.peak_threshold = -12.0
- in processing_microarray: vervang 'PEAK_THRESHOLD' door 'self.peak_threshold' (2x)
"""
import time
import numpy as np
from scipy.io import wavfile
from processor import Processor

# === CONFIG ===
BASE_PATH = "data/phase3_audioData/audiodata_batch_1/reverberant"
RIR_PATH  = "data/phase3_audioData/audiodata_batch_1/reverberant/lma_16kHz_200ms.npz"
GT_PATH   = "data/phase3_audioData/audiodata_batch_1/reverberant"
PAIR = 1
DURATION_SECONDS = 120

BETA_OPTIMAL = 0.97  # uit beta-sweep test

# Thresholds om te testen (in dB t.o.v. globale spectrum maximum)
# -1000 = effectief uitgeschakeld, altijd accepteren
# -3 = zeer strikt, alleen zeer dichte pieken
THRESHOLDS = [-1000.0, -20.0, -15.0, -12.0, -9.0, -6.0, -3.0]

# === DATA LADEN ===
print(f"Laden data voor pair{PAIR} ({DURATION_SECONDS}s reverberant, beta={BETA_OPTIMAL})...")
fs, lma_audio = wavfile.read(f"{BASE_PATH}/pair{PAIR}/mixture_LMA.wav")
_, lma_gt0   = wavfile.read(f"{BASE_PATH}/pair{PAIR}/leftSpeaker_LMA.wav")
_, lma_gt1   = wavfile.read(f"{BASE_PATH}/pair{PAIR}/rightSpeaker_LMA.wav")

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
hop        = 512
print(f"Sample rate: {fs} Hz | {n_frames} chunks van {chunk_size} samples\n")

switch_samples_l = np.where(np.diff(doa_gt_left) != 0)[0] + 1
switch_samples_r = np.where(np.diff(doa_gt_right) != 0)[0] + 1

# === RESULTATEN PER THRESHOLD ===
results = {}

for threshold in THRESHOLDS:
    label = "OFF (no thr)" if threshold <= -100 else f"{threshold:+.0f} dB"
    print(f"--- Testen peak_threshold={label} ---")

    proc = Processor(rir_path=RIR_PATH)
    proc.beta = BETA_OPTIMAL
    proc.peak_threshold = threshold

    doa_est_left  = []
    doa_est_right = []
    t_start = time.time()
    hops_processed = 0

    for i in range(n_frames):
        chunk     = lma_audio[i * chunk_size : (i + 1) * chunk_size, :]
        chunk_gt0 = lma_gt0  [i * chunk_size : (i + 1) * chunk_size, :]
        chunk_gt1 = lma_gt1  [i * chunk_size : (i + 1) * chunk_size, :]

        proc.processing_microarray(chunk, chunk_gt0, chunk_gt1)

        while not proc.data_queue_phase1.empty():
            _, _, angle_left, angle_right, _ = proc.data_queue_phase1.get_nowait()
            sample_idx = int((hops_processed + 0.5) * hop)
            doa_est_left.append((sample_idx, angle_left))
            doa_est_right.append((sample_idx, angle_right))
            hops_processed += 1

    elapsed = time.time() - t_start

    errors_l = []
    errors_r = []
    errors_l_steady     = []
    errors_r_steady     = []
    errors_l_postswitch = []
    errors_r_postswitch = []

    for sample_idx, est in doa_est_left:
        if sample_idx >= len(doa_gt_left):
            continue
        true_angle = doa_gt_left[sample_idx]
        err = abs(est - true_angle)
        errors_l.append(err)
        dist_to_switch = np.min(np.abs(switch_samples_l - sample_idx)) if len(switch_samples_l) > 0 else fs
        if dist_to_switch < fs:
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

    results[threshold] = {
        "label":          label,
        "err_mean":       np.mean(errors_all),
        "err_median":     np.median(errors_all),
        "err_std":        np.std(errors_all),
        "err_steady":     np.mean(np.concatenate([errors_l_steady, errors_r_steady])) if (errors_l_steady or errors_r_steady) else 0.0,
        "err_postswitch": np.mean(np.concatenate([errors_l_postswitch, errors_r_postswitch])) if (errors_l_postswitch or errors_r_postswitch) else 0.0,
        "pct_lt_5deg":    100 * np.mean(errors_all < 5),
        "pct_lt_10deg":   100 * np.mean(errors_all < 10),
        "n":              len(errors_all),
        "elapsed":        elapsed,
    }

    r = results[threshold]
    print(f"  DOA error: mean={r['err_mean']:.2f}° | median={r['err_median']:.2f}° | std={r['err_std']:.2f}°")
    print(f"  Steady-state error: {r['err_steady']:.2f}°  |  Post-switch error: {r['err_postswitch']:.2f}°")
    print(f"  Frames <5°: {r['pct_lt_5deg']:.0f}% | <10°: {r['pct_lt_10deg']:.0f}%")
    print(f"  Verwerkt in {elapsed:.1f}s\n")

# === SAMENVATTINGSTABEL ===
print("=" * 100)
print(f"{'Threshold':>12} | {'Gem err':>8} | {'Med err':>8} | {'Std':>6} | "
      f"{'Steady':>7} | {'Post-sw':>8} | {'<5°':>5} | {'<10°':>5}")
print("-" * 100)
for t, r in results.items():
    print(f"{r['label']:>12} | {r['err_mean']:>7.2f}° | {r['err_median']:>7.2f}° | {r['err_std']:>5.2f}° | "
          f"{r['err_steady']:>6.2f}° | {r['err_postswitch']:>7.2f}° | "
          f"{r['pct_lt_5deg']:>4.0f}% | {r['pct_lt_10deg']:>4.0f}%")
print("=" * 100)

best = min(results, key=lambda t: results[t]["err_mean"])
print(f"\nBeste threshold op basis van gemiddelde DOA-error: {results[best]['label']}")
print(f"  → Gemiddelde error: {results[best]['err_mean']:.2f}°")
print(f"  → Steady-state:     {results[best]['err_steady']:.2f}°")
print(f"  → Post-switch:      {results[best]['err_postswitch']:.2f}°")

# Vergelijk specifiek OFF vs -12 dB (jullie huidige)
off_key = -1000.0
cur_key = -12.0
if off_key in results and cur_key in results:
    diff = results[cur_key]["err_mean"] - results[off_key]["err_mean"]
    print(f"\nVergelijking: threshold -12 dB vs uitgeschakeld:")
    print(f"  → -12 dB:   {results[cur_key]['err_mean']:.2f}° mean error")
    print(f"  → OFF:      {results[off_key]['err_mean']:.2f}° mean error")
    if abs(diff) < 0.05:
        print(f"  → Verschil te klein om relevant te zijn ({diff:+.2f}°) → threshold is niet nuttig")
    elif diff > 0:
        print(f"  → Zonder threshold is {abs(diff):.2f}° BETER → threshold weghalen")
    else:
        print(f"  → Met threshold is {abs(diff):.2f}° beter → threshold behouden")