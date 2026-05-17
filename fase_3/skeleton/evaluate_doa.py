"""
VAD optimalisatie voor reverberant case.
Test verschillende VAD threshold waarden en rapporteert:
- NLMS update percentage (hoeveel % de filter leert)
- SIR statistieken (kwaliteit van de beamformer output)

Doel: vind de threshold die de hoogste SIR geeft met een gezond update percentage.
"""
import time
import numpy as np
from scipy.io import wavfile
from processor import Processor

# === CONFIG ===
BASE_PATH = "data/phase3_audioData/audiodata_batch_1/reverberant"
RIR_PATH  = "data/phase3_audioData/audiodata_batch_1/reverberant/lma_16kHz_200ms.npz"
PAIR = 1
DURATION_SECONDS = 120

# VAD thresholds om te testen
VAD_THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# === DATA LADEN ===
print(f"Laden data voor pair{PAIR} ({DURATION_SECONDS}s reverberant)...")
fs, lma_audio = wavfile.read(f"{BASE_PATH}/pair{PAIR}/mixture_LMA.wav")
_, lma_gt0   = wavfile.read(f"{BASE_PATH}/pair{PAIR}/leftSpeaker_LMA.wav")
_, lma_gt1   = wavfile.read(f"{BASE_PATH}/pair{PAIR}/rightSpeaker_LMA.wav")

n_samples = int(DURATION_SECONDS * fs)
lma_audio = lma_audio[:n_samples]
lma_gt0   = lma_gt0[:n_samples]
lma_gt1   = lma_gt1[:n_samples]

chunk_size = fs // 32
n_frames   = n_samples // chunk_size
print(f"Sample rate: {fs} Hz | {n_frames} chunks van {chunk_size} samples\n")

# === RESULTATEN PER THRESHOLD ===
results = {}

for threshold in VAD_THRESHOLDS:
    print(f"--- Testen vad_threshold={threshold} ---")

    proc = Processor(rir_path=RIR_PATH)
    proc.vad_threshold = threshold

    sir_history = []
    t_start = time.time()

    for i in range(n_frames):
        chunk     = lma_audio[i * chunk_size : (i + 1) * chunk_size, :]
        chunk_gt0 = lma_gt0  [i * chunk_size : (i + 1) * chunk_size, :]
        chunk_gt1 = lma_gt1  [i * chunk_size : (i + 1) * chunk_size, :]

        proc.processing_microarray(chunk, chunk_gt0, chunk_gt1)

        while not proc.data_queue_phase1.empty():
            _, _, _, _, sir = proc.data_queue_phase1.get_nowait()
            if sir != 0.0 and not np.isnan(sir):
                sir_history.append(sir)

    elapsed = time.time() - t_start
    sir_arr = np.array(sir_history) if sir_history else np.array([0.0])

    # VAD statistieken
    upd_l = (proc.vad_update_count_left  / proc.vad_evals * 100) if proc.vad_evals > 0 else 0
    upd_r = (proc.vad_update_count_right / proc.vad_evals * 100) if proc.vad_evals > 0 else 0

    results[threshold] = {
        "upd_l":      upd_l,
        "upd_r":      upd_r,
        "sir_mean":   np.mean(sir_arr),
        "sir_median": np.median(sir_arr),
        "sir_std":    np.std(sir_arr),
        "sir_pct_pos":   100 * np.mean(sir_arr > 0),
        "sir_pct_5db":   100 * np.mean(sir_arr > 5),
        "sir_pct_10db":  100 * np.mean(sir_arr > 10),
        "sir_second_half": np.mean(sir_arr[len(sir_arr)//2:]) if len(sir_arr) > 1 else 0.0,
        "n_sir":      len(sir_arr),
        "elapsed":    elapsed,
    }

    r = results[threshold]
    print(f"  VAD: L={upd_l:.1f}% R={upd_r:.1f}% updaten (= filter leert tijdens stilte)")
    print(f"  SIR: mean={r['sir_mean']:+.2f} dB | median={r['sir_median']:+.2f} dB | "
          f">0dB={r['sir_pct_pos']:.0f}% | >5dB={r['sir_pct_5db']:.0f}% | "
          f">10dB={r['sir_pct_10db']:.0f}%")
    print(f"  SIR 2e helft (na convergentie): {r['sir_second_half']:+.2f} dB")
    print(f"  Verwerkt in {elapsed:.1f}s\n")

# === SAMENVATTINGSTABEL ===
print("=" * 95)
print(f"{'Thresh':>7} | {'Upd L%':>7} | {'Upd R%':>7} | "
      f"{'Gem SIR':>8} | {'Med SIR':>8} | {'>0dB':>6} | {'>5dB':>6} | {'>10dB':>6} | {'2e helft':>9}")
print("-" * 95)
for t, r in results.items():
    print(f"{t:>7.2f} | {r['upd_l']:>6.1f}% | {r['upd_r']:>6.1f}% | "
          f"{r['sir_mean']:>+8.2f} | {r['sir_median']:>+8.2f} | "
          f"{r['sir_pct_pos']:>5.0f}% | {r['sir_pct_5db']:>5.0f}% | "
          f"{r['sir_pct_10db']:>5.0f}% | {r['sir_second_half']:>+9.2f}")
print("=" * 95)

# Beste threshold op basis van mediaan SIR (robuuster dan gemiddelde door uitschieters)
best = max(results, key=lambda t: results[t]["sir_median"])
print(f"\nBeste threshold op basis van mediaan SIR: {best}")
print(f"  → Mediaan SIR: {results[best]['sir_median']:+.2f} dB")
print(f"  → NLMS update: L={results[best]['upd_l']:.1f}% R={results[best]['upd_r']:.1f}%")