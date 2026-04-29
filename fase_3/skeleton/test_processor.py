# test_processor.py
import asyncio
import numpy as np
import scipy.linalg
from scipy import signal
import pickle
from scipy.io import wavfile
from processor import Processor

BASE_PATH = "data/phase3_audioData/audiodata_batch_1/anechoic"

with open(f"{BASE_PATH}/params.pkl", "rb") as f:
    params = pickle.load(f)

fs, lma_audio = wavfile.read(f"{BASE_PATH}/pair1/mixture_LMA.wav")
_, lma_gt0 = wavfile.read(f"{BASE_PATH}/pair1/leftSpeaker_LMA.wav")
_, lma_gt1 = wavfile.read(f"{BASE_PATH}/pair1/rightSpeaker_LMA.wav")

gt = np.load(f"{BASE_PATH}/pair1/gt.npz")
doa_left_gt = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_l"], gt["endSamples_l"])])
doa_right_gt = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_r"], gt["endSamples_r"])])

print(f"Sample rate: {fs} Hz")
print(f"Audio shape: {lma_audio.shape}")
print(f"Totale duur: {lma_audio.shape[0]/fs:.1f} s\n")

proc = Processor()
# Track VAD beslissingen
adapt_count = 0
total_count = 0
chunk_size = fs // 32  # 500 samples per chunk

# Verzamel alle outputs om later te analyseren
all_sig0 = []
all_sig1 = []
all_doa_l = []
all_doa_r = []
all_gt_l = []
all_gt_r = []

n_frames = 200  # ~6 seconden
print(f"\nVerwerken van {n_frames} frames ({n_frames*chunk_size/fs:.1f}s audio)...\n")

#BEGIN TIJDELLIJKE DEBUG
# Sanity test: bereken SIR met dezelfde methode als de diagnose, maar voor één chunk
from processor import build_lut_for_target

rir_data = np.load("data/phase3_audioData/audiodata_batch_1/anechoic/lma_16kHz.npz")
rirs = rir_data["rirs"]
doas_lut = rir_data["thetas"]

# Pak de LUT entry voor 130.5° (linker spreker)
idx = np.argmin(np.abs(doas_lut - 130.5))
print(f"Test met LUT entry: {doas_lut[idx]:.2f}°")
W_FAS_test, B_test = build_lut_for_target(rirs[:, :, idx], L=1024)

# Test: pas FAS toe op één 1024-sample window van gt0 en gt1
test_window = np.sqrt(signal.windows.hann(1024, sym=False))
chunk_gt0 = lma_gt0[:1024, :].astype(float)
chunk_gt1 = lma_gt1[:1024, :].astype(float)

windowed_gt0 = chunk_gt0 * test_window[:, np.newaxis]
windowed_gt1 = chunk_gt1 * test_window[:, np.newaxis]
fft_gt0 = np.fft.rfft(windowed_gt0, n=1024, axis=0)
fft_gt1 = np.fft.rfft(windowed_gt1, n=1024, axis=0)

# FAS only
out_fft_gt0 = np.zeros(513, dtype=complex)
out_fft_gt1 = np.zeros(513, dtype=complex)
for k in range(513):
    out_fft_gt0[k] = np.vdot(W_FAS_test[k, :], fft_gt0[k, :])
    out_fft_gt1[k] = np.vdot(W_FAS_test[k, :], fft_gt1[k, :])

out_gt0 = np.fft.irfft(out_fft_gt0, n=1024) * test_window
out_gt1 = np.fft.irfft(out_fft_gt1, n=1024) * test_window

sir_test = 10 * np.log10(np.var(out_gt0) / np.var(out_gt1))
print(f"Test SIR (FAS naar 130.5°, op eerste 1024 samples): {sir_test:.2f} dB")
print(f"  RMS gt0 door FAS: {np.sqrt(np.mean(out_gt0**2)):.2f}")
print(f"  RMS gt1 door FAS: {np.sqrt(np.mean(out_gt1**2)):.2f}")
#EINDE TIJDELIJKE DEBUG



for i in range(n_frames):
    chunk = lma_audio[i * chunk_size : (i + 1) * chunk_size, :]
    chunk_gt0 = lma_gt0[i * chunk_size : (i + 1) * chunk_size, :]
    chunk_gt1 = lma_gt1[i * chunk_size : (i + 1) * chunk_size, :]
    proc.processing_microarray(chunk, chunk_gt0, chunk_gt1)    # Check interne VAD state
    if len(proc.energy_history) >= 10:
        min_energy = np.min(proc.energy_history)
        current_energy = proc.energy_history[-1]
        if current_energy < min_energy * proc.vad_threshold_factor:
            adapt_count += 1
        total_count += 1
    mid_sample = i * chunk_size + chunk_size // 2
    gt_l = doa_left_gt[mid_sample]
    gt_r = doa_right_gt[mid_sample]

    if not proc.data_queue_phase1.empty():
        sig0, sig1, angle_left, angle_right, sir = proc.data_queue_phase1.get_nowait()
        all_sig0.append(sig0)
        all_sig1.append(sig1)
        all_doa_l.append(angle_left)
        all_doa_r.append(angle_right)
        all_gt_l.append(gt_l)
        all_gt_r.append(gt_r)

        # Print elke 20 frames
        if i % 20 == 0:
            err_l = abs(angle_left - gt_l)
            err_r = abs(angle_right - gt_r)
            print(f"Frame {i:3d}: DOA=({angle_left:5.1f}°, {angle_right:5.1f}°)  |  gt=({gt_l:5.1f}°, {gt_r:5.1f}°)  |  SIR={sir:5.2f} dB")
# ===== Statistieken =====
all_doa_l = np.array(all_doa_l)
all_doa_r = np.array(all_doa_r)
all_gt_l = np.array(all_gt_l)
all_gt_r = np.array(all_gt_r)

print(f"\n===== DOA Statistieken (na convergence, laatste helft) =====")
half = len(all_doa_l) // 2
print(f"Gem. fout links:  {np.mean(np.abs(all_doa_l[half:] - all_gt_l[half:])):.2f}°")
print(f"Gem. fout rechts: {np.mean(np.abs(all_doa_r[half:] - all_gt_r[half:])):.2f}°")

print(f"\n===== VAD Statistieken =====")
if total_count > 0:
    print(f"Adaptatie: {adapt_count}/{total_count} frames ({100*adapt_count/total_count:.1f}%)")

# ===== Audio analyse =====
sig0_concat = np.concatenate(all_sig0)
sig1_concat = np.concatenate(all_sig1)

print(f"\n===== Audio Output =====")
print(f"sig0 (left beam) range: [{sig0_concat.min():.1f}, {sig0_concat.max():.1f}], RMS: {np.sqrt(np.mean(sig0_concat**2)):.2f}")
print(f"sig1 (right beam) range: [{sig1_concat.min():.1f}, {sig1_concat.max():.1f}], RMS: {np.sqrt(np.mean(sig1_concat**2)):.2f}")

# Save voor luistertest
from scipy.io.wavfile import write
# Normaliseer naar int16 range
sig0_norm = (sig0_concat / (np.abs(sig0_concat).max() + 1e-9) * 20000).astype(np.int16)
sig1_norm = (sig1_concat / (np.abs(sig1_concat).max() + 1e-9) * 20000).astype(np.int16)

write("test_output_left_beam.wav", fs, sig0_norm)
write("test_output_right_beam.wav", fs, sig1_norm)
print(f"\nAudio opgeslagen: test_output_left_beam.wav en test_output_right_beam.wav")

# In test_processor.py, na de tracking loop:
import matplotlib.pyplot as plt

# Verzamel alle DOA's
proc_test = Processor()
all_t, all_est_l, all_est_r, all_gt_l, all_gt_r = [], [], [], [], []

for i in range(int(45 * 32)):
    chunk = lma_audio[i * chunk_size : (i + 1) * chunk_size, :]
    chunk_gt0 = lma_gt0[i * chunk_size : (i + 1) * chunk_size, :]
    chunk_gt1 = lma_gt1[i * chunk_size : (i + 1) * chunk_size, :]
    proc_test.processing_microarray(chunk, chunk_gt0, chunk_gt1)
    
    while not proc_test.data_queue_phase1.empty():
        _, _, angle_left, angle_right, _ = proc_test.data_queue_phase1.get_nowait()
        mid = i * chunk_size + chunk_size // 2
        all_t.append(mid / fs)
        all_est_l.append(angle_left)
        all_est_r.append(angle_right)
        all_gt_l.append(doa_left_gt[mid])
        all_gt_r.append(doa_right_gt[mid])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(all_t, all_gt_l, 'g-', label='Ground truth links', linewidth=2)
ax1.plot(all_t, all_est_l, 'b-', label='MUSIC schatting links', alpha=0.7)
ax1.set_ylabel('DOA (graden)'); ax1.set_title('Linker spreker'); ax1.legend(); ax1.grid()

ax2.plot(all_t, all_gt_r, 'g-', label='Ground truth rechts', linewidth=2)
ax2.plot(all_t, all_est_r, 'b-', label='MUSIC schatting rechts', alpha=0.7)
ax2.set_xlabel('Tijd (s)'); ax2.set_ylabel('DOA (graden)'); ax2.set_title('Rechter spreker'); ax2.legend(); ax2.grid()

plt.tight_layout()
plt.savefig('doa_tracking.png', dpi=100)
print("Plot saved as doa_tracking.png")