import numpy as np
import scipy.linalg
from scipy import signal
from scipy.io import wavfile
from processor import build_lut_for_target
import matplotlib.pyplot as plt

# Laad data
fs, lma = wavfile.read("fase_3/skeleton/data/phase3_audioData/audiodata_batch_1/anechoic/pair1/mixture_LMA.wav")
_, gt0 = wavfile.read("fase_3/skeleton/data/phase3_audioData/audiodata_batch_1/anechoic/pair1/leftSpeaker_LMA.wav")
_, gt1 = wavfile.read("fase_3/skeleton/data/phase3_audioData/audiodata_batch_1/anechoic/pair1/rightSpeaker_LMA.wav")
gt = np.load("fase_3/skeleton/data/phase3_audioData/audiodata_batch_1/anechoic/pair1/gt.npz")

print("Eerste sample DOA links:", gt["angles_l"][0])
print("Eerste sample DOA rechts:", gt["angles_r"][0])

# Laad LUT
rir_data = np.load("fase_3/skeleton/data/phase3_audioData/audiodata_batch_1/anechoic/lma_16kHz.npz")
rirs = rir_data["rirs"]
doas = rir_data["thetas"]

# Pak de RIR het dichtst bij de gt linker DOA
target_doa_left = gt["angles_l"][0]   # bv 130.5
target_doa_right = gt["angles_r"][0]  # bv 54.3

idx_left = np.argmin(np.abs(doas - target_doa_left))
idx_right = np.argmin(np.abs(doas - target_doa_right))
print(f"LUT hoek voor links: {doas[idx_left]:.2f}°")
print(f"LUT hoek voor rechts: {doas[idx_right]:.2f}°")

# Bouw beamformers
W_FAS_L, B_L = build_lut_for_target(rirs[:, :, idx_left], L=1024)
W_FAS_R, B_R = build_lut_for_target(rirs[:, :, idx_right], L=1024)

# Pas linker beamformer (alleen FAS, geen NLMS) toe op een chunk
def apply_fas_on_signal(sig, W_FAS, L=1024):
    overlap = L // 2
    window = np.sqrt(signal.windows.hann(L, sym=False))
    _, _, Zxx = signal.stft(sig.T, fs=16000, window=window, nperseg=L, noverlap=overlap)
    nF, nT = Zxx.shape[1], Zxx.shape[2]
    out = np.zeros((nF, nT), dtype=complex)
    for k in range(nF):
        for n in range(nT):
            out[k, n] = np.vdot(W_FAS[k, :], Zxx[:, k, n])
    _, out_t = signal.istft(out, fs=16000, window=window, nperseg=L, noverlap=overlap)
    return out_t[:sig.shape[0]]

# Test op de eerste 5 seconden
N = 5 * 16000
out_L_gt0 = apply_fas_on_signal(gt0[:N].astype(float), W_FAS_L)
out_L_gt1 = apply_fas_on_signal(gt1[:N].astype(float), W_FAS_L)
out_R_gt0 = apply_fas_on_signal(gt0[:N].astype(float), W_FAS_R)
out_R_gt1 = apply_fas_on_signal(gt1[:N].astype(float), W_FAS_R)

# SIR voor linker beam (target = gt0)
sir_L = 10 * np.log10(np.var(out_L_gt0) / np.var(out_L_gt1))
sir_R = 10 * np.log10(np.var(out_R_gt1) / np.var(out_R_gt0))
print(f"\nLinker beam (gericht op {doas[idx_left]:.1f}°): SIR = {sir_L:.2f} dB")
print(f"Rechter beam (gericht op {doas[idx_right]:.1f}°): SIR = {sir_R:.2f} dB")

# En cross-check: wat als ik aanneem dat gt0/gt1 omgewisseld zijn?
sir_L_swapped = 10 * np.log10(np.var(out_L_gt1) / np.var(out_L_gt0))
sir_R_swapped = 10 * np.log10(np.var(out_R_gt0) / np.var(out_R_gt1))
print(f"\nALS gt0=rechts en gt1=links:")
print(f"Linker beam: SIR = {sir_L_swapped:.2f} dB")
print(f"Rechter beam: SIR = {sir_R_swapped:.2f} dB")

rir = rirs[:, 0, idx_left]  # eerste mic, linker hoek
plt.plot(rir[:2000])
plt.title(f"RIR voor {doas[idx_left]:.1f}°, mic 0")
plt.savefig("rir_check.png")
print(f"RIR max op sample: {np.argmax(np.abs(rir))}")
print(f"RIR energie eerste 1024: {np.sum(rir[:1024]**2):.4f}")
print(f"RIR totale energie: {np.sum(rir**2):.4f}")

# Test 1: één enkele 1024-sample frame met rfft/irfft (zoals processor.py doet)
print("\n=== Test 1: Single frame rfft/irfft ===")
test_window = np.sqrt(signal.windows.hann(1024, sym=False))

for start in [0, 1024, 4096, 16000]:  # Verschillende startposities
    chunk_gt0 = gt0[start:start+1024, :].astype(float)
    chunk_gt1 = gt1[start:start+1024, :].astype(float)
    
    windowed_gt0 = chunk_gt0 * test_window[:, np.newaxis]
    windowed_gt1 = chunk_gt1 * test_window[:, np.newaxis]
    fft_gt0 = np.fft.rfft(windowed_gt0, n=1024, axis=0)
    fft_gt1 = np.fft.rfft(windowed_gt1, n=1024, axis=0)
    
    out_fft_gt0 = np.zeros(513, dtype=complex)
    out_fft_gt1 = np.zeros(513, dtype=complex)
    for k in range(513):
        out_fft_gt0[k] = np.vdot(W_FAS_L[k, :], fft_gt0[k, :])
        out_fft_gt1[k] = np.vdot(W_FAS_L[k, :], fft_gt1[k, :])
    
    out_gt0_t = np.fft.irfft(out_fft_gt0, n=1024) * test_window
    out_gt1_t = np.fft.irfft(out_fft_gt1, n=1024) * test_window
    
    var_t = np.var(out_gt0_t)
    var_i = np.var(out_gt1_t)
    if var_t > 0 and var_i > 0:
        sir_frame = 10 * np.log10(var_t / var_i)
        print(f"  Start={start}: SIR linker beam = {sir_frame:.2f} dB (gt0_var={var_t:.1f}, gt1_var={var_i:.1f})")

# Test 2: zelfde frame maar SIR berekend in frequency domain (geen ifft)
print("\n=== Test 2: SIR direct in frequency domain ===")
chunk_gt0 = gt0[:1024, :].astype(float)
chunk_gt1 = gt1[:1024, :].astype(float)
windowed_gt0 = chunk_gt0 * test_window[:, np.newaxis]
windowed_gt1 = chunk_gt1 * test_window[:, np.newaxis]
fft_gt0 = np.fft.rfft(windowed_gt0, n=1024, axis=0)
fft_gt1 = np.fft.rfft(windowed_gt1, n=1024, axis=0)

power_gt0_freq = 0
power_gt1_freq = 0
for k in range(513):
    out_gt0 = np.vdot(W_FAS_L[k, :], fft_gt0[k, :])
    out_gt1 = np.vdot(W_FAS_L[k, :], fft_gt1[k, :])
    power_gt0_freq += np.abs(out_gt0)**2
    power_gt1_freq += np.abs(out_gt1)**2

sir_freq = 10 * np.log10(power_gt0_freq / power_gt1_freq)
print(f"SIR berekend in freq domain: {sir_freq:.2f} dB")

print("\n=== Test 3: probeer alle LUT-hoeken voor de linker spreker ===")
print("De linker spreker staat op DOA 130.5° volgens gt.npz.")
print("Welke LUT-entry geeft de hoogste SIR voor gt0 (links)?\n")

test_window = np.sqrt(signal.windows.hann(1024, sym=False))
chunk_gt0 = gt0[8000:9024, :].astype(float)  # ergens midden in audio
chunk_gt1 = gt1[8000:9024, :].astype(float)
windowed_gt0 = chunk_gt0 * test_window[:, np.newaxis]
windowed_gt1 = chunk_gt1 * test_window[:, np.newaxis]
fft_gt0 = np.fft.rfft(windowed_gt0, n=1024, axis=0)
fft_gt1 = np.fft.rfft(windowed_gt1, n=1024, axis=0)

for i, lut_doa in enumerate(doas):
    W_FAS_test, _ = build_lut_for_target(rirs[:, :, i], L=1024)
    
    p_gt0 = 0
    p_gt1 = 0
    for k in range(513):
        out_gt0 = np.vdot(W_FAS_test[k, :], fft_gt0[k, :])
        out_gt1 = np.vdot(W_FAS_test[k, :], fft_gt1[k, :])
        p_gt0 += np.abs(out_gt0)**2
        p_gt1 += np.abs(out_gt1)**2
    
    sir = 10 * np.log10(p_gt0 / p_gt1) if p_gt1 > 0 else float('inf')
    marker = ""
    if abs(lut_doa - 130.5) < 5:
        marker = "  <-- VERWACHT (gt links DOA = 130.5°)"
    if abs(lut_doa - 54.3) < 5:
        marker = "  <-- DIT IS DE RECHTER SPREKER"
    print(f"  LUT[{i}] = {lut_doa:6.2f}°: SIR voor target=gt0 = {sir:7.2f} dB{marker}")

    print("\n=== Test 4: directe SIR van mic 0 zonder beamforming ===")
chunk_gt0 = gt0[8000:9024, 0].astype(float)  # mic 0
chunk_gt1 = gt1[8000:9024, 0].astype(float)
print(f"RMS gt0 op mic 0: {np.sqrt(np.mean(chunk_gt0**2)):.2f}")
print(f"RMS gt1 op mic 0: {np.sqrt(np.mean(chunk_gt1**2)):.2f}")
print(f"SIR direct (gt0 / gt1): {10*np.log10(np.var(chunk_gt0) / np.var(chunk_gt1)):.2f} dB")

# Probeer ook over de eerste seconde
chunk_gt0_long = gt0[:16000, 0].astype(float)
chunk_gt1_long = gt1[:16000, 0].astype(float)
print(f"\nOver eerste 1s:")
print(f"RMS gt0: {np.sqrt(np.mean(chunk_gt0_long**2)):.2f}")
print(f"RMS gt1: {np.sqrt(np.mean(chunk_gt1_long**2)):.2f}")
print(f"SIR: {10*np.log10(np.var(chunk_gt0_long) / np.var(chunk_gt1_long)):.2f} dB")

# Hoeveel keer is gt1 > 100 RMS in non-overlappende 1024-sample chunks?
print("\nActiviteit per 1024-sample chunk in eerste 5s:")
for start in [0, 1024, 4096, 16000]:
    rms0 = np.sqrt(np.mean(gt0[start:start+1024, 0].astype(float)**2))
    rms1 = np.sqrt(np.mean(gt1[start:start+1024, 0].astype(float)**2))
    print(f"  Start={start}: RMS gt0={rms0:7.1f}, RMS gt1={rms1:7.1f}")

