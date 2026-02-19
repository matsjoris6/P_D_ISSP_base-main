import os
import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATIE & SETUP
# ==========================================
base_folder = os.path.join("sound_files", "head_mounted_rirs")
dry_signal_1_path = os.path.join("sound_files", "part1_track1_dry.wav") 
dry_signal_2_path = os.path.join("sound_files", "part1_track2_dry.wav")

fs_sim = 44100
c = 343.0
d_ear = 0.013    # 1.3 cm
d_front = 0.215  # 21.5 cm

folder_left = "s-30"
folder_right = "s30"

# Haal de werkelijke hoeken uit de strings
angle_L = abs(float(folder_left.replace("s", "")))  # Wordt 30.0
angle_R = float(folder_right.replace("s", ""))      # Wordt 30.0

# "Broadside" voor het frontale paar is recht vooruit (90° op de as yL1-yR1).
# Een bron op +30° (rechts) zit op 90-30 = 60° van de as.
# Een bron op -30° (links) zit op 90+30 = 120° van de as.
true_front_R = 90.0 - angle_R
true_front_L = 90.0 + angle_L

# ==========================================
# 2. SIGNAAL GENERATIE
# ==========================================
try:
    dry_sig_1, _ = sf.read(dry_signal_1_path)
    dry_sig_2, _ = sf.read(dry_signal_2_path)
    
    min_len = min(len(dry_sig_1), len(dry_sig_2))
    dry_sig_1 = dry_sig_1[:min_len]
    dry_sig_2 = dry_sig_2[:min_len]

    # Laad de HMIRs (L1, L2, R1, R2) voor beide richtingen
    yL1_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_L1.wav"))
    yL2_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_L2.wav"))
    yR1_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_R1.wav"))
    yR2_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_R2.wav"))

    yL1_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_L1.wav"))
    yL2_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_L2.wav"))
    yR1_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_R1.wav"))
    yR2_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_R2.wav"))

    # Convolutie: Bron 1 (links) naar alle mics
    yL1_from_left = signal.fftconvolve(dry_sig_1, yL1_rir_left, mode='full')
    yL2_from_left = signal.fftconvolve(dry_sig_1, yL2_rir_left, mode='full')
    yR1_from_left = signal.fftconvolve(dry_sig_1, yR1_rir_left, mode='full')
    yR2_from_left = signal.fftconvolve(dry_sig_1, yR2_rir_left, mode='full')

    # Convolutie: Bron 2 (rechts) naar alle mics
    yL1_from_right = signal.fftconvolve(dry_sig_2, yL1_rir_right, mode='full')
    yL2_from_right = signal.fftconvolve(dry_sig_2, yL2_rir_right, mode='full')
    yR1_from_right = signal.fftconvolve(dry_sig_2, yR1_rir_right, mode='full')
    yR2_from_right = signal.fftconvolve(dry_sig_2, yR2_rir_right, mode='full')

    # Mix per microfoon
    yL1_mix = yL1_from_left + yL1_from_right
    yL2_mix = yL2_from_left + yL2_from_right
    yR1_mix = yR1_from_left + yR1_from_right
    yR2_mix = yR2_from_left + yR2_from_right

    # Bereid de drie combinaties voor (nT, M)
    micsigs_left_ear = np.column_stack((yL1_mix, yL2_mix))
    micsigs_right_ear = np.column_stack((yR1_mix, yR2_mix))
    micsigs_front = np.column_stack((yL1_mix, yR1_mix)) # L1 en R1 voor de grote afstand

except Exception as e:
    print(f"Fout bij het laden van audio/RIRs: {e}")
    exit()

# ==========================================
# 3. WIDEBAND MUSIC (Geforceerd op Q=1)
# ==========================================
def music_wideband_2mics(micsigs, fs, d, Q=1):
    L = 1024
    overlap = L // 2
    
    stft_list = []
    for m in range(micsigs.shape[1]):
        freqs, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window='hann', nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0)
    
    M, nF, nT = stft_data.shape
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    
    mics_pos = np.array([0, d])
    valid_indices = range(1, L // 2)
    pseudospectra = []
    
    for k in valid_indices:
        Y = stft_data[:, k, :]
        Ryy = (Y @ Y.conj().T) / nT
        _, eigvecs = np.linalg.eigh(Ryy)
        En = eigvecs[:, :M-Q] 
        
        omega = 2 * np.pi * freqs[k]
        taus = (mics_pos.reshape(-1, 1) * np.cos(rads)) / c 
        A = np.exp(-1j * omega * taus)
        
        denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
        pseudospectra.append(1.0 / denom)
        
    pseudospectra = np.array(pseudospectra)
    p_geom = np.exp(np.mean(np.log(pseudospectra), axis=0))
    spectrum_db = 10 * np.log10(p_geom / np.max(p_geom))
    
    peaks_indices, _ = signal.find_peaks(spectrum_db)
    
    if len(peaks_indices) > 0:
        sorted_peaks = peaks_indices[np.argsort(spectrum_db[peaks_indices])][-Q:]
        estimated_doas = np.sort(angles[sorted_peaks])
    else:
        estimated_doas = [np.nan]
        
    return angles, spectrum_db, estimated_doas

# ==========================================
# 4. UITVOEREN EN PLOTTEN
# ==========================================
# Run MUSIC voor alle drie de setups
ang_L, spec_L, est_L = music_wideband_2mics(micsigs_left_ear, fs_sim, d_ear, Q=1)
ang_R, spec_R, est_R = music_wideband_2mics(micsigs_right_ear, fs_sim, d_ear, Q=1)
ang_F, spec_F, est_F = music_wideband_2mics(micsigs_front, fs_sim, d_front, Q=1)

# Plotting in 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: Linker Oor
ax1.plot(ang_L, spec_L, 'b-', lw=2)
ax1.axvline(angle_L, color='g', ls='--', label=f'Waarheid (Links): {angle_L}°')
ax1.axvline(est_L[0], color='r', ls=':', label=f'Schatting: {est_L[0]:.1f}°')
ax1.set_title("Linker Oor (yL1, yL2) - Verwacht: Dominerende linker bron")
ax1.set_ylabel("Mag (dB)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Rechter Oor
ax2.plot(ang_R, spec_R, 'b-', lw=2)
ax2.axvline(angle_R, color='g', ls='--', label=f'Waarheid (Rechts): {angle_R}°')
ax2.axvline(est_R[0], color='r', ls=':', label=f'Schatting: {est_R[0]:.1f}°')
ax2.set_title("Rechter Oor (yR1, yR2) - Verwacht: Dominerende rechter bron")
ax2.set_ylabel("Mag (dB)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Frontale Mics
ax3.plot(ang_F, spec_F, 'b-', lw=2)
ax3.axvline(true_front_R, color='g', ls='--', label=f'Waarheid R-bron: {true_front_R}°')
ax3.axvline(true_front_L, color='purple', ls='--', label=f'Waarheid L-bron: {true_front_L}°')
ax3.axvline(est_F[0], color='r', ls=':', label=f'Schatting (Gefaald): {est_F[0]:.1f}°')
ax3.set_title("Frontale Mics (yL1, yR1) - Verwacht: Spatiale aliasing door grote afstand (d=21.5cm)")
ax3.set_ylabel("Mag (dB)")
ax3.set_xlabel("Hoek (graden)")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()