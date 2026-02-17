import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
import os
import sys

# ==========================================
# 0. SETUP & IMPORTS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from package import load_rirs
from package.utils import create_micsigs

# ==========================================
# 1. FUNCTIE: BEREKEN WAARHEID
# ==========================================
def calculate_ground_truth_doa(acoustic_scenario):
    if acoustic_scenario.audioPos is None or len(acoustic_scenario.audioPos) == 0:
        return None
    mics = acoustic_scenario.micPos       
    source = acoustic_scenario.audioPos[0] 
    array_center = np.mean(mics, axis=0)
    dx = source[0] - array_center[0]
    dy = source[1] - array_center[1] 
    dist = np.sqrt(dx**2 + dy**2)
    cos_theta = -dy / dist
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# ==========================================
# 2. FUNCTIE: MUSIC NARROWBAND
# ==========================================
def music_narrowband(micsigs, fs, acoustic_scenario):
    # 1. STFT berekenen [cite: 10, 11]
    L = 1024
    overlap = L // 2
    stft_list = []
    for m in range(micsigs.shape[1]):
        _, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window='hann', nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0)
    
    M, nF, nT = stft_data.shape
    c, d = 343.0, 0.05
    
    # 2. Aantal bronnen (Q) [cite: 36]
    Q = acoustic_scenario.audioPos.shape[0] if acoustic_scenario.audioPos is not None else 0
    
    # 3. Frequentiebin selectie [cite: 14]
    power_spectrum = np.mean(np.abs(stft_data)**2, axis=(0, 2))
    max_bin_idx = np.argmax(power_spectrum[1:]) + 1 
    freqs = np.fft.rfftfreq(2*(nF-1), 1/fs)
    omega = 2 * np.pi * freqs[max_bin_idx]

    # 4. Covariantiematrix Ryy 
    Y = stft_data[:, max_bin_idx, :]
    Ryy = (Y @ Y.conj().T) / nT
    
    # 5. Eigen-decompositie [cite: 17, 35]
    # np.linalg.eigh sorteert van klein naar groot
    eigvals, eigvecs = np.linalg.eigh(Ryy)
    
    # Noise Subspace (En) bevat de M - Q kleinste eigenvectoren [cite: 116, 117]
    En = eigvecs[:, :M-Q] 
    
    # 6. Pseudospectrum berekenen [cite: 118, 121]
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    taus = (np.arange(M).reshape(-1, 1) * d * (-np.cos(rads))) / c 
    A = np.exp(-1j * omega * taus)
    
    denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
    pseudospectrum = 1.0 / denom
    spectrum_db = 10 * np.log10(pseudospectrum / np.max(pseudospectrum))
    
    # 7. Piekdetectie [cite: 37, 40]
    peaks_indices, _ = signal.find_peaks(spectrum_db)
    sorted_peak_indices = peaks_indices[np.argsort(spectrum_db[peaks_indices])][-Q:]
    estimated_doas = np.sort(angles[sorted_peak_indices])
    
    # Retourneer ook alle eigenwaarden (eigvals)
    return angles, spectrum_db, estimated_doas, freqs[max_bin_idx], np.real(eigvals)

# ==========================================
# 3. MAIN PROGRAMMA
# ==========================================
if __name__ == "__main__":
    try:
        rirs_folder = os.path.join(parent_dir, "rirs")
        files = [os.path.join(rirs_folder, f) for f in os.listdir(rirs_folder) if f.endswith('.pkl')]
        latest_rir = max(files, key=os.path.getmtime)
        scenario = load_rirs(latest_rir)

        speech_path = os.path.join(parent_dir, "sound_files (1)", "speech1.wav")
        if not os.path.exists(speech_path):
            speech_path = os.path.join(parent_dir, "sound_files", "speech1.wav")
            
        noise_path = os.path.join(parent_dir, "sound_files (1)", "white_noise1.wav")
        if not os.path.exists(noise_path):
            noise_path = os.path.join(parent_dir, "sound_files", "white_noise1.wav")

        # Situatie 1: Schoon
        micsigs_clean, _, _ = create_micsigs(scenario, [speech_path], [], duration=10.0)
        _, spec_clean, _, f_clean, ev_clean = music_narrowband(micsigs_clean, scenario.fs, scenario)

        # Situatie 2: Ruis [cite: 29]
        if os.path.exists(noise_path):
            micsigs_noisy, _, _ = create_micsigs(scenario, [speech_path], [noise_path], duration=10.0)
        else:
            noise_vol = 0.05
            micsigs_noisy = micsigs_clean + (np.random.randn(*micsigs_clean.shape) * noise_vol)

        angles, spec_noisy, _, f_noisy, ev_noisy = music_narrowband(micsigs_noisy, scenario.fs, scenario)

        # Analyseer resultaten
        real_doa = calculate_ground_truth_doa(scenario)

        # Plot [cite: 18]
        plt.figure(figsize=(12, 7))
        plt.plot(angles, spec_clean, color='blue', label=f'Ruisvrij (f={f_clean:.1f}Hz)')
        plt.plot(angles, spec_noisy, color='orange', linestyle='--', label=f'Met Ruis (f={f_noisy:.1f}Hz)')
        if real_doa is not None:
            plt.axvline(x=real_doa, color='green', label=f'Waarheid ({real_doa:.1f} deg)', alpha=0.5)
        plt.xlim(0, 180)
        plt.ylim(-50, 2)
        plt.xlabel("Hoek (graden)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Vergelijking MUSIC Pseudospectrum")
        plt.legend()
        
        # UITGEBREIDE EIGENWAARDEN OUTPUT [cite: 31, 32]
        print("\n" + "="*50)
        print("VOLLEDIGE EIGENWAARDEN ANALYSE (Gesorteerd van klein naar groot)")
        print("="*50)
        print(f"Schoon scenario: {ev_clean}")
        print(f"Ruis scenario:   {ev_noisy}")
        print("-"*50)
        print(f"Verschil (Ruis - Schoon): {ev_noisy - ev_clean}")
        print("="*50)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Fout: {e}")