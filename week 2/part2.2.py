import numpy as np
import scipy.signal as signal
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
def calculate_ground_truth_doas(acoustic_scenario):
    if acoustic_scenario.audioPos is None or len(acoustic_scenario.audioPos) == 0:
        return []
    
    mics = acoustic_scenario.micPos       
    array_center = np.mean(mics, axis=0)
    
    doas = []
    for source in acoustic_scenario.audioPos:
        dx = source[0] - array_center[0]
        dy = source[1] - array_center[1] 
        dist = np.sqrt(dx**2 + dy**2)
        
        # Volgens de conventie ligt mic 1 bovenaan (0 graden).
        # Een positieve dy (bron ligt hoger dan center) geeft cos(0)=1.
        cos_theta = np.clip(dy / dist, -1.0, 1.0)
        doas.append(np.degrees(np.arccos(cos_theta)))
        
    return np.array(doas)

# ==========================================
# 2. FUNCTIE: MUSIC NARROWBAND
# ==========================================
def music_narrowband(micsigs, fs, acoustic_scenario):
    # 1. STFT berekenen (L=1024, 50% overlap)
    L = 1024
    overlap = L // 2
    stft_list = []
    for m in range(micsigs.shape[1]):
        _, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window='hann', nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0)
    
    M, nF, nT = stft_data.shape
    c, d = 343.0, 0.05
    
    # 2. Bepaal aantal bronnen Q
    Q = acoustic_scenario.audioPos.shape[0] if acoustic_scenario.audioPos is not None else 1
    
    # 3. Selecteer de frequentiebin met het hoogste vermogen (Boven 400Hz!)
    freqs = np.fft.rfftfreq(2*(nF-1), 1/fs)
    power_spectrum = np.mean(np.abs(stft_data)**2, axis=(0, 2))
    
    # Forceer MUSIC om een frequentie met wél genoeg spatiële resolutie te gebruiken
    valid_bins = freqs > 400.0  
    power_spectrum_valid = power_spectrum.copy()
    power_spectrum_valid[~valid_bins] = 0.0
    
    max_bin_idx = np.argmax(power_spectrum_valid)
    f_val = freqs[max_bin_idx]
    omega = 2 * np.pi * f_val

    # 4. Bereken ruimtelijke covariantiematrix Ryy
    Y = stft_data[:, max_bin_idx, :]
    Ryy = (Y @ Y.conj().T) / nT
    
    # 5. Eigen-decompositie en Noise Subspace E(omega)
    eigvals, eigvecs = np.linalg.eigh(Ryy)
    En = eigvecs[:, :M-Q] 
    
    # 6. Pseudospectrum berekenen
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    
    # Gecorrigeerde vertraging passend bij 0 graden = top end-fire
    taus = (np.arange(M).reshape(-1, 1) * d * np.cos(rads)) / c 
    A = np.exp(-1j * omega * taus)
    
    denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
    pseudospectrum = 1.0 / denom
    spectrum_db = 10 * np.log10(pseudospectrum / np.max(pseudospectrum))
    
    # 7. Identificeer de Q grootste pieken
    peaks_indices, _ = signal.find_peaks(spectrum_db)
    
    if len(peaks_indices) >= Q:
        sorted_peak_indices = peaks_indices[np.argsort(spectrum_db[peaks_indices])][-Q:]
        estimated_doas = np.sort(angles[sorted_peak_indices])
    else:
        # Robuuste fallback: als bronnen overlappen in een brede bult, 
        # pak dan de Q lokaal hoogste waarden met tenminste 10 graden afstand.
        sorted_all = np.argsort(spectrum_db)[::-1]
        est_idx = []
        for idx in sorted_all:
            if len(est_idx) == Q: break
            if all(abs(idx - e) > 20 for e in est_idx): 
                est_idx.append(idx)
        estimated_doas = np.sort(angles[est_idx])
    
    return angles, spectrum_db, estimated_doas, f_val, np.real(eigvals)

# ==========================================
# 3. MAIN PROGRAMMA
# ==========================================
if __name__ == "__main__":
    try:
        rirs_folder = os.path.join(parent_dir, "rirs")
        files = [os.path.join(rirs_folder, f) for f in os.listdir(rirs_folder) if f.endswith('.pkl')]
        latest_rir = max(files, key=os.path.getmtime)
        scenario = load_rirs(latest_rir)

        Q = scenario.audioPos.shape[0] if scenario.audioPos is not None else 1

        # Vaste paden naar de audio bestanden
        alle_bestanden = [
            os.path.join(parent_dir, "sound_files", "speech1.wav"),
            os.path.join(parent_dir, "sound_files", "speech2.wav")
        ]
        speech_paths = alle_bestanden[:Q]

        # Micsigs genereren
        micsigs_clean, _, _ = create_micsigs(scenario, speech_paths, [], duration=10.0)
        
        # MUSIC runnen
        angles, spec_clean, est_clean, f_clean, ev_clean = music_narrowband(micsigs_clean, scenario.fs, scenario)
        real_doas = calculate_ground_truth_doas(scenario)

        # ================= PLOTTEN =================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

        # Pseudospectrum plot
        ax1.plot(angles, spec_clean, color='blue', label=f'Ruisvrij (f={f_clean:.1f}Hz)')
        
        for i, r_doa in enumerate(real_doas):
            label = 'Waarheid' if i == 0 else None
            ax1.axvline(x=r_doa, color='green', linestyle='--', label=label)
            
        for i, e_doa in enumerate(est_clean):
            label = 'Schatting' if i == 0 else None
            ax1.axvline(x=e_doa, color='red', linestyle=':', label=label)
            peak_idx = np.argmin(np.abs(angles - e_doa))
            ax1.plot(e_doa, spec_clean[peak_idx], "rx", markersize=8)

        ax1.set_xlim(0, 180)
        ax1.set_ylim(-50, 2)
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title(f"Narrowband MUSIC Pseudospectrum ({len(real_doas)} bronnen)")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Eigenwaarden plot
        indices = np.arange(1, len(ev_clean) + 1)
        ax2.scatter(indices, ev_clean, color='blue', label='Schoon', marker='o')
        ax2.set_yscale('log')
        ax2.set_xticks(indices)
        ax2.set_xlabel("Eigenwaarde Index (1 = kleinste)")
        ax2.set_ylabel("Magnitude (Log schaal)")
        ax2.set_title("Eigenwaarden van de Covariantiematrix Ryy")
        ax2.legend()
        ax2.grid(True, which="both", ls="-", alpha=0.2)

        # ================= CONSOLE OUTPUT =================
        print("\n" + "="*60)
        print(f"DOA ANALYSE ({len(real_doas)} bronnen)")
        print("-" * 60)
        print(f"Echte DOAs:      {np.round(real_doas, 2)}°")
        print(f"Geschatte DOAs:  {np.round(est_clean, 2)}°")
        
        if len(est_clean) == len(real_doas):
            fouten = np.abs(np.sort(real_doas) - np.sort(est_clean))
            print(f"Foutmarge:       {np.round(fouten, 2)}°")
            
        print("\nEIGENWAARDEN")
        print(f"Schoon: {[f'{v:.2e}' for v in ev_clean]}")
        print("="*60)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Fout in uitvoering: {e}")