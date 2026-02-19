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
from package.utils import calculate_ground_truth_doas
# ==========================================
# 2. FUNCTIE: MUSIC WIDEBAND
# ==========================================
def music_wideband(micsigs, fs, acoustic_scenario):
    # 1. Efficiënte STFT berekening (Vectorized over time)
    L = 1024
    overlap = L // 2
    
    stft_list = []
    for m in range(micsigs.shape[1]):
        freqs, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window='hann', nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0) # Shape: (M, nF, nT)
    
    M, nF, nT = stft_data.shape
    c = 343.0
    Q = acoustic_scenario.audioPos.shape[0] if acoustic_scenario.audioPos is not None else 1
    
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    
    # Pre-compute mic coördinaten
    mics_centered = acoustic_scenario.micPos - np.mean(acoustic_scenario.micPos, axis=0)
    px = mics_centered[:, 0].reshape(-1, 1)
    py = mics_centered[:, 1].reshape(-1, 1)
    
    # We itereren over bins k = 2 ... L/2. 
    # In Python (0-based) is k=1 (DC) index 0. L/2 is index 512 (Nyquist).
    # De indices 1 t/m 511 komen exact overeen met k=2 ... L/2
    valid_indices = range(1, L // 2)
    pseudospectra = []
    
    for k in valid_indices:
        # Signaal isoleren voor frequentie k
        Y = stft_data[:, k, :]
        Ryy = (Y @ Y.conj().T) / nT
        
        # Ruissubruimte bepalen
        _, eigvecs = np.linalg.eigh(Ryy)
        En = eigvecs[:, :M-Q] 
        
        # Steering vector bepalen
        omega = 2 * np.pi * freqs[k]
        taus = (px * np.sin(rads) + py * np.cos(rads)) / c 
        A = np.exp(-1j * omega * taus)
        
        # Pseudospectrum P(theta)
        denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
        p_theta = 1.0 / denom
        pseudospectra.append(p_theta)
        
    pseudospectra = np.array(pseudospectra) # Shape: (511, len(angles))
    
    # 2. Geometrisch Gemiddelde (Numeriek stabiele methode) + veel minder gevoelig voor pieken die samensmelten
    # i.p.v. p1 * p2 * .. pn tot de macht 1/N doen we: exp(mean(log(p))) want veel rekenkrachtiger
    log_p = np.log(pseudospectra)
    p_geom = np.exp(np.mean(log_p, axis=0))
    
    # Normalizeer en naar dB
    spectrum_geom_db = 10 * np.log10(p_geom / np.max(p_geom))
    
    # 3. Peak Finding op het gemiddelde spectrum
    peaks_indices, _ = signal.find_peaks(spectrum_geom_db)
    if len(peaks_indices) >= Q:
        sorted_peak_indices = peaks_indices[np.argsort(spectrum_geom_db[peaks_indices])][-Q:]
        estimated_doas = np.sort(angles[sorted_peak_indices])
    else:
        # Fallback indien pieken samensmelten
        sorted_all = np.argsort(spectrum_geom_db)[::-1]
        est_idx = []
        for idx in sorted_all:
            if len(est_idx) == Q: break
            if all(abs(idx - e) > 20 for e in est_idx):
                est_idx.append(idx)
        estimated_doas = np.sort(angles[est_idx])

    # Geef de individuele genormaliseerde spectra ook mee (in dB) voor de plot
    ps_ind_db = 10 * np.log10(pseudospectra / np.max(pseudospectra, axis=1, keepdims=True))

    return angles, ps_ind_db, spectrum_geom_db, estimated_doas

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

        alle_bestanden = [ 
            os.path.join(parent_dir, "sound_files", "speech1.wav"),
            os.path.join(parent_dir, "sound_files", "speech2.wav")
        ]
        speech_paths = alle_bestanden[:Q]

        micsigs_clean, _, _ = create_micsigs(scenario, speech_paths, [], duration=10.0)
        
        # Voer de wideband MUSIC uit
        angles, ps_ind_db, spec_geom, est_wide, = music_wideband(micsigs_clean, scenario.fs, scenario)
        real_doas = calculate_ground_truth_doas(scenario)

        # ================= PLOTTEN =================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # --- PLOT 1: Alle individuele spectra (zeer transparant) ---
        ax1.plot(angles, ps_ind_db.T, color='gray', alpha=0.03) 
        # Fake lijn voor de legend
        ax1.plot([], [], color='gray', alpha=0.5, label='Individuele bins (k=2 ... L/2)') 
        ax1.set_xlim(0, 180)
        ax1.set_ylim(-60, 2)
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title("Alle individuele Pseudospectra over frequenties heen")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- PLOT 2: Geometrisch Gemiddelde ---
        ax2.plot(angles, spec_geom, color='blue', linewidth=2, label='Geometrisch Gemiddeld Spectrum')
        
        for i, r_doa in enumerate(real_doas):
            ax2.axvline(x=r_doa, color='green', linestyle='--', label='Waarheid' if i == 0 else None)
            
        for i, e_doa in enumerate(est_wide):
            peak_idx = np.argmin(np.abs(angles - e_doa))
            ax2.axvline(x=e_doa, color='red', linestyle=':', label='Schatting (Piek)' if i == 0 else None)
            ax2.plot(e_doa, spec_geom[peak_idx], "rx", markersize=8)

        ax2.set_xlim(0, 180)
        ax2.set_ylim(-60, 2)
        ax2.set_xlabel("Hoek (graden)")
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_title("Wideband MUSIC: Geometrisch Gemiddelde Pseudospectrum")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # ================= CONSOLE OUTPUT =================
        print("\n" + "="*60)
        print(f"WIDEBAND DOA ANALYSE ({len(real_doas)} bronnen)")
        print("-" * 60)
        print(f"Echte DOAs:      {np.round(real_doas, 2)}°")
        print(f"Geschatte DOAs:  {np.round(est_wide, 2)}°")
        
        if len(est_wide) == len(real_doas):
            fouten = np.abs(np.sort(real_doas) - np.sort(est_wide))
            print(f"Foutmarge:       {np.round(fouten, 2)}°")
        print("="*60)

    except Exception as e:
        print(f"Fout in uitvoering: {e}")