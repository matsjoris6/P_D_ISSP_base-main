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

# ==========================================
# 1. FUNCTIE: CREATE MICSIGS
# ==========================================
def create_micsigs(acoustic_scenario, speech_filenames, noise_filenames, duration):
    fs_rir = acoustic_scenario.fs
    num_mics = acoustic_scenario.RIRs_audio.shape[1] 
    num_samples = int(duration * fs_rir)
    
    speech_component = np.zeros((num_samples, num_mics))
    noise_component = np.zeros((num_samples, num_mics))

    for src_idx, filename in enumerate(speech_filenames):
        if not os.path.exists(filename): continue 
        audio, fs_src = sf.read(filename)
        if len(audio.shape) > 1: audio = audio[:, 0]
        if fs_src != fs_rir:
            new_len = int(len(audio) * fs_rir / fs_src)
            audio = signal.resample(audio, new_len)
        for mic_idx in range(num_mics):
            rir = acoustic_scenario.RIRs_audio[:, mic_idx, src_idx]
            filtered = signal.fftconvolve(audio, rir, mode='full')
            L = min(len(filtered), num_samples)
            speech_component[:L, mic_idx] += filtered[:L]

    if noise_filenames and acoustic_scenario.RIRs_noise is not None:
        for src_idx, filename in enumerate(noise_filenames):
            if not filename or not os.path.exists(filename): continue
            audio, fs_src = sf.read(filename)
            if len(audio.shape) > 1: audio = audio[:, 0]
            if fs_src != fs_rir: audio = signal.resample(audio, int(len(audio)*fs_rir/fs_src))
            for mic_idx in range(num_mics):
                rir = acoustic_scenario.RIRs_noise[:, mic_idx, src_idx]
                filtered = signal.fftconvolve(audio, rir, mode='full')
                L = min(len(filtered), num_samples)
                noise_component[:L, mic_idx] += filtered[:L]
    
    return speech_component + noise_component, speech_component, noise_component

# ==========================================
# 2. FUNCTIE: BEREKEN WAARHEID (NIEUW)
# ==========================================
def calculate_ground_truth_doa(acoustic_scenario):
    """
    Berekent de werkelijke hoek op basis van de x,y posities in het scenario.
    Conventie: 0 graden = Top (Noord/Positieve Y), 180 graden = Bottom (Zuid/Negatieve Y).
    """
    if acoustic_scenario.audioPos is None or len(acoustic_scenario.audioPos) == 0:
        return None

    # Posities ophalen
    mics = acoustic_scenario.micPos       # [5 x 2]
    source = acoustic_scenario.audioPos[0] # [x, y] van bron 1

    # Het centrum van de array berekenen
    array_center = np.mean(mics, axis=0)

    # Vector van centrum naar bron
    vec = source - array_center
    dx = vec[0]
    dy = vec[1]

    # Bereken hoek t.o.v. de verticale as (Y-as)
    # Cos(theta) = dy / r
    # Als dy positief is (boven array) -> hoek dicht bij 0
    # Als dy negatief is (onder array) -> hoek dicht bij 180
    dist = np.sqrt(dx**2 + dy**2)
    cos_theta = dy / dist
    
    # Clip voor numerieke stabiliteit (zodat het tussen -1 en 1 blijft)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Arccos geeft radialen tussen 0 en pi
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

# ==========================================
# 3. FUNCTIE: MUSIC NARROWBAND
# ==========================================
def music_narrowband(stft_data, fs, acoustic_scenario):
    print("\n--- Start MUSIC Narrowband ---")
    
    M, nF, nT = stft_data.shape
    c = 343.0
    d = 0.05
    
    # --- A. Frequentie Selectie ---
    power_spectrum = np.mean(np.abs(stft_data)**2, axis=(0, 2))
    max_bin_idx = np.argmax(power_spectrum[1:]) + 1 
    freqs = np.fft.rfftfreq(2*(nF-1), 1/fs)
    f_max = freqs[max_bin_idx]
    omega = 2 * np.pi * f_max
    
    print(f"📡 Frequentie piek: {f_max:.2f} Hz")

    # --- B. Covariantiematrix & Eigenwaarden ---
    Y = stft_data[:, max_bin_idx, :]
    Ryy = (Y @ Y.conj().T) / nT
    eigvals, eigvecs = np.linalg.eigh(Ryy)
    En = eigvecs[:, :-1] # Noise subspace
    
    # --- C. Ground Truth Berekenen (NIEUW) ---
    real_doa = calculate_ground_truth_doa(acoustic_scenario)
    
    # --- D. Scannen en Pseudospectrum ---
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    
    m_indices = np.arange(M).reshape(-1, 1)
    cos_thetas = np.cos(rads).reshape(1, -1)
    taus = (m_indices * d * cos_thetas) / c
    A = np.exp(-1j * omega * taus)
    
    proj = En.conj().T @ A
    denom = np.sum(np.abs(proj)**2, axis=0)
    pseudospectrum = 1.0 / denom
    spectrum_db = 10 * np.log10(pseudospectrum / np.max(pseudospectrum))
    
    # --- E. Resultaat ---
    peak_idx = np.argmax(pseudospectrum)
    estimated_doa = angles[peak_idx]
    
    print("-" * 30)
    print(f"🎯 Geschatte DOA: {estimated_doa:.1f}°")
    if real_doa is not None:
        print(f"📍 Echte DOA:      {real_doa:.1f}°")
        print(f"📉 Foutmarge:      {abs(estimated_doa - real_doa):.2f}°")
    print("-" * 30)
    
    # --- F. Plotten ---
    plt.figure(figsize=(10, 6))
    plt.plot(angles, spectrum_db, linewidth=2, label='MUSIC Spectrum')
    
    # Geschatte DOA (Rood)
    plt.stem([estimated_doa], [0], linefmt='r--', markerfmt='ro', basefmt=' ', 
             label=f'Schatting: {estimated_doa:.1f}°')
    
    # Echte DOA (Groen)
    if real_doa is not None:
        plt.axvline(x=real_doa, color='green', linestyle='-', alpha=0.7, 
                    label=f'Waarheid: {real_doa:.1f}°')

    plt.xlim(0, 180)
    plt.ylim(bottom=-60)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Hoek (graden) [0=Top, 180=Bottom]")
    plt.ylabel("Pseudospectrum (dB)")
    plt.title(f"MUSIC Resultaat (f={f_max:.0f}Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return estimated_doa

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    try:
        # Zoek bestanden
        rirs_folder = os.path.join(parent_dir, "rirs")
        if not os.path.exists(rirs_folder): raise FileNotFoundError("Geen 'rirs' map.")
        files = [os.path.join(rirs_folder, f) for f in os.listdir(rirs_folder) if f.endswith('.pkl')]
        if not files: raise FileNotFoundError("Geen RIRs gevonden.")
        latest_rir = max(files, key=os.path.getmtime)
        print(f"📂 RIR geladen: {os.path.basename(latest_rir)}")

        speech_path = None
        possible = [
            os.path.join(parent_dir, "sound_files (1)", "speech1.wav"),
            os.path.join(parent_dir, "sound_files", "speech1.wav"),
            os.path.join(parent_dir, "week 2", "speech1.wav")
        ]
        for p in possible:
            if os.path.exists(p): speech_path = p; break
            
        if not speech_path:
            # Fallback
            sf_dir = os.path.join(parent_dir, "sound_files")
            if os.path.exists(sf_dir):
                wavs = [f for f in os.listdir(sf_dir) if f.endswith('.wav')]
                if wavs: speech_path = os.path.join(sf_dir, wavs[0])
        
        if not speech_path: raise FileNotFoundError("Geen wav bestand.")

        # Uitvoeren
        scenario = load_rirs(latest_rir)
        
        # STFT
        print("Genereren signalen en STFT...")
        micsigs, _, _ = create_micsigs(scenario, [speech_path], [], duration=10.0)
        L = 1024
        stft_list = []
        for m in range(micsigs.shape[1]):
            _, _, Zxx = signal.stft(micsigs[:, m], fs=scenario.fs, window='hann', nperseg=L, noverlap=L//2)
            stft_list.append(Zxx)
        stft_data = np.stack(stft_list, axis=0)
        
        # MUSIC
        music_narrowband(stft_data, scenario.fs, scenario)
        
    except Exception as e:
        print(f"\n❌ Fout: {e}")
        import traceback
        traceback.print_exc()