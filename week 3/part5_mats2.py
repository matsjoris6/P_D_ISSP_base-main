import os
import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==========================================
# 0. PARAMETERS & HEAD-MOUNTED GEOMETRIE
# ==========================================
fs_sim = 44100
c = 343.0

# 2D Posities van de microfoons rond het hoofd (in meters)
x_ear = 0.215 / 2  
y_mic = 0.013 / 2  

# Volgorde: L1 (links voor), L2 (links achter), R1 (rechts voor), R2 (rechts achter)
micPos_all = np.array([
    [-x_ear,  y_mic],  # 0: L1
    [-x_ear, -y_mic],  # 1: L2
    [ x_ear,  y_mic],  # 2: R1
    [ x_ear, -y_mic]   # 3: R2
])

# ==========================================
# 1. SIGNAAL GENERATIE MET HMIRs (s90 en s-90)
# ==========================================
def generate_hmir_signals():
    print("[Info] Laden van HMIRs en genereren van signalen...")
    base_folder = os.path.join("sound_files", "head_mounted_rirs")
    dry_signal_1_path = os.path.join("sound_files", "part2_track1_dry.wav") # Target (Links)
    dry_signal_2_path = os.path.join("sound_files", "part2_track2_dry.wav") # Noise (Rechts)
    
    folder_left = "s-90"
    folder_right = "s90"

    dry_sig_1, _ = sf.read(dry_signal_1_path)
    dry_sig_2, _ = sf.read(dry_signal_2_path)
    
    min_len = min(len(dry_sig_1), len(dry_sig_2), 20 * fs_sim)
    dry_sig_1 = dry_sig_1[:min_len]
    dry_sig_2 = dry_sig_2[:min_len]

    def load_rirs(folder):
        L1, _ = sf.read(os.path.join(base_folder, folder, "HMIR_L1.wav"))
        L2, _ = sf.read(os.path.join(base_folder, folder, "HMIR_L2.wav"))
        R1, _ = sf.read(os.path.join(base_folder, folder, "HMIR_R1.wav"))
        R2, _ = sf.read(os.path.join(base_folder, folder, "HMIR_R2.wav"))
        return L1, L2, R1, R2

    yL1_L, yL2_L, yR1_L, yR2_L = load_rirs(folder_left)
    yL1_R, yL2_R, yR1_R, yR2_R = load_rirs(folder_right)

    # Convoluties uitvoeren en afsnijden op min_len
    speech_matrix = np.column_stack((
        signal.fftconvolve(dry_sig_1, yL1_L, mode='full')[:min_len],
        signal.fftconvolve(dry_sig_1, yL2_L, mode='full')[:min_len],
        signal.fftconvolve(dry_sig_1, yR1_L, mode='full')[:min_len],
        signal.fftconvolve(dry_sig_1, yR2_L, mode='full')[:min_len]
    ))

    noise_matrix = np.column_stack((
        signal.fftconvolve(dry_sig_2, yL1_R, mode='full')[:min_len],
        signal.fftconvolve(dry_sig_2, yL2_R, mode='full')[:min_len],
        signal.fftconvolve(dry_sig_2, yR1_R, mode='full')[:min_len],
        signal.fftconvolve(dry_sig_2, yR2_R, mode='full')[:min_len]
    ))

    # Witte microfoonruis toevoegen (10% van target power) conform opgave 
    Ps_clean = np.var(speech_matrix[:, 0])
    white_noise = np.random.randn(*speech_matrix.shape) * np.sqrt(0.10 * Ps_clean)
    noise_matrix += white_noise

    mic_matrix = speech_matrix + noise_matrix
    vad = np.abs(speech_matrix[:, 0]) > (np.std(speech_matrix[:, 0]) * 1e-3)
    
    return mic_matrix, speech_matrix, noise_matrix, vad

# ==========================================
# 2. FUNCTIE: MUSIC WIDEBAND (2D Aangepast)
# ==========================================
def music_wideband(micsigs, fs, micPos, Q_override=2):
    L = 1024
    overlap = L // 2
    stft_list = [signal.stft(micsigs[:, m], fs=fs, window='hann', nperseg=L, noverlap=overlap)[2] for m in range(micsigs.shape[1])]
    stft_data = np.stack(stft_list, axis=0) 
    M, nF, nT = stft_data.shape
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    
    mics_centered = micPos - np.mean(micPos, axis=0)
    px, py = mics_centered[:, 0].reshape(-1, 1), mics_centered[:, 1].reshape(-1, 1)
    
    pseudospectra = []
    freqs = np.fft.rfftfreq(L, 1/fs)
    for k in range(1, nF - 1):
        Y = stft_data[:, k, :]
        Ryy = (Y @ Y.conj().T) / nT
        _, eigvecs = np.linalg.eigh(Ryy)
        En = eigvecs[:, :M-Q_override] 
        omega = 2 * np.pi * freqs[k]
        # 2D Steering vector 
        taus = (px * np.cos(rads) + py * np.sin(rads)) / c 
        A = np.exp(1j * omega * taus) 
        denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
        pseudospectra.append(1.0 / denom)
        
    p_geom = np.exp(np.mean(np.log(pseudospectra), axis=0))
    spectrum_db = 10 * np.log10(p_geom / np.max(p_geom))
    
    # Fallback: als signal.find_peaks niets vindt, gebruik dan argmax
    peaks_indices, _ = signal.find_peaks(spectrum_db)
    if len(peaks_indices) == 0:
        return np.array([angles[np.argmax(spectrum_db)]])
    
    sorted_peaks = peaks_indices[np.argsort(spectrum_db[peaks_indices])][-Q_override:]
    return np.sort(angles[sorted_peaks])

# ==========================================
# 3. FUNCTIE: GSC IN TIME DOMAIN (COMPLEET)
# ==========================================
def gsc_td_complete(mic, speech, noise, vad, fs, micPos):
    # 1. DOA & DAS BF
    est_doas = music_wideband(mic, fs, micPos, Q_override=2)
    # Richt op bron dichtst bij 180° (links) conform Week 2 Part 5 
    target_doa = est_doas[np.argmin(np.abs(est_doas - 180.0))]
    target_rad = np.radians(target_doa)
    mics_centered = micPos - np.mean(micPos, axis=0)
    taus = (mics_centered[:, 0] * np.cos(target_rad) + mics_centered[:, 1] * np.sin(target_rad)) / c

    def apply_das(signals, delays):
        N, M = signals.shape
        out_sum = np.zeros(N)
        f = np.fft.rfftfreq(N, 1/fs)
        for m in range(M):
            out_sum += np.fft.irfft(np.fft.rfft(signals[:, m]) * np.exp(1j * 2 * np.pi * f * delays[m]), n=N)
        return out_sum / M

    DASout = apply_das(mic, taus)
    
    # 2. Blocking Matrix (Griffiths-Jim) 
    M_mics = mic.shape[1]
    if M_mics == 2:
        B = np.array([[1, -1]])
    else:
        B = np.array([[1, -1, 0, 0], [1, 0, -1, 0], [1, 0, 0, -1]])
    noise_refs = (B @ mic.T).T 

    # 3. NLMS Adaptief Filter met VAD 
    L, mu, delta = 1024, 0.1, 512
    N_samples = len(DASout)
    target_ref = np.pad(DASout, (delta, 0))[:N_samples]
    vad_delayed = np.pad(vad, (delta, 0))[:N_samples]
    W, GSCout, padded_noise = np.zeros((L, M_mics-1)), np.zeros(N_samples), np.pad(noise_refs, ((L-1, 0), (0, 0)))

    for n in range(N_samples):
        X_slice = padded_noise[n : n+L]
        e_n = target_ref[n] - np.sum(W * X_slice)
        GSCout[n] = e_n
        if vad_delayed[n] == 0: # Filter adaptatie UIT tijdens spraak 
            W += (mu * e_n / (np.sum(X_slice**2) + 1e-8)) * X_slice

    # SNR Berekeningen
    SNR_in = 10 * np.log10(np.var(speech[:, 0]) / (np.var(noise[:, 0]) + 1e-10))
    Pn_gsc = np.var(GSCout[vad_delayed == 0])
    SNR_gsc = 10 * np.log10((np.var(GSCout[vad_delayed == 1]) - Pn_gsc) / (Pn_gsc + 1e-10))
    
    return SNR_in, SNR_gsc, GSCout, delta

# ==========================================
# 4. MAIN: UITVOERING & AUDIO EXPORT
# ==========================================
if __name__ == "__main__":
    try:
        mic_f, speech_f, noise_f, vad_f = generate_hmir_signals()
        
        configs = {
            "1_Rechteroor": [2, 3], 
            "2_Linkeroor": [0, 1],  
            "3_Frontaal": [0, 2],   
            "4_Alle_4_Mics": [0, 1, 2, 3]
        }

        print("\n" + "="*60 + "\nConfiguratie    | Input SNR  | GSC SNR (Eind)\n" + "-"*60)
        for name, idx in configs.items():
            snr_i, snr_g, audio_raw, d = gsc_td_complete(
                mic_f[:, idx], speech_f[:, idx], noise_f[:, idx], vad_f, fs_sim, micPos_all[idx]
            )
            print(f"{name:<15} | {snr_i:>8.2f} dB | {snr_g:>11.2f} dB")
            
            # Audio normaliseren en opslaan
            out_audio = np.pad(audio_raw[d:], (0, d))
            out_audio = out_audio / (np.max(np.abs(out_audio)) + 1e-10)
            sf.write(f"Output_{name}.wav", out_audio, fs_sim)
            
            # Plot opslaan
            plt.figure(figsize=(10, 4))
            plt.plot(out_audio, color='green', label='GSC Output')
            plt.title(f"Resultaat: {name}"); plt.grid(True); plt.savefig(f"Plot_{name}.png"); plt.close()

        print("="*60 + "\n✅ Succes! .wav en .png bestanden zijn opgeslagen.")

    except Exception as e:
        import traceback
        print(f"\n❌ Fout opgetreden:\n{e}"); traceback.print_exc()