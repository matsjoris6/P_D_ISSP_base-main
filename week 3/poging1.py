import os
import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt


base_folder = os.path.join("sound_files", "head_mounted_rirs")
dry_signal_1_path = os.path.join("sound_files", "part2_track1_dry.wav") 
dry_signal_2_path = os.path.join("sound_files", "part2_track2_dry.wav")

fs_sim = 44100
c = 343.0
d_ear = 0.013    
d_front = 0.215  

folder_left = "s-30"
folder_right = "s30"

try:
    dry_sig_1, _ = sf.read(dry_signal_1_path)
    dry_sig_2, _ = sf.read(dry_signal_2_path)
    min_len = min(len(dry_sig_1), len(dry_sig_2),20*fs_sim)
    dry_sig_1 = dry_sig_1[:min_len]
    dry_sig_2 = dry_sig_2[:min_len]

    yL1_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_L1.wav"))
    yL2_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_L2.wav"))
    yR1_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_R1.wav"))
    yR2_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_R2.wav"))

    yL1_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_L1.wav"))
    yL2_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_L2.wav"))
    yR1_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_R1.wav"))
    yR2_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_R2.wav"))

    yL1_from_left = signal.fftconvolve(dry_sig_1, yL1_rir_left, mode='full')
    yL2_from_left = signal.fftconvolve(dry_sig_1, yL2_rir_left, mode='full')
    yR1_from_left = signal.fftconvolve(dry_sig_1, yR1_rir_left, mode='full')
    yR2_from_left = signal.fftconvolve(dry_sig_1, yR2_rir_left, mode='full')

    yL1_from_right = signal.fftconvolve(dry_sig_2, yL1_rir_right, mode='full')
    yL2_from_right = signal.fftconvolve(dry_sig_2, yL2_rir_right, mode='full')
    yR1_from_right = signal.fftconvolve(dry_sig_2, yR1_rir_right, mode='full')
    yR2_from_right = signal.fftconvolve(dry_sig_2, yR2_rir_right, mode='full')

    yL1_mix = yL1_from_left + yL1_from_right
    yL2_mix = yL2_from_left + yL2_from_right
    yR1_mix = yR1_from_left + yR1_from_right
    yR2_mix = yR2_from_left + yR2_from_right

except Exception as e:
    print(f"Fout bij het laden van audio/RIRs: {e}")
    exit()

vad = np.abs(yL1_from_left) > (np.std(yL1_from_left) * 1e-3)


def music_wideband_2mics(micsigs, fs, d, Q=1, orientation="x"):
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
        if orientation == "x":
            taus = (mics_pos.reshape(-1, 1) * np.cos(rads)) / c 
        else:
            taus = (mics_pos.reshape(-1, 1) * np.sin(rads)) / c 
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

def MUSIC_wideband_HM(micsigs, fs, Q=2):
    L = 1024
    overlap = L // 2
    stft_list = []
    for m in range(micsigs.shape[1]):
        freqs, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window='hann', nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0) 
    M, nF, nT = stft_data.shape
    
    x_ear = 0.215 / 2  
    y_mic = 0.013 / 2  
    mics_pos = np.array([
        [-x_ear,  y_mic],  
        [-x_ear, -y_mic],  
        [ x_ear,  y_mic],  
        [ x_ear, -y_mic]   
    ])
    px = mics_pos[:, 0].reshape(-1, 1)
    py = mics_pos[:, 1].reshape(-1, 1)
    
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    pseudospectra = []
    for k in range(1, nF - 1):
        Y = stft_data[:, k, :]
        Ryy = (Y @ Y.conj().T) / nT
        _, eigvecs = np.linalg.eigh(Ryy)
        En = eigvecs[:, :M-Q] 
        omega = 2 * np.pi * freqs[k]
        taus = (px * np.cos(rads) + py * np.sin(rads)) / c 
        A = np.exp(1j * omega * taus)
        denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
        pseudospectra.append(1.0 / denom)
        
    pseudospectra = np.array(pseudospectra)
    p_geom = np.exp(np.mean(np.log(pseudospectra), axis=0))
    spectrum_db = 10 * np.log10(p_geom / np.max(p_geom))
    peaks_indices, _ = signal.find_peaks(spectrum_db)
    if len(peaks_indices) >= Q:
        sorted_peaks = peaks_indices[np.argsort(spectrum_db[peaks_indices])][-Q:]
        estimated_doas = np.sort(angles[sorted_peaks])
    else:
        estimated_doas = angles[np.argsort(spectrum_db)[-Q:]]
    return angles, spectrum_db, estimated_doas


def gsc_universal(micsigs, target_doa, fs, vad, is_1d_array=True, d=None, orientation="x"):
    N_samples, M_mics = micsigs.shape
    target_rad = np.radians(target_doa)
    
    # Vertragingen berekenen 
    if is_1d_array:
        
        mics_pos = np.array([0, d])
        if orientation == "x":
            taus = (mics_pos * np.cos(target_rad)) / c
        else:
            taus = (mics_pos * np.sin(target_rad)) / c

    else:
        
        x_ear = 0.215 / 2  
        y_mic = 0.013 / 2  
        mics_pos = np.array([[-x_ear, y_mic], [-x_ear, -y_mic], [x_ear, y_mic], [x_ear, -y_mic]])
        mics_centered = mics_pos - np.mean(mics_pos, axis=0)
        
        taus = (mics_centered[:, 0] * np.cos(target_rad) + mics_centered[:, 1] * np.sin(target_rad)) / c

    # Delay and Sum
    aligned_mic = np.zeros_like(micsigs)
    DASout = np.zeros(N_samples)
    f = np.fft.rfftfreq(N_samples, 1/fs)
    for m in range(M_mics):
        Sig = np.fft.rfft(micsigs[:, m])
        Sig_aligned = Sig * np.exp(-1j * 2 * np.pi * f * taus[m])
        t_aligned = np.fft.irfft(Sig_aligned, n=N_samples)
        aligned_mic[:, m] = t_aligned
        DASout += t_aligned
    DASout /= M_mics

    Pn_das = np.var(DASout[vad == 0])
    Psn_das = np.var(DASout[vad == 1])
    Ps_das = Psn_das - Pn_das
    SNR_das = 10 * np.log10(max(Ps_das, 1e-10) / max(Pn_das, 1e-10))
    # Blocking Matrix
    B = np.zeros((M_mics - 1, M_mics))
    for i in range(M_mics - 1):
        B[i, i] = 1
        B[i, i + 1] = -1
    noise_refs = (B @ aligned_mic.T).T
    

    plt.figure(figsize=(12, 6))

    # Pak de eerste ruisreferentie (bijv. L1 - L2)
    noise_ref_1 = noise_refs[:, 0]

    # Plot het originele signaal en de ruisreferentie over elkaar
    t = np.arange(len(noise_ref_1)) / fs_sim
    plt.plot(t, micsigs[:, 0], color='gray', alpha=0.4, label='Origineel (Mic 1)')
    plt.plot(t, noise_ref_1, color='red', alpha=0.7, label='Ruisreferentie (BM Output)')

    plt.title("Blocking Matrix Check: Wordt de spraak onderdrukt?")
    plt.xlabel("Tijd (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Bereken de onderdrukkingsfactor in dB tijdens spraak (VAD == 1)
    leakage_power = np.var(noise_ref_1[vad == 1])
    original_power = np.var(micsigs[vad == 1, 0])
    suppression = 10 * np.log10(original_power / leakage_power)

    print(f"--- Blocking Matrix Analyse ---")
    print(f"Spraakonderdrukking in de BM: {suppression:.2f} dB")
    if suppression < 10:
        print("WAARSCHUWING: Weinig onderdrukking. De spraak lekt door naar het filter!")
    else:
        print("Goede onderdrukking. Het filter kan veilig updaten.")

    # NLMS met VAD
    L = 1024 
    mu = 0.1
    delta = L // 2
    
    target_ref = np.pad(DASout, (delta, 0))[:N_samples]
    vad_delayed = np.pad(vad, (delta, 0))[:N_samples]
    padded_noise = np.pad(noise_refs, ((L-1, 0), (0, 0)))
    
    W = np.zeros((L, M_mics - 1))
    GSCout = np.zeros(N_samples)
    eps = 1e-8
    
    for n in range(N_samples):
        X_slice = padded_noise[n : n+L]
        y_n = np.sum(W * X_slice)
        e_n = target_ref[n] - y_n
        GSCout[n] = e_n
        
        if vad_delayed[n] == 0:
            power = np.sum(X_slice**2)
            W += (mu * e_n / (power + eps)) * X_slice
            
    # SNR
    Pn = np.var(GSCout[vad_delayed == 0])
    Psn = np.var(GSCout[vad_delayed == 1])
    Ps = Psn - Pn
    if Ps < 1e-10: Ps = 1e-10
    if Pn < 1e-10: Pn = 1e-10
    SNRout = 10 * np.log10(Ps / Pn)
    
    Pn_in = np.var(micsigs[vad == 0, 0])
    Psn_in = np.var(micsigs[vad == 1, 0])
    Ps_in = Psn_in - Pn_in
    if Ps_in < 1e-10: Ps_in = 1e-10
    if Pn_in < 1e-10: Pn_in = 1e-10
    SNRin = 10 * np.log10(Ps_in / Pn_in)
    
    return GSCout, SNRout, SNRin, DASout, SNR_das




# Data combinaties
arrays = {
    "Linkeroor (2 mic)":   {"mics": np.column_stack((yL1_mix, yL2_mix)), "1d": True, "d": d_ear,"orientation": "y"},
    "Rechteroor (2 mic)":  {"mics": np.column_stack((yR1_mix, yR2_mix)), "1d": True, "d": d_ear,"orientation": "y"},
    "Frontaal (2 mic)":    {"mics": np.column_stack((yL1_mix, yR1_mix)), "1d": True, "d": d_front, "orientation": "x"},
    "Alle 4 Mics":         {"mics": np.column_stack((yL1_mix, yL2_mix, yR1_mix, yR2_mix)), "1d": False, "d": 0, "orientation": "2d"}
}

results = {}

for name, config in arrays.items():
    micsigs = config["mics"]
    orientation= config["orientation"]
   
    if config["1d"]:
        _, _, est_doas = music_wideband_2mics(micsigs, fs_sim, config["d"], Q=1,orientation=orientation)
         
        target_doa = est_doas[0]
        print(f"\n[{name}] MUSIC 1D Schatting: {target_doa:.1f}°")
    else:
        _, _, est_doas = MUSIC_wideband_HM(micsigs, fs_sim, Q=2)
        #doa dichtste bij bron
        target_doa = est_doas[np.argmax(est_doas)] 
        print(f"\n[{name}] MUSIC 2D Schattingen: {est_doas}. We kiezen Target DOA: {target_doa:.1f}°")

    
    GSCout, SNRout, SNRin, DAS_signal,SNRdas = gsc_universal(
        micsigs=micsigs,
        target_doa=target_doa,
        fs=fs_sim,
        vad=vad,
        is_1d_array=config["1d"],
        d=config["d"],
        orientation=orientation
    )
    
    results[name] = {"out": GSCout, "snr_in": SNRin, "snr_out": SNRout, "das": DAS_signal, "snr_das": SNRdas}
    print(f"  -> Input SNR:  {SNRin:.2f} dB")
    print(f"  -> DAS SNR:     {SNRdas:.2f} dB (Winst: {SNRdas - SNRin:.2f} dB)")
    print(f"  -> Output SNR: {SNRout:.2f} dB (Winst: {SNRout - SNRin:.2f} dB)")

# Plot 
t = np.arange(len(yL1_mix)) / fs_sim
plt.figure(figsize=(14, 10))
for i, (name, res) in enumerate(results.items()):
    plt.subplot(4, 1, i+1)
    #origineel
    plt.plot(t, arrays[name]["mics"][:, 0], color='gray', alpha=0.5, label='Mic 1')
    #  DAS 
    plt.plot(t, res["das"], color='green', alpha=0.6, label='Alleen DAS Beamformer')
    #gsc
    plt.plot(t, res["out"], color='blue', alpha=0.8, label=f'GSC (SNR: {res["snr_out"]:.1f} dB)')
    plt.title(f"{name} | Winst: {res['snr_out'] - res['snr_in']:.1f} dB")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()