import numpy as np
import scipy.signal as signal
import scipy.linalg
import matplotlib.pyplot as plt
import soundfile as sf
import os
import sys

# Paden instellen
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from package import load_rirs

# ==========================================
# 1. COMPUTE SIR (Officiële Universiteit Functie)
# ==========================================
def compute_sir(y, x1, x2, groundTruth):
    if np.sqrt(np.sum((y - x1 - x2) ** 2)) / np.sqrt(np.sum(y ** 2)) > 0.01:
        print('/!\ Something is wrong, `y` should be the sum of `x1` and `x2`.')  
        return np.nan
    elif np.sum(groundTruth) + np.sum(1 - groundTruth) != len(groundTruth):
        print('/!\ `groundTruth` vector is not binary.')
        return np.nan
    else:
        sir = 10 * np.log10(
            np.var(x1 * groundTruth + x2 * (1 - groundTruth)) /
            np.var(x2 * groundTruth + x1 * (1 - groundTruth))
        )
    return sir

# ==========================================
# 2. MUSIC (Geometrisch - Week 2 methode)
# ==========================================
def music_wideband_geo(micsigs, fs, acoustic_scenario):
    L = 1024
    overlap = L // 2
    window = np.sqrt(signal.windows.hann(L, sym=False))
    
    stft_list = []
    for m in range(micsigs.shape[1]):
        freqs, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window=window, nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0) 
    
    M, nF, nT = stft_data.shape
    c = 343.0
    Q = 2 # Target + Interferentie
    
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    
    mics_centered = acoustic_scenario.micPos - np.mean(acoustic_scenario.micPos, axis=0)
    px = mics_centered[:, 0].reshape(-1, 1)
    py = mics_centered[:, 1].reshape(-1, 1)
    
    valid_indices = range(1, L // 2)
    pseudospectra = []
    
    for k in valid_indices:
        Y = stft_data[:, k, :]
        Ryy = (Y @ Y.conj().T) / nT
        _, eigvecs = np.linalg.eigh(Ryy)
        En = eigvecs[:, :M-Q] 
        
        omega = 2 * np.pi * freqs[k]
        taus = (px * np.sin(rads) + py * np.cos(rads)) / c 
        A = np.exp(-1j * omega * taus)
        
        denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
        pseudospectra.append(1.0 / denom)
        
    p_geom = np.exp(np.mean(np.log(np.array(pseudospectra)), axis=0))
    spectrum_geom_db = 10 * np.log10(p_geom / np.max(p_geom))
    
    peaks_indices, _ = signal.find_peaks(spectrum_geom_db)
    if len(peaks_indices) >= Q:
        sorted_peak_indices = peaks_indices[np.argsort(spectrum_geom_db[peaks_indices])][-Q:]
        estimated_doas = np.sort(angles[sorted_peak_indices])
    else:
        estimated_doas = np.sort(angles[np.argsort(spectrum_geom_db)[-Q:]])
        
    return estimated_doas

# ==========================================
# 3. LUT GENERATOR (Geometrisch)
# ==========================================
def generate_steering_lut_geo(scenario, L, fs):
    """
    Genereert een Look-Up Table op basis van de array geometrie (Geen RIRs).
    """
    angles = np.arange(0, 181, 1) 
    rads = np.radians(angles)
    c = 343.0
    M = scenario.micPos.shape[0]
    nF = L // 2 + 1
    freqs = np.fft.rfftfreq(L, 1/fs)
    
    mics_centered = scenario.micPos - np.mean(scenario.micPos, axis=0)
    px = mics_centered[:, 0].reshape(-1, 1)
    py = mics_centered[:, 1].reshape(-1, 1)
    
    LUT = np.zeros((nF, len(angles), M), dtype=complex)
    
    for k in range(nF):
        if k == 0:
            LUT[k, :, :] = np.ones((len(angles), M))
            continue
        omega = 2 * np.pi * freqs[k]
        taus = (px * np.sin(rads) + py * np.cos(rads)) / c
        A = np.exp(-1j * omega * taus) # Shape: (M, len(angles))
        H = A / (A[0, :] + 1e-12)      # Normaliseer t.o.v. mic 1
        LUT[k, :, :] = H.T 
        
    return LUT, angles

# ==========================================
# 4. BLOCK-BASED FD-GSC MET COMPONENT SEPARATIE
# ==========================================
def gsc_fd_components(target_comp, interf_comp, fs, doa_estimate, LUT, lut_angles, vad_dry, W_adapt_prev=None, mu=0.1): 
    mic = target_comp + interf_comp
    N_samples, M_mics = mic.shape
    
    L = 1024 
    overlap = L // 2 
    window = np.sqrt(signal.windows.hann(L, sym=False))
    
    _, _, Zxx_mic = signal.stft(mic.T, fs, window=window, nperseg=L, noverlap=overlap)
    _, _, Zxx_tar = signal.stft(target_comp.T, fs, window=window, nperseg=L, noverlap=overlap)
    _, _, Zxx_int = signal.stft(interf_comp.T, fs, window=window, nperseg=L, noverlap=overlap)
    
    nF, nT = Zxx_mic.shape[1], Zxx_mic.shape[2]
    
    # VAD gebaseerd op de DROGE spraak (0.05 threshold)
    vad_frames = np.zeros(nT)
    for n in range(nT):
        start_idx = n * (L - overlap)
        end_idx = start_idx + L
        if end_idx <= len(vad_dry):
            vad_frames[n] = 1 if np.mean(vad_dry[start_idx:end_idx]) > 0.05 else 0    

    E_out_mic = np.zeros((nF, nT), dtype=complex)
    E_out_tar = np.zeros((nF, nT), dtype=complex)
    E_out_int = np.zeros((nF, nT), dtype=complex)
    
    # Geheugen behouden!
    W_adapt = W_adapt_prev if W_adapt_prev is not None else np.zeros((nF, M_mics - 1), dtype=complex)
    lut_idx = np.argmin(np.abs(lut_angles - doa_estimate))
    
    eps = 1e-8
    
    for k in range(nF):
        h_k = LUT[k, lut_idx, :].reshape(M_mics, 1)
        
        # FAS Beamformer
        norm_factor = np.vdot(h_k, h_k).real
        w_FAS = h_k / norm_factor if norm_factor > 1e-10 else np.zeros((M_mics, 1), dtype=complex)
            
        # Blocking Matrix
        B_k = scipy.linalg.null_space(h_k.conj().T).conj().T
        
        for n in range(nT):
            # MIX processing
            x_mic = Zxx_mic[:, k, n].reshape(M_mics, 1)
            d_mic = (w_FAS.conj().T @ x_mic)[0, 0]
            u_mic = B_k @ x_mic
            e_mic = d_mic - (W_adapt[k, :].reshape(-1, 1).conj().T @ u_mic)[0, 0]
            E_out_mic[k, n] = e_mic
            
            # LOSSE componenten filtering voor SIR
            x_tar = Zxx_tar[:, k, n].reshape(M_mics, 1)
            x_int = Zxx_int[:, k, n].reshape(M_mics, 1)
            
            E_out_tar[k, n] = (w_FAS.conj().T @ x_tar)[0, 0] - (W_adapt[k, :].reshape(-1, 1).conj().T @ (B_k @ x_tar))[0, 0]
            E_out_int[k, n] = (w_FAS.conj().T @ x_int)[0, 0] - (W_adapt[k, :].reshape(-1, 1).conj().T @ (B_k @ x_int))[0, 0]
            
            # Adaptatie ALLEEN in stiltes
            if vad_frames[n] == 0:
                power = (u_mic.conj().T @ u_mic).real[0, 0]
                W_adapt[k, :] += (mu * np.conj(e_mic) * u_mic / (power + eps)).flatten()

    _, out_mic_td = signal.istft(E_out_mic, fs, window=window, nperseg=L, noverlap=overlap)
    _, out_tar_td = signal.istft(E_out_tar, fs, window=window, nperseg=L, noverlap=overlap)
    _, out_int_td = signal.istft(E_out_int, fs, window=window, nperseg=L, noverlap=overlap)

    return out_mic_td[:N_samples], out_tar_td[:N_samples], out_int_td[:N_samples], W_adapt

# ==========================================
# 5. MAIN PROGRAMMA
# ==========================================
if __name__ == "__main__":
    use_reverb = True 
    fs = 44100
    segment_duration = 10.0 
    segment_samples = int(fs * segment_duration)
    
    rir_files_no_reverb = ["160_20_no_reverb.pkl", "140_40_no_reverb.pkl", "120_60_no_reverb.pkl", "100_40_no_reverb.pkl", "95_85_no_reverb.pkl"]
    rir_files_reverb = ["160_20_reverb.pkl", "140_40_reverb.pkl", "120_60_reverb.pkl", "100_40_reverb.pkl", "95_85_reverb.pkl"]
    
    rir_files = rir_files_reverb if use_reverb else rir_files_no_reverb
    reverb_label = "REVERBERANT (T60=1s)" if use_reverb else "ANECHOIC (T60=0s)"
    
    t_dry, _ = sf.read(os.path.join(parent_dir, "sound_files", "part1_track1_dry.wav"))
    i_dry, _ = sf.read(os.path.join(parent_dir, "sound_files", "part1_track2_dry.wav"))
    
    full_mix_out, full_mic_in = [], []
    sir_in_history, sir_out_history = [], []

    print("\n" + "="*70)
    print(f"START TIME-VARYING FD-GSC EXPERIMENT [{reverb_label}] - GEOMETRIE LUT")
    print("="*70)

    W_adapt_state = None  
    
    for i, rir_file in enumerate(rir_files):
        print(f"\n--- Segment {i+1}/5 (Positie RIR: {rir_file}) ---")
        scenario = load_rirs(os.path.join(parent_dir, "rirs", rir_file))
        
        target_rir = scenario.RIRs_audio[:, :, 0]
        interf_rir = scenario.RIRs_audio[:, :, 1] if scenario.RIRs_audio.shape[2] > 1 else scenario.RIRs_noise[:, :, 0]

        chunk_t = t_dry[i*segment_samples : (i+1)*segment_samples]
        chunk_i = i_dry[i*segment_samples : (i+1)*segment_samples]

        t_comp = np.stack([signal.fftconvolve(chunk_t, target_rir[:,m], mode='full')[:segment_samples] for m in range(5)], axis=1)
        i_comp = np.stack([signal.fftconvolve(chunk_i, interf_rir[:,m], mode='full')[:segment_samples] for m in range(5)], axis=1)
        mic_mix = t_comp + i_comp
        
        # DOA Schatting (Geometrisch)
        est_doas = music_wideband_geo(mic_mix, fs, scenario)
        target_candidates = [d for d in est_doas if 90 <= d <= 180]
        doa_estimate = target_candidates[np.argmin(np.abs(np.array(target_candidates) - 90.0))] if target_candidates else 90.0
        print(f"MUSIC Geschatte Target DOA: {doa_estimate:.1f}°")
        
        # LUT genereren (Geometrisch!)
        LUT, lut_angles = generate_steering_lut_geo(scenario, 1024, fs)
        
        vad_dry = np.abs(chunk_t) > (np.std(chunk_t) * 0.05)
        
        # GSC Filtering
        out_mix, out_tar, out_int, W_adapt_state = gsc_fd_components(
            t_comp, i_comp, fs, doa_estimate, LUT, lut_angles, vad_dry, W_adapt_prev=W_adapt_state, mu=0.1
        )
        
        # SIR Berekening
        ground_truth = np.ones(len(out_tar), dtype=int)
        sir_in = compute_sir(mic_mix[:, 0], t_comp[:, 0], i_comp[:, 0], ground_truth)
        sir_out = compute_sir(out_mix, out_tar, out_int, ground_truth)
        
        sir_in_history.append(sir_in)
        sir_out_history.append(sir_out)
        print(f"SIR In: {sir_in:.2f} dB | SIR Out: {sir_out:.2f} dB (Winst: {sir_out - sir_in:.2f} dB)")
        
        full_mix_out.append(out_mix)
        full_mic_in.append(mic_mix[:, 0]) 

    # Plot genereren
    plt.figure(figsize=(10, 5))
    t_axis = np.arange(1, 6) * 10
    plt.plot(t_axis, sir_in_history, 'o--', color='gray', label='Input SIR (Mic 1)')
    plt.plot(t_axis, sir_out_history, 's-', color='blue', label='Output SIR (FD-GSC - Geometrie)')
    plt.title(f'SIR Verbetering over Tijd - {reverb_label}')
    plt.xlabel('Tijd (s)')
    plt.ylabel('SIR (dB)')
    plt.legend()
    plt.grid()
    plt.show()

    # Exporteer gecombineerde audio
    final_output = np.concatenate(full_mix_out)
    final_mic_in = np.concatenate(full_mic_in)
    sf.write(os.path.join(parent_dir, f"01_Origineel_Mic1_{'reverb' if use_reverb else 'anechoic'}.wav"), final_mic_in / np.max(np.abs(final_mic_in)), fs)
    sf.write(os.path.join(parent_dir, f"02_FD-GSC_Output_{'reverb' if use_reverb else 'anechoic'}.wav"), final_output / np.max(np.abs(final_output)), fs)