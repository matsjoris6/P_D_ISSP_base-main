import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
import os
from package import load_rirs, select_latest_rir

def create_micsigs(acoustic_scenario, speech_filenames, noise_filenames, duration):
    fs_rir = acoustic_scenario.fs
    num_mics = len(acoustic_scenario.RIRs_audio[0])
    num_samples = int(duration * fs_rir)
    
    speech_component = np.zeros((num_samples, num_mics))
    noise_component = np.zeros((num_samples, num_mics))

 

    for src_idx, filename in enumerate(speech_filenames):
        audio, fs_src = sf.read(filename)
        if fs_src != fs_rir:
            new_len = int(len(audio) * fs_rir / fs_src)
            audio = signal.resample(audio, new_len)
        
        for mic_idx in range(num_mics):
            rir = acoustic_scenario.RIRs_audio[:, mic_idx, src_idx]
            filtered = signal.fftconvolve(audio, rir, mode='full')
            
            #  EXTRA CHECK PLOT 
            #if src_idx == 0 and mic_idx == 0:
                #plt.figure(figsize=(10, 5))
                #plt.subplot(2, 1, 1)
                #plt.plot(audio[:fs_rir], color='blue')
                #plt.title("VOOR convolutie (Originele droge spraak - 1 sec)")
                #plt.subplot(2, 1, 2)
                #plt.plot(filtered[:fs_rir], color='red')
                #plt.title("NA convolutie (Met kamerakoestiek/galm - 1 sec)")
                #plt.tight_layout()
                #plt.show() 
            
            L = min(len(filtered), num_samples)
            speech_component[:L, mic_idx] += filtered[:L]

# RUIS 

    if noise_filenames and len(noise_filenames) > 0 and acoustic_scenario.RIRs_noise is not None:
        for src_idx, filename in enumerate(noise_filenames):
            if not filename: continue
            print(f" - Verwerken ruis: {filename}")
            audio, fs_src = sf.read(filename)
            
            if fs_src != fs_rir:
                audio = signal.resample(audio, int(len(audio) * fs_rir / fs_src))
                
            for mic_idx in range(num_mics):
  
                rir = acoustic_scenario.RIRs_noise[:, mic_idx, src_idx]
                filtered = signal.fftconvolve(audio, rir, mode='full')
                
                L = min(len(filtered), num_samples)
                noise_component[:L, mic_idx] += filtered[:L]
    #else:
        #print(" Ruis-verwerking overgeslagen (geen bestanden of geen ruis-RIRs gevonden).")

    mic = speech_component + noise_component

  
    #plt.figure(figsize=(10, 4))
    #plt.plot(mic[:, 0], label="Microfoon 1", alpha=0.6)
    #plt.plot(mic[:, 1], label="Microfoon 2", alpha=0.6)
    #plt.title("Eerste twee microfoonsignalen")
    #plt.legend()
    #plt.savefig("part2_mic_signals.png") 
    
    return mic, speech_component, noise_component

def create_micsigs_modified(acoustic_scenario, speech_filenames, noise_filenames, duration):
    fs_rir = acoustic_scenario.fs
    if fs_rir != 44100:
        raise ValueError(f"Fout: Sampling frequency moet 44.1 kHz zijn, maar is {fs_rir} Hz.")
    num_mics = len(acoustic_scenario.RIRs_audio[0])
    num_samples = int(duration * fs_rir)
    
    speech_component = np.zeros((num_samples, num_mics))
    noise_component = np.zeros((num_samples, num_mics))

 

    for src_idx, filename in enumerate(speech_filenames):
        audio, fs_src = sf.read(filename)
        if fs_src != fs_rir:
            new_len = int(len(audio) * fs_rir / fs_src)
            audio = signal.resample(audio, new_len)
        
        for mic_idx in range(num_mics):
            rir = acoustic_scenario.RIRs_audio[:, mic_idx, src_idx]
            filtered = signal.fftconvolve(audio, rir, mode='full')
            
            #  EXTRA CHECK PLOT 
            #if src_idx == 0 and mic_idx == 0:
                #plt.figure(figsize=(10, 5))
                #plt.subplot(2, 1, 1)
                #plt.plot(audio[:fs_rir], color='blue')
                #plt.title("VOOR convolutie (Originele droge spraak - 1 sec)")
                #plt.subplot(2, 1, 2)
                #plt.plot(filtered[:fs_rir], color='red')
                #plt.title("NA convolutie (Met kamerakoestiek/galm - 1 sec)")
                #plt.tight_layout()
                #plt.show() 
            
            L = min(len(filtered), num_samples)
            speech_component[:L, mic_idx] += filtered[:L]

# RUIS 

    if noise_filenames and len(noise_filenames) > 0 and acoustic_scenario.RIRs_noise is not None:
        for src_idx, filename in enumerate(noise_filenames):
            if not filename: continue
          
            audio, fs_src = sf.read(filename)
            
            if fs_src != fs_rir:
                audio = signal.resample(audio, int(len(audio) * fs_rir / fs_src))
                
            for mic_idx in range(num_mics):
  
                rir = acoustic_scenario.RIRs_noise[:, mic_idx, src_idx]
                filtered = signal.fftconvolve(audio, rir, mode='full')
                
                L = min(len(filtered), num_samples)
                noise_component[:L, mic_idx] += filtered[:L]
    else:
        print("INFO: Ruis-verwerking overgeslagen (geen bestanden of geen ruis-RIRs gevonden).")

    speech_mic1 = speech_component[:, 0]
    vad = np.abs(speech_mic1) > np.std(speech_mic1) * 1e-3
    active_speech = speech_mic1[vad]
    Ps = np.var(active_speech) if len(active_speech) > 0 else 0
    Pn_target = 0.1 * Ps
    white_noise = np.random.normal(0, np.sqrt(Pn_target), (num_samples, num_mics))
       
    noise_component += white_noise
    Pn_actual = np.var(noise_component[:, 0])
    
    if Pn_actual > 0:
        snr_mic1 = 10 * np.log10(Ps / Pn_actual)
        print(f"SNR in de eerste microfoon: {snr_mic1:.2f} dB")
    else:
        print("Kan SNR niet berekenen (Ruis is 0).")

    mic = speech_component + noise_component

  
    #plt.figure(figsize=(10, 4))
    #plt.plot(mic[:, 0], label="Microfoon 1", alpha=0.6)
    #plt.plot(mic[:, 1], label="Microfoon 2", alpha=0.6)
    #plt.title("Eerste twee microfoonsignalen")
    #plt.legend()
    #plt.savefig("part2_mic_signals.png") 
    
    return mic, speech_component, noise_component

def calculate_ground_truth_doas(acoustic_scenario):
    if acoustic_scenario.audioPos is None or len(acoustic_scenario.audioPos) == 0:
        return []
    
    mics = acoustic_scenario.micPos       
    array_center = np.mean(mics, axis=0)
    
    doas = []
    for source in acoustic_scenario.audioPos:
        dx = source[0] - array_center[0]
        dy = source[1] - array_center[1] 
        
        # GECORRIGEERD: Gebruik arctan2 geprojecteerd op het specifieke assenstelsel
        # -dy zorgt dat negatieve y (omhoog) naar 0 graden wijst.
        # -dx zorgt dat negatieve x (links) naar 90 graden wijst.
        hoek_rad = np.arctan2(-dx, -dy)
        hoek_deg = np.degrees(hoek_rad) % 360
        doas.append(hoek_deg)
        
    return np.array(doas)

def music_narrowband(micsigs, fs, acoustic_scenario):
    # 1. STFT berekenen 
    L = 1024
    overlap = L // 2
    stft_list = []
    for m in range(micsigs.shape[1]):
        _, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window='hann', nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0)
    
    M, nF, nT = stft_data.shape
    c, d = 343.0, 0.05
    
    # 2. Aantal bronnen (Q)
    Q = acoustic_scenario.audioPos.shape[0] if acoustic_scenario.audioPos is not None else 0
    
    # 3. Frequentiebin selectie 
    power_spectrum = np.mean(np.abs(stft_data)**2, axis=(0, 2))
    max_bin_idx = np.argmax(power_spectrum[1:]) + 1 
    freqs = np.fft.rfftfreq(2*(nF-1), 1/fs)
    omega = 2 * np.pi * freqs[max_bin_idx]

    # 4. Covariantiematrix Ryy 
    Y = stft_data[:, max_bin_idx, :]
    Ryy = (Y @ Y.conj().T) / nT
    
    # 5. Eigen-decompositie 
    # np.linalg.eigh sorteert van klein naar groot
    eigvals, eigvecs = np.linalg.eigh(Ryy)
    
    # Noise Subspace (En) bevat de M - Q kleinste eigenvectoren
    En = eigvecs[:, :M-Q] 
    
    # 6. Pseudospectrum berekenen 
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    taus = (np.arange(M).reshape(-1, 1) * d * (-np.cos(rads))) / c 
    A = np.exp(-1j * omega * taus)
    
    denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
    pseudospectrum = 1.0 / denom
    spectrum_db = 10 * np.log10(pseudospectrum / np.max(pseudospectrum))
    
    # 7. Piekdetectie 
    peaks_indices, _ = signal.find_peaks(spectrum_db)
    sorted_peak_indices = peaks_indices[np.argsort(spectrum_db[peaks_indices])][-Q:]
    estimated_doas = np.sort(angles[sorted_peak_indices])
    
    # Retourneer ook alle eigenwaarden (eigvals)
    return angles, spectrum_db, estimated_doas, freqs[max_bin_idx], np.real(eigvals)

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
    c = 343.0
    
    # 2. Bepaal aantal bronnen Q
    Q = acoustic_scenario.audioPos.shape[0] if acoustic_scenario.audioPos is not None else 1
    
    # 3. Selecteer de frequentiebin met het hoogste vermogen (Boven 0Hz!)
    freqs = np.fft.rfftfreq(2*(nF-1), 1/fs)
    power_spectrum = np.mean(np.abs(stft_data)**2, axis=(0, 2))
    
    # Forceer MUSIC om een frequentie met wél genoeg spatiële resolutie te gebruiken 
    valid_bins = freqs > 0
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
    
    # GECORRIGEERD: Gebruik de échte (x,y) coördinaten van de microfoons 
    # ten opzichte van het array-centrum om de steering vector op te bouwen.
    mics_centered = acoustic_scenario.micPos - np.mean(acoustic_scenario.micPos, axis=0)
    px = mics_centered[:, 0].reshape(-1, 1) # X-coördinaten
    py = mics_centered[:, 1].reshape(-1, 1) # Y-coördinaten
    
    # Dit berekent de vertraging (tau) perfect voor élke geometrie
    taus = (px * np.sin(rads) + py * np.cos(rads)) / c 
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
        # Robuuste fallback
        sorted_all = np.argsort(spectrum_db)[::-1]
        est_idx = []
        for idx in sorted_all:
            if len(est_idx) == Q: break
            if all(abs(idx - e) > 20 for e in est_idx):
                est_idx.append(idx)
        estimated_doas = np.sort(angles[est_idx])
    
    return angles, spectrum_db, estimated_doas, f_val, np.real(eigvals)

def music_wideband(micsigs, fs, acoustic_scenario, Q_override=None):
    # 1. Efficiënte STFT berekening (Vectorized over time)
    L = 1024    # tunen parameter: lengte van FFT en venster
    overlap = L // 2       # tunen parameter: aantal overlappende samples
    
    # Tuning: Square-root Hanning window in plaats van standaard 'hann'
    window = np.sqrt(signal.windows.hann(L, sym=False))
    
    stft_list = []
    for m in range(micsigs.shape[1]):
        freqs, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window=window, nperseg=L, noverlap=overlap) 
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0) # Shape: (M, nF, nT)
    
    M, nF, nT = stft_data.shape
    c = 343.0
    
    # =========================================================================
    # TUNING: Aantal bronnen (Q)
    # =========================================================================
    if Q_override is not None:
        Q = Q_override
    else:
        num_audio = acoustic_scenario.audioPos.shape[0] if acoustic_scenario.audioPos is not None else 0
        num_noise = acoustic_scenario.noisePos.shape[0] if acoustic_scenario.noisePos is not None else 0
        Q = num_audio + num_noise
    print('Q =', Q)
    
    step_size = 0.1 # TUNING: stapgrootte als aparte variabele voor de fallback berekening verderop
    angles = np.arange(0, 180.5, step_size)    
    rads = np.radians(angles)
    
    # Pre-compute mic coördinaten
    mics_centered = acoustic_scenario.micPos - np.mean(acoustic_scenario.micPos, axis=0)
    px = mics_centered[:, 0].reshape(-1, 1)
    py = mics_centered[:, 1].reshape(-1, 1)
    
    # =========================================================================
    # TUNING: Frequentiebin-selectie voor MUSIC
    # =========================================================================
    f_min = 300   # Minimale frequentie in Hz (Tuning parameter)
    f_max = 4000  # Maximale frequentie in Hz (Tuning parameter)
    
    k_min = int(f_min / (fs / L))
    k_max = int(f_max / (fs / L))
    
    valid_indices = range(max(1, k_min), min(L // 2, k_max))
    # =========================================================================
    
    pseudospectra = [] # FIX: Deze initialisatie ontbrak in jouw originele code!
    
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
        p_theta = 1.0 / (denom + 1e-12) # FIX: + 1e-12 om eventuele deling door nul te voorkomen
        pseudospectra.append(p_theta)
        
    pseudospectra = np.array(pseudospectra) # Shape: (aantal_bins, len(angles))
    
    # 2. Geometrisch Gemiddelde (Numeriek stabiele methode) + veel minder gevoelig voor pieken die samensmelten
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
        # =========================================================================
        # TUNING: Peak Finding (find_peaks) Fallback logica
        # =========================================================================
        # In plaats van het getal '20', baseren we de minimale afstand nu op graden.
        # Bij een step_size van 0.1 was 20 indexen slechts 2 graden afstand. 
        # Door min_dist_deg aan te passen bepaal je de werkelijke resolutie.
        
        min_dist_deg = 10.0 # TUNING PARAMETER: Minimale afstand tussen twee bronnen in graden
        idx_threshold = int(min_dist_deg / step_size)
        
        sorted_all = np.argsort(spectrum_geom_db)[::-1]
        est_idx = []
        for idx in sorted_all:
            if len(est_idx) == Q: break
            if all(abs(idx - e) > idx_threshold for e in est_idx):
                est_idx.append(idx)
        estimated_doas = np.sort(angles[est_idx])

    # Geef de individuele genormaliseerde spectra ook mee (in dB) voor de plot
    ps_ind_db = 10 * np.log10(pseudospectra / np.max(pseudospectra, axis=1, keepdims=True))

    return angles, ps_ind_db, spectrum_geom_db, estimated_doas

def das_bf(acoustic_scenario, speech_filenames, noise_filenames, duration):
    fs = acoustic_scenario.fs
    c = 343.0
    
 
    mic, speech_comp, noise_comp = create_micsigs(
        acoustic_scenario, speech_filenames, noise_filenames, duration
    )
    

    num_audio = acoustic_scenario.audioPos.shape[0] if acoustic_scenario.audioPos is not None else 0
    num_noise = acoustic_scenario.noisePos.shape[0] if acoustic_scenario.noisePos is not None else 0
    Q = num_audio + num_noise

    

    angles, ps_ind_db, spec_geo, est_doas = music_wideband(mic, fs, acoustic_scenario)
    

    target_idx = np.argmin(np.abs(est_doas - 90.0))
    target_doa = est_doas[target_idx]
    print(f"Target DOA: {target_doa:.1f}°")
    

    target_rad = np.radians(target_doa)
    mics_centered = acoustic_scenario.micPos - np.mean(acoustic_scenario.micPos, axis=0)
    px = mics_centered[:, 0]
    py = mics_centered[:, 1]
    
 
    taus = (px * np.sin(target_rad) + py * np.cos(target_rad)) / c 
    
 
    def shift_signals(component_signals):
        M = component_signals.shape[1]
        N = component_signals.shape[0]
        aligned_signals = np.zeros_like(component_signals)
        
        freqs = np.fft.rfftfreq(N, 1/fs)
        
        for m in range(M):
       
            sig_fft = np.fft.rfft(component_signals[:, m])
            # + om delay ongedaan te maken
            shifted_fft = sig_fft * np.exp(1j * 2 * np.pi * freqs * taus[m])

            aligned_signals[:, m] = np.fft.irfft(shifted_fft, n=N)
            


        return aligned_signals 

    # 1. Verschuif de componenten
    aligned_speech = shift_signals(speech_comp)
    aligned_noise = shift_signals(noise_comp)
    
    # voor gsc: Alle uitgelijnde kanalen (spraak + ruis)
    aligned_mics = aligned_speech + aligned_noise
    
    # 3. optellen en schalen we  voor de DAS output
    M_val = aligned_mics.shape[1]
    speechDAS = np.sum(aligned_speech, axis=1) / M_val
    noiseDAS = np.sum(aligned_noise, axis=1) / M_val
    DASout = speechDAS + noiseDAS
    
    #snr berekenen
    vad = np.abs(speech_comp[:, 0]) > np.std(speech_comp[:, 0]) * 1e-3
    
    Ps_das = np.var(speechDAS[vad]) if np.any(vad) else 0
    Pn_das = np.var(noiseDAS)
    
    SNRoutDAS = 10 * np.log10(Ps_das / Pn_das) if Pn_das > 0 else np.nan
    print(f"DAS Output SNR: {SNRoutDAS:.2f} dB")
    

    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(mic[:, 0], color='blue', alpha=0.7)
    plt.title("Microfoon 1 (Ongeloofde mix)")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 1, 2)
    plt.plot(DASout, color='green', alpha=0.7)
    plt.title(f"Delay-and-Sum Output (Georiënteerd op {target_doa:.1f}°)")
    plt.ylabel("Amplitude")
    plt.xlabel("Samples")
    
    plt.tight_layout()
    plt.show()

    return DASout, speechDAS, noiseDAS, SNRoutDAS, aligned_mics

import numpy as np

def compute_sir(y, x1, x2, groundTruth):
    """
    Compute the signal-to-interference ratio for two sources (one target source
    and one interfering source). The script takes into account possible
    switches between which source is the target and which one is the
    interference. The script assumes there is access to each source's
    contribution in the beamformer output.

    Parameters
    ----------
    -y : [N x 1] np.ndarray[float]
        Actual beamformer output signal (`N` is the number of samples).
    -x1 : [N x 1] np.ndarray[float]
        Beamformer output attributed to source 1.
    -x2 : [N x 1] np.ndarray[float]
        Beamformer output attributed to source 2.
    -groundTruth : [N x 1] np.ndarray[int (0 or 1) or float (0. or 1.)]
        Array indicating, for each sample,
        which source is the target: 1=x1, 0=x2.

    Returns
    -------
    -sir : 
    """
    # Sanity check (check whether `y = x1 + x2` based on RMSE of residual)
    if np.sqrt(np.sum((y - x1 - x2) ** 2)) / np.sqrt(np.sum(y ** 2)) > 0.01:
        print('/!\ Something is wrong, `y` should be the sum of `x1` and `x2`.')  
        print('SIR can not be computed -- Returning NaN.')  
        sir = np.nan
    # Input check
    elif np.sum(groundTruth) + np.sum(1 - groundTruth) != len(groundTruth):
        print('/!\ `groundTruth` vector is not binary.')
        print('SIR can not be computed')  
        sir = np.nan
    else:
        sir = 10 * np.log10(
            np.var(x1 * groundTruth + x2 * (1 - groundTruth)) /\
                np.var(x2 * groundTruth + x1 * (1 - groundTruth))
        )

    return 