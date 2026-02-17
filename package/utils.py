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
    else:
        print("INFO: Ruis-verwerking overgeslagen (geen bestanden of geen ruis-RIRs gevonden).")

    mic = speech_component + noise_component

  
    #plt.figure(figsize=(10, 4))
    #plt.plot(mic[:, 0], label="Microfoon 1", alpha=0.6)
    #plt.plot(mic[:, 1], label="Microfoon 2", alpha=0.6)
    #plt.title("Eerste twee microfoonsignalen")
    #plt.legend()
    #plt.savefig("part2_mic_signals.png") 
    
    return mic, speech_component, noise_component

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

