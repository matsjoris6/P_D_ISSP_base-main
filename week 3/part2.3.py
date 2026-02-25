import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from package import load_rirs
from package.utils import create_micsigs
from package.utils import music_wideband

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


rirs_folder = os.path.join("rirs")
files = [os.path.join(rirs_folder, f) for f in os.listdir(rirs_folder) if f.endswith('.pkl')]
latest_rir = max(files, key=os.path.getmtime)
acoustic_scenario = load_rirs(latest_rir)

speech_files = [os.path.join("sound_files", "speech1.wav")]
noise_files = [os.path.join("sound_files", "speech2.wav")] 

duration = 5.0

DASout, speechDAS, noiseDAS, SNRoutDAS,_= das_bf(acoustic_scenario, speech_files, noise_files, duration)

