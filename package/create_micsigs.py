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

