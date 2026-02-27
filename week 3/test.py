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


def gsc_td(acoustic_scenario, speech_filenames, noise_filenames, duration):
    # 1. Roep DAS aan en haal de resultaten (inclusief aligned_mics!) op
    print("Start DAS Beamformer...")
    DASout, speechDAS, noiseDAS, SNRoutDAS, aligned_mics = das_bf(
        acoustic_scenario, speech_filenames, noise_filenames, duration
    )
    
    # Configuratie van de GSC
    L = 1024
    mu = 0.1
    delta = L // 2
    num_samples, M = aligned_mics.shape
    

    # 2. Griffiths-Jim Blocking Matrix

    # Creëer M-1 ruis-referenties (U) door kanaal 2,3,4,etc af te trekken van kanaal 1
    U = np.zeros((num_samples, M - 1))
    for m in range(1, M):
        U[:, m-1] = aligned_mics[:, 0] - aligned_mics[:, m]
        
 
    # 3. Delay op het DAS-signaal (Causaal maken)

    # Plak delta nullen vooraan, gooi de laatste delta eraf (zodat lengte gelijk blijft)
    d = np.concatenate((np.zeros(delta), DASout[:-delta]))
    

    # 4. Het NLMS Adaptieve Filter

    num_noise_channels = M - 1
    w = np.zeros(num_noise_channels * L) # De filtergewichten
    GSCout = np.zeros(num_samples)
    
    print(f"GSC NLMS filter is aan het trainen op {num_samples} samples... (dit kan 10-20 sec duren)")
    # Start bij n=L zodat we altijd L samples uit het verleden kunnen pakken
    for n in range(L, num_samples):
        # Pak het blokje van L samples uit het verleden voor alle noise kanalen en sla plat naar 1D
        x_n = U[n-L:n, :].flatten()
        
        # Bereken de voorspelde ruis (y) via dot-product
        y = np.dot(w, x_n)
        
        # Bereken de fout/error (Dit is je schone signaal!)
        e = d[n] - y
        GSCout[n] = e
        
        # Update filtergewichten (NLMS formule)
        norm_x = np.dot(x_n, x_n)
        w = w + (mu / (norm_x + 1e-6)) * e * x_n

    # 5. SNR Schatten (Aanname van stationaire ruis)

    # Gebruik de pure vertraagde spraak om de VAD te bepalen
    vad = np.abs(speechDAS) > np.std(speechDAS) * 1e-3
    
    # Segmenteer de GSC output in 'ruis' en 'spraak+ruis'
    GSCout_noise_only = GSCout[~vad]
    GSCout_speech_noise = GSCout[vad]
    
    # Bereken vermogens
    Pn = np.var(GSCout_noise_only) if len(GSCout_noise_only) > 0 else 0
    Psn = np.var(GSCout_speech_noise) if len(GSCout_speech_noise) > 0 else 0
    Ps = Psn - Pn # Spraakvermogen is totaal minus ruis
    
    if Pn > 0 and Ps > 0:
        SNRoutGSC = 10 * np.log10(Ps / Pn)
        print(f"GSC Output SNR: {SNRoutGSC:.2f} dB (T.o.v. DAS SNR: {SNRoutDAS:.2f} dB)")
    else:
        SNRoutGSC = np.nan
        print("Fout bij SNR berekening: Vermogen negatief of nul.")


    # 6. Plotten (met delay compensatie)
   
    # Schuif GSCout weer delta samples naar 'links' om hem weer synchroon te laten lopen met DASout
    plot_GSCout = np.zeros_like(GSCout)
    plot_GSCout[:-delta] = GSCout[delta:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(DASout, label="DAS Output (Startpunt - Spraak + Ruis)", color='blue', alpha=0.4)
    plt.plot(plot_GSCout, label="GSC Output (Na adaptief filter)", color='green', alpha=0.8)
    plt.plot(speechDAS, label="Pure Spraak (speechDAS)", color='orange', alpha=0.5, linestyle='--')
    
    plt.title("GSC vs DAS Beamforming")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return GSCout, SNRoutGSC, U


rirs_folder = os.path.join("rirs")
files = [os.path.join(rirs_folder, f) for f in os.listdir(rirs_folder) if f.endswith('.pkl')]
latest_rir = max(files, key=os.path.getmtime)
acoustic_scenario = load_rirs(latest_rir)

# De target spraak
speech_files = [os.path.join("sound_files", "speech1.wav")]

# De 3 verschillende ruisscenario's die we moeten testen
noise_scenarios = [
    "White_noise1.wav",
    "Babble_noise1.wav",
    "speech2.wav"
]

duration = 5.0


for noise_file in noise_scenarios:

    
    noise_files_list = [os.path.join("sound_files", noise_file)]
    

    GSCout, SNRoutGSC, U = gsc_td(acoustic_scenario, speech_files, noise_files_list, duration)
    

    out_name_gsc = f"GSC_Output_{noise_file.split('.')[0]}.wav"
    sf.write(out_name_gsc, GSCout, acoustic_scenario.fs)

    

    out_name_bm = f"BlockingMatrix_Ref_{noise_file.split('.')[0]}.wav"
    sf.write(out_name_bm, U[:, 0], acoustic_scenario.fs)


