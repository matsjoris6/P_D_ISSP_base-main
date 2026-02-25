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
from package.utils import das_bf

def gsc_td(acoustic_scenario, speech_filenames, noise_filenames, duration):


    DASout, speechDAS, noiseDAS, SNRoutDAS, aligned_mics = das_bf(
        acoustic_scenario, speech_filenames, noise_filenames, duration
    )
    
   
    L = 1024
    mu = 0.1
    delta = L // 2
    num_samples, M = aligned_mics.shape
    

    # Creëer M-1 ruis-referenties (U) door kanaal 2,3,4,etc af te trekken van kanaal 1
    U = np.zeros((num_samples, M - 1))
    for m in range(1, M):
        U[:, m-1] = aligned_mics[:, 0] - aligned_mics[:, m]
        

    #Causaal maken
   
  
    d = np.concatenate((np.zeros(delta), DASout[:-delta]))
    

    #Het NLMS Adaptieve Filter
  
    num_noise_channels = M - 1
    w = np.zeros(num_noise_channels * L) # De filtergewichten
    GSCout = np.zeros(num_samples)
    
 
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

#SNR Schatten (Aanname van stationaire ruis)

    # Gebruik de pure vertraagde spraak om de VAD te bepalen
    vad = np.abs(speechDAS) > np.std(speechDAS) * 1e-3
    
    # Segmenteer de GSC output in 'ruis' en 'spraak+ruis'
    GSCout_noise_only = GSCout[~vad]
    GSCout_speech_noise = GSCout[vad]
    
    # Bereken vermogens
    Pn = np.var(GSCout_noise_only) if len(GSCout_noise_only) > 0 else 0
    Psn = np.var(GSCout_speech_noise) if len(GSCout_speech_noise) > 0 else 0
    Ps = Psn - Pn # Spraakvermogen is totaal minus ruis
    
   
    SNRoutGSC = 10 * np.log10(Ps / Pn)
    print(f"GSC Output SNR: {SNRoutGSC:.2f} dB (T.o.v. DAS SNR: {SNRoutDAS:.2f} dB)")


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

    # Sla het uiteindelijke GSC signaal op
    sf.write("GSC_output.wav", plot_GSCout, 44100)
    
    # Sla de eerste kolom van de Blocking Matrix (U) op om te controleren op 'speech leakage'
    sf.write("BlockingMatrix_NoiseRef.wav", U[:, 0], 44100)
    
    return GSCout, SNRoutGSC, U


rirs_folder = os.path.join("rirs")
files = [os.path.join(rirs_folder, f) for f in os.listdir(rirs_folder) if f.endswith('.pkl')]
latest_rir = max(files, key=os.path.getmtime)
acoustic_scenario = load_rirs(latest_rir)

speech_files = [os.path.join("sound_files", "speech1.wav")]
noise_files = [os.path.join("sound_files", "speech2.wav")]


duration = 5.0


GSCout, SNRoutGSC, U = gsc_td(acoustic_scenario, speech_files, noise_files, duration)
    

out_name_gsc = f"GSC_Output_{noise_file.split('.')[0]}.wav"
sf.write(out_name_gsc, GSCout, acoustic_scenario.fs)

    

out_name_bm = f"BlockingMatrix_Ref_{noise_file.split('.')[0]}.wav"
sf.write(out_name_bm, U[:, 0], acoustic_scenario.fs)

