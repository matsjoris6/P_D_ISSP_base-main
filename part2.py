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
            if src_idx == 0 and mic_idx == 0:
                plt.figure(figsize=(10, 5))
                plt.subplot(2, 1, 1)
                plt.plot(audio[:fs_rir], color='blue')
                plt.title("VOOR convolutie (Originele droge spraak - 1 sec)")
                plt.subplot(2, 1, 2)
                plt.plot(filtered[:fs_rir], color='red')
                plt.title("NA convolutie (Met kamerakoestiek/galm - 1 sec)")
                plt.tight_layout()
                plt.show() # Dit opent een venster om te kijken
            
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

  
    plt.figure(figsize=(10, 4))
    plt.plot(mic[:, 0], label="Microfoon 1", alpha=0.6)
    plt.plot(mic[:, 1], label="Microfoon 2", alpha=0.6)
    plt.title("Eerste twee microfoonsignalen")
    plt.legend()
    plt.savefig("part2_mic_signals.png") 
    
    return mic, speech_component, noise_component

if __name__== "__main__":
    
    
   
    speech_files = [os.path.join("sound_files", "speech1.wav")]
    noise_files = [os.path.join("sound_files", "Babble_noise1.wav")]
    
   
    try:
        path_to_rirs = "rirs"
        rir_file = select_latest_rir(path_to_rirs)
        acousticScenario = load_rirs(rir_file)
        print(f"Scenario geladen: {rir_file}")
    except Exception as e:
        print(f"FOUT bij laden RIR: {e}")
      
        exit()

 
    try:
        mics, speech, noise = create_micsigs(
            acoustic_scenario=acousticScenario,
            speech_filenames=speech_files,
            noise_filenames=noise_files,
            duration=3.0
        )
        print("Succes! Check nu of het bestand 'part2_mic_signals.png' is aangemaakt.")
    except Exception as e:
        print(f"FOUT tijdens uitvoeren functie: {e}")
        import traceback
        traceback.print_exc()
    
# Sla de resultaten op als audiobestanden

    fs = acousticScenario.fs
    
    print("Audio exporteren...")
    sf.write('output_clean_speech.wav', speech[:, 0], fs)
    sf.write('output_mic_with_reverb.wav', mics[:, 0], fs)

    print("Klaar! De bestanden 'output_clean_speech.wav' en 'output_mic_with_reverb.wav' staan in je map.")
    plt.figure(figsize=(10, 4))
    plt.plot(mics[50000:51000, 0], label="Mic 1")
    plt.plot(mics[50000:51000, 1], label="Mic 2", alpha=0.7)
    plt.title("Microfoonsignalen bij 8000 Hz (Zoom)")
    plt.legend()
    plt.savefig("problem_8000hz.png")

def TDOA_corr(acoustic_scenario, noise_filenames, segment_duration):
    fs = acoustic_scenario.fs
    

    peak_mic1 = np.argmax(np.abs(acoustic_scenario.RIRs_audio[:, 0, 0]))
    peak_mic2 = np.argmax(np.abs(acoustic_scenario.RIRs_audio[:, 1, 0]))
    ground_truth_tdoa = peak_mic2 - peak_mic1
    
  
    mics, _, _ = create_micsigs(acoustic_scenario, noise_filenames, [], duration=2.0)
    

    N = int(segment_duration * fs)
    sig1 = mics[:N, 0]
    sig2 = mics[:N, 1]

    corr = signal.correlate(sig2, sig1, mode='full')

    lags = np.arange(-N + 1, N)
    

    estimated_tdoa = lags[np.argmax(corr)]
    
  
    plt.figure(figsize=(10, 4))
    plt.plot(lags, corr, label='Cross-correlation function')

    plt.stem([ground_truth_tdoa], [np.max(corr)], linefmt='r--', markerfmt='ro', label='Ground Truth')
    plt.title("TDOA Schatting via Cross-correlation")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()
    

    print(f"Ground Truth TDOA: {ground_truth_tdoa} samples")
    print(f"Estimated TDOA:    {estimated_tdoa} samples")
    print(f"Error:             {estimated_tdoa - ground_truth_tdoa} samples")
    
    return estimated_tdoa, ground_truth_tdoa


try:
    path_to_rirs = "rirs"
    rir_file = select_latest_rir(path_to_rirs)
    acousticScenario = load_rirs(rir_file)
    print(f"Scenario geladen: {rir_file}")
    



    noise_files = [os.path.join("sound_files", "White_noise1.wav")]
    
 
    est, gt = TDOA_corr(acousticScenario, noise_files, segment_duration=0.2)
    

    print(f"\n De error is {est - gt}.")

except Exception as e:
    print(f"Er is iets misgegaan bij het laden of uitvoeren: {e}")


def DOA_corr(acoustic_scenario, noise_filenames, segment_duration):
    fs = acoustic_scenario.fs
    c = acoustic_scenario.c
    pos1 = acoustic_scenario.micPos[0]
    pos2 = acoustic_scenario.micPos[1]
    
    d = np.sqrt(np.sum((pos1 - pos2)**2))    
    
  
    mics, _, _ = create_micsigs(acoustic_scenario, noise_filenames, [], duration=2.0)
    

    N = int(segment_duration * fs)
    sig1 = mics[:N, 0]
    sig2 = mics[:N, 1]

    corr = signal.correlate(sig2, sig1, mode='full')

    lags = np.arange(-N + 1, N)
    

    estimated_tdoa = lags[np.argmax(corr)]

    tau = -estimated_tdoa / fs #samples naar sec

    cos_theta = (tau * c) / d
    #cos_theta = np.clip(cos_theta, -1.0, 1.0) #niet buiten domein door afrondingsfouten
    
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    

    DOAestAll = [theta_deg]
    
    print(f"Geschatte TDOA: {estimated_tdoa} samples")
    print(f"Geschatte DOA:  {theta_deg:.2f} graden")
    git remote set-url origin https://<ZIJN_TOKEN>@github.com/matsjoris6/P_D_ISSP_base-main.git
    return DOAestAll



try:
    path_to_rirs = "rirs"
    rir_file = select_latest_rir(path_to_rirs)
    asc_44k = load_rirs(rir_file)



    speech_files = [os.path.join("sound_files", "speech1.wav")]


    DOAestAll = DOA_corr(asc_44k, speech_files, segment_duration=0.5)





except Exception as e:
    print(f"Fout bij uitvoeren: {e}")



from package import check_plot_tdoas 


doa_in_rad = [theta * np.pi / 180 for theta in DOAestAll]


fig = check_plot_tdoas(
    doaEstTarget=doa_in_rad, 
    doaEstAll=doa_in_rad, 
    asc=acousticScenario
)

# 4. Toon de plot
plt.show()
    