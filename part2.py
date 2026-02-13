import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
import os
from package import load_rirs, select_latest_rir, check_plot_tdoas

def create_micsigs(acoustic_scenario, speech_filenames, noise_filenames, duration):
    fs_rir = acoustic_scenario.fs
    num_mics = len(acoustic_scenario.RIRs_audio[0])
    num_samples = int(duration * fs_rir)
    
    speech_component = np.zeros((num_samples, num_mics))
    noise_component = np.zeros((num_samples, num_mics))

    for src_idx, filename in enumerate(speech_filenames):
        if not os.path.exists(filename):
            print(f"WAARSCHUWING: Bestand niet gevonden: {filename}")
            continue
        audio, fs_src = sf.read(filename)
        if fs_src != fs_rir:
            new_len = int(len(audio) * fs_rir / fs_src)
            audio = signal.resample(audio, new_len)
        
        for mic_idx in range(num_mics):
            rir = acoustic_scenario.RIRs_audio[:, mic_idx, src_idx]
            filtered = signal.fftconvolve(audio, rir, mode='full')
            
            L = min(len(filtered), num_samples)
            speech_component[:L, mic_idx] += filtered[:L]

    if noise_filenames and len(noise_filenames) > 0 and acoustic_scenario.RIRs_noise is not None:
        for src_idx, filename in enumerate(noise_filenames):
            if not filename or not os.path.exists(filename): continue
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
        print("INFO: Ruis-verwerking overgeslagen.")

    mic = speech_component + noise_component
    
    plt.figure(figsize=(10, 4))
    plt.plot(mic[:, 0], label="Microfoon 1", alpha=0.6)
    plt.plot(mic[:, 1], label="Microfoon 2", alpha=0.6)
    plt.title("Eerste twee microfoonsignalen")
    plt.legend()
    plt.savefig("part2_mic_signals.png") 
    
    return mic, speech_component, noise_component

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
    
    print(f"Ground Truth TDOA: {ground_truth_tdoa} samples")
    print(f"Estimated TDOA:    {estimated_tdoa} samples")
    return estimated_tdoa, ground_truth_tdoa

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

    tau = -estimated_tdoa / fs 
    cos_theta = (tau * c) / d
    cos_theta = np.clip(cos_theta, -1.0, 1.0) 
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    
    print(f"Geschatte DOA:  {theta_deg:.2f} graden")
    return [theta_deg]

# --- MAIN EXECUTION ---
if __name__== "__main__":
    DOAestAll = [] # Initialiseer leeg voor het geval dat
    
    try:
        path_to_rirs = "rirs"
        rir_file = select_latest_rir(path_to_rirs)
        acousticScenario = load_rirs(rir_file)
        print(f"Scenario geladen: {rir_file}")

        speech_files = [os.path.join("sound_files (1)", "speech1.wav")]
        noise_files = [os.path.join("sound_files (1)", "Babble_noise1.wav")]

        # Stap 1: Genereer signalen en sla op
        mics, speech, noise = create_micsigs(
            acoustic_scenario=acousticScenario,
            speech_filenames=speech_files,
            noise_filenames=noise_files,
            duration=3.0
        )
        
        fs = acousticScenario.fs
        print("Audio exporteren...")
        sf.write('output_clean_speech.wav', speech[:, 0], fs)
        sf.write('output_mic_with_reverb.wav', mics[:, 0], fs)
        print("Succes! Bestanden opgeslagen.")

        # Stap 2: TDOA
        white_noise = [os.path.join("sound_files (1)", "White_noise1.wav")]
        est, gt = TDOA_corr(acousticScenario, white_noise, segment_duration=0.2)
        print(f"TDOA Error: {est - gt} samples")

        # Stap 3: DOA
        DOAestAll = DOA_corr(acousticScenario, speech_files, segment_duration=0.5)

        # Stap 4: Plotten
        if DOAestAll:
            doa_in_rad = [theta * np.pi / 180 for theta in DOAestAll]
            fig = check_plot_tdoas(
                doaEstTarget=doa_in_rad, 
                doaEstAll=doa_in_rad, 
                asc=acousticScenario
            )
            plt.show()

    except Exception as e:
        print(f"Er is een fout opgetreden: {e}")
        import traceback
        traceback.print_exc()