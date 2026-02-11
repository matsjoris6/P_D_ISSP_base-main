import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
import os
from package import load_rirs, select_latest_rir

def create_micsigs(acoustic_scenario, speech_filenames, noise_filenames, duration):
    """
    Genereert microfoonsignalen (Verbeterde versie zonder .numMics fout)
    """
    # 1. Haal de sampling frequentie (fs) uit het scenario
    fs_rir = acoustic_scenario.fs
    
    # 2. BEPAAL AANTAL MICS DOOR TE TELLEN (Hier ging het mis)
    # We kijken naar de eerste audiobron (index 0) en tellen hoeveel RIRs die heeft.
    num_mics = len(acoustic_scenario.RIRs_audio[0])
    
    # Bepaal het aantal samples
    num_samples = int(duration * fs_rir)
    
    # Maak lege arrays
    mic_signals = np.zeros((num_samples, num_mics))
    speech_component = np.zeros((num_samples, num_mics))
    noise_component = np.zeros((num_samples, num_mics))

    print(f"Start generatie: {num_mics} microfoons gedetecteerd.")

    # --- Verwerk SPRAAK ---
    for src_idx, filename in enumerate(speech_filenames):
        print(f" - Verwerken spraak: {filename}")
        audio, fs_src = sf.read(filename)
        
        # Resample
        if fs_src != fs_rir:
            new_len = int(len(audio) * fs_rir / fs_src)
            audio = signal.resample(audio, new_len)
        
        # Convolutie
        for mic_idx in range(num_mics):
            rir = acoustic_scenario.RIRs_audio[src_idx][mic_idx]
            filtered = signal.fftconvolve(audio, rir, mode='full')
            
            # Optellen (lengte bewaken)
            L = min(len(filtered), num_samples)
            speech_component[:L, mic_idx] += filtered[:L]

    # --- Verwerk RUIS ---
    for src_idx, filename in enumerate(noise_filenames):
        if not filename: continue
        print(f" - Verwerken ruis: {filename}")
        audio, fs_src = sf.read(filename)
        
        if fs_src != fs_rir:
            new_len = int(len(audio) * fs_rir / fs_src)
            audio = signal.resample(audio, new_len)
            
        for mic_idx in range(num_mics):
            # Let op: RIRs_noise gebruiken
            rir = acoustic_scenario.RIRs_noise[src_idx][mic_idx]
            filtered = signal.fftconvolve(audio, rir, mode='full')
            
            L = min(len(filtered), num_samples)
            noise_component[:L, mic_idx] += filtered[:L]

    # Totaal
    mic_signals = speech_component + noise_component

    # Plotten en Opslaan
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(mic_signals[:, 0])
    plt.title("Microfoon 1")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(mic_signals[:, 1])
    plt.title("Microfoon 2")
    plt.tight_layout()
    plt.savefig("part2_mic_signals.png")
    
    return mic_signals, speech_component, noise_component
# ==========================================
# PLAK DIT ONDERAAN JE BESTAND (BUITEN DE FUNCTIE)
# ==========================================

if __name__ == "__main__":
    print("--- Script is gestart ---") 
    
    # 1. Paden instellen
    # Zorg dat de map 'sound_files' bestaat en de wav-bestanden erin zitten!
    speech_files = [os.path.join("sound_files", "speech1.wav")]
    noise_files = [os.path.join("sound_files", "White_noise1.wav")]
    
    # 2. RIR laden
    print("RIRs laden...")
    try:
        path_to_rirs = "rirs"
        rir_file = select_latest_rir(path_to_rirs)
        acousticScenario = load_rirs(rir_file)
        print(f"Scenario geladen: {rir_file}")
    except Exception as e:
        print(f"FOUT bij laden RIR: {e}")
        print("Heb je wel eerst stap 1 gedaan (GUI runnen)?")
        exit()

    # 3. De functie aanroepen
    print("Signalen genereren...")
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