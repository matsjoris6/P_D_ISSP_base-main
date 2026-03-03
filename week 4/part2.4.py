import numpy as np
import scipy.signal as signal
import scipy.linalg
import matplotlib.pyplot as plt
import soundfile as sf
import os
import sys
import importlib.util

# Paden instellen
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)  # Voeg week 4 map toe zodat we part1_4 kunnen importeren

# Importeer vereiste functies
from package import load_rirs
from package.utils import music_wideband

# Importeer de Look-Up Table generator uit je Part 1 script
# In Python: bestand "part1.4.py" kan niet direct geïmporteerd worden (dot in naam)
# Gebruik importlib om het dynamisch te laden
spec = importlib.util.spec_from_file_location("part1_4", os.path.join(current_dir, "part1.4.py"))
part1_4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(part1_4)

# ==========================================
# FUNCTION: COMPUTE SIR (Signal-to-Interference Ratio)
# ==========================================
def compute_sir_metric(target_signal, interferer_signal):
    """
    Berekent de SIR over een gegeven tijdsblok[cite: 61, 63].
    Formule: 10 * log10( var(target) / var(interferer) )
    """
    var_target = np.var(target_signal)
    var_interf = np.var(interferer_signal)
    
    if var_interf < 1e-12:
        return np.nan # Voorkom deling door 0
        
    return 10 * np.log10((var_target + 1e-12) / var_interf)

# ==========================================
# FUNCTION: BLOCK-BASED FD-GSC MET COMPONENT SEPARATIE
# ==========================================
def gsc_fd_components(target_comp, interf_comp, fs, doa_estimate, LUT, lut_angles, mu=0.05):
    """
    Voert de FD-GSC uit, maar filtert de target en interferer componenten
    simultaan door EXACT dezelfde adaptieve filtergewichten om achteraf
    de SIR correct te kunnen berekenen.
    """
    # De effectieve microfoonmix die het adaptieve filter stuurt
    mic = target_comp + interf_comp
    N_samples, M_mics = mic.shape
    
    L = 1024
    overlap = L // 2
    window = np.sqrt(signal.windows.hann(L, sym=False))
    
    # STFT op de mix, de target én de interferer
    _, _, Zxx_mic = signal.stft(mic.T, fs, window=window, nperseg=L, noverlap=overlap)
    _, _, Zxx_tar = signal.stft(target_comp.T, fs, window=window, nperseg=L, noverlap=overlap)
    _, _, Zxx_int = signal.stft(interf_comp.T, fs, window=window, nperseg=L, noverlap=overlap)
    
    nF, nT = Zxx_mic.shape[1], Zxx_mic.shape[2]
    
    # VAD gebaseerd op de pure target
    vad = np.abs(target_comp[:, 0]) > (np.std(target_comp[:, 0]) * 1e-3)
    vad_frames = np.zeros(nT)
    for n in range(nT):
        start_idx = n * (L - overlap)
        end_idx = start_idx + L
        if end_idx <= len(vad):
            vad_frames[n] = 1 if np.mean(vad[start_idx:end_idx]) > 0.05 else 0

    # Output arrays
    E_out_mic = np.zeros((nF, nT), dtype=complex)
    E_out_tar = np.zeros((nF, nT), dtype=complex)
    E_out_int = np.zeros((nF, nT), dtype=complex)
    
    W_adapt = np.zeros((nF, M_mics - 1), dtype=complex)
    lut_idx = np.argmin(np.abs(lut_angles - doa_estimate))
    
    for k in range(nF):
        h_k = LUT[k, lut_idx, :]
        norm_factor = np.vdot(h_k, h_k)
        w_FAS = h_k / norm_factor if np.abs(norm_factor) > 1e-10 else np.zeros(M_mics, dtype=complex)
            
        N_mat = scipy.linalg.null_space(h_k.reshape(1, M_mics).conj())
        B_k = N_mat.conj().T 
        
        for n in range(nT):
            # Berekeningen op de MIX (bepaalt de filter adaptatie)
            x_mic = Zxx_mic[:, k, n]
            y_FAS_mic = np.vdot(w_FAS, x_mic)
            u_mic = B_k @ x_mic
            e_mic = y_FAS_mic - np.vdot(W_adapt[k, :], u_mic)
            E_out_mic[k, n] = e_mic
            
            # Pas EXACT DITZELFDE FILTER toe op de target component
            x_tar = Zxx_tar[:, k, n]
            y_FAS_tar = np.vdot(w_FAS, x_tar)
            u_tar = B_k @ x_tar
            E_out_tar[k, n] = y_FAS_tar - np.vdot(W_adapt[k, :], u_tar)
            
            # Pas EXACT DITZELFDE FILTER toe op de interferer component
            x_int = Zxx_int[:, k, n]
            y_FAS_int = np.vdot(w_FAS, x_int)
            u_int = B_k @ x_int
            E_out_int[k, n] = y_FAS_int - np.vdot(W_adapt[k, :], u_int)
            
            # Update W_adapt ALLEEN op basis van de mix en VAD
            if vad_frames[n] == 0:
                power = np.vdot(u_mic, u_mic).real + 1e-10
                W_adapt[k, :] += mu * u_mic * np.conj(e_mic) / power

    # Synthese
    _, out_tar_td = signal.istft(E_out_tar, fs, window=window, nperseg=L, noverlap=overlap)
    _, out_int_td = signal.istft(E_out_int, fs, window=window, nperseg=L, noverlap=overlap)
    _, out_mic_td = signal.istft(E_out_mic, fs, window=window, nperseg=L, noverlap=overlap)

    return out_mic_td[:N_samples], out_tar_td[:N_samples], out_int_td[:N_samples]

# ==========================================
# MAIN PROGRAMMA
# ==========================================
if __name__ == "__main__":
    # ===============================================
    # CONFIGURATIE: KIES HIER JOUW INSTELLINGEN
    # ===============================================
    use_reverb = False  # Set to True voor reverberant RIRs, False voor dry/no_reverb
    
    fs = 44100
    segment_duration = 10.0 # 10 seconden per positie [cite: 31, 42]
    segment_samples = int(fs * segment_duration)
    
    # 1. Definieer de paden naar je 5 gegenereerde RIRs voor een specifieke T60
    # PAS DEZE NAMEN AAN NAAR HOE JIJ ZE IN DE GUI HEBT GENOEMD!
    # Vervang de oude lijst door deze:
    rir_files_no_reverb = [
        "160_20_no_reverb.pkl",
        "140_40_no_reverb.pkl",
        "120_60_no_reverb.pkl",
        "100_40_no_reverb.pkl",
        "95_85_no_reverb.pkl"
    ]

    # Vervang de oude lijst door deze:
    rir_files_reverb = [
        "160_20_reverb.pkl",
        "140_40_reverb.pkl",
        "120_60_reverb.pkl",
        "100_40_reverb.pkl",
        "95_85_reverb.pkl"
    ]
    
    # Selecteer op basis van de instelling hierboven
    if use_reverb:
        rir_files = rir_files_reverb
        reverb_label = "REVERBERANT"
    else:
        rir_files = rir_files_no_reverb
        reverb_label = "NO_REVERB"
    
    # 2. Laad de droge audio (zorg dat ze minimaal 50 seconden lang zijn!)
    target_audio, _ = sf.read(os.path.join(parent_dir, "sound_files", "part1_track1_dry.wav")) # [cite: 41]
    interf_audio, _ = sf.read(os.path.join(parent_dir, "sound_files", "part1_track2_dry.wav")) # [cite: 41]
    
    # Lijsten om de volledige 50s signalen op te slaan
    full_target_in, full_interf_in = [], []
    full_target_out, full_interf_out, full_mix_out = [], [], []
    
    sir_in_history, sir_out_history, doas_history = [], [], []
    
    print("\n" + "="*70)
    print(f"START TIME-VARYING SCENARIO VERWERKING [{reverb_label}]")
    print(f"RIR Files: {rir_files}")
    print("="*70)
    
    for i, rir_file in enumerate(rir_files):
        print(f"\n--- Verwerken van Segment {i+1}/5 (0-{segment_duration}s) ---")
        
        # Laad scenario
        scenario = load_rirs(os.path.join(parent_dir, "rirs", rir_file))
        
        # DEBUG: Print hoeveel bronnen er zijn gevonden
        print(f"DEBUG: Aantal Audio RIRs: {scenario.RIRs_audio.shape[2]}")
        
        target_rir = scenario.RIRs_audio[:, :, 0] # Altijd bron 1
        
        # Check of de tweede bron in RIRs_audio of RIRs_noise staat
        if scenario.RIRs_audio.shape[2] > 1:
            interf_rir = scenario.RIRs_audio[:, :, 1]
        elif hasattr(scenario, 'RIRs_noise') and scenario.RIRs_noise is not None:
            print("Interferer gevonden in RIRs_noise")
            interf_rir = scenario.RIRs_noise[:, :, 0]
        else:
            raise ValueError(f"Bestand {rir_file} bevat slechts 1 bron. Beiden zijn nodig voor SIR berekening!")
        
        # Snij de juiste 10 seconden uit de bron-audio
        start_idx = i * segment_samples
        end_idx = start_idx + segment_samples
        
        # Zorg voor looping als audio korter is dan 50s
        chunk_target = target_audio[start_idx % len(target_audio) : end_idx % len(target_audio)]
        chunk_interf = interf_audio[start_idx % len(interf_audio) : end_idx % len(interf_audio)]
        if len(chunk_target) < segment_samples: 
            chunk_target = np.pad(chunk_target, (0, segment_samples - len(chunk_target)))
            chunk_interf = np.pad(chunk_interf, (0, segment_samples - len(chunk_interf)))

        # Convolve met de RIRs om de microfooncomponenten te krijgen
        M_mics = target_rir.shape[1]
        target_comp = np.zeros((segment_samples, M_mics))
        interf_comp = np.zeros((segment_samples, M_mics))
        
        for m in range(M_mics):
            t_conv = signal.fftconvolve(chunk_target, target_rir[:, m], mode='full')
            i_conv = signal.fftconvolve(chunk_interf, interf_rir[:, m], mode='full')
            target_comp[:, m] = t_conv[:segment_samples]
            interf_comp[:, m] = i_conv[:segment_samples]
            
        mic_mix = target_comp + interf_comp
        
        # MUSIC DOA Estimation [cite: 43]
        angles_music, _, _, est_doas = music_wideband(mic_mix, fs, scenario)
        doa_estimate = est_doas[np.argmin(np.abs(est_doas - 90.0))] if len(est_doas) > 0 else 90.0
        doas_history.append(doa_estimate)
        print(f"Geschatte Target DOA: {doa_estimate:.1f}°")
        
        # Genereer Look-Up Table (Eenmalig per blok)
        LUT, lut_angles = part1_4.generate_steering_lut(scenario, 1024, fs)
        
        # Run de aangepaste GSC
        out_mix, out_tar, out_int = gsc_fd_components(
            target_comp, interf_comp, fs, doa_estimate, LUT, lut_angles
        )
        
        # SIR Berekeningen [cite: 44, 45]
        sir_in = compute_sir_metric(target_comp[:, 0], interf_comp[:, 0])
        sir_out = compute_sir_metric(out_tar, out_int)
        
        sir_in_history.append(sir_in)
        sir_out_history.append(sir_out)
        
        print(f"Input SIR (Mic 1): {sir_in:.2f} dB")
        print(f"Output SIR (GSC):  {sir_out:.2f} dB (Winst: {sir_out - sir_in:.2f} dB)")
        
        # Opslaan voor de uiteindelijke 50s audio
        full_target_in.append(target_comp[:, 0])
        full_mix_out.append(out_mix)

    # Concateneer alle blokken tot één 50s signaal
    final_input = np.concatenate(full_target_in)
    final_output = np.concatenate(full_mix_out)
    
    # Plot het verloop over tijd
    plt.figure(figsize=(12, 6))
    t_blocks = np.arange(1, len(sir_in_history) + 1) * segment_duration
    plt.plot(t_blocks, sir_in_history, marker='o', linestyle='--', color='gray', label='Input SIR (Mic 1)', linewidth=2)
    plt.plot(t_blocks, sir_out_history, marker='s', linestyle='-', color='blue', label='Output SIR (GSC)', linewidth=2)
    plt.title(f'SIR Evolutie in Tijds-Variërend Scenario [{reverb_label}]')
    plt.xlabel(f'Tijdstip van verandering (s, interval={segment_duration}s)')
    plt.ylabel('SIR (dB)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Audio exporteren met label
    output_filename = f"Part2_Dynamic_Output_{reverb_label}_{len(sir_in_history)}_segments.wav"
    output_path = os.path.join(parent_dir, output_filename)
    sf.write(output_path, final_output / np.max(np.abs(final_output)), fs)
    print(f"\nVerwerking voltooid. Audio opgeslagen als '{output_filename}'")
    print(f"Volledige pad: {output_path}")