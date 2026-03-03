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
sys.path.append(current_dir)

# Importeer vereiste functies
from package import load_rirs
from package.utils import music_wideband

# Importeer de Look-Up Table generator uit part1.4.py
spec = importlib.util.spec_from_file_location("part1_4", os.path.join(current_dir, "part1.4.py"))
part1_4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(part1_4)

# ==========================================
# FUNCTION: COMPUTE SIR (Signal-to-Interference Ratio)
# ==========================================
def compute_sir_metric(target_signal, interferer_signal):
    """
    Berekent de SIR over een gegeven tijdsblok[cite: 61, 63].
    """
    var_target = np.var(target_signal)
    var_interf = np.var(interferer_signal)
    
    if var_interf < 1e-12:
        return np.nan 
        
    return 10 * np.log10((var_target + 1e-12) / (var_interf + 1e-12))

# ==========================================
# FUNCTION: BLOCK-BASED FD-GSC MET COMPONENT SEPARATIE
# ==========================================
def gsc_fd_components(target_comp, interf_comp, fs, doa_estimate, LUT, lut_angles, mu=0.05):
    """
    Filtert target en interferer simultaan door EXACT dezelfde filtergewichten
    om een eerlijke SIR-meting te garanderen[cite: 68, 69].
    """
    mic = target_comp + interf_comp
    N_samples, M_mics = mic.shape
    
    L = 1024
    overlap = L // 2
    window = np.sqrt(signal.windows.hann(L, sym=False))
    
    # STFT op alle componenten
    _, _, Zxx_mic = signal.stft(mic.T, fs, window=window, nperseg=L, noverlap=overlap)
    _, _, Zxx_tar = signal.stft(target_comp.T, fs, window=window, nperseg=L, noverlap=overlap)
    _, _, Zxx_int = signal.stft(interf_comp.T, fs, window=window, nperseg=L, noverlap=overlap)
    
    nF, nT = Zxx_mic.shape[1], Zxx_mic.shape[2]
    
    # VAD op de zuivere target voor controle van adaptatie
    vad = np.abs(target_comp[:, 0]) > (np.std(target_comp[:, 0]) * 1e-3)
    vad_frames = np.zeros(nT)
    for n in range(nT):
        start_idx = n * (L - overlap)
        end_idx = start_idx + L
        if end_idx <= len(vad):
            vad_frames[n] = 1 if np.mean(vad[start_idx:end_idx]) > 0.05 else 0

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
            # MIX (voor adaptatie)
            x_mic = Zxx_mic[:, k, n]
            y_FAS_mic = np.vdot(w_FAS, x_mic)
            u_mic = B_k @ x_mic
            e_mic = y_FAS_mic - np.vdot(W_adapt[k, :], u_mic)
            E_out_mic[k, n] = e_mic
            
            # Pas filter toe op gescheiden componenten voor SIR [cite: 69]
            E_out_tar[k, n] = np.vdot(w_FAS, Zxx_tar[:,k,n]) - np.vdot(W_adapt[k,:], B_k @ Zxx_tar[:,k,n])
            E_out_int[k, n] = np.vdot(w_FAS, Zxx_int[:,k,n]) - np.vdot(W_adapt[k,:], B_k @ Zxx_int[:,k,n])
            
            # NLMS update alleen bij afwezigheid van target (VAD == 0)
            if vad_frames[n] == 0:
                p_u = np.vdot(u_mic, u_mic).real + 1e-10
                W_adapt[k, :] += mu * u_mic * np.conj(e_mic) / p_u

    # Synthese (WOLA)
    _, out_tar_td = signal.istft(E_out_tar, fs, window=window, nperseg=L, noverlap=overlap)
    _, out_int_td = signal.istft(E_out_int, fs, window=window, nperseg=L, noverlap=overlap)
    _, out_mic_td = signal.istft(E_out_mic, fs, window=window, nperseg=L, noverlap=overlap)

    return out_mic_td[:N_samples], out_tar_td[:N_samples], out_int_td[:N_samples]

# ==========================================
# MAIN PROGRAMMA
# ==========================================
if __name__ == "__main__":
    use_reverb = False  # Pas aan voor REVERBERANT (True) of NO_REVERB (False)
    
    fs = 44100
    segment_duration = 10.0 
    segment_samples = int(fs * segment_duration)
    
    # Gebruik jouw specifieke bestandsnamen
    rir_files_no_reverb = ["160_20_no_reverb.pkl", "140_40_no_reverb.pkl", "120_60_no_reverb.pkl", "100_40_no_reverb.pkl", "95_85_no_reverb.pkl"]
    rir_files_reverb = ["160_20_reverb.pkl", "140_40_reverb.pkl", "120_60_reverb.pkl", "100_40_reverb.pkl", "95_85_reverb.pkl"]
    
    rir_files = rir_files_reverb if use_reverb else rir_files_no_reverb
    reverb_label = "REVERBERANT" if use_reverb else "NO_REVERB"
    
    # Laden en NORMALISEREN van audio voor gelijke startpositie
    t_dry, _ = sf.read(os.path.join(parent_dir, "sound_files", "part1_track1_dry.wav"))
    i_dry, _ = sf.read(os.path.join(parent_dir, "sound_files", "part1_track2_dry.wav"))
    t_dry = t_dry / np.max(np.abs(t_dry)) * 0.9
    i_dry = i_dry / np.max(np.abs(i_dry)) * 0.9
    
    full_mix_out, sir_in_history, sir_out_history = [], [], []
    
    print("\n" + "="*70)
    print(f"START TIME-VARYING GSC [{reverb_label}]")
    print("="*70)
    
    for i, rir_file in enumerate(rir_files):
        print(f"\n--- Segment {i+1}/5 ({i*10}-{(i+1)*10}s) ---")
        scenario = load_rirs(os.path.join(parent_dir, "rirs", rir_file))
        
        # Bron detectie (Audio vs Noise RIR)
        target_rir = scenario.RIRs_audio[:, :, 0]
        if scenario.RIRs_audio.shape[2] > 1:
            interf_rir = scenario.RIRs_audio[:, :, 1]
        else:
            interf_rir = scenario.RIRs_noise[:, :, 0]

        # Audio segmenten pakken
        chunk_t = t_dry[i*segment_samples : (i+1)*segment_samples]
        chunk_i = i_dry[i*segment_samples : (i+1)*segment_samples]

        # Convolutie
        t_comp = np.stack([signal.fftconvolve(chunk_t, target_rir[:,m], mode='full')[:segment_samples] for m in range(5)], axis=1)
        i_comp = np.stack([signal.fftconvolve(chunk_i, interf_rir[:,m], mode='full')[:segment_samples] for m in range(5)], axis=1)
        
        # --- VERBETERDE DOA SELECTIE ---
        _, _, _, est_doas = music_wideband(t_comp + i_comp, fs, scenario)
        # Zoek alleen naar kandidaten in het target-gebied (links, 90-180 graden) [cite: 32]
        target_candidates = [d for d in est_doas if d >= 85]
        if target_candidates:
            doa_estimate = target_candidates[np.argmin(np.abs(np.array(target_candidates) - 90.0))]
        else:
            doa_estimate = 90.0
        print(f"Geschatte Target DOA: {doa_estimate:.1f}°")
        
        LUT, lut_angles = part1_4.generate_steering_lut(scenario, 1024, fs)
        out_mix, out_tar, out_int = gsc_fd_components(t_comp, i_comp, fs, doa_estimate, LUT, lut_angles)
        
        # SIR berekening [cite: 63, 71]
        sir_in = compute_sir_metric(t_comp[:, 0], i_comp[:, 0])
        sir_out = compute_sir_metric(out_tar, out_int)
        sir_in_history.append(sir_in)
        sir_out_history.append(sir_out)
        
        print(f"SIR In: {sir_in:.2f} dB | SIR Out: {sir_out:.2f} dB (Winst: {sir_out - sir_in:.2f} dB)")
        full_mix_out.append(out_mix)

    # Plot & Save
    plt.figure(figsize=(10, 5))
    t_axis = np.arange(1, 6) * 10
    plt.plot(t_axis, sir_in_history, 'o--', color='gray', label='Input SIR')
    plt.plot(t_axis, sir_out_history, 's-', color='blue', label='Output SIR (GSC)')
    plt.title(f'SIR Verbetering [{reverb_label}]'); plt.ylabel('SIR (dB)'); plt.legend(); plt.grid(); plt.show()

    final_output = np.concatenate(full_mix_out)
    sf.write(os.path.join(parent_dir, f"GSC_Dynamic_{reverb_label}.wav"), final_output / np.max(np.abs(final_output)), fs)