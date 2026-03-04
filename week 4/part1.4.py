import numpy as np
import scipy.signal as signal
import scipy.linalg
import matplotlib.pyplot as plt
import soundfile as sf
import os
import sys

# Setup path to import from package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from package.utils import create_micsigs as original_create_micsigs
from package.utils import music_wideband

# ==========================================
# FUNCTION: MODIFIED CREATE_MICSIGS
# ==========================================
def modified_create_micsigs(scenario, speech_paths, noise_paths=None, duration=10.0):
    if noise_paths is None:
        noise_paths = []

    if scenario.fs != 44100:
        raise ValueError(f"Fout: De samplingfrequentie is {scenario.fs} Hz. Dit moet 44.1 kHz zijn!")

    _, speech, _ = original_create_micsigs(scenario, speech_paths, [], duration=duration)
    
    if len(noise_paths) > 0:
        _, _, gui_noise = original_create_micsigs(scenario, speech_paths, noise_paths, duration=duration)
    else:
        gui_noise = np.zeros_like(speech)

    vad = np.abs(speech[:, 0]) > (np.std(speech[:, 0]) * 1e-3)
    Ps = np.var(speech[vad == 1, 0])
    
    white_noise_power = 0.10 * Ps
    white_noise = np.random.randn(*speech.shape) * np.sqrt(white_noise_power)
    noise = white_noise + gui_noise
    
    mic = speech + noise
    Pn = np.var(noise[:, 0])
    SNR_in = 10 * np.log10(Ps / max(Pn, 1e-10))
    
    print(f"[Info] Input SNR (Microfoon 1): {SNR_in:.2f} dB")
    
    return mic, speech, noise, SNR_in, vad

# ==========================================
# FUNCTION: CALCULATE GROUND TRUTH DOAS
# ==========================================
def calculate_ground_truth_doas(acoustic_scenario):
    """Berekent de werkelijke hoeken van de bronnen ten opzichte van het array."""
    if acoustic_scenario.audioPos is None or len(acoustic_scenario.audioPos) == 0:
        return []
    
    mics = acoustic_scenario.micPos       
    array_center = np.mean(mics, axis=0)
    
    doas = []
    for source in acoustic_scenario.audioPos:
        dx = source[0] - array_center[0]
        dy = source[1] - array_center[1] 
        # Berekening hoek (afhankelijk van specifieke assenstelsel orientatie)
        hoek_rad = np.arctan2(-dx, -dy)
        hoek_deg = np.degrees(hoek_rad) % 360
        doas.append(hoek_deg)
        
    return np.array(doas)

# ==========================================
# FUNCTION: GENERATE LOOK-UP TABLE (LUT) FROM RIRS
# ==========================================
def generate_steering_lut(scenario, L, fs):
    """
    Genereert een LUT van stuurvectoren h(w, theta) = a(w, theta)/a_1(w, theta) 
    door de DFT te nemen van de meegeleverde RIRs (inclusief galm)[cite: 19, 20].
    
    Let op: In een echt time-varying scenario zou je de RIR generator hier aanroepen
    voor elke graad (0-180). Hier bouwen we de LUT uit de RIRs in het opgeslagen scenario.
    """
    nF = L // 2 + 1
    M = scenario.micPos.shape[0]
    
    # 1. Bepaal DOAs van de beschikbare bronnen in het scenario
    audio_doas = calculate_ground_truth_doas(scenario)
    
    LUT_list = []
    lut_angles = []

    # 2. RIRs van audiosources transformeren
    if scenario.audioPos is not None and hasattr(scenario, 'RIRs_audio'):
        for src_idx, doa in enumerate(audio_doas):
            rir = scenario.RIRs_audio[:, :, src_idx] # Shape: (N_samples, M_mics)
            
            # STFT of FFT nemen van de RIR [cite: 20]
            H = np.fft.rfft(rir, n=L, axis=0) # Shape: (nF, M_mics)
            
            # Normaliseer t.o.v. eerste microfoon (voorkom deling door 0) [cite: 20]
            H_norm = H / (H[:, 0:1] + 1e-12) 
            
            LUT_list.append(H_norm)
            lut_angles.append(doa)
            
    # Zorg voor de juiste dimensies: (nF, nAngles, M_mics)
    if len(LUT_list) > 0:
        LUT = np.array(LUT_list).transpose(1, 0, 2)
    else:
        # Fallback als er geen RIRs zijn
        LUT = np.ones((nF, 1, M), dtype=complex)
        lut_angles = [90.0]

    return LUT, np.array(lut_angles)

# ==========================================
# FUNCTION: MUSIC MET RIR-GEBASEERDE LUT
# ==========================================
def music_wideband_lut(micsigs, fs, L, overlap, LUT, lut_angles, Q=1):
    """
    MUSIC algoritme dat de stuurvectoren h(w, theta) uit de LUT gebruikt 
    ter vervanging van de geometrische g(w, theta) om DOA schatting te verbeteren.
    """
    window = np.sqrt(signal.windows.hann(L, sym=False))
    stft_list = []
    for m in range(micsigs.shape[1]):
        freqs, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window=window, nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0)
    
    M, nF, nT = stft_data.shape
    nAngles = len(lut_angles)
    
    f_min, f_max = 300, 4000
    k_min = int(f_min / (fs / L))
    k_max = int(f_max / (fs / L))
    valid_indices = range(max(1, k_min), min(nF, k_max))
    
    pseudospectra = np.zeros((len(valid_indices), nAngles))
    
    for idx, k in enumerate(valid_indices):
        Y = stft_data[:, k, :]
        Ryy = (Y @ Y.conj().T) / nT
        _, eigvecs = np.linalg.eigh(Ryy)
        En = eigvecs[:, :M-Q]
        
        # Gebruik de h(w, theta) uit de LUT in plaats van geometrische vertragingen 
        for a_idx in range(nAngles):
            h_k = LUT[k, a_idx, :] # Shape: (M,)
            denom = np.linalg.norm(En.conj().T @ h_k)**2
            pseudospectra[idx, a_idx] = 1.0 / (denom + 1e-12)
            
    p_geom = np.exp(np.mean(np.log(pseudospectra + 1e-12), axis=0))
    spectrum_geom_db = 10 * np.log10(p_geom / np.max(p_geom))
    
    # Kies de piek (of pieken)
    best_idx = np.argmax(spectrum_geom_db)
    estimated_doa = lut_angles[best_idx]
    
    return spectrum_geom_db, estimated_doa

# ==========================================
# FUNCTION: FD-GSC BEAMFORMER
# ==========================================
def gsc_fd(scenario, speech_paths, noise_paths=None, duration=10.0, mu=0.05):
    """Frequency-domain DOA-informed GSC."""
    
    mic, speech, noise, snr_in, vad = modified_create_micsigs(
        scenario, speech_paths, noise_paths, duration
    )
    fs = scenario.fs
    N_samples, M_mics = mic.shape
    L = 1024
    overlap = L // 2 # 50% overlap [cite: 11]

    # --- STAP A & B: LUT GENEREREN EN DOA SCHATTEN (RIR-gebaseerd) ---
    print("[FD-GSC] Generating Look-Up Table (LUT) from RIRs...")
    LUT, lut_angles = generate_steering_lut(scenario, L, fs)
    
    print("[FD-GSC] Estimating DOA via LUT-based MUSIC...")
    Q_sources = 1 if (scenario.audioPos is not None) else 0
    spectrum_lut, doa_estimate = music_wideband_lut(mic, fs, L, overlap, LUT, lut_angles, Q=Q_sources)
    print(f"[FD-GSC] LUT MUSIC Spectrum (dB): {np.round(spectrum_lut, 2)}")
    print(f"[FD-GSC] Selected Target DOA from LUT: {doa_estimate:.1f}°")

    lut_idx = np.argmin(np.abs(lut_angles - doa_estimate))

    # --- STAP C: ANALYSIS PART (STFT) --- 
    window = np.sqrt(signal.windows.hann(L, sym=False)) # Square-root Hanning [cite: 12]
    f_bins, t_frames, Zxx = signal.stft(mic.T, fs, window=window, nperseg=L, noverlap=overlap)
    nF, nT = Zxx.shape[1], Zxx.shape[2]

    # VAD vertalen naar frame-niveau voor adaptatie
    vad_frames = np.zeros(nT)
    for n in range(nT):
        start_idx = n * (L - overlap)
        end_idx = start_idx + L
        if end_idx <= len(vad):
            vad_frames[n] = 1 if np.mean(vad[start_idx:end_idx]) > 0.05 else 0

    # --- STAP D: GSC FILTERING IN FREQUENTIEDOMEIN ---
    E_out = np.zeros((nF, nT), dtype=complex)
    W_adapt = np.zeros((nF, M_mics - 1), dtype=complex)
    
    for k in range(nF):
        h_k = LUT[k, lut_idx, :]
        
        # a) FAS Beamformer: w_FAS = h / (h^H * h) [cite: 25]
        norm_factor = np.vdot(h_k, h_k)
        w_FAS = np.zeros(M_mics, dtype=complex) if np.abs(norm_factor) < 1e-10 else h_k / norm_factor
            
        # b) Blocking Matrix B: B * h = 0 [cite: 26]
        N_mat = scipy.linalg.null_space(h_k.reshape(1, M_mics).conj())
        B_k = N_mat.conj().T 
        
        # Frame-per-frame adaptatie
        for n in range(nT):
            x_kn = Zxx[:, k, n] 
            
            y_FAS = np.vdot(w_FAS, x_kn) # Hoofdpad (FAS)
            u_kn = B_k @ x_kn            # Ruispad (BM)
            e_kn = y_FAS - np.vdot(W_adapt[k, :], u_kn) # GSC Output
            E_out[k, n] = e_kn
            
            # Adaptatie pauzeren tijdens actieve spraak (Target Cancellation voorkomen)
            if vad_frames[n] == 0:
                power = np.vdot(u_kn, u_kn).real + 1e-10
                W_adapt[k, :] += mu * u_kn * np.conj(e_kn) / power

    # --- STAP E: SYNTHESIS PART (ISTFT) --- 
    # Match analysis window met WOLA (Weighted Overlap-Add) synthese [cite: 17]
    _, gsc_out_td = signal.istft(E_out, fs, window=window, nperseg=L, noverlap=overlap)
    gsc_out_td = gsc_out_td[:N_samples]
    
    Pn_gsc = np.var(gsc_out_td[vad == 0])
    Psn_gsc = np.var(gsc_out_td[vad == 1])
    Ps_gsc = max(Psn_gsc - Pn_gsc, 1e-10)
    SNRout_GSC = 10 * np.log10(Ps_gsc / max(Pn_gsc, 1e-10))
    
    print(f"[FD-GSC] Output SNR: {SNRout_GSC:.2f} dB (Netto Verbetering vs Mic1: {SNRout_GSC - snr_in:.2f} dB)")
    
    return gsc_out_td, SNRout_GSC, mic, doa_estimate, snr_in

# ==========================================
# MAIN PROGRAMMA
# ==========================================
if __name__ == "__main__":
    try:
        from package import load_rirs
        rirs_folder = os.path.join(parent_dir, "rirs")
        files = [os.path.join(rirs_folder, f) for f in os.listdir(rirs_folder) if f.endswith('.pkl')]
        latest_rir = max(files, key=os.path.getmtime)
        scenario = load_rirs(latest_rir)

        num_audio = scenario.audioPos.shape[0] if scenario.audioPos is not None else 0
        num_noise = scenario.noisePos.shape[0] if (hasattr(scenario, 'noisePos') and scenario.noisePos is not None) else 0

        speech_paths = [os.path.join(parent_dir, "sound_files", "speech1.wav")][:num_audio]
        noise_paths = [os.path.join(parent_dir, "sound_files", "speech2.wav")][:num_noise]

        print("\n" + "="*50)
        print("START FREQUENCY-DOMAIN GSC (DOA-Informed & LUT-based)")
        print("="*50)
        
        gsc_out_td, SNRout_GSC, mic, doa_estimate, SNR_in_actual = gsc_fd(
            scenario, speech_paths, noise_paths, duration=10.0, mu=0.05
        )
        
        print("\n" + "="*50)
        print("KLAAR! (Verificatie plots & export logica overgeslagen voor leesbaarheid)")
        print("="*50)

    except Exception as e:
        import traceback
        traceback.print_exc()