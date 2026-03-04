import numpy as np
import scipy.linalg
import scipy.signal as signal
import scipy.linalg
import soundfile as sf
import matplotlib.pyplot as plt
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from package import load_rirs
from package.utils import create_micsigs as original_create_micsigs


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
    SNR_in = 10 * np.log10(Ps / Pn)
    
    print(f"[Info] Input SNR (Microfoon 1): {SNR_in:.2f} dB")
    
    return mic, speech, noise, SNR_in, vad
def music_wideband(micsigs, fs, acoustic_scenario):
  
    L = 1024
    overlap = L // 2
    
    stft_list = []
    for m in range(micsigs.shape[1]):
        freqs, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window='hann', nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0) # Shape: (M, nF, nT)
    
    M, nF, nT = stft_data.shape
    c = 343.0
    num_audio = acoustic_scenario.audioPos.shape[0] if acoustic_scenario.audioPos is not None else 0
    num_noise = acoustic_scenario.noisePos.shape[0] if acoustic_scenario.noisePos is not None else 0
    Q = num_audio + num_noise
    print('Q =', Q)
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    
    # mic coördinaten
    mics_centered = acoustic_scenario.micPos - np.mean(acoustic_scenario.micPos, axis=0)
    px = mics_centered[:, 0].reshape(-1, 1)
    py = mics_centered[:, 1].reshape(-1, 1)
    
    # We itereren over bins k = 2 ... L/2. 
    # In Python (0-based) is k=1 (DC) index 0. L/2 is index 512 (Nyquist).
    # De indices 1 t/m 511 komen exact overeen met k=2 ... L/2
    valid_indices = range(1, L // 2)
    pseudospectra = []
    
    for k in valid_indices:
        # Signaal isoleren voor frequentie k
        Y = stft_data[:, k, :]
        Ryy = (Y @ Y.conj().T) / nT
        
        # Ruissubruimte bepalen
        _, eigvecs = np.linalg.eigh(Ryy)
        En = eigvecs[:, :M-Q] 
        
        # Steering vector bepalen
        omega = 2 * np.pi * freqs[k]
        taus = (px * np.sin(rads) + py * np.cos(rads)) / c 
        A = np.exp(-1j * omega * taus)
        
        # Pseudospectrum P(theta)
        denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
        p_theta = 1.0 / denom
        pseudospectra.append(p_theta)
        
    pseudospectra = np.array(pseudospectra) # Shape: (511, len(angles))
    
    # 2. Geometrisch Gemiddelde (Numeriek stabiele methode) + veel minder gevoelig voor pieken die samensmelten
    # i.p.v. p1 * p2 * .. pn tot de macht 1/N doen we: exp(mean(log(p))) want veel rekenkrachtiger
    log_p = np.log(pseudospectra)
    p_geom = np.exp(np.mean(log_p, axis=0))
    
    # Normalizeer en naar dB
    spectrum_geom_db = 10 * np.log10(p_geom / np.max(p_geom))
    
    # 3. Peak Finding op het gemiddelde spectrum
    peaks_indices, _ = signal.find_peaks(spectrum_geom_db)
    if len(peaks_indices) >= Q:
        sorted_peak_indices = peaks_indices[np.argsort(spectrum_geom_db[peaks_indices])][-Q:]
        estimated_doas = np.sort(angles[sorted_peak_indices])
    else:
        # Fallback indien pieken samensmelten
        sorted_all = np.argsort(spectrum_geom_db)[::-1]
        est_idx = []
        for idx in sorted_all:
            if len(est_idx) == Q: break
            if all(abs(idx - e) > 20 for e in est_idx):
                est_idx.append(idx)
        estimated_doas = np.sort(angles[est_idx])

    # Geef de individuele genormaliseerde spectra ook mee (in dB) voor de plot
    ps_ind_db = 10 * np.log10(pseudospectra / np.max(pseudospectra, axis=1, keepdims=True))

    return angles, ps_ind_db, spectrum_geom_db, estimated_doas, stft_data

def get_target_rir_from_scenario(scenario, source_index=0):
  
    if not hasattr(scenario, 'RIRs_audio') or scenario.RIRs_audio is None:
        raise ValueError("Kan 'RIRs_audio' niet vinden in het scenario!")
        
    # Snijd de 3D matrix (samples, mics, sources) af op de gewenste source_index (0)
    # Dit geeft een 2D matrix (samples, mics) terug, exact wat de LUT-builder nodig heeft!
    target_rir = scenario.RIRs_audio[:, :, source_index]
    
    return target_rir
def build_lut_for_target(target_rir, L=1024):
    """
    Bouwt de FAS en BM matrices (Look-Up Table) voor de specifieke target RIR per frequentie.
    Volgens de theorie: h(w) = A(w)/A_1(w), w_FAS = h/(h^H h), B * h = 0.
    """
    n_bins = L // 2 + 1
    M_mics = target_rir.shape[1]
    
    H_omega = np.fft.rfft(target_rir, n=L, axis=0)
    
    W_FAS = np.zeros((n_bins, M_mics), dtype=complex)
    B_matrix = np.zeros((n_bins, M_mics - 1, M_mics), dtype=complex)
    
    for k in range(n_bins):
        h_k = H_omega[k, :].reshape(M_mics, 1)
        
        # Normalizeer naar Mic 1 (index 0)
        eps = 1e-12
        A_1 = h_k[0, 0]
        if np.abs(A_1) > eps:
            h_k = h_k / A_1
            
        # FAS Beamformer
        denom = (h_k.conj().T @ h_k)[0, 0]
        if np.abs(denom) > eps:
            W_FAS[k, :] = (h_k / denom).flatten()
        else:
            W_FAS[k, :] = np.ones(M_mics) / M_mics
            
        # Blocking Matrix
        Z = scipy.linalg.null_space(h_k.conj().T)
        if Z.shape[1] > 0:
            B_matrix[k, :, :] = Z.conj().T
            
    return W_FAS, B_matrix

def gsc_fd(scenario, speech_paths, noise_paths=None, duration=10.0, mu=0.05):
    # 1. Genereer Micsigs (Mix, en losse componenten)
    mic, speech, noise, snr_in, vad = modified_create_micsigs(scenario, speech_paths, noise_paths, duration)
    N_samples, M_mics = mic.shape
    L = 1024
    overlap = L // 2

    print(f"\n[FD-GSC] LUT bouwen op basis van exacte RIR...")
    target_rir = get_target_rir_from_scenario(scenario, source_index=0)
    W_FAS_lut, B_lut = build_lut_for_target(target_rir, L=L)

    # 2. STFT voor de Mix én de losse componenten!
    _, _, Zxx_mix = signal.stft(mic.T, fs=scenario.fs, window='hann', nperseg=L, noverlap=overlap)
    _, _, Zxx_tar = signal.stft(speech.T, fs=scenario.fs, window='hann', nperseg=L, noverlap=overlap)
    _, _, Zxx_int = signal.stft(noise.T, fs=scenario.fs, window='hann', nperseg=L, noverlap=overlap)
    
    nF, nT = Zxx_mix.shape[1], Zxx_mix.shape[2]

    # 3. VAD synchroniseren
    vad_frames = np.zeros(nT)
    samples_per_frame = L - overlap
    for n in range(nT):
        start_idx = n * samples_per_frame
        end_idx = start_idx + L
        if end_idx <= N_samples:
            vad_frames[n] = 1 if np.mean(vad[start_idx:end_idx]) > 0.5 else 0

    # 4. GSC Filtering
    E_out_mix = np.zeros((nF, nT), dtype=complex)
    E_out_tar = np.zeros((nF, nT), dtype=complex) # Voor SIR (x1)
    E_out_int = np.zeros((nF, nT), dtype=complex) # Voor SIR (x2)
    FAS_out   = np.zeros((nF, nT), dtype=complex) 
    
    eps = 1e-8
    
    for k in range(nF):
        w_fas = W_FAS_lut[k, :]
        B_k = B_lut[k, :, :]
        w_nlms = np.zeros((M_mics - 1,), dtype=complex)
        
        for n in range(nT):
            # Signalen ophalen
            x_mix = Zxx_mix[:, k, n]
            x_tar = Zxx_tar[:, k, n]
            x_int = Zxx_int[:, k, n]
            
            # --- Mix (Bepaalt de filter updates) ---
            y_fas_mix = np.vdot(w_fas, x_mix)
            u_mix = B_k @ x_mix
            e_mix = y_fas_mix - np.vdot(w_nlms, u_mix)
            E_out_mix[k, n] = e_mix
            FAS_out[k, n] = y_fas_mix
            
            # NLMS Update (enkel tijdens stilte)
            if vad_frames[n] == 0:
                power = np.vdot(u_mix, u_mix).real
                w_nlms += mu * u_mix * np.conj(e_mix) / (power + eps)

            # --- Shadow Filtering (Pas zelfde filter toe op losse spraak/ruis) ---
            y_fas_tar = np.vdot(w_fas, x_tar)
            u_tar = B_k @ x_tar
            E_out_tar[k, n] = y_fas_tar - np.vdot(w_nlms, u_tar) # x1

            y_fas_int = np.vdot(w_fas, x_int)
            u_int = B_k @ x_int
            E_out_int[k, n] = y_fas_int - np.vdot(w_nlms, u_int) # x2

    # 5. Synthese (Inverse STFT)
    _, gsc_out_mix = signal.istft(E_out_mix, fs=scenario.fs, window='hann', nperseg=L, noverlap=overlap)
    _, fas_out_time = signal.istft(FAS_out, fs=scenario.fs, window='hann', nperseg=L, noverlap=overlap)
    _, out_tar = signal.istft(E_out_tar, fs=scenario.fs, window='hann', nperseg=L, noverlap=overlap)
    _, out_int = signal.istft(E_out_int, fs=scenario.fs, window='hann', nperseg=L, noverlap=overlap)
    
    gsc_out_mix = gsc_out_mix[:N_samples]
    fas_out_time = fas_out_time[:N_samples]
    out_tar = out_tar[:N_samples]
    out_int = out_int[:N_samples]

    # Geef nu ALLES terug wat we ooit nodig kunnen hebben!
    return gsc_out_mix, mic, snr_in, fas_out_time, out_tar, out_int, speech, noise

import numpy as np

def compute_sir(y, x1, x2, groundTruth):
    """
    Compute the signal-to-interference ratio for two sources (one target source
    and one interfering source). The script takes into account possible
    switches between which source is the target and which one is the
    interference. The script assumes there is access to each source's
    contribution in the beamformer output.

    Parameters
    ----------
    -y : [N x 1] np.ndarray[float]
        Actual beamformer output signal (`N` is the number of samples).
    -x1 : [N x 1] np.ndarray[float]
        Beamformer output attributed to source 1.
    -x2 : [N x 1] np.ndarray[float]
        Beamformer output attributed to source 2.
    -groundTruth : [N x 1] np.ndarray[int (0 or 1) or float (0. or 1.)]
        Array indicating, for each sample,
        which source is the target: 1=x1, 0=x2.

    Returns
    -------
    -sir : 
    """
    # Sanity check (check whether `y = x1 + x2` based on RMSE of residual)
    if np.sqrt(np.sum((y - x1 - x2) ** 2)) / np.sqrt(np.sum(y ** 2)) > 0.01:
        print('/!\ Something is wrong, `y` should be the sum of `x1` and `x2`.')  
        print('SIR can not be computed -- Returning NaN.')  
        sir = np.nan
    # Input check
    elif np.sum(groundTruth) + np.sum(1 - groundTruth) != len(groundTruth):
        print('/!\ `groundTruth` vector is not binary.')
        print('SIR can not be computed')  
        sir = np.nan
    else:
        sir = 10 * np.log10(
            np.var(x1 * groundTruth + x2 * (1 - groundTruth)) /\
                np.var(x2 * groundTruth + x1 * (1 - groundTruth))
        )

    return sir

# ==========================================
# MAIN PROGRAMMA: DEEL 2 (TIME-VARYING SCENARIO)
# ==========================================
if __name__ == "__main__":
    
    fs = 44100
    segment_duration = 10.0 
    segment_samples = int(fs * segment_duration)
    
    # Audio inladen (zonder extra looping)
    t_dry, _ = sf.read(os.path.join(parent_dir, "sound_files", "part1_track1_dry.wav"))
    i_dry, _ = sf.read(os.path.join(parent_dir, "sound_files", "part1_track2_dry.wav"))
    
    t_dry = t_dry / np.max(np.abs(t_dry)) * 0.9
    i_dry = i_dry / np.max(np.abs(i_dry)) * 0.9

    # De opdracht vereist het vergelijken van ANECHOIC (0s) en REVERBERANT (1s)
    # We laten de code automatisch achter elkaar beide scenario's runnen
    for use_reverb in [False, True]:
        
        reverb_label = "REVERBERANT (T60=1s)" if use_reverb else "ANECHOIC (T60=0s)"
        
        # JOUW RIR BESTANDEN VOOR DE 5 POSITIES
        if not use_reverb:
            rir_files = ["160_20_no_reverb.pkl", "140_40_no_reverb.pkl", "120_60_no_reverb.pkl", "100_40_no_reverb.pkl", "95_85_no_reverb.pkl"]
        else:
            rir_files = ["160_20_reverb.pkl", "140_40_reverb.pkl", "120_60_reverb.pkl", "100_40_reverb.pkl", "95_85_reverb.pkl"]
        
        full_mix_out = []
        full_mic_in = [] 
        sir_in_per_sec = []
        sir_out_per_sec = []
        
        print("\n" + "="*70)
        print(f"STARTING SCENARIO: {reverb_label}")
        print("="*70)
        
        for i, rir_file in enumerate(rir_files):
            print(f"\n--- Segment {i+1}/5 | Tijd: {i*10}s tot {(i+1)*10}s | Bestand: {rir_file} ---")
            
            try:
                scenario = load_rirs(os.path.join(parent_dir, "rirs", rir_file))
            except Exception as e:
                print(f"❌ Kon {rir_file} niet vinden in map 'rirs'. Controleer de naam!")
                sys.exit()

            target_rir = scenario.RIRs_audio[:, :, 0]
            interf_rir = scenario.RIRs_audio[:, :, 1] if scenario.RIRs_audio.shape[2] > 1 else scenario.RIRs_noise[:, :, 0]

            # Knip het juiste 10s stukje audio eruit
            chunk_t = t_dry[i*segment_samples : (i+1)*segment_samples]
            chunk_i = i_dry[i*segment_samples : (i+1)*segment_samples]

            # Voeg akoestiek toe
            t_comp = np.stack([signal.fftconvolve(chunk_t, target_rir[:,m], mode='full')[:segment_samples] for m in range(5)], axis=1)
            i_comp = np.stack([signal.fftconvolve(chunk_i, interf_rir[:,m], mode='full')[:segment_samples] for m in range(5)], axis=1)
            mic_mix = t_comp + i_comp
            
            # --- DOA ESTIMATION & PRINTEN ---
            _, _, _, est_doas, _ = music_wideband(mic_mix, fs, scenario)
            print(f"   [DOA] MUSIC detecteert bronnen op: {est_doas}°")
            
            # --- GSC FILTERING ---
            speech_paths = [os.path.join(parent_dir, "sound_files", "part1_track1_dry.wav")]
            noise_paths = [os.path.join(parent_dir, "sound_files", "part1_track2_dry.wav")]
            
            # Voer GSC uit (gebruikt RIR van het huidige scenario)
            gsc_out_mix, mic, _, _, out_tar, out_int, _, _ = gsc_fd(
                scenario, speech_paths, noise_paths, duration=10.0, mu=0.005
            )
            
            full_mix_out.append(gsc_out_mix)
            full_mic_in.append(mic[:, 0]) 
            
            # --- SIR BEREKENING PER SECONDE ---
            sec_samples = int(fs * 1.0)
            ground_truth = np.ones(sec_samples, dtype=int) 
            
            print("   [SIR] Prestaties per seconde:")
            for s in range(10): 
                start_idx = s * sec_samples
                end_idx = start_idx + sec_samples
                
                # Input SIR
                y_in_sec = t_comp[start_idx:end_idx, 0] + i_comp[start_idx:end_idx, 0]
                sir_in = compute_sir(y_in_sec, t_comp[start_idx:end_idx, 0], i_comp[start_idx:end_idx, 0], ground_truth)
                
                # Output SIR
                sir_out = compute_sir(gsc_out_mix[start_idx:end_idx], out_tar[start_idx:end_idx], out_int[start_idx:end_idx], ground_truth)
                
                sir_in_per_sec.append(sir_in)
                sir_out_per_sec.append(sir_out)
                
                # Print de SIR mooi gestructureerd voor elke seconde
                print(f"         Sec {i*10 + s + 1:02d}: Input SIR = {sir_in:5.1f} dB  -->  GSC Output SIR = {sir_out:5.1f} dB")

        # Plot de SIR over de volle 50 seconden voor dit specifieke scenario
        plt.figure(figsize=(12, 6))
        t_axis = np.arange(1, 51)
        plt.plot(t_axis, sir_in_per_sec, color='gray', linestyle=':', label='Input SIR (Mic 1)')
        plt.plot(t_axis, sir_out_per_sec, color='blue', linewidth=2, label='Output SIR (GSC)')
        
        for wissel in range(10, 50, 10):
            plt.axvline(x=wissel, color='red', linestyle='--', alpha=0.5)
            
        plt.title(f'SIR Verbetering over Tijd - {reverb_label}')
        plt.ylabel('SIR (dB)')
        plt.xlabel('Tijd (seconden)')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Audio opslaan
        final_output = np.concatenate(full_mix_out)
        final_mic_in = np.concatenate(full_mic_in)
        
        # Bestandsnamen opschonen voor OS
        safe_label = reverb_label.replace("=", "").replace("(", "").replace(")", "").replace(" ", "_")
        sf.write(os.path.join(parent_dir, f"01_Origineel_Mic1_{safe_label}.wav"), final_mic_in / np.max(np.abs(final_mic_in)), fs)
        sf.write(os.path.join(parent_dir, f"02_Gefilterd_GSC_{safe_label}.wav"), final_output / np.max(np.abs(final_output)), fs)
        
        print(f"\n✅ Verwerking {reverb_label} voltooid! Audio opgeslagen.")