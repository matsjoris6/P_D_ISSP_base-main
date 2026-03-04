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

def gsc_fd(scenario, speech_paths, noise_paths=None, duration=10.0, vad_threshold=1e-3):
    # 1. Genereer Micsigs
    mic, speech, noise, snr_in, _ = modified_create_micsigs(scenario, speech_paths, noise_paths, duration)
    N_samples, M_mics = mic.shape
    L = 1024
    overlap = L // 2
    
    # Gebruik Square-Root Hanning Window zoals gevraagd in de opdracht
    window_sqrthann = np.sqrt(signal.windows.hann(L, sym=False))

    angles, _, _, est_doas, _ = music_wideband(mic, scenario.fs, scenario)
    target_doa = est_doas[np.argmin(np.abs(est_doas - 90.0))]
    print(f"\n[FD-GSC] Target DOA: {target_doa:.1f}°")

    # 3. Look-Up Table (LUT) genereren voor FAS en BM op basis van de RIR
    target_rir = get_target_rir_from_scenario(scenario, source_index=0)
    W_FAS_lut, B_lut = build_lut_for_target(target_rir, L=L)

    # 4. STFT uitvoeren met het nieuwe window
    _, _, Zxx_mix = signal.stft(mic.T, fs=scenario.fs, window=window_sqrthann, nperseg=L, noverlap=overlap)
    _, _, Zxx_tar = signal.stft(speech.T, fs=scenario.fs, window=window_sqrthann, nperseg=L, noverlap=overlap)
    
    nF, nT = Zxx_mix.shape[1], Zxx_mix.shape[2]

    # --- NIEUW: IDEALE FREQUENTIEDOMEIN VAD (Subband VAD) ---
    Zxx_tar_mic1 = Zxx_tar[0, :, :]
    thresholds = np.std(np.abs(Zxx_tar_mic1), axis=1, keepdims=True) * vad_threshold + 1e-6
    VAD_FD = np.abs(Zxx_tar_mic1) > thresholds

    # Print VAD Statistieken
    actieve_bins = np.sum(VAD_FD)
    totaal_bins = nF * nT
    percentage_spraak = (actieve_bins / totaal_bins) * 100
    percentage_adaptatie = 100 - percentage_spraak
    print(f"  [VAD-FD] Spraak in {percentage_spraak:.1f}% van de tijd-frequentie bins | Filter leert in {percentage_adaptatie:.1f}% van de bins.")

    # 5. GSC Filtering in het Frequentiedomein
    E_out = np.zeros((nF, nT), dtype=complex)
    FAS_out = np.zeros((nF, nT), dtype=complex)
    
    mu = 0.05
    eps = 1e-8
    
    # Frequenty-domein filtering
    for k in range(nF):
        w_fas = W_FAS_lut[k, :].reshape(M_mics, 1)
        B_k = B_lut[k, :, :]
        w_nlms = np.zeros((M_mics - 1, 1), dtype=complex)
        
        for n in range(nT):
            y_kn = Zxx_mix[:, k, n].reshape(M_mics, 1)
            
            # FAS Pad (Target)
            d_kn = (w_fas.conj().T @ y_kn)[0, 0]
            FAS_out[k, n] = d_kn
            
            # BM Pad (Noise refs)
            x_kn = B_k @ y_kn
            
            # Filter output & Error
            y_out_kn = (w_nlms.conj().T @ x_kn)[0, 0]
            e_kn = d_kn - y_out_kn
            E_out[k, n] = e_kn
            
            # NLMS Update (enkel tijdens stilte op basis van de Subband VAD!)
            if VAD_FD[k, n] == False:
                power = np.real((x_kn.conj().T @ x_kn)[0, 0])
                w_nlms += mu * np.conj(e_kn) * x_kn / (power + eps)

    # 6. Synthese (Inverse STFT) met hetzelfde square-root window
    _, fas_out_time = signal.istft(FAS_out, fs=scenario.fs, window=window_sqrthann, nperseg=L, noverlap=overlap)
    _, gsc_out_time = signal.istft(E_out, fs=scenario.fs, window=window_sqrthann, nperseg=L, noverlap=overlap)
    
    fas_out_time = fas_out_time[:N_samples]
    gsc_out_time = gsc_out_time[:N_samples]
    
    # 7. SNR Berekening (We gebruiken hier nog de oude tijdsdomein VAD puur voor de output-berekening)
    vad_time = np.abs(speech[:, 0]) > (np.std(speech[:, 0]) * 1e-3)
    
    def calc_snr(sig, vad_mask):
        Ps = np.var(sig[vad_mask == 1])
        Pn = np.var(sig[vad_mask == 0])
        return 10 * np.log10(max(Ps, 1e-10) / max(Pn, 1e-10))
        
    snr_fas = calc_snr(fas_out_time, vad_time)
    snr_gsc = calc_snr(gsc_out_time, vad_time)
    
    print(f"  -> Input SNR:  {snr_in:.2f} dB")
    print(f"  -> FAS SNR:    {snr_fas:.2f} dB (Winst: {snr_fas - snr_in:.2f} dB)")
    print(f"  -> GSC SNR:    {snr_gsc:.2f} dB (Winst: {snr_gsc - snr_in:.2f} dB)")

    # Plot genereren
    t = np.arange(N_samples) / scenario.fs
    
    plt.figure(figsize=(14, 7))
    plt.plot(t, mic[:, 0], color='gray', alpha=0.5, label=f'Microfoon 1 (Input SNR: {snr_in:.1f} dB)')
    plt.plot(t, fas_out_time, color='dodgerblue', alpha=0.6, label=f'FAS Beamformer (SNR: {snr_fas:.1f} dB)')
    plt.plot(t, gsc_out_time, color='green', alpha=0.9, linewidth=1.2, label=f'GSC FD Output (SNR: {snr_gsc:.1f} dB)')
    
    plt.xlabel("Tijd (s)")
    plt.ylabel("Amplitude")
    plt.title("Signaalvergelijking: Origineel vs. FAS-BF vs. Frequency-Domain GSC")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    return gsc_out_time, mic, snr_in, fas_out_time

if __name__ == "__main__":
    try:
        rirs_folder = os.path.join(parent_dir, "rirs")
        files = [os.path.join(rirs_folder, f) for f in os.listdir(rirs_folder) if f.endswith('.pkl')]
        latest_rir = max(files, key=os.path.getmtime)
        scenario = load_rirs(latest_rir)

        num_audio = scenario.audioPos.shape[0] if scenario.audioPos is not None else 0
        num_noise = scenario.noisePos.shape[0] if (hasattr(scenario, 'noisePos') and scenario.noisePos is not None) else 0

        alle_spraak_bestanden = [
            os.path.join(parent_dir, "sound_files", "speech1.wav"),
            os.path.join(parent_dir, "sound_files", "speech2.wav")
        ]
        speech_paths = alle_spraak_bestanden[:num_audio]

        # Gelokaliseerde ruisbestand instellen
        gekozen_ruisbestand = "speech2.wav" 
        ruis_pad = os.path.join(parent_dir, "sound_files", gekozen_ruisbestand)
        alle_ruis_bestanden = [ruis_pad]
        noise_paths = alle_ruis_bestanden[:num_noise]

        # --- NIEUW: Voer de Frequentiedomein GSC uit ---
        # Let op: we vangen hier de 4 return-waarden op!
        gsc_out_time, mic, snr_in, fas_out_time = gsc_fd(scenario, speech_paths, noise_paths, duration=10.0)

        # --- AUDIO OPSLAAN ---
        # Origineel
        mic_audio = mic[:, 0] / np.max(np.abs(mic[:, 0]))
        path_mic = os.path.join(parent_dir, "Luister_01_Origineel_Mic1.wav")
        sf.write(path_mic, mic_audio, scenario.fs)
        
        # Verbeterd met FAS (ter vervanging van de DAS)
        fas_audio = fas_out_time / np.max(np.abs(fas_out_time))
        path_fas = os.path.join(parent_dir, "Luister_02_Verbeterd_FAS.wav")
        sf.write(path_fas, fas_audio, scenario.fs)
        
        # Volledige GSC Output
        gsc_audio = gsc_out_time / np.max(np.abs(gsc_out_time))
        path_gsc = os.path.join(parent_dir, "Luister_03_GSC_FD_Filter.wav")
        sf.write(path_gsc, gsc_audio, scenario.fs)
        
        print("\n Audiobestanden succesvol opgeslagen in je parent directory!")

    except Exception as e:
        import traceback
        print(f"\n❌ Fout in uitvoering: {e}")
        traceback.print_exc()