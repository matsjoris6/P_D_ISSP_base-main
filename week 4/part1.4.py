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
    SNR_in = 10 * np.log10(Ps / Pn)
    
    print(f"[Info] Input SNR (Microfoon 1): {SNR_in:.2f} dB")
    
    return mic, speech, noise, SNR_in, vad

# ==========================================
# FUNCTION: GENERATE LOOK-UP TABLE (LUT)
# ==========================================
def generate_steering_lut(scenario, L, fs):
    """
    Genereert een Look-Up Table (LUT) van stuurvectoren voor elke DOA en frequentie.
    Omdat we geen gemeten RIRs hebben voor álle 180 graden, gebruiken we de 
    array-geometrie om de theoretische stuurvectoren te berekenen.
    """
    angles = np.arange(0, 181, 1) # LUT voor 0 tot 180 graden
    rads = np.radians(angles)
    c = 343.0
    M = scenario.micPos.shape[0]
    nF = L // 2 + 1
    freqs = np.fft.rfftfreq(L, 1/fs)
    
    mics_centered = scenario.micPos - np.mean(scenario.micPos, axis=0)
    px = mics_centered[:, 0].reshape(-1, 1)
    py = mics_centered[:, 1].reshape(-1, 1)
    
    LUT = np.zeros((nF, len(angles), M), dtype=complex)
    
    for k in range(nF):
        omega = 2 * np.pi * freqs[k]
        # Bereken delays voor ALLE hoeken (Vectorized)
        taus = (px * np.sin(rads) + py * np.cos(rads)) / c
        A = np.exp(-1j * omega * taus) # Shape: (M, len(angles))
        
        # Normaliseer ten opzichte van de eerste microfoon: a(omega)/a_1(omega)
        H = A / (A[0, :] + 1e-12)
        
        # Opslaan in vorm: (nF, nAngles, M_mics)
        LUT[k, :, :] = H.T 
        
    return LUT, angles

# ==========================================
# FUNCTION: FD-GSC BEAMFORMER
# ==========================================
def gsc_fd(scenario, speech_paths, noise_paths=None, duration=10.0, mu=0.05):
    """
    Frequency-domain DOA-informed GSC.
    """
    # 1. Signalen genereren
    mic, speech, noise, snr_in, vad = modified_create_micsigs(
        scenario, speech_paths, noise_paths, duration
    )
    fs = scenario.fs
    N_samples, M_mics = mic.shape
    L = 1024
    overlap = L // 2 # 50% overlap [cite: 11]
    
    # --- STAP A: MUSIC DOA SCHATTING EERST ---
    print("[FD-GSC] Estimating DOA via MUSIC...")
    angles_music, ps_ind_db, spec_geo, est_doas = music_wideband(mic, fs, scenario)
    
    if len(est_doas) > 0:
        doa_estimate = est_doas[np.argmin(np.abs(est_doas - 90.0))]
    else:
        doa_estimate = 90.0
    print(f"[FD-GSC] MUSIC Schattingen: {est_doas}")
    print(f"[FD-GSC] Selected Target DOA: {doa_estimate:.1f}°")

    # --- STAP B: LOOK-UP TABLE (LUT) GENEREREN ---
    print("[FD-GSC] Generating Look-Up Table (LUT) for Steering Vectors...")
    LUT, lut_angles = generate_steering_lut(scenario, L, fs)
    
    # Zoek de index in de LUT die het dichtst bij onze MUSIC schatting ligt
    lut_idx = np.argmin(np.abs(lut_angles - doa_estimate))
    print(f"[FD-GSC] Extracting Steering Vector from LUT for angle {lut_angles[lut_idx]}°")

    # --- STAP C: ANALYSIS PART (STFT) --- 
    window = np.sqrt(signal.windows.hann(L, sym=False)) # [cite: 12]
    f_bins, t_frames, Zxx = signal.stft(mic.T, fs, window=window, nperseg=L, noverlap=overlap)
    nF, nT = Zxx.shape[1], Zxx.shape[2]

    # VAD vertalen naar frame-niveau
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
        # Haal de stuurvector voor deze specifieke bin en geschatte hoek uit de LUT 
        h_k = LUT[k, lut_idx, :]
        
        # a) FAS Beamformer wegingen: w_FAS = h / (h^H * h) [cite: 25]
        norm_factor = np.vdot(h_k, h_k)
        if np.abs(norm_factor) < 1e-10:
            w_FAS = np.zeros(M_mics, dtype=complex)
        else:
            w_FAS = h_k / norm_factor
            
        # b) Blocking Matrix B berekenen: B * h = 0 [cite: 26, 29]
        N_mat = scipy.linalg.null_space(h_k.reshape(1, M_mics).conj())
        B_k = N_mat.conj().T 
        
        # Frame-per-frame verwerking
        for n in range(nT):
            x_kn = Zxx[:, k, n] 
            
            y_FAS = np.vdot(w_FAS, x_kn) # Hoofdpad (FAS)
            u_kn = B_k @ x_kn            # Ruispad (BM)
            e_kn = y_FAS - np.vdot(W_adapt[k, :], u_kn) # Output
            E_out[k, n] = e_kn
            
            # Adaptatie pauzeren tijdens spraak
            if vad_frames[n] == 0:
                power = np.vdot(u_kn, u_kn).real + 1e-10
                W_adapt[k, :] += mu * u_kn * np.conj(e_kn) / power

    # --- STAP E: SYNTHESIS PART (ISTFT) --- [cite: 17]
    _, gsc_out_td = signal.istft(E_out, fs, window=window, nperseg=L, noverlap=overlap)
    gsc_out_td = gsc_out_td[:N_samples]
    
    Pn_gsc = np.var(gsc_out_td[vad == 0])
    Psn_gsc = np.var(gsc_out_td[vad == 1])
    Ps_gsc = max(Psn_gsc - Pn_gsc, 1e-10)
    SNRout_GSC = 10 * np.log10(Ps_gsc / max(Pn_gsc, 1e-10))
    
    print(f"[FD-GSC] Output SNR: {SNRout_GSC:.2f} dB (Netto Verbetering vs Mic1: {SNRout_GSC - snr_in:.2f} dB)")

    # (PLOTS ZIJN HIER WEGGELATEN OM DE CODE COMPACT TE HOUDEN - JE KUNT JOUW EIGEN PLOT LOGICA HIER TERUGZETTEN ALS JE WILT)
    
    return gsc_out_td, SNRout_GSC, mic, E_out, doa_estimate, angles_music, spec_geo, est_doas, snr_in

import matplotlib.pyplot as plt

# ==========================================
# FUNCTION: VERIFICATION PLOTS
# ==========================================
def plot_fd_gsc_verification(mic, gsc_out_td, fs, snr_in, snr_out, angles_music, spec_geo, est_doas, doa_estimate):
    """
    Genereert verificatieplots voor de FD-GSC:
    1. Tijdsdomein overlay (Input vs Output)
    2. Spectrogram van de Input
    3. Spectrogram van de Output
    4. MUSIC DOA Spectrum
    """
    t = np.arange(len(gsc_out_td)) / fs

    fig = plt.figure(figsize=(16, 12))

    # 1. Tijdsdomein: Origineel vs GSC Output
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, mic[:len(gsc_out_td), 0], color='lightgray', label=f'Mic 1 Input (SNR: {snr_in:.1f} dB)')
    ax1.plot(t, gsc_out_td, color='blue', alpha=0.8, linewidth=0.8, label=f'FD-GSC Output (SNR: {snr_out:.1f} dB)')
    ax1.set_title('1. Tijdsdomein Verificatie: Input vs. FD-GSC Output')
    ax1.set_xlabel('Tijd (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2 & 3. Spectrograms met gelijk kleurenschema
    # Bereken beide spectrogrammen eerst
    f1, t1, Sxx1 = signal.spectrogram(mic[:len(gsc_out_td), 0], fs=fs, nperseg=1024, noverlap=512)
    f2, t2, Sxx2 = signal.spectrogram(gsc_out_td, fs=fs, nperseg=1024, noverlap=512)
    
    # Converteer naar dB schaal
    Sxx1_dB = 10 * np.log10(Sxx1 + 1e-10)
    Sxx2_dB = 10 * np.log10(Sxx2 + 1e-10)
    
    # Bepaal gemeenschappelijke min/max voor beide spectrogrammen
    vmin = min(np.min(Sxx1_dB), np.min(Sxx2_dB))
    vmax = max(np.max(Sxx1_dB), np.max(Sxx2_dB))
    
    # 2. Spectrogram: Input (Mic 1)
    ax2 = plt.subplot(3, 2, 3)
    im1 = ax2.pcolormesh(t1, f1, Sxx1_dB, shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)
    cbar1 = plt.colorbar(im1, ax=ax2)
    cbar1.set_label('Power (dB)')
    ax2.set_title('2A. Spectrogram: Originele Input (Mic 1)')
    ax2.set_xlabel('Tijd (s)')
    ax2.set_ylabel('Frequentie (Hz)')
    ax2.set_ylim(0, fs/2)

    # 3. Spectrogram: FD-GSC Output
    ax3 = plt.subplot(3, 2, 4)
    im2 = ax3.pcolormesh(t2, f2, Sxx2_dB, shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)
    cbar2 = plt.colorbar(im2, ax=ax3)
    cbar2.set_label('Power (dB)')
    ax3.set_title('2B. Spectrogram: FD-GSC Output (Let op ruisonderdrukking)')
    ax3.set_xlabel('Tijd (s)')
    ax3.set_ylabel('Frequentie (Hz)')
    ax3.set_ylim(0, fs/2)

    # 4. MUSIC DOA Spectrum
    ax4 = plt.subplot(3, 1, 3)
    ax4.plot(angles_music, spec_geo, 'b-', linewidth=1.5)
    ax4.axvline(doa_estimate, color='red', linestyle='--', linewidth=2, label=f'Gekozen Target DOA: {doa_estimate:.1f}°')
    
    # Plot alle gevonden pieken
    for doa in est_doas:
        ax4.axvline(doa, color='green', linestyle=':', alpha=0.7, label='Gedetecteerde bron')
        
    ax4.set_title('3. DOA Verificatie: MUSIC Spectrum')
    ax4.set_xlabel('Hoek (graden)')
    ax4.set_ylabel('Spectrum (dB)')
    
    # Verwijder dubbele labels in de legenda
    handles, labels = ax4.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax4.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 180)

    plt.tight_layout()
    plt.show()

# ==========================================
# FUNCTION: EXPORT AUDIO FILES
# ==========================================
def export_audio(output_dir, gsc_out_td, mic_input, fs, doa_estimate, snr_in, snr_out):
    os.makedirs(output_dir, exist_ok=True)
    mic_normalized = mic_input[:, 0] / (np.max(np.abs(mic_input[:, 0])) + 1e-10)
    gsc_normalized = gsc_out_td / (np.max(np.abs(gsc_out_td)) + 1e-10)
    
    sf.write(os.path.join(output_dir, f"01_FD-GSC_Input_SNR{snr_in:.1f}dB.wav"), mic_normalized, fs)
    sf.write(os.path.join(output_dir, f"02_FD-GSC_Output_SNR{snr_out:.1f}dB_DOA{doa_estimate:.1f}deg.wav"), gsc_normalized, fs)

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
        
        # Voer de FD-GSC uit
        gsc_out_td, SNRout_GSC, mic, E_out, doa_estimate, angles_music, spec_geo, est_doas, SNR_in_actual = gsc_fd(
            scenario, speech_paths, noise_paths, duration=10.0, mu=0.05
        )
        
        # Roep de plot functie aan
        print("\n[FD-GSC] Genereren van verificatie plots...")
        plot_fd_gsc_verification(
            mic, gsc_out_td, scenario.fs, SNR_in_actual, SNRout_GSC, 
            angles_music, spec_geo, est_doas, doa_estimate
        )

        print("\n" + "="*50)
        print("EXPORTING AUDIO FILES")
        print("="*50)
        
        export_audio(os.path.join(parent_dir, "FD-GSC_Output"), gsc_out_td, mic, scenario.fs, doa_estimate, SNR_in_actual, SNRout_GSC)

    except Exception as e:
        import traceback
        traceback.print_exc()