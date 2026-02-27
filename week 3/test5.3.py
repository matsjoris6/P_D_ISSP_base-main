import os
import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt


base_folder = os.path.join("sound_files", "head_mounted_rirs")
dry_signal_1_path = os.path.join("sound_files", "part2_track1_dry.wav") 
dry_signal_2_path = os.path.join("sound_files", "part2_track2_dry.wav")

fs_sim = 44100
c = 343.0
d_ear = 0.013    # 1.3 cm
d_front = 0.215  # 21.5 cm

folder_left = "s-30"
folder_right = "s30"


angle_L = abs(float(folder_left.replace("s", "")))  
angle_R = float(folder_right.replace("s", ""))      


true_front_R = 90.0 - angle_R
true_front_L = 90.0 + angle_L


try:
    dry_sig_1, _ = sf.read(dry_signal_1_path)
    dry_sig_2, _ = sf.read(dry_signal_2_path)
    
    min_len = min(len(dry_sig_1), len(dry_sig_2))
    dry_sig_1 = dry_sig_1[:min_len]
    dry_sig_2 = dry_sig_2[:min_len]

    yL1_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_L1.wav"))
    yL2_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_L2.wav"))
    yR1_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_R1.wav"))
    yR2_rir_left, _ = sf.read(os.path.join(base_folder, folder_left, "HMIR_R2.wav"))

    yL1_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_L1.wav"))
    yL2_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_L2.wav"))
    yR1_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_R1.wav"))
    yR2_rir_right, _ = sf.read(os.path.join(base_folder, folder_right, "HMIR_R2.wav"))

    yL1_from_left = signal.fftconvolve(dry_sig_1, yL1_rir_left, mode='full')
    yL2_from_left = signal.fftconvolve(dry_sig_1, yL2_rir_left, mode='full')
    yR1_from_left = signal.fftconvolve(dry_sig_1, yR1_rir_left, mode='full')
    yR2_from_left = signal.fftconvolve(dry_sig_1, yR2_rir_left, mode='full')

    yL1_from_right = signal.fftconvolve(dry_sig_2, yL1_rir_right, mode='full')
    yL2_from_right = signal.fftconvolve(dry_sig_2, yL2_rir_right, mode='full')
    yR1_from_right = signal.fftconvolve(dry_sig_2, yR1_rir_right, mode='full')
    yR2_from_right = signal.fftconvolve(dry_sig_2, yR2_rir_right, mode='full')

  
    yL1_mix = yL1_from_left + yL1_from_right
    yL2_mix = yL2_from_left + yL2_from_right
    yR1_mix = yR1_from_left + yR1_from_right
    yR2_mix = yR2_from_left + yR2_from_right

    micsigs_left_ear = np.column_stack((yL1_mix, yL2_mix))
    micsigs_right_ear = np.column_stack((yR1_mix, yR2_mix))
    micsigs_front = np.column_stack((yL1_mix, yR1_mix)) # L1 en R1 voor de grote afstand

except Exception as e:
    print(f"Fout bij het laden van audio/RIRs: {e}")
    exit()

def music_wideband_2mics(micsigs, fs, d, Q=1):
    L = 1024
    overlap = L // 2
    
    stft_list = []
    for m in range(micsigs.shape[1]):
        freqs, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window='hann', nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0)
    
    M, nF, nT = stft_data.shape
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    
    mics_pos = np.array([0, d])
    valid_indices = range(1, L // 2)
    pseudospectra = []
    
    for k in valid_indices:
        Y = stft_data[:, k, :]
        Ryy = (Y @ Y.conj().T) / nT
        _, eigvecs = np.linalg.eigh(Ryy)
        En = eigvecs[:, :M-Q] 
        
        omega = 2 * np.pi * freqs[k]
        taus = (mics_pos.reshape(-1, 1) * np.cos(rads)) / c 
        A = np.exp(-1j * omega * taus)
        
        denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
        pseudospectra.append(1.0 / denom)
        
    pseudospectra = np.array(pseudospectra)
    p_geom = np.exp(np.mean(np.log(pseudospectra), axis=0))
    spectrum_db = 10 * np.log10(p_geom / np.max(p_geom))
    
    peaks_indices, _ = signal.find_peaks(spectrum_db)
    
    if len(peaks_indices) > 0:
        sorted_peaks = peaks_indices[np.argsort(spectrum_db[peaks_indices])][-Q:]
        estimated_doas = np.sort(angles[sorted_peaks])
    else:
        estimated_doas = [np.nan]
        
    return angles, spectrum_db, estimated_doas
def MUSIC_wideband_HM(micsigs, fs, Q=2):
  
    L = 1024
    overlap = L // 2
    
    
    stft_list = []
    for m in range(micsigs.shape[1]):
        freqs, _, Zxx = signal.stft(micsigs[:, m], fs=fs, window='hann', nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0) 
    
    M, nF, nT = stft_data.shape
    c = 343.0
    
    
    x_ear = 0.215 / 2  
    y_mic = 0.013 / 2  
    
    mics_pos = np.array([
        [-x_ear,  y_mic],  
        [-x_ear, -y_mic],  
        [ x_ear,  y_mic],  
        [ x_ear, -y_mic]   
    ])
    
    px = mics_pos[:, 0].reshape(-1, 1)
    py = mics_pos[:, 1].reshape(-1, 1)
    
    angles = np.arange(0, 180.5, 0.5)
    rads = np.radians(angles)
    
    pseudospectra = []
    
   
    for k in range(1, nF - 1):
     
        Y = stft_data[:, k, :]
        Ryy = (Y @ Y.conj().T) / nT
        
        _, eigvecs = np.linalg.eigh(Ryy)
   
        En = eigvecs[:, :M-Q] 
        
        omega = 2 * np.pi * freqs[k]
        
        
        taus = (px * np.cos(rads) + py * np.sin(rads)) / c 
        A = np.exp(1j * omega * taus)
        
        denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
        pseudospectra.append(1.0 / denom)
        
  
    pseudospectra = np.array(pseudospectra)
    p_geom = np.exp(np.mean(np.log(pseudospectra), axis=0))
    spectrum_db = 10 * np.log10(p_geom / np.max(p_geom))
    

    peaks_indices, _ = signal.find_peaks(spectrum_db)
    
    if len(peaks_indices) >= Q:
        sorted_peaks = peaks_indices[np.argsort(spectrum_db[peaks_indices])][-Q:]
        estimated_doas = np.sort(angles[sorted_peaks])
    else:

        estimated_doas = angles[np.argsort(spectrum_db)[-Q:]]
        
    return angles, spectrum_db, estimated_doas
def das_bf_hm(micsigs, speech_comp, noise_comp, fs, mics_pos, target_doa_deg):
    """
    Delay-and-Sum beamformer voor een willekeurig microfoonarray.
    mics_pos : [M x 2] array van microfoonposities (gecentreerd)
    target_doa_deg : geschatte DOA van de doelbron (graden)
    """
    target_rad = np.radians(target_doa_deg)
    px = mics_pos[:, 0]
    py = mics_pos[:, 1]
    taus = (px * np.cos(target_rad) + py * np.sin(target_rad)) / c

    def apply_das(sigs, delays):
        M_val = sigs.shape[1]
        N     = sigs.shape[0]
        out_sum     = np.zeros(N)
        out_aligned = np.zeros((N, M_val))
        freqs = np.fft.rfftfreq(N, 1 / fs)
        for m in range(M_val):
            S = np.fft.rfft(sigs[:, m])
            S_shifted = S * np.exp(1j * 2 * np.pi * freqs * delays[m])
            t_aligned = np.fft.irfft(S_shifted, n=N)
            out_aligned[:, m] = t_aligned
            out_sum += t_aligned
        return out_sum / M_val, out_aligned

    speechDAS, speech_aligned = apply_das(speech_comp, taus)
    noiseDAS,  noise_aligned  = apply_das(noise_comp,  taus)
    DASout      = speechDAS + noiseDAS
    aligned_mic = speech_aligned + noise_aligned

    # VAD op basis van de eerste spraakreferentie
    vad = np.abs(speech_comp[:, 0]) > np.std(speech_comp[:, 0]) * 1e-3
    Ps  = np.var(speechDAS[vad]) if np.any(vad) else 1e-10
    Pn  = np.var(noiseDAS)
    Pn  = max(Pn, 1e-10)
    SNR = 10 * np.log10(Ps / Pn)

    return DASout, speechDAS, noiseDAS, SNR, aligned_mic, vad, taus

def gsc_hm(micsigs, speech_comp, noise_comp, fs, mics_pos,
           target_doa_deg, label=""):
    """
    Generalized Sidelobe Canceler met VAD voor een head-mounted array.
    Griffiths-Jim blocking matrix + NLMS adaptief filter.
    """
    print(f"\n{'='*55}")
    print(f"GSC met VAD – {label}")
    print(f"{'='*55}")

    DASout, speechDAS, noiseDAS, SNRoutDAS, aligned_mic, vad, taus = das_bf_hm(
        micsigs, speech_comp, noise_comp, fs, mics_pos, target_doa_deg
    )
    print(f"[DAS] SNR: {SNRoutDAS:.2f} dB")

    N_samples, M_mics = aligned_mic.shape

    # --- Blocking Matrix (Griffiths-Jim) ---
    B = np.zeros((M_mics - 1, M_mics))
    for i in range(M_mics - 1):
        B[i, 0]     =  1
        B[i, i + 1] = -1

    noise_refs = (B @ aligned_mic.T).T  # [N x (M-1)]

    # --- NLMS instellingen ---
    L_fir = 1024
    mu    = 0.1
    delta = L_fir // 2

    target_ref = np.pad(DASout, (delta, 0))[:N_samples]
    vad_delayed = np.pad(vad,    (delta, 0))[:N_samples]

    W      = np.zeros((L_fir, M_mics - 1))
    GSCout = np.zeros(N_samples)

    padded_noise = np.pad(noise_refs, ((L_fir - 1, 0), (0, 0)))
    eps = 1e-8

    print(f"[GSC] NLMS filtering (L={L_fir}, mu={mu}, delta={delta}) ...")
    for n in range(N_samples):
        X_slice = padded_noise[n: n + L_fir]
        y_n     = np.sum(W * X_slice)
        e_n     = target_ref[n] - y_n
        GSCout[n] = e_n

        # Filter update ALLEEN tijdens stilte (VAD = 0) → voorkomt target cancellation
        if vad_delayed[n] == 0:
            power = np.sum(X_slice ** 2)
            W += (mu * e_n / (power + eps)) * X_slice

    print("[GSC] Filtering voltooid!")

    # --- SNR berekening ---
    Pn_gsc  = np.var(GSCout[vad_delayed == 0])
    Psn_gsc = np.var(GSCout[vad_delayed == 1])
    Ps_gsc  = max(Psn_gsc - Pn_gsc, 1e-10)
    Pn_gsc  = max(Pn_gsc, 1e-10)
    SNRoutGSC = 10 * np.log10(Ps_gsc / Pn_gsc)

    # SNR eerste microfoon (referentie)
    speech_ref = speech_comp[:, 0]
    noise_ref  = noise_comp[:, 0]
    Ps_mic1 = np.var(speech_ref[vad]) if np.any(vad) else 1e-10
    Pn_mic1 = max(np.var(noise_ref), 1e-10)
    SNR_mic1 = 10 * np.log10(Ps_mic1 / Pn_mic1)

    print(f"[GSC] Output SNR:        {SNRoutGSC:.2f} dB")
    print(f"[GSC] Verbetering t.o.v. Mic 1: {SNRoutGSC - SNR_mic1:.2f} dB")

    # --- Plot ---
    t = np.arange(N_samples) / fs
    GSCout_plot = np.pad(GSCout[delta:], (0, delta))

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"GSC met VAD – {label}", fontsize=14, fontweight='bold')

    axes[0].plot(t, micsigs[:, 0], color='gray', alpha=0.7,
                 label=f'Microfoon 1 (SNR: {SNR_mic1:.1f} dB)')
    axes[0].set_title("Origineel Microfoonsignaal (Mic 1)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, DASout, color='dodgerblue', alpha=0.7,
                 label=f'DAS BF (SNR: {SNRoutDAS:.1f} dB)')
    axes[1].plot(t, speechDAS, color='cyan', alpha=0.6, linestyle='--',
                 label='Spraakcomponent DAS')
    axes[1].set_title("Delay-and-Sum Beamformer")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, GSCout_plot, color='green', alpha=0.9, linewidth=1.2,
                 label=f'GSC Output (SNR: {SNRoutGSC:.1f} dB)')
    axes[2].set_title("Generalized Sidelobe Canceler (met VAD)")
    axes[2].set_xlabel("Tijd (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return GSCout_plot, SNRoutGSC, DASout, SNRoutDAS, noise_refs

# ==========================================
# 7. MICROFOONPOSITIES DEFINIEREN
# ==========================================
x_ear = d_front / 2
y_mic = d_ear   / 2

# Posities van alle 4 microfoons (gecentreerd op het hoofd)
pos_L1 = np.array([-x_ear,  y_mic])  # Linker oor, mic 1
pos_L2 = np.array([-x_ear, -y_mic])  # Linker oor, mic 2
pos_R1 = np.array([ x_ear,  y_mic])  # Rechter oor, mic 1
pos_R2 = np.array([ x_ear, -y_mic])  # Rechter oor, mic 2

# ==========================================
# 8. SIGNAALMATRICES OPBOUWEN
# ==========================================
N = len(yL1_mix)

# Micsignalen per configuratie
micsigs_right = np.column_stack((yR1_mix, yR2_mix))
micsigs_left  = np.column_stack((yL1_mix, yL2_mix))
micsigs_front = np.column_stack((yL1_mix, yR1_mix))
micsigs_all4  = np.column_stack((yL1_mix, yL2_mix, yR1_mix, yR2_mix))

# Spraak- en ruiscomponenten per configuratie
speech_right = np.column_stack((speech_R1[:N], speech_R2[:N]))
noise_right  = np.column_stack((noise_R1[:N],  noise_R2[:N]))

speech_left  = np.column_stack((speech_L1[:N], speech_L2[:N]))
noise_left   = np.column_stack((noise_L1[:N],  noise_L2[:N]))

speech_front = np.column_stack((speech_L1[:N], speech_R1[:N]))
noise_front  = np.column_stack((noise_L1[:N],  noise_R1[:N]))

speech_all4  = np.column_stack((speech_L1[:N], speech_L2[:N],
                                 speech_R1[:N], speech_R2[:N]))
noise_all4   = np.column_stack((noise_L1[:N],  noise_L2[:N],
                                 noise_R1[:N],  noise_R2[:N]))

# ==========================================
# 9. DOA SCHATTING PER CONFIGURATIE
# ==========================================
print("\n--- DOA-schatting via Wideband MUSIC ---")

# Rechter oor (Q=1: schat enkel de dominante richting)
_, _, est_R = music_wideband_2mics(micsigs_right, fs_sim, d_ear, Q=1)
target_R    = est_R[0]
print(f"Rechter oor – geschatte DOA: {target_R:.1f}° (waarde: {true_front_R}°)")

# Linker oor
_, _, est_L = music_wideband_2mics(micsigs_left, fs_sim, d_ear, Q=1)
target_L    = est_L[0]
print(f"Linker oor  – geschatte DOA: {target_L:.1f}° (waarde: {true_front_L}°)")

# Frontale mics
_, _, est_F = music_wideband_2mics(micsigs_front, fs_sim, d_front, Q=1)
target_F    = est_F[0]
print(f"Frontale mics – geschatte DOA: {target_F:.1f}°")

# 4-mic array
_, _, est_4 = music_wideband_4mics_hm(micsigs_all4, fs_sim, Q=2)
# Kies de bron dichtstbij 90° als doelbron (recht vooruit)
target_4    = est_4[np.argmin(np.abs(est_4 - 90.0))]
print(f"4-Mic array – geschatte DOAs: {est_4}, doelbron: {target_4:.1f}°")

# ==========================================
# 10. GSC UITVOEREN PER MICROFOONCONFIGUIRATIE
# ==========================================

# --- 10a: Rechter oor (yR1, yR2) ---
pos_right = np.array([pos_R1, pos_R2])
pos_right_c = pos_right - np.mean(pos_right, axis=0)

GSCout_R, SNR_R, DASout_R, SNR_DAS_R, noise_refs_R = gsc_hm(
    micsigs_right, speech_right, noise_right,
    fs_sim, pos_right_c, target_R,
    label=f"Rechter Oor (yR1, yR2) – Doelbron @ {target_R:.1f}°"
)

# --- 10b: Linker oor (yL1, yL2) ---
pos_left  = np.array([pos_L1, pos_L2])
pos_left_c = pos_left - np.mean(pos_left, axis=0)

GSCout_L, SNR_L, DASout_L, SNR_DAS_L, noise_refs_L = gsc_hm(
    micsigs_left, speech_left, noise_left,
    fs_sim, pos_left_c, target_L,
    label=f"Linker Oor (yL1, yL2) – Doelbron @ {target_L:.1f}°"
)

# --- 10c: Frontale mics (yL1, yR1) ---
pos_frontal   = np.array([pos_L1, pos_R1])
pos_frontal_c = pos_frontal - np.mean(pos_frontal, axis=0)

GSCout_F, SNR_F, DASout_F, SNR_DAS_F, noise_refs_F = gsc_hm(
    micsigs_front, speech_front, noise_front,
    fs_sim, pos_frontal_c, target_F,
    label=f"Frontale Mics (yL1, yR1) – Doelbron @ {target_F:.1f}°"
)

# --- 10d: Alle 4 microfoons ---
pos_all4   = np.array([pos_L1, pos_L2, pos_R1, pos_R2])
pos_all4_c = pos_all4 - np.mean(pos_all4, axis=0)

GSCout_4, SNR_4, DASout_4, SNR_DAS_4, noise_refs_4 = gsc_hm(
    micsigs_all4, speech_all4, noise_all4,
    fs_sim, pos_all4_c, target_4,
    label=f"4-Microfoon Array (yL1, yL2, yR1, yR2) – Doelbron @ {target_4:.1f}°"
)

# ==========================================
# 11. VERGELIJKINGSPLOT: SNR OVERZICHT
# ==========================================
labels    = ['Rechter Oor\n(yR1,yR2)', 'Linker Oor\n(yL1,yL2)',
             'Frontaal\n(yL1,yR1)', '4-Mic Array\n(alle)']
snr_das   = [SNR_DAS_R, SNR_DAS_L, SNR_DAS_F, SNR_DAS_4]
snr_gsc   = [SNR_R,     SNR_L,     SNR_F,     SNR_4]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, snr_das, width, label='DAS Beamformer', color='dodgerblue', alpha=0.8)
bars2 = ax.bar(x + width/2, snr_gsc, width, label='GSC met VAD',    color='green',      alpha=0.8)

ax.set_xlabel("Microfoonconfiguiratie")
ax.set_ylabel("Output SNR (dB)")
ax.set_title("SNR Vergelijking: DAS vs. GSC per Microfoonconfiguiratie")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, axis='y', alpha=0.4)

for bar in bars1:
    ax.annotate(f'{bar.get_height():.1f} dB',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', fontsize=9)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.1f} dB',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# ==========================================
# 12. AUDIO OPSLAAN
# ==========================================
print("\n--- Audio bestanden opslaan ---")

def save_audio(sig, path, fs):
    sig_norm = sig / (np.max(np.abs(sig)) + 1e-10)
    sf.write(path, sig_norm, fs)
    print(f"✅ Opgeslagen: {path}")

save_audio(micsigs_right[:, 0],  "Luister_Mic1_RechterOor.wav",   fs_sim)
save_audio(DASout_R,              "Luister_DAS_RechterOor.wav",    fs_sim)
save_audio(GSCout_R,              "Luister_GSC_RechterOor.wav",    fs_sim)

save_audio(micsigs_left[:, 0],   "Luister_Mic1_LinkerOor.wav",    fs_sim)
save_audio(DASout_L,              "Luister_DAS_LinkerOor.wav",     fs_sim)
save_audio(GSCout_L,              "Luister_GSC_LinkerOor.wav",     fs_sim)

save_audio(micsigs_front[:, 0],  "Luister_Mic1_Frontaal.wav",     fs_sim)
save_audio(DASout_F,              "Luister_DAS_Frontaal.wav",      fs_sim)
save_audio(GSCout_F,              "Luister_GSC_Frontaal.wav",      fs_sim)

save_audio(micsigs_all4[:, 0],   "Luister_Mic1_4Mic.wav",         fs_sim)
save_audio(DASout_4,              "Luister_DAS_4Mic.wav",          fs_sim)
save_audio(GSCout_4,              "Luister_GSC_4Mic.wav",          fs_sim)

# ==========================================
# 13. EINDOVERZICHT
# ==========================================
print("\n" + "="*55)
print("EINDOVERZICHT: SNR PER CONFIGURATIE")
print("="*55)
print(f"{'Configuratie':<30} {'DAS SNR':>10} {'GSC SNR':>10} {'Verbetering':>12}")
print("-"*55)
configs = [("Rechter Oor (yR1,yR2)",       SNR_DAS_R, SNR_R),
           ("Linker Oor (yL1,yL2)",         SNR_DAS_L, SNR_L),
           ("Frontaal (yL1,yR1)",           SNR_DAS_F, SNR_F),
           ("4-Mic Array (alle 4)",         SNR_DAS_4, SNR_4)]
for name, das_snr, gsc_snr in configs:
    print(f"{name:<30} {das_snr:>9.1f}° {gsc_snr:>9.1f}° {gsc_snr - das_snr:>+11.1f} dB")
print("="*55)
