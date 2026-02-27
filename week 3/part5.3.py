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

folder_left = "s-60"
folder_right = "s60"
angle_L = abs(float(folder_left.replace("s", "")))  
angle_R = float(folder_right.replace("s", ""))      

true_front_R = 90.0 - angle_R # Ruis
true_front_L = 90.0 + angle_L # Target spraak


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

    # Bron 1 (Links) is TARGET SPRAAK
    yL1_from_left = signal.fftconvolve(dry_sig_1, yL1_rir_left, mode='full')
    yL2_from_left = signal.fftconvolve(dry_sig_1, yL2_rir_left, mode='full')
    yR1_from_left = signal.fftconvolve(dry_sig_1, yR1_rir_left, mode='full')
    yR2_from_left = signal.fftconvolve(dry_sig_1, yR2_rir_left, mode='full')

    # Bron 2 (Rechts) is RUIS
    yL1_from_right = signal.fftconvolve(dry_sig_2, yL1_rir_right, mode='full')
    yL2_from_right = signal.fftconvolve(dry_sig_2, yL2_rir_right, mode='full')
    yR1_from_right = signal.fftconvolve(dry_sig_2, yR1_rir_right, mode='full')
    yR2_from_right = signal.fftconvolve(dry_sig_2, yR2_rir_right, mode='full')

    # De microfoon signalen (Mix)
    yL1_mix = yL1_from_left + yL1_from_right
    yL2_mix = yL2_from_left + yL2_from_right
    yR1_mix = yR1_from_left + yR1_from_right
    yR2_mix = yR2_from_left + yR2_from_right

except Exception as e:
    print(f"Fout bij het laden van audio/RIRs: {e}")
    exit()


x_ear = 0.215 / 2  
y_mic = 0.013 / 2  

pos_L1 = [-x_ear,  y_mic]
pos_L2 = [-x_ear, -y_mic]
pos_R1 = [ x_ear,  y_mic]
pos_R2 = [ x_ear, -y_mic]


arrays = {
    "Linkeroor (yL1, yL2)": {
        "mics": np.column_stack((yL1_mix, yL2_mix)),
        "pos": np.array([pos_L1, pos_L2])
    },
    "Rechteroor (yR1, yR2)": {
        "mics": np.column_stack((yR1_mix, yR2_mix)),
        "pos": np.array([pos_R1, pos_R2])
    },
    "Frontaal (yL1, yR1)": {
        "mics": np.column_stack((yL1_mix, yR1_mix)),
        "pos": np.array([pos_L1, pos_R1])
    },
    "Alle 4 Microfoons": {
        "mics": np.column_stack((yL1_mix, yL2_mix, yR1_mix, yR2_mix)),
        "pos": np.array([pos_L1, pos_L2, pos_R1, pos_R2])
    }
}

# Ideale VAD baseren op schone target spraak op linker hoofd-microfoon
vad = np.abs(yL1_from_left) > (np.std(yL1_from_left) * 1e-3)


def gsc_head_mounted(micsigs, mics_pos, target_doa, fs, vad):
    N_samples, M_mics = micsigs.shape
    

    target_rad = np.radians(target_doa)
    mics_centered = mics_pos - np.mean(mics_pos, axis=0)
   
    taus = (mics_centered[:, 0] * np.cos(target_rad) + mics_centered[:, 1] * np.sin(target_rad)) / c
    
    aligned_mic = np.zeros_like(micsigs)
    DASout = np.zeros(N_samples)
    f = np.fft.rfftfreq(N_samples, 1/fs)
    
    for m in range(M_mics):
        Sig = np.fft.rfft(micsigs[:, m])
        Sig_aligned = Sig * np.exp(1j * 2 * np.pi * f * taus[m])
        t_aligned = np.fft.irfft(Sig_aligned, n=N_samples)
        aligned_mic[:, m] = t_aligned
        DASout += t_aligned
    DASout /= M_mics

    B = np.zeros((M_mics - 1, M_mics))
    for i in range(M_mics - 1):
        B[i, i] = 1
        B[i, i + 1] = -1
    noise_refs = (B @ aligned_mic.T).T
    
   
    L = 1024 
    mu = 0.1
    delta = L // 2
    
    target_ref = np.pad(DASout, (delta, 0))[:N_samples]
    vad_delayed = np.pad(vad, (delta, 0))[:N_samples]
    padded_noise = np.pad(noise_refs, ((L-1, 0), (0, 0)))
    
    W = np.zeros((L, M_mics - 1))
    GSCout = np.zeros(N_samples)
    eps = 1e-8
    
    for n in range(N_samples):
        X_slice = padded_noise[n : n+L]
        y_n = np.sum(W * X_slice)
        e_n = target_ref[n] - y_n
        GSCout[n] = e_n
        
        if vad_delayed[n] == 0:
            power = np.sum(X_slice**2)
            W += (mu * e_n / (power + eps)) * X_slice
            
    #  SNR Berekening 
    Pn = np.var(GSCout[vad_delayed == 0])
    Psn = np.var(GSCout[vad_delayed == 1])
    Ps = Psn - Pn
    if Ps < 1e-10: Ps = 1e-10
    if Pn < 1e-10: Pn = 1e-10
    SNRout = 10 * np.log10(Ps / Pn)
    
    # SNR Input ter vergelijking (Mic 1 van deze specifieke array)
    Pn_in = np.var(micsigs[vad == 0, 0])
    Psn_in = np.var(micsigs[vad == 1, 0])
    Ps_in = Psn_in - Pn_in
    if Ps_in < 1e-10: Ps_in = 1e-10
    if Pn_in < 1e-10: Pn_in = 1e-10
    SNRin = 10 * np.log10(Ps_in / Pn_in)
    
    return GSCout, SNRout, SNRin, noise_refs


print(f"\nTarget DOA ingesteld op: {true_front_L}° (Linker Bron)")
print("-" * 50)

results = {}
t = np.arange(len(yL1_mix)) / fs_sim

plt.figure(figsize=(14, 8))

for i, (name, data) in enumerate(arrays.items()):
    
    GSCout, SNRout, SNRin, noise_refs = gsc_head_mounted(
        micsigs=data["mics"], 
        mics_pos=data["pos"], 
        target_doa=true_front_L, 
        fs=fs_sim, 
        vad=vad
    )
    
    results[name] = {"out": GSCout, "snr": SNRout, "snr_in": SNRin}
    print(f"  -> Input SNR:  {SNRin:.2f} dB | Output SNR: {SNRout:.2f} dB")
    
    # 2. AUDIO EXPORT (Direct in de lus)
    clean_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    
    # GSC Resultaat (Het gefilterde geluid)
    out_norm = GSCout / (np.max(np.abs(GSCout)) + 1e-9)
    sf.write(f"Resultaat_GSC_{clean_name}.wav", out_norm, fs_sim)
    
    # CHECK 3: Lekkage (Wat gaat er het filter in?)
    # We pakken de eerste ruisreferentie uit de Blocking Matrix
    lekkage = noise_refs[:, 0]
    lekkage_norm = lekkage / (np.max(np.abs(lekkage)) + 1e-9)
    sf.write(f"Check3_Lekkage_{clean_name}.wav", lekkage_norm, fs_sim)
    
    # 3. PLOTTEN
    plt.subplot(4, 1, i+1)
    plt.plot(t, data["mics"][:, 0], color='gray', alpha=0.5, label='Input (Mic 1)')
    plt.plot(t, GSCout, color='green' if '4' in name else 'blue', alpha=0.8, label=f'GSC Output')
    plt.title(f"{name} | Netto: {(SNRout - SNRin):.2f} dB")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

# Referentie opslaan
ref_mic = yL1_mix / (np.max(np.abs(yL1_mix)) + 1e-9)
sf.write("Referentie_MicL1.wav", ref_mic, fs_sim)

plt.xlabel("Tijd (s)")
plt.tight_layout()
plt.show()

