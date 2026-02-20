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
d_ear = 0.013    
d_front = 0.215  

folder_left = "s-90"
folder_right = "s90"


angle_L = abs(float(folder_left.replace("s", "")))  
angle_R = float(folder_right.replace("s", ""))      

true_front_R =  angle_R
true_front_L = angle_L

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
    micsigs_front = np.column_stack((yL1_mix, yR1_mix)) 

except Exception as e:
    print(f"Fout bij het laden van audio/RIRs: {e}")
    exit()


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
       #if freqs[k] > 2000.0:
           #continue
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


micsigs_all_4 = np.column_stack((yL1_mix, yL2_mix, yR1_mix, yR2_mix))


ang_4, spec_4, est_4 = MUSIC_wideband_HM(micsigs_all_4, fs_sim, Q=2)

plt.figure(figsize=(10, 6))


plt.plot(ang_4, spec_4, color='purple', linewidth=2.5, label='4-Mic Pseudospectrum $\\bar{p}(\\theta)$')


plt.axvline(true_front_R, color='green', linestyle='--', linewidth=1.5, label=f'Waarheid Rechts: {true_front_R}°')
plt.axvline(true_front_L, color='teal', linestyle='--', linewidth=1.5, label=f'Waarheid Links: {true_front_L}°')


colors_est = ['red', 'orange']
for i, e_doa in enumerate(est_4):

    plt.axvline(e_doa, color=colors_est[i % 2], linestyle=':', linewidth=2, label=f'Schatting {i+1}: {e_doa:.1f}°')
    
 
    peak_idx = np.argmin(np.abs(ang_4 - e_doa))
    plt.plot(e_doa, spec_4[peak_idx], marker="x", color=colors_est[i % 2], markersize=10, markeredgewidth=3)

plt.title("Wideband MUSIC: 4-Mic Head-Mounted Array (2 Bronnen)", fontsize=14)
plt.xlabel("Hoek (graden) [90° = Recht vooruit]", fontsize=12)
plt.ylabel("Magnitude (dB)", fontsize=12)
plt.xlim(0, 180)
plt.ylim(np.min(spec_4) - 5, 5) 
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()


plt.show()


print("\n" + "="*50)
print("VERGELIJKING: 4-MICROFOON ARRAY")
print("="*50)
print(f"Ware hoeken:        {true_front_R:.1f}° en {true_front_L:.1f}°")

if len(est_4) == 2:
    print(f"Geschatte hoeken:   {est_4[0]:.1f}° en {est_4[1]:.1f}°")
    
  
    ware_hoeken = np.sort([true_front_R, true_front_L])
    geschatte_hoeken = np.sort(est_4)
    fouten = np.abs(ware_hoeken - geschatte_hoeken)
    
    print("-" * 50)
    print(f"Foutmarge Bron 1:   {fouten[0]:.1f}°")
    print(f"Foutmarge Bron 2:   {fouten[1]:.1f}°")
else:
    print(f"Let op: Het algoritme heeft slechts {len(est_4)} piek(en) gevonden: {est_4}")
print("="*50)