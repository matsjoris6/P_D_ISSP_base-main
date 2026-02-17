import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
from scipy.signal import find_peaks
import sounddevice as sd
import os

# ==========================================
# 1. FUNCTIES VOOR LADEN & MIXEN
# ==========================================
def load_hmirs_for_angle(base_folder, angle_name):
    """Laadt de 4 wav bestanden voor een specifieke hoek."""
    folder_path = os.path.join(base_folder, angle_name)
    if not os.path.exists(folder_path):
        print(f"Let op: Map '{angle_name}' niet gevonden in {base_folder}")
        return None

    # Volgorde: L1, L2, R1, R2
    filenames = ["HMIR_L1.wav", "HMIR_L2.wav", "HMIR_R1.wav", "HMIR_R2.wav"]
    rirs = []
    for f in filenames:
        data, fs = sf.read(os.path.join(folder_path, f))
        rirs.append(data)
    
    # Maak er een matrix van (Lengte x 4)
    min_len = min([len(r) for r in rirs])
    rir_matrix = np.zeros((min_len, 4))
    for i in range(4):
        rir_matrix[:, i] = rirs[i][:min_len]
        
    return rir_matrix, fs

def create_mixed_scene(speech1, rirs1, speech2, rirs2):
    """Convolueer spraak met RIRs en tel ze bij elkaar op."""
    # Lengte bepalen
    len1 = len(speech1) + len(rirs1) - 1
    len2 = len(speech2) + len(rirs2) - 1
    total_len = max(len1, len2)
    
    # 4 kanalen output
    mixed_sig = np.zeros((total_len, 4))
    
    print("Bezig met convolutie en mixen (even geduld)...")
    for mic in range(4):
        # Bron 1 (Links)
        s1_filtered = signal.fftconvolve(speech1, rirs1[:, mic], mode='full')
        # Bron 2 (Rechts)
        s2_filtered = signal.fftconvolve(speech2, rirs2[:, mic], mode='full')
        
        # Optellen (padding indien nodig)
        L = min(len(s1_filtered), total_len)
        mixed_sig[:L, mic] += s1_filtered[:L]
        
        L = min(len(s2_filtered), total_len)
        mixed_sig[:L, mic] += s2_filtered[:L]
        
    return mixed_sig

# ==========================================
# 2. DOAs SCHATTEN (MAX-HOLD + PRE-WHITENING)
# ==========================================
def estimate_doa_frontal(mixed_sig, fs):
    """
    Schat DOAs gebruikmakend van het frontale paar (Mic 0 en Mic 2).
    Gebruikt Pre-whitening en Max-Hold om 2 bronnen te vinden.
    """
    # Frontaal paar: L1 (index 0) en R1 (index 2)
    sigL = mixed_sig[:, 0]
    sigR = mixed_sig[:, 2]
    
    d = 0.215 # 21.5 cm afstand
    c = 343.0
    
    # 1. Pre-Whitening (belangrijk voor spraak!)
    b, a = [1.0, -0.95], [1.0]
    sigL = signal.lfilter(b, a, sigL)
    sigR = signal.lfilter(b, a, sigR)
    
    # 2. Max-Hold Correlatie
    segment_time = 0.5 # sec
    N = int(segment_time * fs)
    step = int(N / 2)
    lags = np.arange(-N + 1, N)
    max_curve = np.zeros(len(lags))
    
    print("Analyseren van DOA's...")
    for t in range(0, len(sigL) - N, step):
        segL = sigL[t:t+N]
        segR = sigR[t:t+N]
        
        if np.sum(segL**2) < 1e-4: continue # Stilte overslaan
        
        # Correlatie: R vs L
        corr = signal.correlate(segR, segL, mode='full')
        if np.max(np.abs(corr)) > 0:
            corr /= np.max(np.abs(corr))
            
        max_curve = np.maximum(max_curve, np.abs(corr))
        
    # 3. Piek Detectie (Zoek de 2 hoogste)
    # Fysieke limiet berekenen
    max_lag_samples = int(np.ceil((d/c)*fs)) + 2
    
    peaks, props = find_peaks(max_curve, height=0.3, distance=20)
    
    # Filteren op limiet
    valid_peaks = [p for p in peaks if abs(lags[p]) <= max_lag_samples]
    # Sorteren op hoogte
    valid_peaks.sort(key=lambda x: max_curve[x], reverse=True)
    top_2 = valid_peaks[:2]
    
    # 4. Plotten
    plt.figure(figsize=(10, 5))
    plt.plot(lags, max_curve, color='purple', label='Max-Hold Correlatie')
    plt.axvspan(-max_lag_samples, max_lag_samples, color='green', alpha=0.1, label='Fysieke Limiet')
    plt.title(f"DOA Schatting (Frontaal Paar L1-R1) - {len(top_2)} Bronnen Gevonden")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Norm. Correlatie")
    plt.xlim([-max_lag_samples*2, max_lag_samples*2])
    plt.grid(True, linestyle='--')
    
    print("\n--- Resultaten ---")
    for p in top_2:
        lag = lags[p]
        # Hoek berekenen
        # Let op: Lag > 0 betekent dat R later is dan L -> Bron is Links
        # Formule: tau = lag / fs. cos(theta) = (tau * c) / d
        tau = -lag / fs 
        arg = np.clip((tau * c) / d, -1.0, 1.0)
        deg = np.degrees(np.arccos(arg))
        
        print(f"Piek gevonden op Lag {lag}: {deg:.1f}°")
        plt.plot(lag, max_curve[p], 'ro')
        plt.text(lag, max_curve[p] + 0.05, f"{deg:.0f}°", color='red', fontweight='bold', ha='center')

    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    # PADEN AANPASSEN AAN JOUW SYSTEEM
    base_folder = "sound_files (1)" 
    if not os.path.exists(base_folder): base_folder = "sound_files"
    
    rir_folder = os.path.join(base_folder, "head_mounted_rirs")
    
    # 1. Laad Audio
    file1 = os.path.join(base_folder, "part1_track1_dry.wav") # Bron Links
    file2 = os.path.join(base_folder, "part1_track2_dry.wav") # Bron Rechts
    
    # (Fallback naamgeving checken)
    if not os.path.exists(file1): file1 = os.path.join(base_folder, "part1 track1 dry.wav")
    if not os.path.exists(file2): file2 = os.path.join(base_folder, "part1 track2 dry.wav")
    
    s1, fs = sf.read(file1); s1 = s1[:,0] if s1.ndim > 1 else s1
    s2, fs = sf.read(file2); s2 = s2[:,0] if s2.ndim > 1 else s2
    
    # Zorg dat ze even lang zijn (voor het mixen makkelijk te maken)
    min_len = min(len(s1), len(s2))
    s1 = s1[:min_len]
    s2 = s2[:min_len]

    # 2. Laad RIRs (Links op -60, Rechts op +60)
    # Check je mapnamen: s-60 is links, s60 is rechts
    print("Laden van RIRs...")
    rirs_left, _ = load_hmirs_for_angle(rir_folder, "s-90") 
    rirs_right, _ = load_hmirs_for_angle(rir_folder, "s30")

    if rirs_left is not None and rirs_right is not None:
        # 3. Mixen
        # Track 1 komt van Links (-60), Track 2 komt van Rechts (+60)
        mixed_mics = create_mixed_scene(s1, rirs_left, s2, rirs_right)
        
        # 4. Binaural Luisteren (L1 en R1)
        # Genormaliseerd afspelen
        binaural_sig = np.column_stack((mixed_mics[:, 0], mixed_mics[:, 2]))
        max_val = np.max(np.abs(binaural_sig))
        if max_val > 0: binaural_sig /= max_val
        
        print("\n--- Audio Afspelen (Binaural) ---")
        print("Je hoort nu Track 1 Links en Track 2 Rechts tegelijk.")
        try:
            sd.play(binaural_sig, fs)
            # sd.wait() # Uncomment als je wil wachten tot het klaar is
        except:
            print("Kon niet afspelen, check je audio device.")
            
        # Sla op voor zekerheid
        sf.write("binaural_mix_output.wav", binaural_sig, fs)

        # 5. DOA Schatten
        estimate_doa_frontal(mixed_mics, fs)