import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
from scipy.signal import find_peaks
import os
from package import load_rirs, select_latest_rir, check_plot_tdoas

# --- NIEUWE FUNCTIE: ANALYSEER AUTOCORRELATIE ---
def analyze_autocorrelation(speech_file, noise_file, fs=44100):
    print("\n--- Start Autocorrelatie Analyse ---")
    
    # 1. Laad signalen (kort stukje is genoeg, bv 1 seconde)
    sig_speech, _ = sf.read(speech_file)
    if len(sig_speech.shape) > 1: sig_speech = sig_speech[:, 0]
    
    # Als er geen ruisbestand is, genereren we zelf ruis ter vergelijking
    if noise_file and os.path.exists(noise_file):
        sig_noise, _ = sf.read(noise_file)
        if len(sig_noise.shape) > 1: sig_noise = sig_noise[:, 0]
    else:
        sig_noise = np.random.randn(len(sig_speech))

    # Pak een fragment van 2048 samples (typische frame grootte)
    N = 2048
    start = int(len(sig_speech)/2) # Midden in het bestand
    fragment_speech = sig_speech[start:start+N]
    fragment_noise = sig_noise[0:N] # Ruis is overal hetzelfde
    
    # 2. Pre-Whitening Filter toepassen op spraak
    b, a = [1.0, -0.95], [1.0]
    fragment_speech_white = signal.lfilter(b, a, fragment_speech)

    # 3. Bereken Autocorrelaties
    # (Mode='full' geeft lengte 2N-1, piek zit in het midden)
    auto_noise = signal.correlate(fragment_noise, fragment_noise, mode='full')
    auto_speech = signal.correlate(fragment_speech, fragment_speech, mode='full')
    auto_speech_white = signal.correlate(fragment_speech_white, fragment_speech_white, mode='full')

    # Normaliseren naar 1.0 voor eerlijke vergelijking
    auto_noise /= np.max(np.abs(auto_noise))
    auto_speech /= np.max(np.abs(auto_speech))
    auto_speech_white /= np.max(np.abs(auto_speech_white))

    # Lags as voor plot (in samples)
    lags = np.arange(-N + 1, N)

    # 4. Plotten
    plt.figure(figsize=(10, 8))
    
    # Plot 1: Witte Ruis (De Referentie)
    plt.subplot(3, 1, 1)
    plt.plot(lags, auto_noise, color='grey')
    plt.title("Autocorrelatie: Witte Ruis (Ideaal)")
    plt.ylabel("Correlatie")
    plt.grid(True)
    plt.xlim([-100, 100]) # Zoom in op het centrum
    plt.text(50, 0.8, "Scherpe Piek!", color='green', fontweight='bold')

    # Plot 2: Spraak Origineel (Het Probleem)
    plt.subplot(3, 1, 2)
    plt.plot(lags, auto_speech, color='red')
    plt.title("Autocorrelatie: Spraak (Ongefilterd)")
    plt.ylabel("Correlatie")
    plt.grid(True)
    plt.xlim([-100, 100])
    plt.text(50, 0.8, "Brede Heuvel...", color='red', fontweight='bold')

    # Plot 3: Spraak met Pre-Whitening (De Oplossing)
    plt.subplot(3, 1, 3)
    plt.plot(lags, auto_speech_white, color='blue')
    plt.title("Autocorrelatie: Spraak (Met Pre-Whitening Filter)")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlatie")
    plt.grid(True)
    plt.xlim([-100, 100])
    plt.text(50, 0.8, "Weer Scherp!", color='blue', fontweight='bold')

    plt.tight_layout()
    plt.show()

# --- BESTAANDE FUNCTIES (Ongewijzigd gelaten voor werking) ---

def create_micsigs(acoustic_scenario, speech_filenames, noise_filenames, duration):
    fs_rir = acoustic_scenario.fs
    if acoustic_scenario.RIRs_audio is None: return None, None, None

    num_mics = acoustic_scenario.RIRs_audio.shape[1]
    num_sources = acoustic_scenario.RIRs_audio.shape[2]
    num_samples = int(duration * fs_rir)
    
    files = []
    while len(files) < num_sources: files.extend(speech_filenames)
    files = files[:num_sources]

    speech_sig = np.zeros((num_samples, num_mics))
    
    for src_idx, filename in enumerate(files):
        if not os.path.exists(filename): continue
        audio, fs_src = sf.read(filename)
        if len(audio.shape) > 1: audio = audio[:, 0]
        if np.max(np.abs(audio)) > 0: audio = audio / np.max(np.abs(audio))
        if fs_src != fs_rir: audio = signal.resample(audio, int(len(audio) * fs_rir / fs_src))
        
        for mic_idx in range(num_mics):
            rir = acoustic_scenario.RIRs_audio[:, mic_idx, src_idx]
            filt = signal.fftconvolve(audio, rpe[1]
    target_peaks = force_num_peaks if force_num_peaks else acoustic_scenario.RIRs_audio.shape[2]

    pos = acoustic_scenario.micPos
    if pos.shape[0] == 3: d = np.sqrt(np.sum((pos[:, 0] - pos[:, 1])**2))
    else: d = np.sqrt(np.sum((pos[0] - pos[1])**2))
    limit_samples = int(np.ceil((d / c) * fs)) + 2 

    mics, _, _ = create_micsigs(acoustic_scenariir, mode='full')[:num_samples]
            speech_sig[:, mic_idx] += filt

    return speech_sig, None, None

def TDOA_corr(acoustic_scenario, noise_filenames, segment_duration, force_num_peaks=None):
    fs = acoustic_scenario.fs
    c = getattr(acoustic_scenario, 'c', 343.0)
    num_mics = acoustic_scenario.RIRs_audio.shape[1]
    target_peaks = force_num_peaks if force_num_peaks else acoustic_scenario.RIRs_audio.shape[2]

    pos = acoustic_scenario.micPos
    if pos.shape[0] == 3: d = np.sqrt(np.sum((pos[:, 0] - pos[:, 1])**2))
    else: d = np.sqrt(np.sum((pos[0] - pos[1])**2))
    limit_samples = int(np.ceil((d / c) * fs)) + 2 

    mics, _, _ = create_micsigs(acoustic_scenario, noise_filenames, [], duration=5.0)
    if mics is None: return

    # Pre-Whitening
    b, a = [1.0, -0.95], [1.0]
    mics = signal.lfilter(b, a, mics, axis=0)

    N = int(segment_duration * fs)
    lags = np.arange(-N + 1, N)
    step = int(N / 2)
    
    fig, axes = plt.subplots(num_mics - 1, 1, figsize=(8, 2 * (num_mics-1)), sharex=True)
    if num_mics - 1 == 1: axes = [axes]

    print(f"\n--- TDOA Analyse (Max Lag: {limit_samples} samples) ---")

    for i in range(num_mics - 1):
        sigA, sigB = mics[:, i], mics[:, i+1]
        max_curve = np.zeros(len(lags))

        for t in range(0, len(sigA) - N, step):
            segA, segB = sigA[t:t+N], sigB[t:t+N]
            if np.sum(segA**2) < 1e-4: continue 
            corr = signal.correlate(segB, segA, mode='full')
            if np.max(np.abs(corr)) > 0: corr /= np.max(np.abs(corr))
            max_curve = np.maximum(max_curve, np.abs(corr))

        peaks, props = find_peaks(max_curve, height=0.3, distance=10)
        valid_peaks = [p for p in peaks if abs(lags[p]) <= limit_samples]
        valid_peaks.sort(key=lambda x: max_curve[x], reverse=True)
        top_peaks = valid_peaks[:target_peaks]

        ax = axes[i]
        ax.plot(lags, max_curve, color='purple')
        ax.axvspan(-limit_samples, limit_samples, color='green', alpha=0.1)
        ax.set_title(f"Mic {i+1}-{i+2}")
        ax.set_xlim([-limit_samples*3, limit_samples*3])
        ax.grid(True, linestyle='--')

        for p in top_peaks:
            lag = lags[p]
            deg = np.degrees(np.arccos(np.clip((-lag/fs * c) / d, -1, 1)))
            ax.plot(lag, max_curve[p], 'ro')
            ax.text(lag, max_curve[p], f"{deg:.0f}°", color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()

# --- MAIN ---
if __name__== "__main__":
    try:
        rir_file = select_latest_rir("rirs")
        scenario = load_rirs(rir_file)
        
        folder = "sound_files (1)" if os.path.exists("sound_files (1)") else "sound_files"
        speech_file = os.path.join(folder, "part1_track1_dry.wav")
        noise_file = os.path.join(folder, "whitenoise_signal_1.wav")
        
        # Fallbacks voor bestandsnamen
        if not os.path.exists(speech_file): speech_file = os.path.join(folder, "part1 track1 dry.wav")
        if not os.path.exists(noise_file): noise_file = os.path.join(folder, "whitenoise sig 1.wav")
        
        files = [speech_file, os.path.join(folder, "part1_track2_dry.wav")]
        if not os.path.exists(files[1]): files[1] = os.path.join(folder, "part1 track2 dry.wav")

        # STAP 1: Laat eerst de autocorrelatie analyse zien (zoals de prof vroeg)
        analyze_autocorrelation(speech_file, noise_file)

        # STAP 2: Doe daarna de normale TDOA analyse
        TDOA_corr(scenario, files, segment_duration=0.5, force_num_peaks=2)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()