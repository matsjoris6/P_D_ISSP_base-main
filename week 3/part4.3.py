import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
import os
import sys

# ==========================================
# 0. SETUP & IMPORTS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from package import load_rirs
from package.utils import create_micsigs as original_create_micsigs
from package.utils import music_wideband

# ==========================================
# 1. FUNCTIE: GEMODIFICEERDE CREATE_MICSIGS
# ==========================================
def modified_create_micsigs(scenario, speech_paths, noise_paths=None, duration=10.0):
    if noise_paths is None:
        noise_paths = []

    if scenario.fs != 44100:
        raise ValueError(f"Fout: De samplingfrequentie is {scenario.fs} Hz. Dit moet 44.1 kHz zijn!")

    # Tijdelijk de terminal-output blokkeren om de verwarrende 'INFO' waarschuwing te verbergen
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        _, speech, _ = original_create_micsigs(scenario, speech_paths, [], duration=duration)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout  # Terminal output weer aanzetten
    
    # Ruis inladen met RIRs
    if len(noise_paths) > 0:
        _, _, gui_noise = original_create_micsigs(scenario, speech_paths, noise_paths, duration=duration)
    else:
        gui_noise = np.zeros_like(speech)

    vad = np.abs(speech[:, 0]) > (np.std(speech[:, 0]) * 1e-3)
    Ps = np.var(speech[vad == 1, 0])
    
    # We gebruiken uitsluitend de gelokaliseerde ruis uit de GUI
    noise = gui_noise
    mic = speech + noise
    
    Pn = np.var(noise[:, 0])
    if Pn < 1e-10: Pn = 1e-10 
    SNR_in = 10 * np.log10(Ps / Pn)
    
    print(f"[Info] Input SNR (Microfoon 1): {SNR_in:.2f} dB")
    
    return mic, speech, noise, SNR_in, vad

# ==========================================
# 2. FUNCTIE: DAS BEAMFORMER 
# ==========================================
def das_bf(scenario, speech_paths, noise_paths=None, duration=10.0):
    if noise_paths is None:
        noise_paths = []

    mic, speech, noise, snr_in, vad = modified_create_micsigs(scenario, speech_paths, noise_paths, duration)

    num_audio = scenario.audioPos.shape[0] if scenario.audioPos is not None else 0
    num_noise = scenario.noisePos.shape[0] if (hasattr(scenario, 'noisePos') and scenario.noisePos is not None) else 0
    Q = num_audio + num_noise
    if Q == 0: Q = 1
    
    # MUSIC oproepen
    angles, ps_ind_db, spectrum_geom_db, est_doas = music_wideband(mic, scenario.fs, scenario)

    # Kies de DOA die het dichtst bij 90 graden ligt
    target_doa = est_doas[np.argmin(np.abs(est_doas - 90.0))]
    print(f"[DAS BF] Target bron gedetecteerd door MUSIC op: {target_doa:.1f}°")

    target_rad = np.radians(target_doa)
    c = 343.0
    mics_centered = scenario.micPos - np.mean(scenario.micPos, axis=0)
    taus = (mics_centered[:, 0] * np.sin(target_rad) + mics_centered[:, 1] * np.cos(target_rad)) / c

    def apply_das(signals, delays, fs):
        M = signals.shape[1]
        N = signals.shape[0]
        out_sum = np.zeros(N)
        out_aligned = np.zeros((N, M))
        f = np.fft.rfftfreq(N, 1/fs)
        for m in range(M):
            Sig = np.fft.rfft(signals[:, m])
            Sig_aligned = Sig * np.exp(1j * 2 * np.pi * f * delays[m])
            t_aligned = np.fft.irfft(Sig_aligned, n=N)
            out_aligned[:, m] = t_aligned
            out_sum += t_aligned
        return out_sum / M, out_aligned

    speechDAS, speech_aligned = apply_das(speech, taus, scenario.fs)
    noiseDAS, noise_aligned = apply_das(noise, taus, scenario.fs)
    
    DASout = speechDAS + noiseDAS
    aligned_mic = speech_aligned + noise_aligned  

    P_speech_das = np.var(speechDAS[vad == 1])
    P_noise_das = np.var(noiseDAS)
    if P_noise_das < 1e-10: P_noise_das = 1e-10
    SNRoutDAS = 10 * np.log10(P_speech_das / P_noise_das)
    print(f"[DAS BF] Output SNR: {SNRoutDAS:.2f} dB")
    
    return DASout, speechDAS, noiseDAS, SNRoutDAS, mic, aligned_mic, vad, snr_in

# ==========================================
# 3. FUNCTIE: GSC IN TIME DOMAIN (NLMS)
# ==========================================
def gsc_td(scenario, speech_paths, noise_paths=None, duration=10.0):
    print("\n" + "="*50)
    print("START GENERALIZED SIDELOBE CANCELER (GSC)")
    print("="*50)

    DASout, speechDAS, noiseDAS, SNRoutDAS, mic, aligned_mic, vad, snr_in = das_bf(
        scenario, speech_paths, noise_paths, duration=duration
    )

    N_samples, M_mics = aligned_mic.shape

    print("[GSC] Genereren van Noise References (Blocking Matrix B)...")
    B = np.zeros((M_mics - 1, M_mics))
    for i in range(M_mics - 1):
        B[i, 0] = 1
        B[i, i + 1] = -1
        
    noise_refs = (B @ aligned_mic.T).T  

    L = 1024
    mu = 0.15
    delta = L // 2

    target_ref = np.pad(DASout, (delta, 0))[:N_samples]
    
    W = np.zeros((L, M_mics - 1))
    GSCout = np.zeros(N_samples)
    
    print(f"[GSC] Start NLMS adaptieve filtering (L={L}, mu={mu}, delta={delta})...")
    print("[GSC] ⚠️ Adaptatie pauzeert automatisch tijdens spraak (Target Cancellation fix)")

    padded_noise = np.pad(noise_refs, ((L-1, 0), (0, 0)))
    eps = 1e-8
    vad_delayed = np.pad(vad, (delta, 0))[:N_samples]

    for n in range(N_samples):
        X_slice = padded_noise[n : n+L]
        
        y_n = np.sum(W * X_slice)
        e_n = target_ref[n] - y_n
        GSCout[n] = e_n
        
        # Filter updatet alleen als de VAD aangeeft dat er GEEN spraak is
        if vad_delayed[n] == 0:
            power = np.sum(X_slice**2)
            W += (mu * e_n / (power + eps)) * X_slice

    print("[GSC] Filtering voltooid!")

    Pn_gsc = np.var(GSCout[vad_delayed == 0])
    Psn_gsc = np.var(GSCout[vad_delayed == 1])
    Ps_gsc = Psn_gsc - Pn_gsc
    if Ps_gsc < 0: Ps_gsc = 1e-10 
    if Pn_gsc < 1e-10: Pn_gsc = 1e-10
    
    SNRoutGSC = 10 * np.log10(Ps_gsc / Pn_gsc)
    print(f"[GSC] Output SNR: {SNRoutGSC:.2f} dB (Netto Verbetering vs Mic1: {SNRoutGSC - snr_in:.2f} dB)")

    # Alles plotten in 1 mooie, transparante grafiek
    t = np.arange(N_samples) / scenario.fs
    GSCout_plot = np.pad(GSCout[delta:], (0, delta))
    
    plt.figure(figsize=(14, 7))
    plt.plot(t, mic[:, 0], color='gray', alpha=0.5, label=f'Microfoon 1 (Input SNR: {snr_in:.1f} dB)')
    plt.plot(t, DASout, color='dodgerblue', alpha=0.6, label=f'DAS Beamformer (SNR: {SNRoutDAS:.1f} dB)')
    plt.plot(t, GSCout_plot, color='green', alpha=0.9, linewidth=1.2, label=f'GSC Output (SNR: {SNRoutGSC:.1f} dB)')
    
    plt.xlabel("Tijd (s)")
    plt.ylabel("Amplitude")
    plt.title("Signaalvergelijking: Origineel vs. DAS-BF vs. GSC")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    return GSCout, SNRoutGSC, mic, DASout, GSCout_plot, noise_refs

# ==========================================
# 4. MAIN PROGRAMMA
# ==========================================
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
        gekozen_ruisbestand = "speech1.wav" 
        ruis_pad = os.path.join(parent_dir, "sound_files", gekozen_ruisbestand)
        alle_ruis_bestanden = [ruis_pad]
        noise_paths = alle_ruis_bestanden[:num_noise]

        # Voer de volledige GSC uit
        GSCout, SNRoutGSC, mic, DASout, GSCout_plot, noise_refs = gsc_td(scenario, speech_paths, noise_paths, duration=10.0)

        # --- AUDIO OPSLAAN ---
        print("\n--- AUDIO BESTANDEN OPSLAAN ---")
        mic_audio = mic[:, 0] / np.max(np.abs(mic[:, 0]))
        path_mic = os.path.join(parent_dir, "Luister_01_Origineel_Mic1.wav")
        sf.write(path_mic, mic_audio, scenario.fs)
        
        das_audio = DASout / np.max(np.abs(DASout))
        path_das = os.path.join(parent_dir, "Luister_02_Verbeterd_DAS.wav")
        sf.write(path_das, das_audio, scenario.fs)
        
        gsc_audio = GSCout_plot / np.max(np.abs(GSCout_plot))
        path_gsc = os.path.join(parent_dir, "Luister_03_GSC_Filter.wav")
        sf.write(path_gsc, gsc_audio, scenario.fs)
        
        bm_audio = noise_refs[:, 0] / np.max(np.abs(noise_refs[:, 0]))
        path_bm = os.path.join(parent_dir, "Luister_04_BlockingMatrix.wav")
        sf.write(path_bm, bm_audio, scenario.fs)
        
        print("✅ Alle 4 de audiobestanden zijn succesvol opgeslagen in de map!")

    except Exception as e:
        import traceback
        print(f"\n❌ Fout in uitvoering: {e}")
        traceback.print_exc()