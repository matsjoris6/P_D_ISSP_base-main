import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from package import load_rirs
from package.utils import music_wideband
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



def das_bf(scenario, speech_paths, noise_paths=None, duration=10.0, show_plot=False):
    if noise_paths is None:
        noise_paths = []

    mic, speech, noise, snr_in, vad = modified_create_micsigs(scenario, speech_paths, noise_paths, duration)

    num_audio = scenario.audioPos.shape[0] if scenario.audioPos is not None else 0
    num_noise = scenario.noisePos.shape[0] if (hasattr(scenario, 'noisePos') and scenario.noisePos is not None) else 0
    Q = num_audio + num_noise
    if Q == 0: Q = 1
    
    _, _, _, est_doas = music_wideband(mic, scenario.fs, scenario)

    target_doa = est_doas[np.argmin(np.abs(est_doas - 90.0))]
    print(f"[DAS BF] Target bron gedetecteerd op: {target_doa:.1f}°")

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
    SNRoutDAS = 10 * np.log10(P_speech_das / P_noise_das)
    print(f"[DAS BF] Output SNR: {SNRoutDAS:.2f} dB")
    
    if show_plot:
        t = np.arange(len(mic[:, 0])) / scenario.fs
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, mic[:, 0], color='gray', label=f'Microfoon 1 (SNR: {snr_in:.1f} dB)')
        plt.legend(loc='upper right')
        plt.subplot(2, 1, 2)
        plt.plot(t, DASout, color='blue', label=f'DAS Beamformer (SNR: {SNRoutDAS:.1f} dB)')
        plt.legend(loc='upper right')
        plt.show()

    return DASout, speechDAS, noiseDAS, SNRoutDAS, mic, aligned_mic, vad, snr_in


def gsc_td(scenario, speech_paths, noise_paths=None, duration=10.0):
  

    #  Run DAS BF 
    DASout, speechDAS, noiseDAS, SNRoutDAS, mic, aligned_mic, vad, snr_in = das_bf(
        scenario, speech_paths, noise_paths, duration=duration, show_plot=False
    )

    N_samples, M_mics = aligned_mic.shape

    #  Blocking Matrix (Griffiths-Jim)
 
    B = np.zeros((M_mics - 1, M_mics))
    for i in range(M_mics - 1):
        B[i, 0] = 1
        B[i, i + 1] = -1
        
    noise_refs = (B @ aligned_mic.T).T  

    #  NLMS Adaptief Filter Instellingen
    L = 1024
    mu = 0.1
    delta = L // 2

    target_ref = np.pad(DASout, (delta, 0))[:N_samples]
    
    W = np.zeros((L, M_mics - 1))
    GSCout = np.zeros(N_samples)
    

    padded_noise = np.pad(noise_refs, ((L-1, 0), (0, 0)))
    eps = 1e-8

    for n in range(N_samples):
        X_slice = padded_noise[n : n+L]
        y_n = np.sum(W * X_slice)
        e_n = target_ref[n] - y_n
        GSCout[n] = e_n
        power = np.sum(X_slice**2)
        W += (mu * e_n / (power + eps)) * X_slice

 

    #  SNR Berekenen met VAD op het GSC signaal 
    vad_delayed = np.pad(vad, (delta, 0))[:N_samples]
    Pn_gsc = np.var(GSCout[vad_delayed == 0])
    Psn_gsc = np.var(GSCout[vad_delayed == 1])
    Ps_gsc = Psn_gsc - Pn_gsc
    if Ps_gsc < 0: Ps_gsc = 1e-10 
    
    SNRoutGSC = 10 * np.log10(Ps_gsc / Pn_gsc)
    print(f"[GSC] Output SNR: {SNRoutGSC:.2f} dB (Netto Verbetering vs Mic1: {SNRoutGSC - snr_in:.2f} dB)")

    #  Plotting
    t = np.arange(N_samples) / scenario.fs
    GSCout_plot = np.pad(GSCout[delta:], (0, delta))
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, mic[:, 0], color='gray', label=f'Microfoon 1 (Input SNR: {snr_in:.1f} dB)')
    plt.ylabel("Amplitude")
    plt.title("Oorspronkelijk Microfoonsignaal (Referentie)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(t, DASout, color='blue', alpha=0.6, label=f'DAS Beamformer (SNR: {SNRoutDAS:.1f} dB)')
    plt.plot(t, speechDAS, color='cyan', alpha=0.8, linestyle='--', label='Zuivere Spraakcomponent (SpeechDAS)')
    plt.ylabel("Amplitude")
    plt.title("Delay-and-Sum (Deel 1 GSC)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(t, GSCout_plot, color='green', label=f'GSC Output (SNR: {SNRoutGSC:.1f} dB)')
    plt.xlabel("Tijd (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Generalized Sidelobe Canceler (Adaptieve Onderdrukking)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    return GSCout, SNRoutGSC, mic, DASout, GSCout_plot, noise_refs


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
         
        ]
        speech_paths = alle_spraak_bestanden[:num_audio]

       
        gekozen_ruisbestand = "Babble_noise1.wav" 
        
        alle_ruis_bestanden = [
            os.path.join(parent_dir, "sound_files", gekozen_ruisbestand)
        ]
        noise_paths = alle_ruis_bestanden[:num_noise]

       
        GSCout, SNRoutGSC, mic, DASout, GSCout_plot, noise_refs = gsc_td(scenario, speech_paths, noise_paths, duration=10.0)

        # Audio opslaan 
       
        
        #  Origineel Signaal (Mic 1)
        mic_audio = mic[:, 0] / np.max(np.abs(mic[:, 0]))
        path_mic = os.path.join(parent_dir, "Luister_01_Origineel_Mic1.wav")
        sf.write(path_mic, mic_audio, scenario.fs)
        print(f" Mic 1 Audio opgeslagen: {path_mic}")

        #  DAS Beamformer Signaal
        das_audio = DASout / np.max(np.abs(DASout))
        path_das = os.path.join(parent_dir, "Luister_02_Verbeterd_DAS.wav")
        sf.write(path_das, das_audio, scenario.fs)
        print(f" DAS Audio opgeslagen: {path_das}")

        # GSC Audio opslaan
        gsc_audio = GSCout_plot / np.max(np.abs(GSCout_plot))
        path_gsc = os.path.join(parent_dir, "Luister_03_GSC_Filter.wav")
        sf.write(path_gsc, gsc_audio, scenario.fs)
        print(f" GSC Audio opgeslagen: {path_gsc}")

        #  Blocking Matrix Audio opslaan
        bm_audio = noise_refs[:, 0] / np.max(np.abs(noise_refs[:, 0]))
        path_bm = os.path.join(parent_dir, "Luister_04_BlockingMatrix.wav")
        sf.write(path_bm, bm_audio, scenario.fs)
        print(f" Blocking Matrix Audio opgeslagen: {path_bm}")

    except Exception as e:
        import traceback
        print(f"Fout in uitvoering: {e}")
        traceback.print_exc()
