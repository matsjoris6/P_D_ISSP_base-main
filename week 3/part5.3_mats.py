"""
Week 3 - Part 5: DOA-informed GSC with VAD - Head-Mounted Microphone Case
==========================================================================
Applies GSC+VAD to all mic subset configurations:
  1. Right ear:  yR1, yR2
  2. Left ear:   yL1, yL2
  3. Frontal:    yL1, yR1
  4. All 4 mics: yL1, yL2, yR1, yR2

Sources:
  - Target (speech):   part1_track1_dry.wav  from s-90  (LEFT side → 180°)
  - Interferer (noise):part1_track2_dry.wav  from s90   (RIGHT side →  0°)
"""

import os
import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt

# ============================================================
# 0.  GLOBAL PARAMETERS & HEAD-MOUNTED GEOMETRY
# ============================================================
fs_sim  = 44100
c       = 343.0

# Physical microphone positions (metres) – 2-D, origin at head centre
# Convention:  x-axis = left(−) to right(+),  y-axis = front(+) to back(−)
x_ear   = 0.215 / 2     # half inter-ear distance  = 10.75 cm
y_mic   = 0.013 / 2     # half intra-ear distance  =  0.65 cm

micPos_all = np.array([
    [-x_ear,  y_mic],   # 0 : L1  (left-front)
    [-x_ear, -y_mic],   # 1 : L2  (left-back)
    [ x_ear,  y_mic],   # 2 : R1  (right-front)
    [ x_ear, -y_mic],   # 3 : R2  (right-back)
])

# True DOAs in the 0-180° convention used throughout (90° = straight ahead)
# s-90  → left side   →  90° + 90° = 180°
# s90   → right side  →  90° − 90° =   0°
TRUE_DOA_TARGET     = 180.0   # left source  (target / speech)
TRUE_DOA_INTERFERER =   0.0   # right source (interferer / noise)

# ============================================================
# 1.  LOAD HMIRs AND BUILD MICROPHONE SIGNALS
# ============================================================
def load_signals():
    base_folder       = os.path.join("sound_files", "head_mounted_rirs")
    dry_target_path   = os.path.join("sound_files", "part1_track1_dry.wav")
    dry_noise_path    = os.path.join("sound_files", "part1_track2_dry.wav")
    folder_left       = "s-90"   # target source (left)
    folder_right      = "s90"    # interferer   (right)

    dry_target, _ = sf.read(dry_target_path)
    dry_noise,  _ = sf.read(dry_noise_path)
    min_len        = min(len(dry_target), len(dry_noise), fs_sim * 10)   # 10 s cap
    dry_target     = dry_target[:min_len]
    dry_noise      = dry_noise [:min_len]

    def load_rirs(folder):
        def rd(name):
            return sf.read(os.path.join(base_folder, folder, name))[0]
        return rd("HMIR_L1.wav"), rd("HMIR_L2.wav"), rd("HMIR_R1.wav"), rd("HMIR_R2.wav")

    L1_t, L2_t, R1_t, R2_t = load_rirs(folder_left)
    L1_n, L2_n, R1_n, R2_n = load_rirs(folder_right)

    def conv(dry, rir):
        return signal.fftconvolve(dry, rir, mode='full')[:min_len]

    # Speech matrix  [N × 4]  — target convolved through each HMIR
    speech = np.column_stack([
        conv(dry_target, L1_t),
        conv(dry_target, L2_t),
        conv(dry_target, R1_t),
        conv(dry_target, R2_t),
    ])

    # Noise matrix  [N × 4]  — interferer convolved through each HMIR
    noise = np.column_stack([
        conv(dry_noise, L1_n),
        conv(dry_noise, L2_n),
        conv(dry_noise, R1_n),
        conv(dry_noise, R2_n),
    ])

    # Add spatially uncorrelated white mic noise (10 % of target power in mic 0)
    vad  = np.abs(speech[:, 0]) > np.std(speech[:, 0]) * 1e-3
    Ps   = np.var(speech[vad, 0])
    white = np.random.randn(*speech.shape) * np.sqrt(0.10 * Ps)
    noise = noise + white

    mic = speech + noise
    return mic, speech, noise, vad

# ============================================================
# 2.  MUSIC WIDEBAND FOR HEAD-MOUNTED ARRAYS (any subset)
# ============================================================
def music_wideband_hm(micsigs, micPos, Q=1):
    """
    Wideband MUSIC adapted for arbitrary subset of head-mounted mics.

    Parameters
    ----------
    micsigs : [N × M] array
    micPos  : [M × 2] array  – actual positions of the M selected mics
    Q       : int            – number of sources to estimate (≤ M-1)

    Returns
    -------
    angles        : (nAngle,) deg grid 0–180
    spectrum_db   : (nAngle,) dB-normalised geometrically averaged pseudospectrum
    estimated_doas: (Q,) estimated DOAs in degrees
    """
    L       = 1024
    overlap = L // 2
    M       = micsigs.shape[1]

    # Clamp Q so noise subspace is at least 1 vector
    Q = min(Q, M - 1)
    if Q < 1:
        Q = 1

    # STFT per channel
    stft_list = []
    for m in range(M):
        freqs, _, Zxx = signal.stft(micsigs[:, m], fs=fs_sim,
                                    window='hann', nperseg=L, noverlap=overlap)
        stft_list.append(Zxx)
    stft_data = np.stack(stft_list, axis=0)          # [M × nF × nT]
    _, nF, nT = stft_data.shape

    # Centre mic positions
    mics_c = micPos - np.mean(micPos, axis=0)
    px     = mics_c[:, 0].reshape(-1, 1)             # [M × 1]
    py     = mics_c[:, 1].reshape(-1, 1)

    angles = np.arange(0, 180.5, 0.5)
    rads   = np.radians(angles)

    pseudospectra = []
    for k in range(1, nF - 1):
        Y   = stft_data[:, k, :]                      # [M × nT]
        Ryy = (Y @ Y.conj().T) / nT                   # [M × M]

        _, eigvecs = np.linalg.eigh(Ryy)
        En  = eigvecs[:, :M - Q]                      # noise eigenvectors [M × (M-Q)]

        omega = 2 * np.pi * freqs[k]
        taus  = (px * np.cos(rads) + py * np.sin(rads)) / c   # [M × nAngle]
        A     = np.exp(1j * omega * taus)                      # steering matrix

        denom = np.sum(np.abs(En.conj().T @ A) ** 2, axis=0)
        pseudospectra.append(1.0 / (denom + 1e-30))

    pseudospectra = np.array(pseudospectra)            # [nF-2 × nAngle]
    p_geom        = np.exp(np.mean(np.log(pseudospectra + 1e-30), axis=0))
    spectrum_db   = 10 * np.log10(p_geom / (np.max(p_geom) + 1e-30))

    peaks_idx, _ = signal.find_peaks(spectrum_db)
    if len(peaks_idx) >= Q:
        sorted_p      = peaks_idx[np.argsort(spectrum_db[peaks_idx])][-Q:]
        estimated_doas = np.sort(angles[sorted_p])
    else:
        estimated_doas = np.sort(angles[np.argsort(spectrum_db)[-Q:]])

    return angles, spectrum_db, estimated_doas

# ============================================================
# 3.  DAS BEAMFORMER
# ============================================================
def apply_das(signals, micPos, target_doa_deg, fs):
    """
    Delay-and-sum toward target_doa_deg.
    Returns (das_output [N], aligned_signals [N × M]).
    """
    N, M  = signals.shape
    rad   = np.radians(target_doa_deg)
    mics_c = micPos - np.mean(micPos, axis=0)
    # Propagation delays: τ_m = (x_m·cos θ + y_m·sin θ) / c
    taus  = (mics_c[:, 0] * np.cos(rad) + mics_c[:, 1] * np.sin(rad)) / c

    f          = np.fft.rfftfreq(N, 1.0 / fs)
    out_sum    = np.zeros(N)
    out_aligned = np.zeros((N, M))
    for m in range(M):
        S_aligned         = np.fft.rfft(signals[:, m]) * np.exp(1j * 2 * np.pi * f * taus[m])
        ch                = np.fft.irfft(S_aligned, n=N)
        out_aligned[:, m] = ch
        out_sum           += ch
    return out_sum / M, out_aligned

# ============================================================
# 4.  GSC WITH VAD  (one config)
# ============================================================
def gsc_vad(mic_sub, speech_sub, noise_sub, vad, micPos_sub, config_name,
            Q_music=1, mu=0.05, L_fir=1024):
    """
    Full pipeline: MUSIC → DAS → blocking matrix → NLMS with VAD gate.

    Returns
    -------
    snr_in  : input SNR  (dB)
    snr_das : DAS output SNR (dB)
    snr_gsc : GSC output SNR (dB)
    gsc_out : GSC output waveform (delay-compensated)
    das_out : DAS output waveform
    """
    N, M = mic_sub.shape
    delta = L_fir // 2

    # --- SNR at input mic 0 ---
    Ps_in  = np.var(speech_sub[vad, 0])
    Pn_in  = np.var(noise_sub[:, 0])
    snr_in = 10 * np.log10(Ps_in / (Pn_in + 1e-10))

    # --- MUSIC DOA estimation (target = closest to 180°) ---
    _, _, est_doas = music_wideband_hm(mic_sub, micPos_sub, Q=Q_music)
    target_doa     = est_doas[np.argmin(np.abs(est_doas - TRUE_DOA_TARGET))]
    print(f"  [{config_name}] MUSIC → target DOA = {target_doa:.1f}°  "
          f"(truth = {TRUE_DOA_TARGET:.0f}°)")

    # --- DAS beamformer ---
    das_speech, speech_aligned = apply_das(speech_sub, micPos_sub, target_doa, fs_sim)
    das_noise,  noise_aligned  = apply_das(noise_sub,  micPos_sub, target_doa, fs_sim)
    das_out = das_speech + das_noise

    P_das_s  = np.var(das_speech[vad])
    P_das_n  = np.var(das_noise)
    snr_das  = 10 * np.log10(P_das_s / (P_das_n + 1e-10))

    # --- Blocking Matrix (Griffiths-Jim) ---
    aligned_mic = speech_aligned + noise_aligned        # [N × M]
    B           = np.zeros((M - 1, M))
    for i in range(M - 1):
        B[i, 0]     =  1
        B[i, i + 1] = -1
    noise_refs = (B @ aligned_mic.T).T                  # [N × (M-1)]

    # --- NLMS with VAD gate ---
    target_ref  = np.pad(das_out, (delta, 0))[:N]
    vad_delayed = np.pad(vad.astype(int), (delta, 0))[:N].astype(bool)

    W          = np.zeros((L_fir, M - 1))
    gsc_raw    = np.zeros(N)
    padded_ref = np.pad(noise_refs, ((L_fir - 1, 0), (0, 0)))

    for n in range(N):
        X   = padded_ref[n: n + L_fir]                 # [L_fir × (M-1)]
        e_n = target_ref[n] - np.sum(W * X)
        gsc_raw[n] = e_n
        if not vad_delayed[n]:                          # update only during noise-only
            power = np.sum(X ** 2) + 1e-8
            W    += (mu * e_n / power) * X

    # Compensate for delay delta
    gsc_out = np.pad(gsc_raw[delta:], (0, delta))

    # --- GSC output SNR ---
    Pn_gsc  = np.var(gsc_raw[~vad_delayed])
    Psn_gsc = np.var(gsc_raw[ vad_delayed])
    Ps_gsc  = max(Psn_gsc - Pn_gsc, 1e-10)
    snr_gsc = 10 * np.log10(Ps_gsc / (Pn_gsc + 1e-10))

    return snr_in, snr_das, snr_gsc, gsc_out, das_out

# ============================================================
# 5.  MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Week 3 – Part 5: Head-Mounted GSC with VAD")
    print("=" * 60)

    print("\n[1/2] Loading signals …")
    mic_all, speech_all, noise_all, vad = load_signals()
    print(f"      Signal length: {mic_all.shape[0]/fs_sim:.1f} s  |  "
          f"Active speech frames: {vad.mean()*100:.1f} %")

    # --- Configurations ---
    configs = {
        "Right ear (R1,R2)":  {"idx": [2, 3], "Q_music": 1},
        "Left ear  (L1,L2)":  {"idx": [0, 1], "Q_music": 1},
        "Frontal   (L1,R1)":  {"idx": [0, 2], "Q_music": 1},
        "All 4 mics":         {"idx": [0, 1, 2, 3], "Q_music": 2},
    }

    results = {}
    print("\n[2/2] Running GSC for each configuration …\n")

    fig, axes = plt.subplots(len(configs), 1, figsize=(14, 4 * len(configs)), sharex=True)
    t = np.arange(mic_all.shape[0]) / fs_sim

    for ax, (name, cfg) in zip(axes, configs.items()):
        idx       = cfg["idx"]
        mic_sub   = mic_all   [:, idx]
        speech_sub= speech_all[:, idx]
        noise_sub = noise_all [:, idx]
        pos_sub   = micPos_all [idx]

        snr_in, snr_das, snr_gsc, gsc_out, das_out = gsc_vad(
            mic_sub, speech_sub, noise_sub, vad, pos_sub, name,
            Q_music=cfg["Q_music"]
        )
        results[name] = (snr_in, snr_das, snr_gsc)

        # Scale DAS & GSC to mic amplitude for visual comparison
        scale = np.std(mic_sub[:, 0]) / (np.std(das_out) + 1e-10)
        ax.plot(t, mic_sub[:, 0], color='gray',      alpha=0.45, lw=0.8,
                label=f'Mic 0  (SNR in : {snr_in:.1f} dB)')
        ax.plot(t, das_out * scale, color='dodgerblue', alpha=0.7, lw=0.9,
                label=f'DAS-BF (SNR out: {snr_das:.1f} dB)')
        ax.plot(t, gsc_out * scale, color='green',    alpha=0.9, lw=1.1,
                label=f'GSC    (SNR out: {snr_gsc:.1f} dB)')
        ax.set_title(name, fontsize=11)
        ax.set_ylabel("Amplitude")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Save audio
        def save(arr, fname):
            arr = arr / (np.max(np.abs(arr)) + 1e-10)
            sf.write(fname, arr, fs_sim)

        tag = name.replace(" ", "_").replace("(","").replace(")","").replace(",","")
        save(mic_sub[:, 0], f"Output_{tag}_mic0.wav")
        save(das_out,       f"Output_{tag}_DAS.wav")
        save(gsc_out,       f"Output_{tag}_GSC.wav")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Week 3 Part 5 – Head-Mounted GSC with VAD", fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.savefig("week3_part5_gsc_hm_plot.png", dpi=120)
    plt.show()

    # --- Summary table ---
    print("\n" + "=" * 70)
    print(f"{'Configuration':<22} | {'SNR in':>8} | {'SNR DAS':>9} | {'SNR GSC':>9} | {'Gain GSC':>9}")
    print("-" * 70)
    for name, (si, sd, sg) in results.items():
        print(f"{name:<22} | {si:>7.1f}° | {sd:>8.1f}° | {sg:>8.1f}° | {sg-si:>+8.1f}°")
    print("=" * 70)

    print("\n✅  Done. Plots saved to 'week3_part5_gsc_hm_plot.png'.")
    print("    Audio files saved as Output_<config>_<stage>.wav")