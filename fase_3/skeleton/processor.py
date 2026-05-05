import asyncio
import numpy as np
import scipy.linalg
from scipy import signal

def compute_sir(y, x1, x2, groundTruth):
    """
    y = totale beamformer output
    x1 = bijdrage bron 1 in y, x2 = bijdrage bron 2 in y
    groundTruth = per sample 1 als x1 target, 0 als x2 target
    """
    if np.sqrt(np.sum((y - x1 - x2) ** 2)) / (np.sqrt(np.sum(y ** 2)) + 1e-12) > 0.05:
        return np.nan
    if np.sum(groundTruth) + np.sum(1 - groundTruth) != len(groundTruth):
        return np.nan
    target_var = np.var(x1 * groundTruth + x2 * (1 - groundTruth))
    interf_var = np.var(x2 * groundTruth + x1 * (1 - groundTruth))
    if interf_var < 1e-12:
        return np.nan
    return 10 * np.log10(target_var / interf_var)

_LMA_COORDS = np.array([
    [4.699999999999999, 2.2],
    [4.699999999999999, 2.3000000000000003],
    [4.699999999999999, 2.4],
    [4.699999999999999, 2.5],
    [4.699999999999999, 2.6],
])



_RIR_PATH = "data/phase3_audioData/audiodata_batch_1/anechoic/lma_16kHz.npz"
def build_lut_for_target(target_rir, L=1024):
    """Bouwt FAS beamformer en Blocking matrix voor één RIR."""
    n_bins = L // 2 + 1
    M_mics = target_rir.shape[1]

    H_omega = np.fft.rfft(target_rir, n=L, axis=0)

    W_FAS = np.zeros((n_bins, M_mics), dtype=complex)
    B_matrix = np.zeros((n_bins, M_mics - 1, M_mics), dtype=complex)

    for k in range(n_bins):
        h_k = H_omega[k, :].reshape(M_mics, 1)

        eps = 1e-12
        A_1 = h_k[0, 0]
        if np.abs(A_1) > eps:
            h_k = h_k / A_1
        else:
            h_k = h_k / (A_1 + eps)

        denom = (h_k.conj().T @ h_k)[0, 0]
        if np.abs(denom) > eps:
            W_FAS[k, :] = (h_k / denom).flatten()
        else:
            W_FAS[k, :] = np.ones(M_mics) / M_mics

        Z = scipy.linalg.null_space(h_k.conj().T)
        if Z.shape[1] > 0:
            B_matrix[k, :, :] = Z.conj().T
    return W_FAS, B_matrix
class Processor:
    def __init__(self, fs=16000, rir_path=_RIR_PATH):
        self.attended_left = 1

        # Output 'pipes'
        self.data_queue_phase1 = asyncio.Queue()
        self.data_queue_phase2 = asyncio.Queue()
        self.data_queue_phase3 = asyncio.Queue()

        
        self.fs = fs
        self.M = 5  # Aantal LMA microfoons
        self.L = 1024  # FFT Window size
        self.beta = 0.95  
        self.c = 343.0
        self.Q = 2  # Aantal sprekers
        self.mu = 0.01  # NLMS stapgrootte

        # Audio buffer voor de sliding window
        self.audio_buffer = np.zeros((self.L, self.M))

        # Ryy opslag: (aantal frequentie bins, M, M)
        self.num_bins = self.L // 2 + 1
        self.Ryy = np.zeros((self.num_bins, self.M, self.M), dtype=complex)

        # Pre-compute MUSIC variabelen
        self.angles = np.arange(0, 180.5, 0.5)
        self.rads = np.radians(self.angles)
        
        mic_pos = _LMA_COORDS
            
        mics_centered = mic_pos - np.mean(mic_pos, axis=0)
        self.px = mics_centered[:, 0].reshape(-1, 1)
        self.py = mics_centered[:, 1].reshape(-1, 1)

        rir_data = np.load(rir_path)
        rirs = rir_data["rirs"]   # (nSamples, nMics, nRIRs) shape: (22050, 5, 20)
        doas = rir_data["thetas"]   # (nRIRs,) shape: (20,) — 20 vooraf berekende hoeken
        print(f"Beschikbare hoeken: {doas}")
        self.lut_angles = np.array(doas)
        self.lut = {}
        for i, angle in enumerate(doas):
            rir = rirs[:, :, i]
            W_FAS, B = build_lut_for_target(rir, L=self.L)
            self.lut[float(angle)] = (W_FAS, B)

        # NLMS gewichten per richting (links en rechts), per frequentiebin
        self.w_nlms_left = np.zeros((self.num_bins, self.M - 1), dtype=complex)
        self.w_nlms_right = np.zeros((self.num_bins, self.M - 1), dtype=complex)

        # Buffers voor SIR berekening (parallelle GSC op gt signalen)
        self.audio_buffer_gt0 = np.zeros((self.L, self.M))
        self.audio_buffer_gt1 = np.zeros((self.L, self.M))

        # SIR over niet-overlappende 1-seconde vensters
        self.sir_window_samples = int(self.fs * 1.0)  # 1 seconde

        # Buffers die zich vullen tot 1s, dan SIR berekend en gereset
        self.sir_buf_y_left = []
        self.sir_buf_y_right = []
        self.sir_buf_L_gt0 = []
        self.sir_buf_L_gt1 = []
        self.sir_buf_R_gt0 = []
        self.sir_buf_R_gt1 = []
        self.sir_buf_count = 0

        # Laatst berekende SIR (blijft staan tussen vensters voor frontend)
        self._last_sir = 0.0
        self._last_sir_left = 0.0     # LATER WEGHALEN
        self._last_sir_right = 0.0    # LATER WEGHALEN


        # Window voor STFT (sqrt-Hann voor perfecte reconstructie)
        self.window = np.sqrt(signal.windows.hann(self.L, sym=False))

        # VAD state voor streaming adaptatie
        self.energy_history = []
        self.energy_history_size = 50    
        self.vad_threshold_factor = 5   # adapteer als energie < factor × min_energy
    
    def _get_lut_for_angle(self, doa):
        """Pakt dichtstbijzijnde hoek uit de LUT."""
        closest_angle = self.lut_angles[np.argmin(np.abs(self.lut_angles - doa))]
        return self.lut[float(closest_angle)]
    
    def _apply_gsc_frame(self, frame_fft, W_FAS, B, w_nlms, update_filter):
        """
        Past FD-GSC toe op één FFT frame (vector over alle M mics per bin).
        
        frame_fft : (num_bins, M) complex - FFT van huidig frame
        W_FAS : (num_bins, M) - FAS weights
        B : (num_bins, M-1, M) - blocking matrix
        w_nlms : (num_bins, M-1) - NLMS weights (wordt in-place geüpdatet)
        update_filter : bool - of NLMS moet leren (alleen bij geen-speech)
        out_fft : (num_bins,) - gefilterde output in freq domein
        """
        out_fft = np.zeros(self.num_bins, dtype=complex)
        eps = 1e-8

        for k in range(self.num_bins):
            x = frame_fft[k, :]  # (M,)
            y_fas = np.vdot(W_FAS[k, :], x)
            u = B[k, :, :] @ x   # (M-1,)
            e = y_fas - np.vdot(w_nlms[k, :], u)
            out_fft[k] = e

            if update_filter:
                power = np.vdot(u, u).real
                w_nlms[k, :] += self.mu * u * np.conj(e) / (power + eps)

        return out_fft

    def processing_microarray(self, lma, lma_gt0=None, lma_gt1=None):
        chunk_size = lma.shape[0]

        #Update de sliding window met de nieuwe chunk
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_size, axis=0)
        self.audio_buffer[-chunk_size:, :] = lma

        # Update sliding windows voor gt signalen (voor SIR)
        if lma_gt0 is not None and lma_gt1 is not None:
            self.audio_buffer_gt0 = np.roll(self.audio_buffer_gt0, -chunk_size, axis=0)
            self.audio_buffer_gt0[-chunk_size:, :] = lma_gt0
            self.audio_buffer_gt1 = np.roll(self.audio_buffer_gt1, -chunk_size, axis=0)
            self.audio_buffer_gt1[-chunk_size:, :] = lma_gt1

        # Toepassen van een window functie (Hann) voor rfft
        windowed_frame = self.audio_buffer * self.window[:, np.newaxis]

        # Fast Fourier Transform op het huidige frame
        frame_fft = np.fft.rfft(windowed_frame, n=self.L, axis=0)
        freqs = np.fft.rfftfreq(self.L, d=1/self.fs)

        pseudospectra = []
        valid_indices = range(1, self.L // 2)

        #Exponentiële middeling en DOA schatting per frequentiebin
        for k in valid_indices:
            # Vector van observaties voor frequentie k
            Y = frame_fft[k, :].reshape(self.M, 1)
            R_k = Y @ Y.conj().T

            # Exponentiële update van Ryy
            self.Ryy[k] = self.beta * self.Ryy[k] + (1 - self.beta) * R_k

            # Ruissubruimte bepalen
            _, eigvecs = np.linalg.eigh(self.Ryy[k])
            En = eigvecs[:, :self.M - self.Q]

            # Steering vector bepalen
            omega = 2 * np.pi * freqs[k]
            taus = (self.px * np.sin(self.rads) + self.py * np.cos(self.rads)) / self.c
            A = np.exp(-1j * omega * taus)

            # Pseudospectrum 
            denom = np.sum(np.abs(En.conj().T @ A)**2, axis=0)
            p_theta = 1.0 / denom
            pseudospectra.append(p_theta)

        #  Geometrisch Gemiddelde en Peak Finding
        pseudospectra = np.array(pseudospectra)
        pseudospectra = np.clip(pseudospectra, 1e-10, None)  # voorkomt log(0) = -inf
        log_p = np.log(pseudospectra)
        p_geom = np.exp(np.mean(log_p, axis=0))
        spectrum_geom_db = 10 * np.log10(p_geom / np.max(p_geom))

        peaks_indices, _ = signal.find_peaks(spectrum_geom_db)
        
        # Selecteer de twee hoogste pieken
       
        if len(peaks_indices) >= self.Q:
            sorted_peak_indices = peaks_indices[np.argsort(spectrum_geom_db[peaks_indices])][-self.Q:]
            estimated_doas = np.sort(self.angles[sorted_peak_indices])
        else:
            return  # Niets sturen, wacht op volgende frame
        

        # angle_left altijd in [180 -> 90]  en angle_right in [90 -> 0]
        angle_right = estimated_doas[0] if estimated_doas[0] <= 90 else estimated_doas[1]
        angle_left = estimated_doas[1] if estimated_doas[1] > 90 else estimated_doas[0]

        # DEBUG: print toewijzing
        if not hasattr(self, "_debug_printed"):
            print(f"angle_left={angle_left} (verwacht >90), angle_right={angle_right} (verwacht <90)")
            self._debug_printed = True


        #Streaming VAD
        frame_energy = np.mean(self.audio_buffer[-chunk_size:, 0] ** 2)
        self.energy_history.append(frame_energy)
        if len(self.energy_history) > self.energy_history_size:
            self.energy_history.pop(0)

        if len(self.energy_history) >= 10:
            min_energy = np.min(self.energy_history)
            update_filter = frame_energy < min_energy * self.vad_threshold_factor
        else:
            update_filter = False  # Niet adapteren voor de buffer betrouwbaar is

        #FD-GSC: twee beamformers
        W_FAS_L, B_L = self._get_lut_for_angle(angle_left)
        W_FAS_R, B_R = self._get_lut_for_angle(angle_right)

        # Target links: steer naar links, cancel rechts
        out_fft_left = self._apply_gsc_frame(
            frame_fft, W_FAS_L, B_L, self.w_nlms_left, update_filter
        )
        # Target rechts: steer naar rechts, cancel links
        out_fft_right = self._apply_gsc_frame(
            frame_fft, W_FAS_R, B_R, self.w_nlms_right, update_filter
        )

        # Inverse FFT + window om tijd-domein output te krijgen
        out_time_left = np.fft.irfft(out_fft_left, n=self.L) * self.window
        out_time_right = np.fft.irfft(out_fft_right, n=self.L) * self.window

        # Pak alleen de laatste chunk_size samples (nieuwste data)
        sig0 = out_time_left[-chunk_size:].astype(np.float32)   # linker spreker versterkt
        sig1 = out_time_right[-chunk_size:].astype(np.float32)  # rechter spreker versterkt

        # ===== SIR berekening: per 1-seconde venster =====
        sir = self._last_sir  # standaard: hou laatste waarde vast tussen vensters

        if lma_gt0 is not None and lma_gt1 is not None:
            # Pas beamformers toe op gt0 en gt1 (geen NLMS update)
            windowed_gt0 = self.audio_buffer_gt0 * self.window[:, np.newaxis]
            windowed_gt1 = self.audio_buffer_gt1 * self.window[:, np.newaxis]
            fft_gt0 = np.fft.rfft(windowed_gt0, n=self.L, axis=0)
            fft_gt1 = np.fft.rfft(windowed_gt1, n=self.L, axis=0)

            w_left_eval = self.w_nlms_left.copy()
            out_fft_L_gt0 = self._apply_gsc_frame(fft_gt0, W_FAS_L, B_L, w_left_eval, False)
            out_fft_L_gt1 = self._apply_gsc_frame(fft_gt1, W_FAS_L, B_L, w_left_eval, False)
            out_L_gt0 = (np.fft.irfft(out_fft_L_gt0, n=self.L) * self.window)[-chunk_size:]
            out_L_gt1 = (np.fft.irfft(out_fft_L_gt1, n=self.L) * self.window)[-chunk_size:]

            w_right_eval = self.w_nlms_right.copy()
            out_fft_R_gt0 = self._apply_gsc_frame(fft_gt0, W_FAS_R, B_R, w_right_eval, False)
            out_fft_R_gt1 = self._apply_gsc_frame(fft_gt1, W_FAS_R, B_R, w_right_eval, False)
            out_R_gt0 = (np.fft.irfft(out_fft_R_gt0, n=self.L) * self.window)[-chunk_size:]
            out_R_gt1 = (np.fft.irfft(out_fft_R_gt1, n=self.L) * self.window)[-chunk_size:]

            # Verzamel in 1-seconde buffers
            self.sir_buf_y_left.append(sig0)
            self.sir_buf_y_right.append(sig1)
            self.sir_buf_L_gt0.append(out_L_gt0)
            self.sir_buf_L_gt1.append(out_L_gt1)
            self.sir_buf_R_gt0.append(out_R_gt0)
            self.sir_buf_R_gt1.append(out_R_gt1)
            self.sir_buf_count += chunk_size

            # Zodra we 1 seconde hebben verzameld: bereken SIR en reset
            if self.sir_buf_count >= self.sir_window_samples:
                y_left_full = np.concatenate(self.sir_buf_y_left)
                y_right_full = np.concatenate(self.sir_buf_y_right)
                L_gt0_full = np.concatenate(self.sir_buf_L_gt0)
                L_gt1_full = np.concatenate(self.sir_buf_L_gt1)
                R_gt0_full = np.concatenate(self.sir_buf_R_gt0)
                R_gt1_full = np.concatenate(self.sir_buf_R_gt1)

                gt_vec = np.ones(len(L_gt0_full))

                sir_left = compute_sir(y_left_full, L_gt0_full, L_gt1_full, gt_vec)     # LATER WEGHALEN
                sir_right = compute_sir(y_right_full, R_gt1_full, R_gt0_full, gt_vec)   # LATER WEGHALEN

                if self.attended_left:
                    sir = sir_left
                else:
                    sir = sir_right

                if np.isnan(sir):
                    sir = 0.0

                self._last_sir = sir
                self._last_sir_left = sir_left if not np.isnan(sir_left) else 0.0    # LATER WEGHALEN
                self._last_sir_right = sir_right if not np.isnan(sir_right) else 0.0 # LATER WEGHALEN

                # Reset buffers
                self.sir_buf_y_left = []
                self.sir_buf_y_right = []
                self.sir_buf_L_gt0 = []
                self.sir_buf_L_gt1 = []
                self.sir_buf_R_gt0 = []
                self.sir_buf_R_gt1 = []
                self.sir_buf_count = 0

                # LATER WEGHALEN: debug print elke seconde
                print(f"  [SIR-1s] left={self._last_sir_left:.2f} dB, right={self._last_sir_right:.2f} dB, attended={'L' if self.attended_left else 'R'} -> {sir:.2f} dB")

        # # LATER WEGHALEN: debug print elke 50 frames
        # if not hasattr(self, "_sir_print_counter"):
        #     self._sir_print_counter = 0
        # self._sir_print_counter += 1
        # if self._sir_print_counter % 50 == 0:
        #     print(f"[SIR] left={sir_left:.2f} dB, right={sir_right:.2f} dB, attended={'L' if self.attended_left else 'R'} -> {sir:.2f} dB")

        # if lma_gt0 is not None and lma_gt1 is not None and self._sir_print_counter % 50 == 1:
        #     rms_sig0 = np.sqrt(np.mean(sig0**2))
        #     rms_L_gt0 = np.sqrt(np.mean(out_L_gt0**2))
        #     rms_L_gt1 = np.sqrt(np.mean(out_L_gt1**2))
        #     rms_R_gt0 = np.sqrt(np.mean(out_R_gt0**2))
        #     rms_R_gt1 = np.sqrt(np.mean(out_R_gt1**2))
        #     print(f"  RMS check: sig0={rms_sig0:.1f}, L_gt0={rms_L_gt0:.1f}, L_gt1={rms_L_gt1:.1f}, R_gt0={rms_R_gt0:.1f}, R_gt1={rms_R_gt1:.1f}")
            
        #     # Check sanity: y vs x1+x2
        #     res_L = sig0 - (out_L_gt0 + out_L_gt1)
        #     print(f"  Residual L: ||y-x1-x2|| / ||y|| = {np.sqrt(np.sum(res_L**2)) / (np.sqrt(np.sum(sig0**2)) + 1e-12):.4f}")
                
        self.data_queue_phase1.put_nowait((sig0, sig1, angle_left, angle_right, sir))

        # Final output selectie
        sig_out = sig0 if self.attended_left else sig1
        speaker = round(np.random.random())
        self.data_queue_phase3.put_nowait((speaker, sig_out))


    def processing_eeg_gt_audio(self, eeg, sig_left_clean, sig_right_clean):
        # Preprocessing, Prediction, etc...
        # Using your own gsc out vs oracle?

        pred_prob = np.random.random() * 0.8 + 0.1
        self.attended_left = round(pred_prob)

        self.data_queue_phase2.put_nowait(pred_prob)
