import asyncio
import numpy as np
import scipy.linalg
from scipy import signal
import logging
logging.getLogger('brian2').setLevel(logging.ERROR)
import brian2
brian2.prefs.codegen.target = 'numpy'
from brian2 import Hz, kHz
from brian2hears import Sound, erbspace, Gammatone, Filterbank
from math import gcd
import tensorflow as tf
import time

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

class EnvelopeFromGammatoneFilterbank(Filterbank):
    """Converts the output of a GammatoneFilterbank to an envelope."""
    def __init__(self, source):
        super().__init__(source)
        self.nchannels = 1

    def buffer_apply(self, input_):
        abs_input = np.abs(input_)
        compressed = abs_input ** 0.6
        envelope = np.sum(compressed, axis=1, keepdims=True)
        return envelope


def compute_audio_envelope(audio_data, sr_in, sr_out=64, lowcut=1.0, highcut=32.0):
    
    sound = Sound(audio_data.reshape(-1, 1), samplerate=sr_in * Hz)
    cf = erbspace(50 * Hz, 5 * kHz, 28)
    gammatone_filterbank = Gammatone(sound, cf)
    envelope_calc = EnvelopeFromGammatoneFilterbank(gammatone_filterbank)
    envelope = envelope_calc.process().flatten()

    sos = signal.butter(N=4, Wn=[lowcut, highcut], btype='bandpass', fs=sr_in, output='sos')
    envelope_filtered = signal.sosfiltfilt(sos, envelope)

    g = gcd(int(sr_in), sr_out)
    envelope_downsampled = signal.resample_poly(envelope_filtered, sr_out // g, int(sr_in) // g)
    return envelope_downsampled


def preprocess_eeg(eeg_data, fs_in, fs_out=64, lowcut=1.0, highcut=32.0):
    
    sos = signal.butter(N=4, Wn=[lowcut, highcut], btype='bandpass', fs=fs_in, output='sos')
    eeg_filtered = signal.sosfiltfilt(sos, eeg_data, axis=0)

    g = gcd(int(fs_in), fs_out)
    eeg_downsampled = signal.resample_poly(eeg_filtered, fs_out // g, int(fs_in) // g, axis=0)
    return eeg_downsampled


class Processor:
    def __init__(self, fs=16000, rir_path="data/phase3_audioData/audiodata_batch_1/anechoic/lma_16kHz.npz"):
        self.attended_left = 1

        # Output 'pipes'
        self.data_queue_phase1 = asyncio.Queue()
        self.data_queue_phase2 = asyncio.Queue()
        self.data_queue_phase3 = asyncio.Queue()

        
        self.fs = fs
        
        self.L = 1024  # FFT Window size
        self.beta = 0.98
        self.c = 343.0
        self.Q = 2  # Aantal sprekers
        self.mu = 0.01  # NLMS stapgrootte

        rir_data = np.load(rir_path)
        rirs = rir_data["rirs"]
        doas = rir_data["thetas"]
        self.M = rirs.shape[1]
        print(f"[Processor] {self.M} microfoons gedetecteerd uit RIR")
        # Audio buffer voor de sliding window
        self.audio_buffer = np.zeros((self.L, self.M))

        # Ryy opslag: (aantal frequentie bins, M, M)
        self.num_bins = self.L // 2 + 1
        self.Ryy = np.zeros((self.num_bins, self.M, self.M), dtype=complex)

        

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

   

        # === Streaming framing parameters ===
        self.hop = self.L // 2   # 512 samples = 50% overlap

        # Input accumulators (wachten tot we hop samples hebben)
        self.input_accumulator = np.zeros((0, self.M))
        self.input_accumulator_gt0 = np.zeros((0, self.M))
        self.input_accumulator_gt1 = np.zeros((0, self.M))

        # Output overlap-add buffers (één per beam)
        self.ola_buffer_left = np.zeros(self.L)
        self.ola_buffer_right = np.zeros(self.L)
        self.ola_buffer_L_gt0 = np.zeros(self.L)
        self.ola_buffer_L_gt1 = np.zeros(self.L)
        self.ola_buffer_R_gt0 = np.zeros(self.L)
        self.ola_buffer_R_gt1 = np.zeros(self.L)

        
        # Fallback hoeken voor de "koude start" (als er nog niet gesproken is)
        self.last_angle_left = 135.0  
        self.last_angle_right = 45.0
        self.peak_threshold = -12.0
        # AAD model laden (Phase 2 Dilated CNN, 5s window) 
        model_path = "models/generic_dilated_alle_proefpersonen_beste_pieter_3laag_5sec_VERVOLG.keras"
        self.aad_model = tf.keras.models.load_model(model_path)
        self.aad_window_samples = 5*64   # 5s × 64Hz
        self.eeg_fs_in = 128            # raw EEG sample rate

        # AAD input mode
        self.use_beamformer_for_aad = False  # False = clean speech (oracle), True = beamformer output
        if self.use_beamformer_for_aad:
            self.audio_fs_in = 16000  # beamformer output is 16 kHz
        else:
            self.audio_fs_in = 48000  # clean stimuli zijn 48 kHz

        # Ring buffer voor beamformer → AAD (5 seconden bij 16 kHz)
        self.beamformer_buffer_max_samples = 5 * self.fs  # 80000 samples
        self.beamformer_buffer_left = np.zeros(self.beamformer_buffer_max_samples, dtype=np.float32)
        self.beamformer_buffer_right = np.zeros(self.beamformer_buffer_max_samples, dtype=np.float32)
        self.beamformer_buffer_filled = 0  # hoeveel samples zijn al geschreven (tot max)

        # AAD EMA Filter parameters
        self.ema_alpha = 0.3
        self.ema_filtered = 0.5  # Start op 50% (volledige twijfel)


        
        self.lut_angles = np.array(doas)
        self.A_lut = np.zeros((self.num_bins, self.M, len(doas)), dtype=complex)
        self.lut = {}
        
        for i, angle in enumerate(doas):
            rir = rirs[:, :, i]
            
            #  MUSIC: Bouw de gemeten steering vector H_omega voor deze hoek
            H_omega = np.fft.rfft(rir, n=self.L, axis=0)
            for k in range(self.num_bins):
                h_k = H_omega[k, :].reshape(self.M, 1)
                A_1 = h_k[0, 0] # Normaliseer op mic 1
                if np.abs(A_1) > 1e-12:
                    h_k = h_k / A_1
                else:
                    h_k = h_k / (A_1 + 1e-12)
                self.A_lut[k, :, i] = h_k.flatten()

            # GSC: Bouw de FAS en Blocking matrix
            W_FAS, B = build_lut_for_target(rir, L=self.L)
            self.lut[float(angle)] = (W_FAS, B)

        #voor vad
        self.noise_floor_left = None
        self.noise_floor_right = None
        self.alpha_up = 0.95     # langzaam stijgen (noise floor groeit voorzichtig)
        self.alpha_down = 0.8    # snel dalen (snel reageren op stilte) eerst 0.5
        self.vad_threshold = 0.75 # spraak = × noise floor

        # VAD statistieken (per beam) ENKEL VOOR TEST
        self.vad_evals = 0
        self.vad_update_count_left = 0
        self.vad_update_count_right = 0
        #EINDE TEST
        self.sir_history_left  = []
        self.sir_history_right = []

        rir_data = np.load(rir_path)
        print(f"[DEBUG] RIR loaded: {rir_path}")
        print(f"[DEBUG] RIR shape: {rir_data['rirs'].shape}")
        print(f"[DEBUG] M={self.M}, RIR mics={rir_data['rirs'].shape[1]}")
     
    def _get_lut_for_angle(self, doa):
        """Pakt dichtstbijzijnde hoek uit de LUT."""
        closest_angle = self.lut_angles[np.argmin(np.abs(self.lut_angles - doa))]
        return self.lut[float(closest_angle)]
    
    def _apply_gsc_frame(self, frame_fft, W_FAS, B, w_nlms, update_filter):
        """
        Gevectoriseerde FD-GSC voor maximale real-time snelheid.
        """
        # 1. FAS output: dot product over de microfoon as (vdot equivalent)
        y_fas = np.sum(np.conj(W_FAS) * frame_fft, axis=1)
        
        # 2. Blocking matrix output: B @ X per frequentiebin
        u = np.einsum('nij,nj->ni', B, frame_fft)
        
        # 3. Adaptive filter vermenigvuldiging
        y_bm = np.sum(np.conj(w_nlms) * u, axis=1)
        
        # 4. Error signal (jouw uiteindelijke gefilterde audio)
        e = y_fas - y_bm
        
        # 5. Filter update (vectorized)
        if update_filter:
            power = np.real(np.sum(np.conj(u) * u, axis=1))
            w_nlms += self.mu * u * np.conj(e)[:, np.newaxis] / (power[:, np.newaxis] + 1e-8)

        return e

    def processing_microarray(self, lma, lma_gt0=None, lma_gt1=None):
        # Accumuleer incoming server-chunks
        self.input_accumulator = np.vstack([self.input_accumulator, lma])
        if lma_gt0 is not None and lma_gt1 is not None:
            self.input_accumulator_gt0 = np.vstack([self.input_accumulator_gt0, lma_gt0])
            self.input_accumulator_gt1 = np.vstack([self.input_accumulator_gt1, lma_gt1])

        # Verwerk zoveel hops als beschikbaar
        while self.input_accumulator.shape[0] >= self.hop:
            # Pak één hop uit de accumulator
            hop_samples = self.input_accumulator[:self.hop, :]
            self.input_accumulator = self.input_accumulator[self.hop:, :]

            # Schuif sliding window
            self.audio_buffer = np.roll(self.audio_buffer, -self.hop, axis=0)
            self.audio_buffer[-self.hop:, :] = hop_samples

            # Idem voor gt buffers
            has_gt = (self.input_accumulator_gt0.shape[0] >= self.hop and
                    self.input_accumulator_gt1.shape[0] >= self.hop)
            if has_gt:
                hop_gt0 = self.input_accumulator_gt0[:self.hop, :]
                hop_gt1 = self.input_accumulator_gt1[:self.hop, :]
                self.input_accumulator_gt0 = self.input_accumulator_gt0[self.hop:, :]
                self.input_accumulator_gt1 = self.input_accumulator_gt1[self.hop:, :]
                self.audio_buffer_gt0 = np.roll(self.audio_buffer_gt0, -self.hop, axis=0)
                self.audio_buffer_gt0[-self.hop:, :] = hop_gt0
                self.audio_buffer_gt1 = np.roll(self.audio_buffer_gt1, -self.hop, axis=0)
                self.audio_buffer_gt1[-self.hop:, :] = hop_gt1

            if has_gt:
                rms_left = np.sqrt(np.mean(self.audio_buffer_gt0[:, 0] ** 2))
                rms_right = np.sqrt(np.mean(self.audio_buffer_gt1[:, 0] ** 2))
    
                # Initialiseer noise floor met eerste hop
                if self.noise_floor_left is None:
                    self.noise_floor_left = rms_left
                    self.noise_floor_right = rms_right
                else:
                    # Asymmetrische EMA: snel naar beneden, langzaam omhoog
                    # → noise floor "klikt vast" op stilte-energie
                    if rms_left < self.noise_floor_left:
                        self.noise_floor_left = self.alpha_down * self.noise_floor_left + (1 - self.alpha_down) * rms_left
                    else:
                        self.noise_floor_left = self.alpha_up * self.noise_floor_left + (1 - self.alpha_up) * rms_left
            
                    if rms_right < self.noise_floor_right:
                        self.noise_floor_right = self.alpha_down * self.noise_floor_right + (1 - self.alpha_down) * rms_right
                    else:
                        self.noise_floor_right = self.alpha_up * self.noise_floor_right + (1 - self.alpha_up) * rms_right
        
                vad_left = rms_left > self.vad_threshold * self.noise_floor_left
                vad_right = rms_right > self.vad_threshold * self.noise_floor_right
            else:
                vad_left = False
                vad_right = False

            update_filter_left = not vad_left
            update_filter_right = not vad_right
        
            # Statistieken bijhouden BEGIN TEST
            self.vad_evals += 1
            if update_filter_left:
                self.vad_update_count_left += 1
            if update_filter_right:
                self.vad_update_count_right += 1    
            #EINDE TEST

            #  Analysis: window + FFT
            windowed = self.audio_buffer * self.window[:, np.newaxis]
            frame_fft = np.fft.rfft(windowed, n=self.L, axis=0)
            freqs = np.fft.rfftfreq(self.L, d=1/self.fs)

            #  MUSIC DOA 
            valid_k = np.arange(1, self.L // 2)

            Y = frame_fft[valid_k, :, np.newaxis] 
            R_k_all = Y @ Y.conj().transpose(0, 2, 1) 
            
            self.Ryy[valid_k] = self.beta * self.Ryy[valid_k] + (1 - self.beta) * R_k_all

            _, eigvecs = np.linalg.eigh(self.Ryy[valid_k])
            En = eigvecs[:, :, :self.M - self.Q] 

            # Vermenigvuldig met self.A_lut 
            En_H_A = En.conj().transpose(0, 2, 1) @ self.A_lut[valid_k] 
            denom = np.sum(np.abs(En_H_A) ** 2, axis=1) 
            pseudospectra = 1.0 / denom 

            pseudospectra = np.clip(pseudospectra, 1e-10, None)
            log_p = np.log(pseudospectra)
            p_geom = np.exp(np.mean(log_p, axis=0)) # Resulteert in exact 20 datapunten
            spectrum_geom_db = 10 * np.log10(p_geom / np.max(p_geom))
            
            # Direct Mappen op de 20 hoeken (Geen find_peaks meer nodig)
            # Direct Mappen op de 20 hoeken 
            left_mask = self.lut_angles > 90
            right_mask = self.lut_angles <= 90
            
            # Argmax pakt simpelweg het hoogste punt per kant
            best_left_idx = np.argmax(np.where(left_mask, spectrum_geom_db, -np.inf))
            best_right_idx = np.argmax(np.where(right_mask, spectrum_geom_db, -np.inf))
                
            # Accepteer altijd de sterkste locatie (geen threshold restricties meer)
            self.last_angle_left = self.lut_angles[best_left_idx]
            self.last_angle_right = self.lut_angles[best_right_idx]

            angle_left = self.last_angle_left
            angle_right = self.last_angle_right

            # FD-GSC
            W_FAS_L, B_L = self._get_lut_for_angle(angle_left)
            W_FAS_R, B_R = self._get_lut_for_angle(angle_right)

            out_fft_left = self._apply_gsc_frame(frame_fft, W_FAS_L, B_L, self.w_nlms_left, update_filter_left)
            out_fft_right = self._apply_gsc_frame(frame_fft, W_FAS_R, B_R, self.w_nlms_right, update_filter_right)

            # Synthesis: IFFT + window + overlap-add
            out_time_left = np.fft.irfft(out_fft_left, n=self.L) * self.window
            out_time_right = np.fft.irfft(out_fft_right, n=self.L) * self.window

            self.ola_buffer_left += out_time_left
            self.ola_buffer_right += out_time_right

            sig0_hop = self.ola_buffer_left[:self.hop].copy()
            sig1_hop = self.ola_buffer_right[:self.hop].copy()

            self.ola_buffer_left = np.concatenate([self.ola_buffer_left[self.hop:], np.zeros(self.hop)])
            self.ola_buffer_right = np.concatenate([self.ola_buffer_right[self.hop:], np.zeros(self.hop)])

            # SIR berekening
            sir = self._last_sir
            if has_gt:
                windowed_gt0 = self.audio_buffer_gt0 * self.window[:, np.newaxis]
                windowed_gt1 = self.audio_buffer_gt1 * self.window[:, np.newaxis]
                fft_gt0 = np.fft.rfft(windowed_gt0, n=self.L, axis=0)
                fft_gt1 = np.fft.rfft(windowed_gt1, n=self.L, axis=0)

                w_left_eval = self.w_nlms_left.copy()
                out_fft_L_gt0 = self._apply_gsc_frame(fft_gt0, W_FAS_L, B_L, w_left_eval, False)
                out_fft_L_gt1 = self._apply_gsc_frame(fft_gt1, W_FAS_L, B_L, w_left_eval, False)
                # Overlap-add voor SIR-signalen (zelfde principe als de hoofd-output)
                out_L_gt0_full = np.fft.irfft(out_fft_L_gt0, n=self.L) * self.window
                out_L_gt1_full = np.fft.irfft(out_fft_L_gt1, n=self.L) * self.window
                self.ola_buffer_L_gt0 += out_L_gt0_full
                self.ola_buffer_L_gt1 += out_L_gt1_full
                out_L_gt0_hop = self.ola_buffer_L_gt0[:self.hop].copy()
                out_L_gt1_hop = self.ola_buffer_L_gt1[:self.hop].copy()
                self.ola_buffer_L_gt0 = np.concatenate([self.ola_buffer_L_gt0[self.hop:], np.zeros(self.hop)])
                self.ola_buffer_L_gt1 = np.concatenate([self.ola_buffer_L_gt1[self.hop:], np.zeros(self.hop)])

                w_right_eval = self.w_nlms_right.copy()
                out_fft_R_gt0 = self._apply_gsc_frame(fft_gt0, W_FAS_R, B_R, w_right_eval, False)
                out_fft_R_gt1 = self._apply_gsc_frame(fft_gt1, W_FAS_R, B_R, w_right_eval, False)
                out_R_gt0_full = np.fft.irfft(out_fft_R_gt0, n=self.L) * self.window
                out_R_gt1_full = np.fft.irfft(out_fft_R_gt1, n=self.L) * self.window
                self.ola_buffer_R_gt0 += out_R_gt0_full
                self.ola_buffer_R_gt1 += out_R_gt1_full
                out_R_gt0_hop = self.ola_buffer_R_gt0[:self.hop].copy()
                out_R_gt1_hop = self.ola_buffer_R_gt1[:self.hop].copy()
                self.ola_buffer_R_gt0 = np.concatenate([self.ola_buffer_R_gt0[self.hop:], np.zeros(self.hop)])
                self.ola_buffer_R_gt1 = np.concatenate([self.ola_buffer_R_gt1[self.hop:], np.zeros(self.hop)])
                
                self.sir_buf_y_left.append(sig0_hop)
                self.sir_buf_y_right.append(sig1_hop)
                self.sir_buf_L_gt0.append(out_L_gt0_hop)
                self.sir_buf_L_gt1.append(out_L_gt1_hop)
                self.sir_buf_R_gt0.append(out_R_gt0_hop)
                self.sir_buf_R_gt1.append(out_R_gt1_hop)
                self.sir_buf_count += self.hop

                if self.sir_buf_count >= self.sir_window_samples and len(self.sir_buf_y_left) > 0:
                    y_left_full = np.concatenate(self.sir_buf_y_left)
                    y_right_full = np.concatenate(self.sir_buf_y_right)
                    L_gt0_full = np.concatenate(self.sir_buf_L_gt0)
                    L_gt1_full = np.concatenate(self.sir_buf_L_gt1)
                    R_gt0_full = np.concatenate(self.sir_buf_R_gt0)
                    R_gt1_full = np.concatenate(self.sir_buf_R_gt1)
                    gt_vec = np.ones(len(L_gt0_full))

                    sir_left = compute_sir(y_left_full, L_gt0_full, L_gt1_full, gt_vec)
                    sir_right = compute_sir(y_right_full, R_gt1_full, R_gt0_full, gt_vec)
                    sir = sir_left if self.attended_left else sir_right
                    if np.isnan(sir):
                        sir = 0.0
                    self._last_sir = sir
                    self._last_sir_left = sir_left if not np.isnan(sir_left) else 0.0
                    self._last_sir_right = sir_right if not np.isnan(sir_right) else 0.0

                    self.sir_history_left.append(self._last_sir_left)
                    self.sir_history_right.append(self._last_sir_right)

                    self.sir_buf_y_left = []
                    self.sir_buf_y_right = []
                    self.sir_buf_L_gt0 = []
                    self.sir_buf_L_gt1 = []
                    self.sir_buf_R_gt0 = []
                    self.sir_buf_R_gt1 = []
                    self.sir_buf_count = 0

                    print(f"  [SIR-1s] left={self._last_sir_left:.2f} dB, right={self._last_sir_right:.2f} dB, attended={'L' if self.attended_left else 'R'} -> {sir:.2f} dB")

            # Push output naar de queues (per hop)
            self.data_queue_phase1.put_nowait(
                (sig0_hop.astype(np.float32), sig1_hop.astype(np.float32),
                angle_left, angle_right, sir)
            )

            # Roll & schrijf nieuwe hop in ring buffer (efficiënt, geen reallocate)
            self.beamformer_buffer_left = np.roll(self.beamformer_buffer_left, -self.hop)
            self.beamformer_buffer_left[-self.hop:] = sig0_hop.astype(np.float32)
            self.beamformer_buffer_right = np.roll(self.beamformer_buffer_right, -self.hop)
            self.beamformer_buffer_right[-self.hop:] = sig1_hop.astype(np.float32)
            self.beamformer_buffer_filled = min(self.beamformer_buffer_filled + self.hop, 
                                                self.beamformer_buffer_max_samples)

            sig_out = sig0_hop if self.attended_left else sig1_hop
            speaker = 0 if self.attended_left else 1
            self.data_queue_phase3.put_nowait((speaker, sig_out.astype(np.float32)))

    def processing_eeg_gt_audio(self, eeg, sig_left_clean, sig_right_clean):
        """
        eeg : (N_eeg, 64) at 128 Hz, ~5 seconden
        sig_left_clean, sig_right_clean : (N_audio,) at 16000 Hz, ~5 seconden
        """
        # === BEAMFORMER MODE: overschrijf clean speech met eigen beamformer output ===
        if self.use_beamformer_for_aad:
            if self.beamformer_buffer_filled < self.beamformer_buffer_max_samples:
                # Buffer nog niet vol → skip
                self.data_queue_phase2.put_nowait(0.5)
                return
            sig_left_clean = self.beamformer_buffer_left.copy()
            sig_right_clean = self.beamformer_buffer_right.copy()
        # === EINDE BEAMFORMER MODE ===


        t0 = time.time()
        # Preprocessing 
        eeg_proc = preprocess_eeg(eeg, fs_in=self.eeg_fs_in, fs_out=64)
        t1=time.time()
        env_left = compute_audio_envelope(sig_left_clean.astype(np.float32),
                                        sr_in=self.audio_fs_in, sr_out=64)
        t2=time.time()
        env_right = compute_audio_envelope(sig_right_clean.astype(np.float32),
                                        sr_in=self.audio_fs_in, sr_out=64)
        t3=time.time()

        # Truncate naar exact aad_window_samples
        n = self.aad_window_samples
        eeg_proc = eeg_proc[:n] #tegen afrondingseffecten bij resample
        env_left = env_left[:n]
        env_right = env_right[:n]

        # Naar model formaat
        eeg_in = eeg_proc[np.newaxis, :, :].astype(np.float32)         # (1, 320, 64) want keras verwacht 3d tensor dus extra batch dimensie
        env1_in = env_left[np.newaxis, :, np.newaxis].astype(np.float32) #(1,320,1 kanaal)
        env2_in = env_right[np.newaxis, :, np.newaxis].astype(np.float32)

        # Predictie
        pred = self.aad_model([eeg_in, env1_in, env2_in], training=False) #gebruik model zelf als functie
        pred_prob =1.0-float(pred[0, 0])
        t4=time.time()
        print(f"\n--- AAD Timing Breakdown ---")
        print(f"EEG Preprocessing:   {(t1 - t0)*1000:.1f} ms")
        print(f"Audio L Envelope:    {(t2 - t1)*1000:.1f} ms")
        print(f"Audio R Envelope:    {(t3 - t2)*1000:.1f} ms")
        print(f"Model Inference:     {(t4 - t3)*1000:.1f} ms")
        print(f"TOTALE AAD TIJD:     {(t4 - t0)*1000:.1f} ms\n")

        #  EMA Filter toepassen (mengt de nieuwe voorspelling met de historie)
        self.ema_filtered = (self.ema_alpha * pred_prob) + ((1.0 - self.ema_alpha) * self.ema_filtered)

        #  Beslissing baseren op de gefilterde (stabielere) kans
        self.attended_left = round(self.ema_filtered)

        #  Stuur de gefilterde kans naar de frontend voor een vloeiendere grafiek
        self.data_queue_phase2.put_nowait(self.ema_filtered)