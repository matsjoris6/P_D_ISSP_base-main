import asyncio
import os
import numpy as np
import scipy.linalg
from scipy import signal
from collections import deque
import logging
logging.getLogger('brian2').setLevel(logging.ERROR)
import brian2
brian2.prefs.codegen.target = 'cython'
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

_LMA_COORDS = np.array([
    [4.699999999999999, 2.2],
    [4.699999999999999, 2.3000000000000003],
    [4.699999999999999, 2.4],
    [4.699999999999999, 2.5],
    [4.699999999999999, 2.6],
])



from config import RIR_PATH as _RIR_PATH   # anechoic of reverberant, bepaald door SCENARIO in config.py
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
    # Reset Brian2's global object registry zodat herhaalde aanroepen
    # geen conflicterende NeuronGroup/Network-objecten ophopen.
    brian2.start_scope()
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
    def __init__(self, fs=16000, rir_path=_RIR_PATH):
        self.attended_left = 1

        # Output 'pipes'
        self.data_queue_phase1 = asyncio.Queue()
        self.data_queue_phase2 = asyncio.Queue()
        self.data_queue_phase3 = asyncio.Queue()

        
        self.fs = fs
        self.M = 5  # Aantal LMA microfoons
        self.L = 1024  # FFT Window size
        self.beta = 0.85  
        self.c = 343.0
        self.Q = 2  # Aantal sprekers
        self.mu = 0.01  # NLMS stapgrootte

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

        # AAD model laden — instellingen komen uit config.py (ACTIVE_MODEL)
        from config import MODEL_PATH, EEG_WINDOW_SAMPLES, EMA_ALPHA, SCHMITT_THRESHOLD, SCHMITT_HYSTERESIS, ACTIVE_MODEL, USE_GSC_AUDIO_FOR_AAD
        print(f"[INFO] AAD model: {ACTIVE_MODEL}  ({MODEL_PATH})")
        self.aad_model = tf.keras.models.load_model(MODEL_PATH)
        self.aad_window_samples = EEG_WINDOW_SAMPLES  # bijv. 320 (5s) of 640 (10s)
        self.eeg_fs_in   = 128    # raw EEG sample rate
        self.audio_fs_in = 48000  # clean speech sample rate (stimuli)

        # ── Audio-bron voor AAD ───────────────────────────────────────────────
        # False: clean speech (stimuli, 48 kHz) — ideale omstandigheid, standaard.
        # True : GSC-output (beamformer, 16 kHz) — realistisch, levert extra punten.
        self.use_gsc_audio_for_aad = USE_GSC_AUDIO_FOR_AAD
        if self.use_gsc_audio_for_aad:
            print("[INFO] AAD audio-bron: GSC-output van de beamformer (16 kHz)")
        else:
            print("[INFO] AAD audio-bron: clean speech van de stimuli (48 kHz)")

        # Rolling buffer voor GSC-audio (gevuld door processing_microarray).
        # Bevat de laatste WINDOW_SEC seconden aan beamformer-output (@ self.fs = 16 kHz).
        # Grootte: EEG_WINDOW_SAMPLES samples @ 64 Hz × (16000/64) = WINDOW_SEC × 16000.
        _gsc_buf_len = EEG_WINDOW_SAMPLES * (self.fs // 64)   # bijv. 320 × 250 = 80 000
        self._gsc_buf_left  = deque(maxlen=_gsc_buf_len)
        self._gsc_buf_right = deque(maxlen=_gsc_buf_len)

        # ── Filter toggles ────────────────────────────────────────────────────
        # Zet op True voor reverb, False voor anechoisch (of om uit te zetten)
        self.use_aad_filter = True    # EMA + Schmitt (parameters uit config.py)
        self.use_doa_filter = True    # Causaal mediaan N=63 op DOA-schattingen
        self.use_vad_filter = False   # Reverb-geoptimaliseerde VAD-parameters

        # ── AAD filter: EMA + Schmitt trigger (uit config.py) ────────────────
        self.ema_alpha          = EMA_ALPHA
        self.ema_filtered       = 0.5              # interne staat, niet aanpassen
        self.schmitt_threshold  = SCHMITT_THRESHOLD
        self.schmitt_hysteresis = SCHMITT_HYSTERESIS
        self.schmitt_state      = 0

        # ── DOA filter: causaal op de MUSIC-schattingen ──────────────────────
        # Kies filter type:  "median"  of  "ema"  of  "none"
        # Beste uit test_doa_filter.py: "median" met N=63
        self.doa_filter_type = "median"
        self.doa_filter_N    = 63      # mediaan: 3, 7, 15, 31, 63, 127, 255
        self.doa_ema_alpha   = 0.4     # ema: 0.2 (traag/stabiel) … 0.8 (snel/reactief)

        # Ring-buffer voor mediaan filter
        self.doa_buf_left  = deque(maxlen=self.doa_filter_N)
        self.doa_buf_right = deque(maxlen=self.doa_filter_N)
        # EMA-staat voor ema filter
        self.doa_ema_left  = None
        self.doa_ema_right = None

        # PRE-COMPUTING: RIR Steering Vectors & GSC Filters
        rir_data = np.load(rir_path)
        rirs = rir_data["rirs"]   
        doas = rir_data["thetas"] 
        
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

        # ── VAD parameters ───────────────────────────────────────────────────
        # Beste waarden uit test_vad.py voor reverb:
        #   alpha_up=0.99, alpha_down=0.9, vad_threshold=0.5
        # Anechoisch (snellere reactie, minder conservatief):
        #   alpha_up=0.95, alpha_down=0.8, vad_threshold=0.5
        self.noise_floor_left  = None
        self.noise_floor_right = None
        self.vad_threshold     = 0.5
        if self.use_vad_filter:   # reverb-geoptimaliseerd
            self.alpha_up   = 0.99
            self.alpha_down = 0.9
        else:                     # anechoisch
            self.alpha_up   = 0.95
            self.alpha_down = 0.8

        # ── Statistieken voor eindrapport ─────────────────────────────────────
        self._stat_doa_left_raw      = []   # ruwe MUSIC-schatting links (per hop)
        self._stat_doa_right_raw     = []   # ruwe MUSIC-schatting rechts (per hop)
        self._stat_doa_left_filt     = []   # na DOA-filter links (per hop)
        self._stat_doa_right_filt    = []   # na DOA-filter rechts (per hop)
        self._stat_vad_left          = []   # VAD-beslissing links (True=spraak, per hop)
        self._stat_vad_right         = []   # VAD-beslissing rechts (True=spraak, per hop)
        self._stat_sir_left          = []   # SIR links per 1s-venster
        self._stat_sir_right         = []   # SIR rechts per 1s-venster
        self._stat_aad_raw_decisions = []   # ruwe modelkans: 0=L/1=R (hoge pred_prob = RIGHT)
        self._stat_aad_filt_decisions= []   # na AAD-filter (EMA+Schmitt) → 0=L/1=R
        self._stat_aad_gt_labels     = []   # GT attended speaker per venster (0=L, 1=R)
        self._stat_aad_times_ms      = []   # verwerkingstijd AAD in ms
        self._doa_gt_raw             = None # ruwe gt.npz data voor MAE-berekening (optioneel)
        self._aad_hop_seconds        = 5    # stap tussen opeenvolgende inferenties (ingesteld door processing.py)
 
    def _get_lut_for_angle(self, doa):
        """Pakt dichtstbijzijnde hoek uit de LUT."""
        closest_angle = self.lut_angles[np.argmin(np.abs(self.lut_angles - doa))]
        return self.lut[float(closest_angle)]

    def set_doa_gt_raw(self, gt_npz_path):
        """
        Laad ruwe DOA ground truth uit gt.npz voor MAE-berekening in eindstatistiek.
        Wordt aangeroepen vanuit processing.py na verbinding met server.
        """
        if os.path.exists(gt_npz_path):
            self._doa_gt_raw = np.load(gt_npz_path)
            print(f"[INFO] DOA ground truth geladen: {gt_npz_path}")
        else:
            print(f"[WARN] DOA ground truth niet gevonden: {gt_npz_path} (MAE niet beschikbaar)")

    def record_aad_gt(self, window_gt):
        """
        Sla de ground truth op voor het zojuist verwerkte AAD-venster.
        window_gt : 0.0 = links geattendeerd, 1.0 = rechts geattendeerd.
        Wordt aangeroepen vanuit processing.py na processing_eeg_gt_audio().
        """
        self._stat_aad_gt_labels.append(float(window_gt))
    
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
            PEAK_THRESHOLD = -12.0
            
            # Deel de 20 hoeken op in links en rechts
            left_mask = self.lut_angles > 90
            right_mask = self.lut_angles <= 90
            
            # np.where negeert de foute kant (-inf), argmax pakt simpelweg het hoogste punt
            best_left_idx = np.argmax(np.where(left_mask, spectrum_geom_db, -np.inf))
            best_right_idx = np.argmax(np.where(right_mask, spectrum_geom_db, -np.inf))
                
            # Check of de gevonden piek hard genoeg is (boven threshold)
            if spectrum_geom_db[best_left_idx] > PEAK_THRESHOLD:
                self.last_angle_left = self.lut_angles[best_left_idx]
            if spectrum_geom_db[best_right_idx] > PEAK_THRESHOLD:
                self.last_angle_right = self.lut_angles[best_right_idx]

            # ── DOA filter ───────────────────────────────────────────────────
            # Stel in via: self.use_doa_filter, self.doa_filter_type, self.doa_filter_N / self.doa_ema_alpha
            # "median" N=63 : causaal venster van 63 hops — verwijdert uitschieters, robuust tegen reverb
            # "ema"   α=0.4 : exponentieel gemiddelde — geen aanlooptijd, maar reageert trager op echte hoekwijzigingen
            # "none"        : ruwe MUSIC met last-valid fallback — snel, maar onstabiel bij reverb
            raw_l = self.last_angle_left
            raw_r = self.last_angle_right

            if self.use_doa_filter and self.doa_filter_type == "median":
                self.doa_buf_left.append(raw_l)
                self.doa_buf_right.append(raw_r)
                angle_left  = float(np.median(self.doa_buf_left))
                angle_right = float(np.median(self.doa_buf_right))
            elif self.use_doa_filter and self.doa_filter_type == "ema":
                if self.doa_ema_left is None:
                    self.doa_ema_left, self.doa_ema_right = raw_l, raw_r
                self.doa_ema_left  = self.doa_ema_alpha * raw_l + (1 - self.doa_ema_alpha) * self.doa_ema_left
                self.doa_ema_right = self.doa_ema_alpha * raw_r + (1 - self.doa_ema_alpha) * self.doa_ema_right
                angle_left  = self.doa_ema_left
                angle_right = self.doa_ema_right
            else:
                angle_left  = raw_l
                angle_right = raw_r

            # Statistieken bijhouden (per hop)
            self._stat_doa_left_raw.append(raw_l)
            self._stat_doa_right_raw.append(raw_r)
            self._stat_doa_left_filt.append(angle_left)
            self._stat_doa_right_filt.append(angle_right)
            self._stat_vad_left.append(bool(vad_left) if has_gt else None)
            self._stat_vad_right.append(bool(vad_right) if has_gt else None)

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

            # Vul de GSC-audio rolling buffer (voor AAD als use_gsc_audio_for_aad=True).
            # sig0_hop = linkerbeam, sig1_hop = rechterbeam — beide @ self.fs (16 kHz).
            if self.use_gsc_audio_for_aad:
                self._gsc_buf_left.extend(sig0_hop.astype(np.float32))
                self._gsc_buf_right.extend(sig1_hop.astype(np.float32))

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
                    self._stat_sir_left.append(self._last_sir_left)
                    self._stat_sir_right.append(self._last_sir_right)

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
            sig_out = sig0_hop if self.attended_left else sig1_hop
            speaker = 0 if self.attended_left else 1
            self.data_queue_phase3.put_nowait((speaker, sig_out.astype(np.float32)))

    def processing_eeg_gt_audio(self, eeg, sig_left_clean, sig_right_clean):
        """
        eeg : (N_eeg, 64) at 128 Hz, ~WINDOW_SEC seconden
        sig_left_clean, sig_right_clean : (N_audio,) at 48000 Hz — clean speech (stimuli).

        Als use_gsc_audio_for_aad=True, worden sig_left_clean/sig_right_clean genegeerd
        en wordt de GSC rolling buffer (_gsc_buf_left/_gsc_buf_right) gebruikt in plaats
        daarvan. Die buffer bevat de laatste WINDOW_SEC seconden beamformer-output @ 16 kHz.
        """
        t0 = time.time()
        # Preprocessing EEG
        eeg_proc = preprocess_eeg(eeg, fs_in=self.eeg_fs_in, fs_out=64)
        t1 = time.time()

        # ── Audio-bron selectie ───────────────────────────────────────────────
        if self.use_gsc_audio_for_aad and len(self._gsc_buf_left) == self._gsc_buf_left.maxlen:
            # GSC-output: snapshot van de rolling buffer (@ self.fs = 16 kHz)
            audio_left  = np.array(self._gsc_buf_left,  dtype=np.float32)
            audio_right = np.array(self._gsc_buf_right, dtype=np.float32)
            audio_fs    = self.fs          # 16 000 Hz

            # DEBUG: swap GSC links/rechts om te testen of er een DOA-swap is.
            # Aanzetten via --gsc_swap CLI of via processor.gsc_swap = True.
            if getattr(self, "gsc_swap", False):
                audio_left, audio_right = audio_right, audio_left
        else:
            # Clean speech van de stimuli (@ self.audio_fs_in = 48 kHz) — standaard.
            # Ook als fallback als de GSC buffer nog niet vol is (eerste WINDOW_SEC).
            audio_left  = sig_left_clean.astype(np.float32)
            audio_right = sig_right_clean.astype(np.float32)
            audio_fs    = self.audio_fs_in  # 48 000 Hz

        env_left = compute_audio_envelope(audio_left,  sr_in=audio_fs, sr_out=64)
        t2 = time.time()
        env_right = compute_audio_envelope(audio_right, sr_in=audio_fs, sr_out=64)
        t3 = time.time()

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
        totaal_ms = (t4 - t0) * 1000
        self._stat_aad_times_ms.append(totaal_ms)

        # Ruwe beslissing (voor filter): hoge pred_prob = RIGHT (1.0), lage = LEFT (0.0)
        # Convention: pred_prob = 1 - model_output → hoog = model denkt RIGHT
        # (zelfde als Schmitt trigger: ema_filtered >= 0.65 → state=1=RIGHT)
        self._stat_aad_raw_decisions.append(0.0 if (pred_prob < 0.5) else 1.0)

        if self.use_aad_filter:
            # EMA smoother: dempt snelle schommelingen in de ruwe modelkans.
            # Schmitt trigger: wisselt pas van kant als kans drempel ± hysterese overschrijdt.
            # → Samen voorkomen ze flickering bij twijfelgevallen.
            self.ema_filtered = self.ema_alpha * pred_prob + (1 - self.ema_alpha) * self.ema_filtered
            if self.schmitt_state == 0 and self.ema_filtered >= self.schmitt_threshold + self.schmitt_hysteresis:
                self.schmitt_state = 1
            elif self.schmitt_state == 1 and self.ema_filtered <= self.schmitt_threshold - self.schmitt_hysteresis:
                self.schmitt_state = 0
            self.attended_left = (self.schmitt_state == 0)
            queue_val = float(self.schmitt_state)
        else:
            # Geen filter: directe drempel op ruwe kans (gevoelig voor ruis)
            # hoge pred_prob (≥0.5) = RIGHT → attended_left = False
            self.attended_left = (pred_prob < 0.5)
            self.ema_filtered  = pred_prob   # expose ruwe kans voor UI-plot
            queue_val = 0.0 if self.attended_left else 1.0

        # Gefilterd besluit bijhouden voor vergelijking in statistieken
        self._stat_aad_filt_decisions.append(0.0 if self.attended_left else 1.0)

        return pred_prob

    def print_statistics(self):
        """Eindrapport — scorecard voor de huidige run."""
        W    = 68
        SEP  = "=" * W
        SEP2 = "-" * W

        # ────────────────────────────────────────────────────────────────────
        print(f"\n{SEP}")
        print(f"  EINDSTATISTIEKEN LIVE-RUN")
        print(SEP)

        # Filter configuratie
        aad_cfg = (f"AAN  —  EMA α={self.ema_alpha},  Schmitt {self.schmitt_threshold} ± {self.schmitt_hysteresis}"
                   if self.use_aad_filter else "UIT  —  drempel 0.5 op ruwe kans")
        aad_audio = ("GSC-output  (beamformer, 16 kHz  — realistisch)"
                     if self.use_gsc_audio_for_aad else
                     "Clean speech  (stimuli, 48 kHz  — ideaal)")
        if self.use_doa_filter:
            doa_cfg = (f"AAN  —  Mediaan N={self.doa_filter_N}" if self.doa_filter_type == "median"
                       else f"AAN  —  EMA α={self.doa_ema_alpha}")
        else:
            doa_cfg = "UIT  —  ruwe MUSIC"
        vad_cfg = (f"AAN  —  α_up={self.alpha_up},  α_down={self.alpha_down},  thresh={self.vad_threshold}"
                   if self.use_vad_filter else f"UIT  —  α_up={self.alpha_up},  α_down={self.alpha_down}")

        print(f"  AAD-filter  :  {aad_cfg}")
        print(f"  AAD-audio   :  {aad_audio}")
        print(f"  DOA-filter  :  {doa_cfg}")
        print(f"  VAD-filter  :  {vad_cfg}")

        # ── DOA ─────────────────────────────────────────────────────────────
        if self._stat_doa_left_filt:
            fl  = np.array(self._stat_doa_left_filt)
            fr  = np.array(self._stat_doa_right_filt)
            jfl = np.abs(np.diff(fl))
            jfr = np.abs(np.diff(fr))

            # Ground truth MAE berekenen als gt.npz geladen is
            has_doa_gt = False
            if self._doa_gt_raw is not None:
                try:
                    n_hops = len(fl)
                    gt = self._doa_gt_raw
                    dur_l = np.diff(np.insert(gt["endSamples_l"], 0, 0))
                    dur_r = np.diff(np.insert(gt["endSamples_r"], 0, 0))
                    gl = np.concatenate([np.repeat(float(a), int(n)) for a, n in zip(gt["angles_l"], dur_l)])
                    gr = np.concatenate([np.repeat(float(a), int(n)) for a, n in zip(gt["angles_r"], dur_r)])
                    idx = np.minimum(np.arange(n_hops) * self.hop + self.hop // 2, len(gl) - 1)
                    gt_l_hops = gl[idx]
                    gt_r_hops = gr[idx]
                    half_step = float(np.min(np.abs(np.diff(np.sort(self.lut_angles))))) / 2
                    mae_l     = np.mean(np.abs(fl - gt_l_hops))
                    mae_r     = np.mean(np.abs(fr - gt_r_hops))
                    exact_l   = np.mean(np.abs(fl - gt_l_hops) <= half_step) * 100
                    exact_r   = np.mean(np.abs(fr - gt_r_hops) <= half_step) * 100
                    has_doa_gt = True
                except Exception as e:
                    print(f"  [WARN] DOA GT MAE mislukt: {e}")

            print(f"\n{SEP}")
            print(f"  DOA  —  {len(fl)} frames  (~{len(fl)*self.hop//self.fs} s)")
            print(SEP2)
            if has_doa_gt:
                #          label    max-sprong  >10°-sprongen   MAE vs GT   % Exact GT
                print(f"  {'':7}  {'Max sprong':>11}  {'>10° sprongen':>14}  {'MAE vs GT':>10}  {'% Exact GT':>11}")
                print(SEP2)
                print(f"  {'Links':<7}  {np.max(jfl):>10.1f}°  {np.sum(jfl>10):>13}x  {mae_l:>9.1f}°  {exact_l:>10.1f}%")
                print(f"  {'Rechts':<7}  {np.max(jfr):>10.1f}°  {np.sum(jfr>10):>13}x  {mae_r:>9.1f}°  {exact_r:>10.1f}%")
            else:
                print(f"  {'':7}  {'Max sprong':>11}  {'>10° sprongen':>14}")
                print(SEP2)
                print(f"  {'Links':<7}  {np.max(jfl):>10.1f}°  {np.sum(jfl>10):>13}x")
                print(f"  {'Rechts':<7}  {np.max(jfr):>10.1f}°  {np.sum(jfr>10):>13}x")
                print(f"  (GT niet beschikbaar — start met reverberant data voor MAE)")

        # ── SIR ─────────────────────────────────────────────────────────────
        if self._stat_sir_left:
            sl   = np.array(self._stat_sir_left)
            sr   = np.array(self._stat_sir_right)
            half = max(len(sl) // 2, 1)
            tl   = np.mean(sl[half:]) - np.mean(sl[:half])
            tr   = np.mean(sr[half:]) - np.mean(sr[:half])
            att_l = self.attended_left
            sa    = sl if att_l else sr
            ta    = np.mean(sa[half:]) - np.mean(sa[:half])

            print(f"\n{SEP}")
            print(f"  SIR  —  {len(sl)} vensters × 1 s  (hogere SIR = betere scheiding)")
            print(SEP2)
            # label=7  Gem/Med/Min/Max: "{:>6.1f} dB"=9 chars  Trend: "{:>+7.1f} dB"=10 chars
            print(f"  {'':7}  {'Gem':>9}  {'Med':>9}  {'Min':>9}  {'Max':>9}  {'Trend':>10}")
            print(SEP2)
            print(f"  {'Links':<7}  {np.mean(sl):>6.1f} dB  {np.median(sl):>6.1f} dB"
                  f"  {np.min(sl):>6.1f} dB  {np.max(sl):>6.1f} dB  {tl:>+7.1f} dB")
            print(f"  {'Rechts':<7}  {np.mean(sr):>6.1f} dB  {np.median(sr):>6.1f} dB"
                  f"  {np.min(sr):>6.1f} dB  {np.max(sr):>6.1f} dB  {tr:>+7.1f} dB")
            print(SEP2)
            print(f"  Aandacht ({'Links' if att_l else 'Rechts'})  :  gem {np.mean(sa):>5.1f} dB  |  trend {ta:>+5.1f} dB")

        # ── VAD ─────────────────────────────────────────────────────────────
        vad_l = [v for v in self._stat_vad_left  if v is not None]
        vad_r = [v for v in self._stat_vad_right if v is not None]
        if vad_l:
            vl = np.array(vad_l)
            vr = np.array(vad_r)
            print(f"\n{SEP}")
            print(f"  VAD  —  {len(vl)} frames met GT-signalen")
            print(SEP2)
            #         label    % spraak    % GSC-update
            print(f"  {'':7}  {'% Spraak':>9}  {'% GSC-update':>13}")
            print(SEP2)
            print(f"  {'Links':<7}  {np.mean(vl)*100:>8.1f}%  {(1-np.mean(vl))*100:>12.1f}%")
            print(f"  {'Rechts':<7}  {np.mean(vr)*100:>8.1f}%  {(1-np.mean(vr))*100:>12.1f}%")

        # ── AAD ─────────────────────────────────────────────────────────────
        if self._stat_aad_filt_decisions:
            fil      = np.array(self._stat_aad_filt_decisions)
            sw       = int(np.sum(np.diff(fil) != 0))
            n_win    = len(fil)
            hop_s    = self._aad_hop_seconds
            interval = n_win * hop_s / max(sw, 1)

            hop_s = self._aad_hop_seconds
            print(f"\n{SEP}")
            print(f"  AAD  —  {n_win} inferenties  (venster 5 s,  stap {hop_s} s,  totaal ~{n_win*hop_s} s)")
            print(SEP2)

            # Accuracy t.o.v. GT attended speaker (als beschikbaar)
            gt = np.array(self._stat_aad_gt_labels) if self._stat_aad_gt_labels else None
            if gt is not None and len(gt) > 0:
                n_acc = min(len(fil), len(gt))
                accuracy = np.mean(fil[:n_acc] == gt[:n_acc]) * 100
                print(f"  {'Accuracy':<16}:  {accuracy:.1f}%  (over {n_acc} vensters)")
                print(f"  {'':16}   ↳ venster-accuracy: 1 beslissing vs majority-vote GT over {self._aad_hop_seconds*5} s")
                print(f"  {'':16}   ↳ UI avg_accuracy : 1 beslissing vs elke GT-sample in de {self._aad_hop_seconds} s hop")
                print(f"  {'':16}     (UI is lager bij sprekerswissels — gemengde GT-samples tellen mee)")

            print(f"  {'Switches':<16}:  {sw:>4}x  (gem. elke {interval:.0f} s)")

            if self._stat_aad_times_ms:
                t = np.array(self._stat_aad_times_ms)
                print(f"  {'Timing':<16}:  {np.mean(t):>4.0f} ms gem"
                      f"  |  {np.median(t):>4.0f} ms med"
                      f"  |  {np.min(t):>4.0f} ms min"
                      f"  |  {np.max(t):>4.0f} ms max")
                budget_ms = self._aad_hop_seconds * 1000
                print(f"  {'Marge':<16}:  {budget_ms/np.mean(t):.1f}× (budget {budget_ms} ms / stap)")

        print(f"\n{SEP}\n")
