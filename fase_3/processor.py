"""Ingevulde versie van het skeleton's processor.py voor fase 3 week 1.

Past de algoritmes uit deadline1/week4.ipynb (FD-GSC + MUSIC) toe in streaming-mode op
de chunks die door processing.py worden aangeleverd. Een LSTM (fase 2) wordt nog niet
ingebouwd; processing_eeg_gt_audio behoudt de placeholder zodat het skeleton blijft draaien
tot het dilated+lstm model van Colab binnen is.

Voor de actual algoritmes: zie fase_3/algorithms/.
"""
import asyncio
import os
import pickle

import numpy as np

# Importeer onze streaming wrappers (zelfde algoritmes als deadline1/week4.ipynb,
# alleen herschreven om frame-by-frame te werken zoals fase 3 vraagt).
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from algorithms.lut_builder import build_lut_from_rirs
from algorithms.streaming_doa import StreamingMUSIC, split_left_right
from algorithms.streaming_gsc import StreamingFDGSC
from algorithms.streaming_sir import StreamingSIR


# Default audio-parameters (zoals afgesproken: L=512 voor 32ms frames bij fs=16kHz)
DEFAULT_FS = 16000
DEFAULT_L = 512
DEFAULT_HOP = 256
DEFAULT_BETA = 0.92  # exponentiele middeling R_yy (Part 2)
DEFAULT_MU = 0.001   # NLMS step (klein gehouden om target-leakage te vermijden bij 16 kHz data)
DEFAULT_NUM_MICS_LMA = 5

# Data-pad: kan via env var PHASE3_DATA_DIR geconfigureerd worden, anders default.
DEFAULT_DATA_DIR = os.environ.get(
    "PHASE3_DATA_DIR",
    "/Users/macbookmats/Desktop/P_D_ISSP_base-main/documents_and_given_code/phase_3/phase3_audioData/audiodata_batch_1/anechoic",
)


class Processor:
    """Streaming processor voor fase 3.

    Werkt op een vaste chunk-grootte zoals het skeleton (default 32 chunks/sec van 500 samples
    audio elk bij fs=16 kHz). Houdt R_yy(omega), w_nlms en STFT-buffers aan tussen chunks.
    """

    def __init__(self, data_dir=None, fs=DEFAULT_FS, L=DEFAULT_L, hop=DEFAULT_HOP,
                 beta=DEFAULT_BETA, mu=DEFAULT_MU,
                 doa_update_every=4, bin_range="auto", combine="geometric"):
        # State voor fase 2 / 3 (welke spreker is "attended")
        self.attended_left = 1

        # Output-queues (door het skeleton geconsumeerd)
        self.data_queue_phase1 = asyncio.Queue()
        self.data_queue_phase2 = asyncio.Queue()
        self.data_queue_phase3 = asyncio.Queue()

        # Audio-config
        self.fs = fs
        self.L = L
        self.hop = hop
        self.beta = beta
        self.mu = mu
        self.doa_update_every = doa_update_every

        # Laad RIRs + mic-config voor LUT-opbouw
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        self.data_dir = data_dir
        params, rirs, thetas = self._load_array_setup(data_dir)
        self.mic_pos = np.array(params["LMAcoords"])
        M = self.mic_pos.shape[0]
        self.M = M

        # Bin-range bepalen
        if bin_range == "full":
            br = (1, L // 2)
        elif bin_range == "auto":
            dists = [np.linalg.norm(self.mic_pos[i] - self.mic_pos[j])
                     for i in range(M) for j in range(i + 1, M)]
            d_min = min(dists) if dists else 0.1
            f_alias = 343.0 / (2.0 * d_min)
            k_alias = int(round(f_alias / (fs / L)))
            br = (2, max(8, min(L // 2, k_alias)))
        else:
            br = bin_range  # tuple

        # LUT (FAS BF + Blocking Matrix) voor alle gemeten RIR-hoeken
        self.lut, self.angles_lut = build_lut_from_rirs(rirs, thetas, L=L)

        # Streaming MUSIC met exponentiele R_yy middeling (Part 2)
        self.music = StreamingMUSIC(
            mic_pos=self.mic_pos,
            fs=fs,
            L=L,
            num_sources=2,
            beta=beta,
            bin_range=br,
            combine=combine,
        )

        # Twee FD-GSCs: een per spreker (Part 1 + Part 4)
        self.gsc_left = StreamingFDGSC(self.lut, self.angles_lut, M, L=L, hop=hop, mu=mu, side="left")
        self.gsc_right = StreamingFDGSC(self.lut, self.angles_lut, M, L=L, hop=hop, mu=mu, side="right")

        # Per-frame SIR-trackers (Part 3)
        self.sir_left = StreamingSIR(fs=fs, window_seconds=2.0)
        self.sir_right = StreamingSIR(fs=fs, window_seconds=2.0)

        # Persistente DOA-schattingen (raw MUSIC, niet de LUT-snapped)
        self.raw_doa_left = float("nan")
        self.raw_doa_right = float("nan")

        # MUSIC-buffer voor sliding STFT (50% overlap)
        self.music_buf = np.zeros((0, M), dtype=np.float64)
        from scipy import signal as _sig
        self._music_window = np.sqrt(_sig.windows.hann(L, sym=False))

        # Chunk-teller
        self.chunk_count = 0

    @staticmethod
    def _load_array_setup(data_dir):
        """Laad params.pkl + lma_*kHz*.npz (anechoic of reverberant)."""
        with open(os.path.join(data_dir, "params.pkl"), "rb") as f:
            params = pickle.load(f)
        # zoek het lma_*.npz bestand
        rir_filename = None
        for fname in os.listdir(data_dir):
            if fname.startswith("lma_") and fname.endswith(".npz"):
                rir_filename = fname
                break
        if rir_filename is None:
            raise FileNotFoundError(f"Geen lma_*.npz gevonden in {data_dir}")
        rir_data = np.load(os.path.join(data_dir, rir_filename))
        return params, rir_data["rirs"], rir_data["thetas"]

    # -------------------------------------------------------------- #
    #                        Audio-pipeline                          #
    # -------------------------------------------------------------- #
    def processing_microarray(self, lma, lma_gt_0=None, lma_gt_1=None):
        """Verwerk één LMA-chunk.

        Parameters
        ----------
        lma      : (chunk_n, M) int16   -- microfoonsignalen (mix)
        lma_gt_0 : (chunk_n, M) int16 of None -- bijdrage van left speaker (oracle, voor SIR)
        lma_gt_1 : (chunk_n, M) int16 of None -- bijdrage van right speaker (oracle, voor SIR)
        """
        ci = self.chunk_count
        self.chunk_count += 1

        mix_c = lma.astype(np.float32)
        tar_l = lma_gt_0.astype(np.float32) if lma_gt_0 is not None else None  # left = target voor gsc_left
        tar_r = lma_gt_1.astype(np.float32) if lma_gt_1 is not None else None  # right = target voor gsc_right

        # ---- Part 2: Dynamic DOA estimation met exp. R_yy averaging ----
        self.music_buf = np.concatenate([self.music_buf, mix_c.astype(np.float64)], axis=0)
        while self.music_buf.shape[0] >= self.L:
            frame = self.music_buf[: self.L, :] * self._music_window[:, None]
            Y = np.fft.rfft(frame, n=self.L, axis=0)
            self.music.update(Y)
            self.music_buf = self.music_buf[self.hop :, :]

        if ci % self.doa_update_every == 0 and self.music.initialized:
            doas = self.music.estimate_doas()
            l, r = split_left_right(doas)
            if not np.isnan(l):
                self.raw_doa_left = l
            if not np.isnan(r):
                self.raw_doa_right = r
            self.gsc_left.set_doa(self.raw_doa_left)
            self.gsc_right.set_doa(self.raw_doa_right)

        # ---- Part 1 + 4: twee streaming FD-GSCs voor moving sources ----
        # gsc_left: target = left, interferer = right
        out_mix_L, out_tar_L, out_int_L = self.gsc_left.process_chunk(mix_c, tar_l, tar_r)
        # gsc_right: target = right, interferer = left
        out_mix_R, out_tar_R, out_int_R = self.gsc_right.process_chunk(mix_c, tar_r, tar_l)

        # ---- Part 3: SIR per chunk ----
        sir_l = self.sir_left.update(out_tar_L, out_int_L) if tar_l is not None else float("nan")
        sir_r = self.sir_right.update(out_tar_R, out_int_R) if tar_r is not None else float("nan")
        # Frontend laat 1 SIR-waarde zien -> kies die van de huidig gevolgde spreker
        sir_active = sir_l if self.attended_left else sir_r

        # Phase 1 output: links- en rechts-getargete streams + DOAs + SIR
        # (int16 voor frontend; clip op 32767)
        beam_left_i16 = np.clip(np.nan_to_num(out_mix_L), -32768, 32767).astype(np.int16)
        beam_right_i16 = np.clip(np.nan_to_num(out_mix_R), -32768, 32767).astype(np.int16)

        self.data_queue_phase1.put_nowait((
            beam_left_i16,
            beam_right_i16,
            float(self.raw_doa_left) if not np.isnan(self.raw_doa_left) else 90.0,
            float(self.raw_doa_right) if not np.isnan(self.raw_doa_right) else 90.0,
            float(sir_active) if not np.isnan(sir_active) else 0.0,
        ))

        # Phase 3 output: select welke spreker te outputten op basis van attended_left
        sig_out = beam_left_i16 if self.attended_left else beam_right_i16
        speaker = 0 if self.attended_left else 1
        self.data_queue_phase3.put_nowait((speaker, sig_out))

    # -------------------------------------------------------------- #
    #                     EEG-pipeline (placeholder)                 #
    # -------------------------------------------------------------- #
    def processing_eeg_gt_audio(self, eeg, sig_left_clean, sig_right_clean):
        """PLACEHOLDER voor fase 2 LSTM (dilated+lstm uit Colab).

        TODO: wanneer mats het Colab-model uploadt:
        1) load model (tf.keras.models.load_model(...))
        2) extract envelopes uit sig_left_clean / sig_right_clean (Gammatone of Hilbert+LP)
        3) run model.predict([eeg, env1, env2]) -> pred_prob
        4) self.attended_left = round(pred_prob)

        Voor week 1 van fase 3 (audio integration) is een placeholder voldoende -- de
        EEG/AAD-integratie is grotendeels week 2-werk.
        """
        # placeholder zoals het skeleton
        pred_prob = float(np.random.random() * 0.8 + 0.1)
        self.attended_left = int(round(pred_prob))
        self.data_queue_phase2.put_nowait(pred_prob)
