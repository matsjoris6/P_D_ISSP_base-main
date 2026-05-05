"""Ingevulde versie van het skeleton's processor.py voor fase 3 week 1.

Past de algoritmes uit deadline1/week4.ipynb (FD-GSC + MUSIC) toe in streaming-mode op
de chunks die door processing.py worden aangeleverd. Een LSTM (fase 2) wordt nog niet
ingebouwd; processing_eeg_gt_audio behoudt de placeholder zodat het skeleton blijft
draaien tot het dilated+lstm model van Colab binnen is.

Pipeline-overzicht (per chunk binnenkomst van het skeleton):
    1. processing_microarray(lma, lma_gt_0, lma_gt_1) wordt aangeroepen door processing.py
    2. STFT-frames extraheren -> MUSIC update R_yy(omega)
    3. Elke `doa_update_every` chunks: MUSIC peak picking -> set_doa(...) op beide GSCs
    4. GSC LINKS: target=spreker0, interferer=spreker1 -> output naar phase1 queue
    5. GSC RECHTS: target=spreker1, interferer=spreker0 -> output naar phase1 queue
    6. SIR per spreker bijwerken (oracle target/interferer paden)
    7. Phase3 queue: selecteer welke spreker op basis van attended_left (uit fase 2)

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
from algorithms.streaming_doa import StreamingMUSIC, RIRSteeringMUSIC, DOATracker, split_left_right
from algorithms.streaming_gsc import StreamingFDGSC
from algorithms.streaming_sir import StreamingSIR


# ---- Default audio-parameters ----
# L=512 -> 32ms frames bij fs=16kHz. Vergelijkbaar met week4 (op 44.1 kHz).
# Korter (L=256) zou meer reactiviteit geven maar slechtere DOA-resolutie.
# Langer (L=1024) zou stabieler zijn maar trager bij sprekerverplaatsing.
DEFAULT_FS = 16000
DEFAULT_L = 512
DEFAULT_HOP = 256  # 50% overlap voor perfect-reconstruction OLA met sqrt(hann)
DEFAULT_BETA = 0.92  # exp. middelingsconstante R_yy (Part 2). Zie README beta-analyse.
DEFAULT_MU = 0.001   # NLMS-step. Klein t.o.v. week4 (0.1) om target-leakage bij 16kHz
                     # data met bewegende sprekers te beheersen.
DEFAULT_NUM_MICS_LMA = 5  # phase 3 LMA = 5 mics

# Data-pad: kan via env var PHASE3_DATA_DIR geconfigureerd worden, anders default.
DEFAULT_DATA_DIR = os.environ.get(
    "PHASE3_DATA_DIR",
    "/Users/macbookmats/Desktop/P_D_ISSP_base-main/documents_and_given_code/phase_3/phase3_audioData/audiodata_batch_1/anechoic",
)


class Processor:
    """Streaming processor voor fase 3.

    Werkt op een vaste chunk-grootte zoals het skeleton (default 32 chunks/sec van 500
    samples audio elk bij fs=16kHz). Houdt R_yy(omega), w_nlms en STFT-buffers aan
    tussen chunks zodat de algoritmes echt "streaming" zijn (Part 1 vereiste).
    """

    def __init__(self, data_dir=None, fs=DEFAULT_FS, L=DEFAULT_L, hop=DEFAULT_HOP,
                 beta=DEFAULT_BETA, mu=DEFAULT_MU,
                 doa_update_every=4, bin_range="auto", combine="geometric",
                 sv_model="rir", snr_weight=False, use_fb=None,
                 doa_tracker_alpha=0.3, doa_tracker_window=5, doa_tracker_outlier=30.0):
        """
        Parameters
        ----------
        data_dir : pad naar scenario-folder (anechoic of reverberant). Bepaalt de RIRs
                   waaruit de LUT wordt gebouwd. Als None: env var of default.
        fs, L, hop : audio + STFT parameters (zie module-level defaults).
        beta : exp. R_yy middelingsconstante. 0.92 is empirisch optimaal voor fase 3
               (zie beta-analyse in README).
        mu   : NLMS-step. 0.001 voorkomt target-leakage; zie README mu-analyse.
        doa_update_every : update DOA elke N chunks. 4 = 1/8 sec @ 32 chunks/s. Vaker
                           is duurder en niet nodig (sprekers bewegen langzaam t.o.v.
                           audio-snelheid).
        bin_range : 'auto' (= bins onder spatial-aliasing limiet), 'full' (week4-style,
                    alle bins), of tuple (k_min, k_max).
        combine : 'geometric' (week4) of 'arithmetic' pseudospectrum-combiner.
        """
        # State voor fase 2 / 3 (welke spreker is "attended").
        # Bij start altijd links; wordt geupdate door processing_eeg_gt_audio (LSTM).
        self.attended_left = 1

        # Output-queues (door het skeleton geconsumeerd in processing.py)
        self.data_queue_phase1 = asyncio.Queue()  # GSC + DOA + SIR
        self.data_queue_phase2 = asyncio.Queue()  # AAD pred_prob (placeholder)
        self.data_queue_phase3 = asyncio.Queue()  # Geselecteerde spreker-output

        # Audio-config
        self.fs = fs
        self.L = L
        self.hop = hop
        self.beta = beta
        self.mu = mu
        self.doa_update_every = doa_update_every

        # ---- Laad RIRs + mic-config voor LUT-opbouw ----
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        self.data_dir = data_dir
        params, rirs, thetas = self._load_array_setup(data_dir)
        self.mic_pos = np.array(params["LMAcoords"])
        M = self.mic_pos.shape[0]
        self.M = M

        # ---- Bin-range bepalen voor MUSIC ----
        # Spatial aliasing treedt op boven f = c/(2*d_min). Boven die frequentie kan
        # MUSIC niet meer eenduidig de DOA vinden -> liever weglaten.
        if bin_range == "full":
            br = (1, L // 2)
        elif bin_range == "auto":
            dists = [np.linalg.norm(self.mic_pos[i] - self.mic_pos[j])
                     for i in range(M) for j in range(i + 1, M)]
            d_min = min(dists) if dists else 0.1
            f_alias = 343.0 / (2.0 * d_min)
            k_alias = int(round(f_alias / (fs / L)))
            # k_min=2 om DC en zeer-lage bins (geen DOA-info) over te slaan.
            br = (2, max(8, min(L // 2, k_alias)))
        else:
            br = bin_range  # tuple (k_min, k_max)

        # ---- LUT (FAS BF + Blocking Matrix) voor alle gemeten RIR-hoeken ----
        # Eenmalige offline opbouw; daarna runtime alleen dict-lookup.
        self.lut, self.angles_lut = build_lut_from_rirs(rirs, thetas, L=L)

        # ---- Streaming MUSIC met exponentiele R_yy middeling (Part 2) ----
        # sv_model:
        #   "rir" (default) -> gebruik gemeten RIRs als steering vectors. Veel
        #                      accurater dan plane wave op deze fase 3 dataset
        #                      omdat de mics niet ideaal omnidirectioneel zijn
        #                      en near-field effecten meegenomen worden.
        #   "planewave"    -> oude variant met far-field plane wave model.
        #                      Behouden voor A/B-vergelijking en debug.
        # snr_weight: per-bin power-weighted pseudospectrum (zie RIRSteeringMUSIC).
        # use_fb : Forward-Backward averaging voor coherente multipath.
        #          None -> auto: aan voor reverberant data, uit voor anechoic.
        if use_fb is None:
            use_fb = "reverberant" in str(data_dir).lower()
        self.sv_model = sv_model
        if sv_model == "rir":
            self.music = RIRSteeringMUSIC(
                rirs=rirs,
                thetas=thetas,
                fs=fs,
                L=L,
                num_sources=2,
                beta=beta,
                bin_range=br,
                combine=combine,
                snr_weight=snr_weight,
                use_fb=use_fb,
            )
        elif sv_model == "planewave":
            self.music = StreamingMUSIC(
                mic_pos=self.mic_pos,
                fs=fs,
                L=L,
                num_sources=2,
                beta=beta,
                bin_range=br,
                combine=combine,
            )
        else:
            raise ValueError(f"Onbekende sv_model: {sv_model!r} (gebruik 'rir' of 'planewave')")

        # ---- Twee FD-GSCs: één per spreker (Part 1 + Part 4) ----
        # gsc_left: targets de LINKER spreker (DOA in (90,180]).
        # gsc_right: targets de RECHTER spreker (DOA in [0,90]).
        self.gsc_left = StreamingFDGSC(self.lut, self.angles_lut, M, L=L, hop=hop, mu=mu, side="left")
        self.gsc_right = StreamingFDGSC(self.lut, self.angles_lut, M, L=L, hop=hop, mu=mu, side="right")

        # ---- DOA outlier-rejecting smoother (Fase D) ----
        # Onderdrukt single-frame uitschieters (stilte, transitions) zonder
        # echte sprekerverplaatsingen te dempen.
        self.doa_tracker = DOATracker(
            window=doa_tracker_window,
            alpha=doa_tracker_alpha,
            outlier_thresh=doa_tracker_outlier,
        )

        # ---- Per-spreker SIR-trackers (Part 3) ----
        # Window 2s voor stabiele waarden in de demo-plot. Voor real-time monitoring
        # in een hoorapparaat zou je 0.5s gebruiken (sneller reagerend).
        self.sir_left = StreamingSIR(fs=fs, window_seconds=2.0)
        self.sir_right = StreamingSIR(fs=fs, window_seconds=2.0)

        # Persistente DOA-schattingen (raw MUSIC-output, NIET de LUT-snapped).
        # Houden we vast tussen DOA-updates zodat we de waarde kunnen blijven loggen
        # ook in chunks waar we MUSIC niet runnen (om rekentijd te besparen).
        self.raw_doa_left = float("nan")
        self.raw_doa_right = float("nan")
        # Gesmoothede DOA-schattingen (door DOATracker geproduceerd, naar GSC).
        self.smooth_doa_left = float("nan")
        self.smooth_doa_right = float("nan")

        # MUSIC-buffer voor sliding STFT (50% overlap)
        self.music_buf = np.zeros((0, M), dtype=np.float64)
        from scipy import signal as _sig
        self._music_window = np.sqrt(_sig.windows.hann(L, sym=False))

        # Chunk-teller (voor doa_update_every modulo)
        self.chunk_count = 0

    @staticmethod
    def _load_array_setup(data_dir):
        """Laad params.pkl + lma_*.npz uit het scenario-pad.

        Anechoic: lma_16kHz.npz
        Reverberant: lma_16kHz_200ms.npz (langere RIRs door T60=200ms reflectie)
        """
        with open(os.path.join(data_dir, "params.pkl"), "rb") as f:
            params = pickle.load(f)
        # Zoek het juiste lma_*.npz bestand. Voor reverberant scenario zit
        # "200ms" in de naam; we pakken het bestand dat hoort bij data_dir.
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
        """Verwerk één LMA-chunk: MUSIC + 2× FD-GSC + SIR.

        Parameters
        ----------
        lma      : (chunk_n, M) int16   -- microfoonsignalen (mix)
        lma_gt_0 : (chunk_n, M) int16 of None -- bijdrage van left speaker (oracle, voor SIR)
        lma_gt_1 : (chunk_n, M) int16 of None -- bijdrage van right speaker (oracle, voor SIR)
        """
        ci = self.chunk_count
        self.chunk_count += 1

        # int16 -> float32 conversie. We laten de schaal als 16-bit ints staan
        # zodat output ook int16 is voor de frontend (clip-veilig binnen ±32767).
        mix_c = lma.astype(np.float32)
        tar_l = lma_gt_0.astype(np.float32) if lma_gt_0 is not None else None  # left = target voor gsc_left
        tar_r = lma_gt_1.astype(np.float32) if lma_gt_1 is not None else None  # right = target voor gsc_right

        # ---------- Part 2: Dynamic DOA estimation met exp. R_yy averaging ----------
        # Voeg chunk toe aan MUSIC-buffer en process zoveel volle STFT-frames als mogelijk.
        self.music_buf = np.concatenate([self.music_buf, mix_c.astype(np.float64)], axis=0)
        while self.music_buf.shape[0] >= self.L:
            frame = self.music_buf[: self.L, :] * self._music_window[:, None]
            Y = np.fft.rfft(frame, n=self.L, axis=0)
            self.music.update(Y)             # update R_yy(omega) recursief
            self.music_buf = self.music_buf[self.hop :, :]  # shift met hop (50% overlap)

        # DOA-schatting niet elke chunk (eigh is duur) -- elke doa_update_every chunks.
        if ci % self.doa_update_every == 0 and self.music.initialized:
            doas = self.music.estimate_doas()
            l, r = split_left_right(doas)
            # Behoud vorige schatting als 1 zijde geen piek geeft (= conservatief)
            if not np.isnan(l):
                self.raw_doa_left = l
            if not np.isnan(r):
                self.raw_doa_right = r
            # Run de outlier-rejecting smoother. Geeft NaN -> NaN door zonder update.
            self.smooth_doa_left, self.smooth_doa_right = self.doa_tracker.update(
                self.raw_doa_left, self.raw_doa_right
            )
            # Push gesmoothede DOA naar beide GSCs. set_doa snapt naar dichtstbijzijnde
            # LUT-hoek en update W_FAS/B alleen als hoek WIJZIGT (cache-vriendelijk).
            # Smoothed i.p.v. raw -> minder LUT-cache flips bij ruis-bursts.
            if not np.isnan(self.smooth_doa_left):
                self.gsc_left.set_doa(self.smooth_doa_left)
            if not np.isnan(self.smooth_doa_right):
                self.gsc_right.set_doa(self.smooth_doa_right)

        # ---------- Part 1 + 4: twee streaming FD-GSCs voor moving sources ----------
        # gsc_left: target = left, interferer = right
        out_mix_L, out_tar_L, out_int_L = self.gsc_left.process_chunk(mix_c, tar_l, tar_r)
        # gsc_right: target = right, interferer = left
        out_mix_R, out_tar_R, out_int_R = self.gsc_right.process_chunk(mix_c, tar_r, tar_l)

        # ---------- Part 3: SIR per chunk (sliding 2s window) ----------
        # Alleen mogelijk als oracle paden beschikbaar -- in productie zou je dit
        # weglaten of een blind SIR-schatter gebruiken.
        sir_l = self.sir_left.update(out_tar_L, out_int_L) if tar_l is not None else float("nan")
        sir_r = self.sir_right.update(out_tar_R, out_int_R) if tar_r is not None else float("nan")
        # Frontend toont 1 SIR-waarde -> kies die van de huidig gevolgde spreker.
        sir_active = sir_l if self.attended_left else sir_r

        # ---------- Phase 1 output -> server -> frontend ----------
        # int16 voor frontend (waveform-rendering verwacht 16-bit), clip op ±32767.
        beam_left_i16 = np.clip(np.nan_to_num(out_mix_L), -32768, 32767).astype(np.int16)
        beam_right_i16 = np.clip(np.nan_to_num(out_mix_R), -32768, 32767).astype(np.int16)

        # Voor frontend: smoothed DOA (vloeiend op plot). Fall back naar raw als nog
        # geen smoothed beschikbaar (cold start). Pas op als raw ook NaN -> 90 default.
        doa_l_out = self.smooth_doa_left if not np.isnan(self.smooth_doa_left) else self.raw_doa_left
        doa_r_out = self.smooth_doa_right if not np.isnan(self.smooth_doa_right) else self.raw_doa_right
        self.data_queue_phase1.put_nowait((
            beam_left_i16,
            beam_right_i16,
            # NaN -> 90 (broadside-default) zodat frontend altijd een float ontvangt.
            float(doa_l_out) if not np.isnan(doa_l_out) else 90.0,
            float(doa_r_out) if not np.isnan(doa_r_out) else 90.0,
            float(sir_active) if not np.isnan(sir_active) else 0.0,
        ))

        # ---------- Phase 3 output: geselecteerde spreker ----------
        # attended_left wordt door processing_eeg_gt_audio (placeholder/LSTM) gezet.
        sig_out = beam_left_i16 if self.attended_left else beam_right_i16
        speaker = 0 if self.attended_left else 1
        self.data_queue_phase3.put_nowait((speaker, sig_out))

    # -------------------------------------------------------------- #
    #                     EEG-pipeline (placeholder)                 #
    # -------------------------------------------------------------- #
    def processing_eeg_gt_audio(self, eeg, sig_left_clean, sig_right_clean):
        """PLACEHOLDER voor fase 2 LSTM (dilated+lstm uit Colab).

        Voor week 1 van fase 3 (audio integration) is een placeholder voldoende -- de
        EEG/AAD-integratie is grotendeels week 2-werk.

        Demo-vriendelijke placeholder:
        - Alterneert tussen sprekers met een soepel verloop (i.p.v. random per window)
        - Geeft pred_prob rond 0.85 / 0.15 zodat de frontend een nette curve toont
        - Switcht elke ~30s (10 windows van 3s) zodat de demo beide sprekers laat zien

        TODO wanneer Pieter het Colab-model uploadt:
        1) load model: tf.keras.models.load_model(...)
        2) extract envelopes uit sig_left_clean / sig_right_clean
           (Gammatone-bank, of Hilbert + lowpass 8Hz)
        3) run model.predict([eeg, env1, env2]) -> pred_prob (in [0,1])
        4) self.attended_left = round(pred_prob)
        """
        # Stable demo placeholder: switch every 10 windows = 30s audio (window = 3s).
        # Voegt kleine ruis toe rond de "ideale" waarde om realisme te suggereren.
        if not hasattr(self, "_aad_window_count"):
            self._aad_window_count = 0
        self._aad_window_count += 1

        cycle_idx = self._aad_window_count // 10
        attended_left = (cycle_idx % 2 == 0)

        # pred_prob = "kans dat links de gevolgde spreker is".
        base = 0.85 if attended_left else 0.15
        # Kleine random variatie zodat lijn niet kunstmatig vlak is in demo.
        noise = float(np.random.uniform(-0.05, 0.05))
        pred_prob = float(np.clip(base + noise, 0.0, 1.0))

        self.attended_left = int(attended_left)
        self.data_queue_phase2.put_nowait(pred_prob)
