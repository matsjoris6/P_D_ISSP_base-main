"""AAD LSTM+dilated wrapper voor live predictie.

Wrappert het Colab-getrainde Keras-model `hybrid_v3_BEST.keras`. Het model verwacht
3 inputs van shape (None, 640, X):
    - EEG_Input  : (None, 640, 64)   -> 64 EEG-kanalen, 5s @ 128Hz
    - Env1_Input : (None, 640, 1)    -> envelope linker spreker, 5s @ 128Hz
    - Env2_Input : (None, 640, 1)    -> envelope rechter spreker, 5s @ 128Hz
en geeft 1 output: pred_prob in [0, 1] (waarschijnlijkheid attended_left).

Voor live gebruik:
- 5s sliding-window met 1s hop (default)
- Audio-envelopes: gammatone-bank (28 bands ERB) -> Hilbert magnitude per band ->
  power-law compression (^0.6) -> sum -> 8Hz lowpass -> downsample naar 128Hz
- Buffert per stream tot venster vol is, dan predict
"""
import os
import numpy as np
import scipy.signal as ss


def _erb(f):
    """Equivalent Rectangular Bandwidth in Hz (Glasberg & Moore 1990)."""
    return 24.7 * (4.37e-3 * f + 1.0)


def _erb_space(low_hz, high_hz, n_bands):
    """ERB-spaced center frequencies tussen low_hz en high_hz."""
    # Standaard ERB-rate transformatie
    erb_low = 21.4 * np.log10(4.37e-3 * low_hz + 1.0)
    erb_high = 21.4 * np.log10(4.37e-3 * high_hz + 1.0)
    erb_centers = np.linspace(erb_low, erb_high, n_bands)
    centers_hz = (10 ** (erb_centers / 21.4) - 1.0) / 4.37e-3
    return centers_hz


class GammatoneEnvelope:
    """Gammatone-bank envelope-extractor voor speech-signalen.

    Pipeline (per audio-chunk):
      1. Filter via gammatone-bank (28 bands, 80-6000 Hz ERB-spaced)
      2. Per band: |hilbert(x)| -> instantaneous envelope
      3. Power-law compressie envelope^0.6 (auditieve compressie)
      4. Som over banden -> single-channel envelope
      5. Lowpass 32Hz Butterworth (anti-alias voor downsample)
      6. Downsample naar fs_target via decimatie

    Implementatie is stateless tussen chunks (filterbank state niet bewaard) maar
    voor 5s chunks is dat geen probleem; minor edge-effecten alleen in eerste/laatste
    paar samples van elke chunk.
    """

    def __init__(self, fs_audio=16000, fs_target=128, n_bands=28,
                 low_hz=80.0, high_hz=6000.0):
        self.fs_audio = fs_audio
        self.fs_target = fs_target
        self.n_bands = n_bands
        self.centers = _erb_space(low_hz, high_hz, n_bands)

        # Anti-alias lowpass voor decimatie (cutoff = 0.4 * Nyquist target)
        self._aa_sos = ss.butter(8, 0.4 * (fs_target / 2), btype="low",
                                  fs=fs_audio, output="sos")

    def _gammatone_filter(self, x, cf):
        """Pas Gammatone IIR-filter toe voor center frequency cf."""
        # scipy.signal.gammatone retourneert (b, a) IIR coefficients
        b, a = ss.gammatone(cf, ftype="iir", fs=self.fs_audio)
        return ss.lfilter(b, a, x)

    def __call__(self, audio):
        """Extract envelope. audio: 1D float, shape (N,) @ fs_audio.
        Returns: 1D float, shape (M,) @ fs_target waarbij M = N * fs_target/fs_audio.
        """
        x = np.asarray(audio, dtype=np.float64)
        if x.ndim != 1:
            x = x.flatten()

        # --- Bouw envelope: gammatone -> hilbert magnitude -> compress -> sum ---
        env = np.zeros_like(x)
        for cf in self.centers:
            band = self._gammatone_filter(x, cf)
            mag = np.abs(ss.hilbert(band))
            env += np.power(mag + 1e-12, 0.6)
        env /= self.n_bands

        # --- Anti-alias lowpass + downsample ---
        env_lp = ss.sosfilt(self._aa_sos, env)
        # Decimatie-factor (integer)
        factor = int(round(self.fs_audio / self.fs_target))
        if factor < 1:
            factor = 1
        env_ds = env_lp[::factor]
        # Clip voor overflow-veiligheid (IIR-filters kunnen grote transients geven
        # bij random-noise of stilte-segmenten; heeft geen effect op echte spraak).
        env_ds = np.nan_to_num(env_ds, nan=0.0, posinf=1e6, neginf=-1e6)
        return env_ds.astype(np.float32)


class HilbertEnvelope:
    """Eenvoudige Hilbert+lowpass envelope (fallback als gammatone te traag is).

    Pipeline:
      1. |hilbert(x)| -> instantaneous magnitude
      2. Power-law compressie ^0.6
      3. Lowpass 8Hz (delta-band, AAD-traditie)
      4. Downsample naar fs_target

    ~10x sneller dan GammatoneEnvelope, vergelijkbare AAD-prestatie volgens
    de meeste literatuur.
    """

    def __init__(self, fs_audio=16000, fs_target=128, lp_hz=8.0):
        self.fs_audio = fs_audio
        self.fs_target = fs_target
        self._lp_sos = ss.butter(4, lp_hz, btype="low", fs=fs_audio, output="sos")

    def __call__(self, audio):
        x = np.asarray(audio, dtype=np.float64).flatten()
        env = np.power(np.abs(ss.hilbert(x)) + 1e-12, 0.6)
        env_lp = ss.sosfilt(self._lp_sos, env)
        factor = int(round(self.fs_audio / self.fs_target))
        return env_lp[::factor].astype(np.float32)


class AADLSTM:
    """Wrapper rond Keras dilated+LSTM AAD-model met sliding window.

    Usage:
        aad = AADLSTM("path/to/model.keras")
        for each_1s_chunk in stream:
            pred = aad.update(eeg_chunk, sig_left, sig_right)
            if pred is not None:
                # pred is in [0, 1]: P(attended_left)
                attended = pred >= 0.5

    Het model verwacht (N, 640, ...) shapes (5s @ 128Hz EEG fs). update() buffert
    inkomende chunks en triggert een nieuwe predictie elke `hop_s` seconden zodra
    er minstens `window_s` seconden in de buffer staan.

    Args:
        model_path    : pad naar .keras of .h5 model
        fs_audio      : sample rate van de audio-stimuli (default 48000 Hz — de
                        fase-3 stimuli WAVs zijn altijd 48 kHz, NIET de mic-rate).
                        OPGELET: dit is de fs van audio1/audio2 die de server stuurt,
                        niet de LMA-microfoonsignalen (die zijn 16 kHz).
        fs_eeg        : EEG sample rate (default 128)
        n_eeg_channels: aantal EEG-kanalen (default 64)
        window_s      : predictie-venster in seconden (default 5.0)
        hop_s         : predictie-hop in seconden (default 1.0)
        envelope      : 'gammatone' | 'hilbert' (default 'gammatone')
        normalize_eeg : als True, z-score normaliseert het EEG per venster per kanaal.
                        Aanbevolen voor modellen zonder interne BatchNorm-laag
                        (bv. generic_dilated). Voor hybrid_v3 niet nodig (heeft
                        EEG_BN_Input intern).
    """

    def __init__(self, model_path, fs_audio=48000, fs_eeg=128, n_eeg_channels=64,
                 window_s=5.0, hop_s=1.0, envelope="gammatone", normalize_eeg=False):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AAD model niet gevonden: {model_path}")

        # Lazy import van TF (kost ~5s; alleen als nodig)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        self._tf = tf

        print(f"[AADLSTM] Model laden: {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        # Print input shapes voor sanity check
        inputs = self.model.inputs if hasattr(self.model, "inputs") else [self.model.input]
        print(f"[AADLSTM] Inputs: {[(i.name, tuple(i.shape)) for i in inputs]}")

        # Auto-detect of model interne normalisatie heeft (BatchNorm op EEG input)
        self._model_has_eeg_bn = self._detect_eeg_batchnorm()
        if self._model_has_eeg_bn:
            print("[AADLSTM] Model heeft interne EEG BatchNorm → externe normalisatie niet nodig")
        else:
            print("[AADLSTM] Model heeft GEEN interne EEG BatchNorm")
            if not normalize_eeg:
                print("[AADLSTM] WAARSCHUWING: normalize_eeg=False maar model heeft geen BN. "
                      "Overweeg normalize_eeg=True voor betere accuracy.")

        self.normalize_eeg = normalize_eeg
        self.fs_audio = fs_audio
        self.fs_eeg = fs_eeg
        self.n_eeg_channels = n_eeg_channels
        self.window_samples_eeg = int(round(window_s * fs_eeg))
        self.hop_samples_eeg = int(round(hop_s * fs_eeg))

        # Sanity-check: decimatiefactor voor envelope
        self._decim_factor = int(round(fs_audio / fs_eeg))
        print(f"[AADLSTM] Audio fs={fs_audio} Hz, EEG fs={fs_eeg} Hz, "
              f"decimatiefactor={self._decim_factor}, "
              f"venster={self.window_samples_eeg} EEG-samples ({window_s}s)")

        # Envelope-extractor (gammatone of hilbert)
        if envelope == "gammatone":
            self.env_extractor = GammatoneEnvelope(fs_audio=fs_audio, fs_target=fs_eeg)
        elif envelope == "hilbert":
            self.env_extractor = HilbertEnvelope(fs_audio=fs_audio, fs_target=fs_eeg)
        else:
            raise ValueError(f"Onbekende envelope: {envelope}")

        # Buffers (groeien tot window_samples, dan slide)
        self.eeg_buf = np.zeros((0, n_eeg_channels), dtype=np.float32)
        self.env_l_buf = np.zeros(0, dtype=np.float32)
        self.env_r_buf = np.zeros(0, dtype=np.float32)
        self._last_pred = 0.5
        self._n_predictions = 0

    def _detect_eeg_batchnorm(self):
        """Kijk of het model een BatchNorm-laag heeft op de EEG-tak."""
        try:
            for layer in self.model.layers:
                class_name = layer.__class__.__name__.lower()
                # BatchNormalization → "batchnormalization" (geen underscore)
                if "batchnorm" in class_name:
                    if "eeg" in layer.name.lower():
                        return True
            return False
        except Exception:
            return False

    def reset(self):
        self.eeg_buf = np.zeros((0, self.n_eeg_channels), dtype=np.float32)
        self.env_l_buf = np.zeros(0, dtype=np.float32)
        self.env_r_buf = np.zeros(0, dtype=np.float32)
        self._last_pred = 0.5

    def update(self, eeg_chunk, sig_left, sig_right):
        """Voeg nieuwe chunk toe en doe predictie als window vol is.

        Args:
            eeg_chunk: (n_samples_eeg, n_channels) float array @ fs_eeg
            sig_left:  (n_samples_audio,) float array @ fs_audio
            sig_right: (n_samples_audio,) float array @ fs_audio

        Returns:
            pred_prob in [0, 1] als nieuwe predictie werd gedaan,
            anders None (window nog niet vol of geen volle hop sinds laatste pred).
        """
        # 1) Audio -> envelope @ fs_eeg
        env_l = self.env_extractor(np.asarray(sig_left, dtype=np.float32))
        env_r = self.env_extractor(np.asarray(sig_right, dtype=np.float32))

        # 2) EEG normaliseren naar float32 (model verwacht typisch float)
        eeg = np.asarray(eeg_chunk, dtype=np.float32)
        if eeg.ndim == 1:
            # Soms komt EEG als (N,) als er maar 1 kanaal is — niet onze case
            eeg = eeg.reshape(-1, 1)

        # 3) Append aan buffers
        self.eeg_buf = np.vstack([self.eeg_buf, eeg])
        self.env_l_buf = np.concatenate([self.env_l_buf, env_l])
        self.env_r_buf = np.concatenate([self.env_r_buf, env_r])

        # 4) Synchroniseer buffer-lengtes (envelope vs eeg drift door rounding)
        min_len = min(self.eeg_buf.shape[0], len(self.env_l_buf), len(self.env_r_buf))
        self.eeg_buf = self.eeg_buf[-min_len:] if min_len > 0 else self.eeg_buf
        self.env_l_buf = self.env_l_buf[-min_len:] if min_len > 0 else self.env_l_buf
        self.env_r_buf = self.env_r_buf[-min_len:] if min_len > 0 else self.env_r_buf

        # 5) Genoeg data voor predictie?
        if min_len < self.window_samples_eeg:
            return None  # buffer nog niet vol

        # 6) Pak laatste window en doe predictie
        eeg_w = self.eeg_buf[-self.window_samples_eeg :]      # (640, 64)
        env_l_w = self.env_l_buf[-self.window_samples_eeg :]  # (640,)
        env_r_w = self.env_r_buf[-self.window_samples_eeg :]  # (640,)

        # Optionele EEG z-score normalisatie per venster per kanaal.
        # Aanbevolen voor modellen zonder interne BatchNorm (bv. generic_dilated).
        if self.normalize_eeg:
            mu = eeg_w.mean(axis=0, keepdims=True)
            sigma = eeg_w.std(axis=0, keepdims=True) + 1e-8
            eeg_w = (eeg_w - mu) / sigma

        # Reshape naar batch + channel: (1, 640, 64) en (1, 640, 1)
        eeg_in = eeg_w[np.newaxis, ...]
        env_l_in = env_l_w[np.newaxis, :, np.newaxis]
        env_r_in = env_r_w[np.newaxis, :, np.newaxis]

        pred = self.model.predict([eeg_in, env_l_in, env_r_in], verbose=0)
        pred_prob = float(np.asarray(pred).flatten()[0])

        # NaN-guard: als model NaN geeft (bv. bij extreme input of slechte
        # normalisatie), val terug op de vorige predictie.
        if np.isnan(pred_prob) or np.isinf(pred_prob):
            pred_prob = self._last_pred
        else:
            pred_prob = float(np.clip(pred_prob, 0.0, 1.0))

        self._last_pred = pred_prob
        self._n_predictions += 1

        # 7) Sliding-window: gooi de oudste hop_samples weg
        self.eeg_buf = self.eeg_buf[self.hop_samples_eeg :]
        self.env_l_buf = self.env_l_buf[self.hop_samples_eeg :]
        self.env_r_buf = self.env_r_buf[self.hop_samples_eeg :]

        return pred_prob

    @property
    def last_pred(self):
        return self._last_pred
