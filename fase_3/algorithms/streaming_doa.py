"""Streaming wideband MUSIC met exponentieel-gemiddelde R_yy(omega, k).

Implementeert Part 2 van fase 3 week 1:
    R_yy(omega, k) = beta * R_yy(omega, k-1) + (1 - beta) * y(omega, k) y^H(omega, k)

Het pseudo-spectrum wordt opgebouwd uit alle bins (geometrisch gemiddelde, zoals in
deadline1/week4.ipynb) en de Q grootste pieken worden geretourneerd als geschatte DOAs.
"""
import numpy as np
import scipy.signal as signal


class StreamingMUSIC:
    """Online wideband MUSIC.

    Per binnenkomende STFT-frame wordt R_yy(omega) recursief geupdate.
    estimate_doas() berekent het geometrisch gemiddelde pseudo-spectrum
    over een frequentiebereik en retourneert Q pieken.
    """

    def __init__(
        self,
        mic_pos,
        fs=16000,
        L=512,
        num_sources=2,
        beta=0.95,
        angle_grid=None,
        bin_range=None,
        diag_load=1e-2,
        combine="geometric",
    ):
        """
        Parameters
        ----------
        mic_pos : ndarray (M, 2)  -- mic-coordinaten (x, y) in meter
        fs      : samplingfrequentie audio (Hz)
        L       : STFT-lengte
        num_sources : aantal te schatten bronnen Q (2 voor pair-geval)
        beta    : exponentiele middelingsconstante (0 < beta < 1).
                  Hoog (0.99) -> traag maar stabiel; laag (0.8) -> snel adapterend, ruisiger.
        angle_grid : np.ndarray of None. Als None: 0..180 in 0.5-graden stappen.
        bin_range : (k_min, k_max). Als None: auto-berekend op basis van mic-spacing
                    (alle bins onder de spatial-aliasing limiet f_alias = c/(2*d_max)).
        diag_load : kleine factor voor diagonal loading van R_yy (numerieke stabiliteit).
        combine   : 'arithmetic' (per-bin normalized + mean) of 'geometric' (week4 stijl)
        """
        self.mic_pos = np.asarray(mic_pos)
        self.M = self.mic_pos.shape[0]
        self.fs = fs
        self.L = L
        self.n_bins = L // 2 + 1
        self.Q = num_sources
        self.beta = beta
        self.diag_load = diag_load
        self.combine = combine

        if angle_grid is None:
            angle_grid = np.arange(0, 180.5, 0.5)
        self.angles = angle_grid
        self.rads = np.radians(self.angles)

        if bin_range is None:
            # Default exact zoals deadline1/week4.ipynb: alle bins van 1 tot L/2.
            # Voor 16 kHz data met 10 cm mic-spacing kunnen bins boven de spatial-
            # aliasing limiet (~bin 56) ruis introduceren -- pass dan een aangepaste
            # bin_range door (zie test_week1.py).
            bin_range = (1, L // 2)
        self.k_min, self.k_max = bin_range
        self.valid_bins = np.arange(self.k_min, self.k_max)

        # mic-offsets t.o.v. centrum (zoals week4.ipynb)
        mics_centered = self.mic_pos - np.mean(self.mic_pos, axis=0)
        self.px = mics_centered[:, 0].reshape(-1, 1)
        self.py = mics_centered[:, 1].reshape(-1, 1)

        # Pre-compute steering-vector matrices A_k voor elke bin in valid_bins
        # A_k shape: (M, len(angles))
        self.freqs = np.fft.rfftfreq(L, d=1.0 / fs)
        self._A = {}
        c = 343.0
        for k in self.valid_bins:
            omega = 2 * np.pi * self.freqs[k]
            taus = (self.px * np.sin(self.rads) + self.py * np.cos(self.rads)) / c
            self._A[k] = np.exp(-1j * omega * taus)

        # State: R_yy(omega) per bin
        self.Ryy = np.zeros((self.n_bins, self.M, self.M), dtype=complex)
        self.initialized = False

    def reset(self):
        self.Ryy = np.zeros((self.n_bins, self.M, self.M), dtype=complex)
        self.initialized = False

    def update(self, Y_frame):
        """Update R_yy met een STFT-frame.

        Parameters
        ----------
        Y_frame : ndarray (n_bins, M) complex  -- STFT van een enkel tijdframe
        """
        if not self.initialized:
            for k in range(self.n_bins):
                yk = Y_frame[k, :].reshape(self.M, 1)
                self.Ryy[k] = yk @ yk.conj().T
            self.initialized = True
        else:
            for k in range(self.n_bins):
                yk = Y_frame[k, :].reshape(self.M, 1)
                self.Ryy[k] = self.beta * self.Ryy[k] + (1.0 - self.beta) * (yk @ yk.conj().T)

    def estimate_doas(self, return_spectrum=False):
        """Bereken pseudo-spectrum + Q DOA-pieken uit huidige R_yy.

        Returns
        -------
        doas_sorted : ndarray (Q,) gesorteerd op hoek (graden)
        spectrum_db : ndarray (len(angles),) in dB (alleen als return_spectrum=True)
        """
        if not self.initialized:
            return (np.full(self.Q, np.nan), None) if return_spectrum else np.full(self.Q, np.nan)

        pseudospectra = np.zeros((len(self.valid_bins), len(self.angles)))
        for i, k in enumerate(self.valid_bins):
            R = self.Ryy[k] + self.diag_load * np.trace(self.Ryy[k]).real / self.M * np.eye(self.M)
            _, eigvecs = np.linalg.eigh(R)
            En = eigvecs[:, : self.M - self.Q]  # noise subspace
            A = self._A[k]
            denom = np.sum(np.abs(En.conj().T @ A) ** 2, axis=0)
            pseudospectra[i] = 1.0 / (denom + 1e-30)

        if self.combine == "geometric":
            log_p = np.log(pseudospectra + 1e-30)
            p_combined = np.exp(np.mean(log_p, axis=0))
        else:
            # arithmetic: normaliseer per bin (zodat luide bins niet domineren) en gemiddelde
            ps_norm = pseudospectra / (np.max(pseudospectra, axis=1, keepdims=True) + 1e-30)
            p_combined = np.mean(ps_norm, axis=0)
        spectrum_db = 10 * np.log10(p_combined / np.max(p_combined) + 1e-30)

        peaks_idx, _ = signal.find_peaks(spectrum_db)
        if len(peaks_idx) >= self.Q:
            top = peaks_idx[np.argsort(spectrum_db[peaks_idx])][-self.Q :]
            doas = np.sort(self.angles[top])
        elif len(peaks_idx) > 0:
            # Minder pieken gevonden dan Q -> geef wat we hebben terug, rest NaN
            doas = np.full(self.Q, np.nan)
            real = np.sort(self.angles[peaks_idx])
            doas[: len(real)] = real
        else:
            doas = np.full(self.Q, np.nan)

        if return_spectrum:
            return doas, spectrum_db
        return doas


def split_left_right(doas):
    """Verdeel een lijst geschatte DOAs in (left_angle, right_angle).

    Linker spreker zit per definitie in (90, 180], rechter in [0, 90] (zie data-readme).
    Als MUSIC slechts 1 zijde vindt, retourneert die zijde de waarde en de andere np.nan
    (de aanroeper houdt dan typisch de vorige schatting aan).
    """
    if doas is None or len(doas) == 0:
        return np.nan, np.nan
    arr = np.asarray(doas, dtype=float)
    arr = arr[~np.isnan(arr)]
    left_cands = arr[arr > 90.0]
    right_cands = arr[arr <= 90.0]
    if len(left_cands) > 0:
        # kies de hoek het verst van 90 (typisch sterkste piek aan die kant)
        left = float(left_cands[np.argmax(np.abs(left_cands - 90.0))])
    else:
        left = np.nan
    if len(right_cands) > 0:
        right = float(right_cands[np.argmax(np.abs(right_cands - 90.0))])
    else:
        right = np.nan
    return left, right
