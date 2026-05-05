"""Per-frame SIR voor streaming GSC-output (Part 3 van fase 3 week 1).

SIR (Signal-to-Interference Ratio) in dB:
    SIR = 10 * log10( var(target_through_BF) / var(interferer_through_BF) )

Waarom sliding window i.p.v. instantaneous SIR?
- Per-sample var() heeft geen statistische betekenis (1 sample = 0 variantie).
- Per-frame (L=512 = 32ms) kan ruisig zijn bij stille fragmenten.
- Sliding window van 0.5-2 seconden geeft een stabiele schatting die TOCH meebeweegt
  met de bewegende sprekers en de NLMS-aanpassing.

We hebben TWEE manieren om SIR te berekenen:
1) StreamingSIR -- streaming, schuifvenster, voor real-time monitoring tijdens runtime.
2) compute_sir_full -- offline, exact zoals computeSIR.py uit week4, voor evaluatie.

In test_week1.py worden beide gebruikt: streaming voor de plot, compute_sir_full voor
de globale eindscore.
"""
import numpy as np


class StreamingSIR:
    """Schuifvenster-SIR voor real-time monitoring.

    Voed het de TARGET-bijdrage en INTERFERER-bijdrage van de GSC-output (oracle paden,
    apart bijgehouden in StreamingFDGSC). De SIR wordt berekend over de laatste
    `window_seconds` seconden, dus reageert hij op DOA-fouten en NLMS-leakage maar is
    niet zo wisselvallig als per-frame.
    """

    def __init__(self, fs=16000, window_seconds=0.5):
        """
        Parameters
        ----------
        fs : samplingfrequentie audio (Hz)
        window_seconds : lengte van schuifvenster.
                         Korter (0.5s): reactiever maar ruisiger.
                         Langer (2s): stabieler maar trager bij sprekerwissel.
                         Default 0.5s -- de processor.py overschrijft naar 2s voor
                         de demo-plots (minder ruis op de SIR-curve).
        """
        self.fs = fs
        self.window_n = int(round(fs * window_seconds))
        self.tar_buf = np.zeros(0, dtype=np.float64)
        self.int_buf = np.zeros(0, dtype=np.float64)
        self.last_sir = np.nan

    def update(self, tar_samples, int_samples):
        """Voeg samples toe aan schuifvenster en bereken SIR.

        Returns
        -------
        sir : float (dB) of np.nan als nog onvoldoende of NaN-input (warmup-fase).
        """
        # Filter NaN-samples weg (warmup output van GSC bevat NaN voor target/int).
        if np.any(np.isnan(tar_samples)) or np.any(np.isnan(int_samples)):
            mask = ~(np.isnan(tar_samples) | np.isnan(int_samples))
            tar_samples = tar_samples[mask]
            int_samples = int_samples[mask]
            if tar_samples.size == 0:
                return np.nan

        self.tar_buf = np.concatenate([self.tar_buf, tar_samples])
        self.int_buf = np.concatenate([self.int_buf, int_samples])

        # Hou alleen laatste window_n samples (FIFO).
        if self.tar_buf.shape[0] > self.window_n:
            self.tar_buf = self.tar_buf[-self.window_n :]
            self.int_buf = self.int_buf[-self.window_n :]

        # Wacht tot er genoeg data is voor een stabiele schatting (>= 1/4 venster).
        if self.tar_buf.shape[0] < self.window_n // 4:
            return np.nan

        var_t = float(np.var(self.tar_buf))
        var_i = float(np.var(self.int_buf))
        if var_i < 1e-30:
            # Interferer praktisch nul (typisch wanneer NLMS hem perfect gecanceld heeft
            # of beide stil zijn). Niet definieerbaar -> NaN.
            return np.nan
        sir = 10.0 * np.log10(var_t / var_i)
        self.last_sir = sir
        return sir


def compute_sir_full(y, x_target, x_interferer, ground_truth=None):
    """Volledige SIR-berekening identiek aan deadline1/week4.ipynb cell 8 / computeSIR.py.

    Bedoeld voor finale evaluatie over een hele opname. `ground_truth` mag None zijn
    (dan wordt aangenomen dat x_target altijd target is, geen swap).

    Parameters
    ----------
    y           : (N,) gemixte BF-output (alleen voor signature-compat met week4; niet gebruikt)
    x_target    : (N,) target-bijdrage door BF
    x_interferer: (N,) interferer-bijdrage door BF
    ground_truth: (N,) of None -- 1=target, 0=interferer per sample (voor switching)
    """
    if ground_truth is None:
        var_t = float(np.var(x_target))
        var_i = float(np.var(x_interferer))
    else:
        # Met switching: bij sample n waar gt=1 telt x_target als target; bij gt=0 swap.
        gt = np.asarray(ground_truth)
        var_t = float(np.var(x_target * gt + x_interferer * (1 - gt)))
        var_i = float(np.var(x_interferer * gt + x_target * (1 - gt)))

    if var_i < 1e-30:
        return np.nan
    return 10.0 * np.log10(var_t / var_i)
