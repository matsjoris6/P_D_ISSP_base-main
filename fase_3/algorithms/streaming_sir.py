"""Per-frame SIR voor streaming GSC-output.

SIR (in dB) = 10 * log10( var(target_through_BF) / var(interferer_through_BF) )

We berekenen dit over een tijdvenster (default 0.5s) zodat de waarde stabiel is maar
toch kan variëren met de bewegende sprekers en de NLMS-aanpassing.
"""
import numpy as np


class StreamingSIR:
    """Schuifvenster-SIR. Voed het de target/interferer-componenten van de GSC-output."""

    def __init__(self, fs=16000, window_seconds=0.5):
        self.fs = fs
        self.window_n = int(round(fs * window_seconds))
        self.tar_buf = np.zeros(0, dtype=np.float64)
        self.int_buf = np.zeros(0, dtype=np.float64)
        self.last_sir = np.nan

    def update(self, tar_samples, int_samples):
        """Voeg samples toe en bereken SIR zodra venster vol is.

        Returns
        -------
        sir : float (dB) of np.nan als nog onvoldoende of NaN-input.
        """
        # Negeer NaN-input (warmup-fase)
        if np.any(np.isnan(tar_samples)) or np.any(np.isnan(int_samples)):
            mask = ~(np.isnan(tar_samples) | np.isnan(int_samples))
            tar_samples = tar_samples[mask]
            int_samples = int_samples[mask]
            if tar_samples.size == 0:
                return np.nan

        self.tar_buf = np.concatenate([self.tar_buf, tar_samples])
        self.int_buf = np.concatenate([self.int_buf, int_samples])

        # Hou laatste window_n samples
        if self.tar_buf.shape[0] > self.window_n:
            self.tar_buf = self.tar_buf[-self.window_n :]
            self.int_buf = self.int_buf[-self.window_n :]

        if self.tar_buf.shape[0] < self.window_n // 4:
            # nog onvoldoende voor stabiele schatting
            return np.nan

        var_t = float(np.var(self.tar_buf))
        var_i = float(np.var(self.int_buf))
        if var_i < 1e-30:
            return np.nan
        sir = 10.0 * np.log10(var_t / var_i)
        self.last_sir = sir
        return sir


def compute_sir_full(y, x_target, x_interferer, ground_truth=None):
    """Volledige SIR-berekening identiek aan deadline1/week4.ipynb cell 8 / computeSIR.py.

    Voor finale evaluatie. `ground_truth` mag None zijn (dan wordt aangenomen dat
    x_target altijd target is, geen swap).
    """
    if ground_truth is None:
        var_t = float(np.var(x_target))
        var_i = float(np.var(x_interferer))
    else:
        gt = np.asarray(ground_truth)
        var_t = float(np.var(x_target * gt + x_interferer * (1 - gt)))
        var_i = float(np.var(x_interferer * gt + x_target * (1 - gt)))

    if var_i < 1e-30:
        return np.nan
    return 10.0 * np.log10(var_t / var_i)
