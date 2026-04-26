"""LUT builder voor FD-GSC: FAS-beamformer + Blocking Matrix per gemeten RIR-hoek.

Identiek aan deadline1/week4.ipynb cell 4 (build_lut_for_target / build_lut_all_angles),
maar herverpakt zodat we de hele LUT in één call kunnen opbouwen vanuit lma_16kHz.npz.
"""
import numpy as np
import scipy.linalg


def build_lut_for_target(target_rir, L=512):
    """Bouw FAS BF (W_FAS) en Blocking Matrix (B) voor 1 RIR-target.

    Parameters
    ----------
    target_rir : ndarray (n_samples_rir, M_mics)
    L : STFT-lengte (n_bins = L//2 + 1)

    Returns
    -------
    W_FAS : ndarray (n_bins, M_mics) complex  -- per-bin FAS gewicht
    B     : ndarray (n_bins, M_mics-1, M_mics) complex -- per-bin Blocking Matrix
    """
    n_bins = L // 2 + 1
    M = target_rir.shape[1]

    H = np.fft.rfft(target_rir, n=L, axis=0)  # (n_bins, M)

    W_FAS = np.zeros((n_bins, M), dtype=complex)
    B = np.zeros((n_bins, M - 1, M), dtype=complex)
    eps = 1e-12

    for k in range(n_bins):
        h_k = H[k, :].reshape(M, 1)

        a1 = h_k[0, 0]
        if np.abs(a1) > eps:
            h_k = h_k / a1
        else:
            h_k = h_k / (a1 + eps)

        denom = (h_k.conj().T @ h_k)[0, 0]
        if np.abs(denom) > eps:
            W_FAS[k, :] = (h_k / denom).flatten()
        else:
            W_FAS[k, :] = np.ones(M) / M

        Z = scipy.linalg.null_space(h_k.conj().T)
        if Z.shape[1] >= M - 1:
            B[k, :, :] = Z[:, : M - 1].conj().T

    return W_FAS, B


def build_lut_from_rirs(rirs, thetas, L=512):
    """Bouw LUT voor alle hoeken in `thetas` uit lma_16kHz.npz / hma_16kHz.npz.

    Parameters
    ----------
    rirs   : ndarray (n_samples_rir, M, n_angles)  -- uit lma_16kHz.npz['rirs']
    thetas : ndarray (n_angles,)                    -- uit lma_16kHz.npz['thetas']
    L      : STFT-lengte

    Returns
    -------
    lut : dict[float] -> (W_FAS, B)  -- key = hoek in graden
    angles_arr : ndarray (n_angles,) sorteerd op hoek (handig voor snap-to-LUT)
    """
    order = np.argsort(thetas)
    lut = {}
    for idx in order:
        angle = float(thetas[idx])
        target_rir = rirs[:, :, idx]  # (n_samples_rir, M)
        lut[angle] = build_lut_for_target(target_rir, L=L)
    angles_arr = np.array(sorted(lut.keys()))
    return lut, angles_arr


def snap_angle_to_lut(angle, angles_arr):
    """Vind dichtstbijzijnde hoek uit de LUT-grid voor een geschatte DOA."""
    return float(angles_arr[np.argmin(np.abs(angles_arr - angle))])
