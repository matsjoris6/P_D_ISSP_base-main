"""LUT builder voor FD-GSC: FAS-beamformer + Blocking Matrix per gemeten RIR-hoek.

Identiek aan deadline1/week4.ipynb cell 4 (build_lut_for_target / build_lut_all_angles),
maar herverpakt zodat we de hele LUT in één call kunnen opbouwen vanuit lma_16kHz.npz.

Concept:
- Voor elke GEMETEN RIR-hoek (typisch ~20 hoeken) bouwen we een set BF-gewichten:
    W_FAS : Filter-And-Sum beamformer gericht op die hoek (target-pad)
    B     : Blocking matrix (M-1 × M) waarvan elke rij orthogonaal staat op de RIR
            -> output van B bevat ALLES BEHALVE de target = "interferer-only" subspace
- Online doen we runtime DOA-schatting (MUSIC), snappen die naar de dichtstbijzijnde
  LUT-hoek en gebruiken de cached W_FAS/B daarvoor in de GSC.

Waarom offline LUT i.p.v. online berekenen?
- scipy.linalg.null_space (voor B) is duur: O(M^3) per bin per hoek.
- 20 hoeken × 257 bins = 5140 SVD-operaties bij opstart, < 1s.
- Tijdens runtime is het slechts een dict-lookup -> ms-snel.
"""
import numpy as np
import scipy.linalg


def build_lut_for_target(target_rir, L=512):
    """Bouw FAS BF (W_FAS) en Blocking Matrix (B) voor 1 RIR-target.

    Parameters
    ----------
    target_rir : ndarray (n_samples_rir, M_mics) -- gemeten RIR voor 1 hoek
    L : STFT-lengte (n_bins = L//2 + 1)

    Returns
    -------
    W_FAS : ndarray (n_bins, M_mics) complex  -- per-bin FAS gewicht
    B     : ndarray (n_bins, M_mics-1, M_mics) complex -- per-bin Blocking Matrix
    """
    n_bins = L // 2 + 1
    M = target_rir.shape[1]

    # FFT van RIR per microfoon. H[k, m] = transfer-functie van bron naar mic m bij bin k.
    H = np.fft.rfft(target_rir, n=L, axis=0)  # (n_bins, M)

    W_FAS = np.zeros((n_bins, M), dtype=complex)
    B = np.zeros((n_bins, M - 1, M), dtype=complex)
    eps = 1e-12

    for k in range(n_bins):
        h_k = H[k, :].reshape(M, 1)

        # Normaliseer t.o.v. mic 0 (referentie). Hierdoor is W_FAS[0] = 1/M en blijft
        # het BF-output unity-gain voor de target. Standaard week4-conventie.
        a1 = h_k[0, 0]
        if np.abs(a1) > eps:
            h_k = h_k / a1
        else:
            h_k = h_k / (a1 + eps)

        # FAS BF: W = h / (h^H h) -> output = W^H y = (h^H y)/(h^H h)
        # Voor target alleen: y = s*h -> output = s. Dus unity-gain richting target.
        denom = (h_k.conj().T @ h_k)[0, 0]
        if np.abs(denom) > eps:
            W_FAS[k, :] = (h_k / denom).flatten()
        else:
            # Singulier (RIR ~0 in deze bin): doe gewoon mean over mics als safety.
            W_FAS[k, :] = np.ones(M) / M

        # Blocking matrix: rijen orthogonaal op h_k -> B @ (s*h_k) = 0.
        # null_space geeft een orthonormale basis voor de nullruimte van h_k^H.
        # We willen B met M-1 rijen (zodat B @ x M-1-dim "interferer subspace" geeft).
        # Conventie: B's RIJEN zijn null-vectors -> B = Z[:, :M-1].conj().T met Z = null(h^H).
        Z = scipy.linalg.null_space(h_k.conj().T)
        if Z.shape[1] >= M - 1:
            B[k, :, :] = Z[:, : M - 1].conj().T

    return W_FAS, B


def build_lut_from_rirs(rirs, thetas, L=512):
    """Bouw LUT voor alle hoeken in `thetas` uit lma_16kHz.npz / hma_16kHz.npz.

    Parameters
    ----------
    rirs   : ndarray (n_samples_rir, M, n_angles) -- gemeten RIRs uit lma_16kHz.npz['rirs']
    thetas : ndarray (n_angles,)                  -- bijhorende hoeken (graden)
    L      : STFT-lengte

    Returns
    -------
    lut : dict[float] -> (W_FAS, B)  -- key = hoek in graden
    angles_arr : ndarray (n_angles,) gesorteerd op hoek (handig voor snap-to-LUT)
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
    """Vind dichtstbijzijnde hoek uit de LUT-grid voor een geschatte DOA.

    Waarom snappen i.p.v. interpoleren?
    - De LUT bevat GEMETEN RIRs die beam-patterns en kamer-effecten correct vangen.
    - Tussen twee meetpunten interpoleren in het frequentiedomein is niet fysisch
      correct (elke RIR heeft eigen reflectie-patroon).
    - Snappen naar dichtstbijzijnde geeft een accurater BF zelfs als de DOA niet
      exact match. De resterende mismatch wordt opgevangen door NLMS.
    """
    return float(angles_arr[np.argmin(np.abs(angles_arr - angle))])
