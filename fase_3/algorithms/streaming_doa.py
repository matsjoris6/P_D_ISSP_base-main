"""Streaming wideband MUSIC met exponentieel-gemiddelde R_yy(omega, k).

Implementeert Part 2 van fase 3 week 1 (UNIVERSITEITS-VEREISTE):
    R_yy(omega, k) = beta * R_yy(omega, k-1) + (1 - beta) * y(omega, k) y^H(omega, k)

Waarom deze recursieve formule?
- In een statische opname kan je R_yy = (1/N) * sum y_n y_n^H berekenen over een hele
  opname. Voor BEWEGENDE sprekers moet R_yy zich aanpassen aan de huidige positie,
  dus we gebruiken een exponentieel-gemiddelde dat oude frames "vergeet" met
  tijdconstante tau ~ 1/(1-beta) frames.
- beta=0.92 -> tijdconstante ~12.5 frames @ hop=256, fs=16kHz = ~0.2s. Snel genoeg
  voor sprekerverplaatsing, traag genoeg om ruis weg te middelen.

Het pseudo-spectrum wordt opgebouwd uit alle bins in self.valid_bins en de Q grootste
pieken worden geretourneerd als geschatte DOAs (= ZELFDE wideband-MUSIC als week 4).

PERFORMANCE: alle inner-loops zijn gevectoriseerd via einsum/batch-eigh. Dit verandert
NIETS aan het algoritme -- alleen de uitvoeringssnelheid.
"""
import numpy as np
import scipy.signal as signal


class StreamingMUSIC:
    """Online wideband MUSIC.

    Per binnenkomende STFT-frame wordt R_yy(omega) recursief geupdate.
    estimate_doas() berekent het geometrisch gemiddelde pseudo-spectrum over een
    frequentiebereik en retourneert Q pieken.
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
        mic_pos : ndarray (M, 2)  -- mic-coordinaten (x, y) in meter (uit params.pkl)
        fs      : samplingfrequentie audio (Hz)
        L       : STFT-lengte (n_bins = L//2 + 1)
        num_sources : aantal te schatten bronnen Q (2 voor pair-geval)
        beta    : exponentiele middelingsconstante (0 < beta < 1).
                  Hoog (0.99) -> traag maar stabiel; laag (0.8) -> snel adapterend, ruisiger.
        angle_grid : np.ndarray of None. Als None: 0..180 in 0.5-graden stappen
                     (= zelfde resolutie als week4.ipynb).
        bin_range : (k_min, k_max). Als None: alle bins (week4-compatibel).
                    Voor 16 kHz / 10cm spacing kan je beter 'auto' gebruiken in
                    Processor (zie auto-aliasing-detectie daar).
        diag_load : kleine factor voor diagonal loading van R_yy. Voorkomt singulariteit
                    in eigh() wanneer een bin slechts 1 dominante bron heeft (rang-deficient).
        combine   : 'arithmetic' (per-bin normalized + mean) of 'geometric' (week4 stijl).
                    Geometric is robuuster: voorkomt dat 1 luide bin het hele spectrum
                    domineert.
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
            bin_range = (1, L // 2)
        self.k_min, self.k_max = bin_range
        self.valid_bins = np.arange(self.k_min, self.k_max)
        self.K = len(self.valid_bins)  # aantal te gebruiken bins

        # Mic-offsets t.o.v. centrum (zoals week4.ipynb). Het "centrum" is willekeurig
        # gekozen want MUSIC is invariant onder array-translatie -- de DOA hangt enkel
        # af van RELATIEVE faseverschillen tussen mics.
        mics_centered = self.mic_pos - np.mean(self.mic_pos, axis=0)
        self.px = mics_centered[:, 0].reshape(-1, 1)
        self.py = mics_centered[:, 1].reshape(-1, 1)

        # Pre-compute steering vectors A_k(theta) voor elke bin in valid_bins.
        # Steering vector: a_k(theta) = exp(-j * omega_k * (px*sin(theta) + py*cos(theta))/c)
        # Dit is de far-field plane-wave assumptie: bron op oneindig, geluid komt aan
        # als platte golf onder hoek theta.
        # We stacken ALLE bins in 1 tensor zodat estimate_doas() volledig vectoriseert.
        self.freqs = np.fft.rfftfreq(L, d=1.0 / fs)
        c = 343.0  # geluidssnelheid (m/s) bij ~20 graden Celsius
        omegas_K = 2 * np.pi * self.freqs[self.valid_bins]  # (K,)
        # taus shape: (M, A) -- voor 1 bin (omega-onafhankelijk; afhang van geometrie)
        taus = (self.px * np.sin(self.rads) + self.py * np.cos(self.rads)) / c
        # A_stacked shape: (K, M, A) -- vermenigvuldig met -j*omega_k per bin
        self.A_stacked = np.exp(-1j * omegas_K[:, None, None] * taus[None, :, :])

        # State: R_yy(omega) per bin. Wordt frame-gewijs ge-update in update().
        self.Ryy = np.zeros((self.n_bins, self.M, self.M), dtype=complex)
        self.initialized = False

    def reset(self):
        """Reset R_yy state (handig tussen pairs in evaluatie)."""
        self.Ryy = np.zeros((self.n_bins, self.M, self.M), dtype=complex)
        self.initialized = False

    def update(self, Y_frame):
        """Update R_yy met 1 STFT-frame -- VOLLEDIG GEVECTORISEERD over bins.

        Implementeert: R_yy(omega, k) = beta*R_yy(omega, k-1) + (1-beta)*y*y^H

        Parameters
        ----------
        Y_frame : ndarray (n_bins, M) complex  -- STFT van een enkel tijdframe
        """
        # Buiten-product per bin: outer[k] = Y[k] @ Y[k]^H, vorm (n_bins, M, M)
        # Equivalent met: for k: outer[k] = Y[k][:,None] @ Y[k][None,:].conj()
        outer = np.einsum('km,kn->kmn', Y_frame, Y_frame.conj())

        if not self.initialized:
            # Eerste frame: gewoon initialiseren (geen middeling mogelijk).
            self.Ryy = outer
            self.initialized = True
        else:
            # Recursieve exponentiele middeling. Dit is DE expliciete vereiste van Part 2.
            self.Ryy = self.beta * self.Ryy + (1.0 - self.beta) * outer

    def estimate_doas(self, return_spectrum=False):
        """Bereken pseudo-spectrum + Q DOA-pieken uit huidige R_yy.

        Wideband MUSIC pipeline (per bin):
          1. R_yy(omega) -> eigendecompositie
          2. noise subspace E_n = eigenvecs corresponderend met M-Q kleinste eigenwaarden
          3. pseudospectrum P(theta) = 1 / ||E_n^H a(theta)||^2
          4. combineer P over alle bins (geometric mean = product in log-domein)
          5. find peaks -> Q grootste = DOAs

        Returns
        -------
        doas_sorted : ndarray (Q,) gesorteerd op hoek (graden)
        spectrum_db : ndarray (len(angles),) in dB (alleen als return_spectrum=True)
        """
        if not self.initialized:
            return (np.full(self.Q, np.nan), None) if return_spectrum else np.full(self.Q, np.nan)

        # Selecteer alleen valid bins voor MUSIC (sla aliasing bins of DC over).
        # R_batch shape: (K, M, M).
        R_batch = self.Ryy[self.valid_bins]

        # ---- Diagonal loading ---- (numerieke stabilisatie)
        # Voegt epsilon * trace(R)/M toe aan de diagonaal. Voorkomt rang-deficient R
        # als een bin maar 1 dominante bron heeft (anders divergeert eigh).
        trace_K = np.einsum('kii->k', R_batch).real  # (K,)
        load = self.diag_load * trace_K / self.M     # (K,)
        eye = np.eye(self.M)
        R_loaded = R_batch + load[:, None, None] * eye[None, :, :]

        # ---- Batch eigendecompositie ----
        # eigvecs shape: (K, M, M), eigenvalues sorteren oplopend (eigh-conventie).
        # De M-Q KLEINSTE eigenwaarden corresponderen met de noise-subspace
        # (signaal ligt in de Q grootste eigenwaarden bij hoge SNR).
        _, eigvecs = np.linalg.eigh(R_loaded)
        En = eigvecs[:, :, : self.M - self.Q]  # noise subspace per bin: (K, M, M-Q)

        # ---- Pseudospectrum ----
        # P_k(theta) = 1 / || E_n^H @ a_k(theta) ||^2
        # proj shape: (K, M-Q, A) waar A = aantal kandidaat-hoeken.
        # |proj|^2 sommeren over noise-dim geeft || ||^2 per (k, theta).
        proj = np.einsum('kmn,kma->kna', En.conj(), self.A_stacked)
        denom = np.sum(np.abs(proj) ** 2, axis=1)  # (K, A)
        pseudospectra = 1.0 / (denom + 1e-30)      # (K, A)

        # ---- Combineer over bins ----
        if self.combine == "geometric":
            # Geometric mean over bins = mean in log-domein. Robuuster: 1 outlier-bin
            # kan het spectrum niet domineren omdat we MULTIPLICEREN i.p.v. optellen.
            log_p = np.log(pseudospectra + 1e-30)
            p_combined = np.exp(np.mean(log_p, axis=0))
        else:
            # Arithmetic met per-bin normalisatie: voorkomt ook dat luide bins
            # domineren, maar minder robuust dan geometric voor wideband.
            ps_norm = pseudospectra / (np.max(pseudospectra, axis=1, keepdims=True) + 1e-30)
            p_combined = np.mean(ps_norm, axis=0)

        spectrum_db = 10 * np.log10(p_combined / np.max(p_combined) + 1e-30)

        # ---- Pieken zoeken ----
        # find_peaks geeft lokale maxima -- we kiezen de Q hoogste.
        peaks_idx, _ = signal.find_peaks(spectrum_db)
        if len(peaks_idx) >= self.Q:
            top = peaks_idx[np.argsort(spectrum_db[peaks_idx])][-self.Q :]
            doas = np.sort(self.angles[top])
        elif len(peaks_idx) > 0:
            # Minder pieken dan Q -> geef wat we hebben terug, rest NaN.
            # In split_left_right behoudt de aanroeper dan de vorige schatting.
            doas = np.full(self.Q, np.nan)
            real = np.sort(self.angles[peaks_idx])
            doas[: len(real)] = real
        else:
            doas = np.full(self.Q, np.nan)

        if return_spectrum:
            return doas, spectrum_db
        return doas


class RIRSteeringMUSIC:
    """Wideband MUSIC met GEMETEN-RIR steering vectors (drop-in vervanger van StreamingMUSIC).

    Verschil met StreamingMUSIC: i.p.v. een far-field plane wave model
        a(theta, k) = exp(-j * omega_k * (p . [sin(theta), cos(theta)]) / c)
    gebruiken we de FFT van de gemeten RIR per (mic, hoek):
        a_rir(theta_idx, k) = H[k, :, theta_idx]
    waarbij H = FFT(RIR, n=L) per mic per gemeten hoek.

    Voordelen:
    - Houdt rekening met niet-ideale mic-respons (niet omnidirectioneel)
    - Vangt near-field effecten (LMA mics op ~10cm spacing)
    - Bevat reflectie-patronen die in de meting zijn opgeslagen
    - Match met de werkelijke physical setup -> systematische bias verdwijnt

    Beperkingen:
    - Hoek-grid is beperkt tot gemeten RIR-hoeken (~36 i.p.v. 361)
      -> wordt gemitigeerd met parabolic peak refinement (Fase C)
    - Voor sprekers ver van gemeten hoeken kan extrapolatie minder accuraat zijn

    De rest van het algoritme (R_yy update, eigh, pseudospectrum, peak picking) is
    identiek aan StreamingMUSIC -- ALLEEN de steering vector model wijzigt.

    Extra Fase B-features (allemaal opt-in via constructor):
    - snr_weight=True   : per-bin power-weighted pseudospectrum-combinatie
    - use_fb=True       : Forward-Backward averaging in update() -> decorreleer multipath
    """

    def __init__(
        self,
        rirs,
        thetas,
        fs=16000,
        L=512,
        num_sources=2,
        beta=0.95,
        bin_range=None,
        diag_load=1e-2,
        combine="geometric",
        snr_weight=False,
        use_fb=False,
        peak_refine=False,
        quality_gate_dB=0.0,
        eigvalue_ratio_thresh=0.0,
    ):
        # Defaults voor fase 3 dataset:
        # - peak_refine=False: GT-hoeken zijn IDENTIEK aan LUT-hoeken (bronnen op RIR-meetposities).
        #   Refinement schuift dan noise-driven weg van het correcte antwoord.
        #   Voor "echte" datasets met bronnen buiten het meetgrid is peak_refine=True waardevol.
        # - quality_gate_dB=0.0: gate uit. Drempel >0 weigert lage-confidence schattingen
        #   (handig voor stilte-detectie maar verliest geldig signaal als drempel te hoog).
        # - eigvalue_ratio_thresh=0.0: gate uit. >0 verlaagt Q naar 1 als tweede bron
        #   te zwak is. Te aggressief op deze dataset (zachte momenten zijn nog steeds
        #   geldige 2-bron observaties).
        """
        Parameters
        ----------
        rirs    : ndarray (n_samples_rir, M, n_angles) -- gemeten RIRs uit lma_*.npz
        thetas  : ndarray (n_angles,) -- bijhorende hoeken in graden
        fs, L, num_sources, beta, bin_range, diag_load, combine : zoals StreamingMUSIC
        snr_weight : (Fase B1) per-bin power-weighting in pseudospectrum-combinatie
        use_fb     : (Fase B2) Forward-Backward averaging in update() voor coherente sources
        """
        self.M = rirs.shape[1]
        self.fs = fs
        self.L = L
        self.n_bins = L // 2 + 1
        self.Q = num_sources
        self.beta = beta
        self.diag_load = diag_load
        self.combine = combine
        self.snr_weight = snr_weight
        self.use_fb = use_fb
        self.peak_refine = peak_refine
        self.quality_gate_dB = quality_gate_dB
        self.eigvalue_ratio_thresh = eigvalue_ratio_thresh

        # Hoek-grid = gemeten RIR-hoeken (gesorteerd voor consistente output)
        order = np.argsort(thetas)
        self.angles = np.asarray(thetas)[order].astype(float)
        rirs_sorted = rirs[:, :, order]

        if bin_range is None:
            bin_range = (1, L // 2)
        self.k_min, self.k_max = bin_range
        self.valid_bins = np.arange(self.k_min, self.k_max)
        self.K = len(self.valid_bins)

        # ---- Pre-compute steering vectors uit gemeten RIRs ----
        # H[k, m, theta_idx] = FFT(rir[:, m, theta_idx], n=L)
        H = np.fft.rfft(rirs_sorted, n=L, axis=0)  # (n_bins, M, n_angles)

        # Normaliseer per-bin per-hoek t.o.v. mic 0 (zelfde conventie als lut_builder).
        # Globale magnitude-schaling per bin doet er voor MUSIC niet toe (project op
        # noise-subspace), maar relatieve mic-fasen wel -- die behouden we.
        eps = 1e-12
        H_ref = H[:, 0:1, :]                         # (n_bins, 1, n_angles)
        H_norm = H / (H_ref + eps)                   # (n_bins, M, n_angles)
        # Selecteer valid bins (zelfde aliasing-strategie als StreamingMUSIC)
        self.A_stacked = H_norm[self.valid_bins].astype(complex)  # (K, M, n_angles)

        # State: R_yy(omega) per bin -- frame-gewijs ge-update in update()
        self.Ryy = np.zeros((self.n_bins, self.M, self.M), dtype=complex)
        self.initialized = False

        # Pre-compute J voor FB averaging (anti-diagonal exchange matrix)
        if self.use_fb:
            self._J = np.fliplr(np.eye(self.M))

    def reset(self):
        """Reset R_yy state (handig tussen pairs in evaluatie)."""
        self.Ryy = np.zeros((self.n_bins, self.M, self.M), dtype=complex)
        self.initialized = False

    def update(self, Y_frame):
        """Update R_yy met 1 STFT-frame (gevectoriseerd over bins).

        Implementeert: R_yy(omega, k) = beta*R_yy(omega, k-1) + (1-beta)*y*y^H
        Optioneel gevolgd door Forward-Backward averaging (use_fb=True):
            R_FB = 0.5 * (R + J*R^*.J)  -- decorreleert coherente sources

        Parameters
        ----------
        Y_frame : ndarray (n_bins, M) complex  -- STFT van een tijdframe
        """
        outer = np.einsum('km,kn->kmn', Y_frame, Y_frame.conj())
        if not self.initialized:
            self.Ryy = outer
            self.initialized = True
        else:
            self.Ryy = self.beta * self.Ryy + (1.0 - self.beta) * outer

        if self.use_fb:
            # R_FB = 0.5 * (R + J @ R^* @ J) per bin -- decorreleert multipath.
            # einsum: (M,M) @ (K,M,M) @ (M,M) -> (K,M,M)
            R_conj = self.Ryy.conj()
            R_flipped = np.einsum('ij,kjl,lm->kim', self._J, R_conj, self._J)
            self.Ryy = 0.5 * (self.Ryy + R_flipped)

    def estimate_doas(self, return_spectrum=False):
        """Bereken pseudospectrum + Q DOA-pieken uit huidige R_yy.

        Pipeline (per bin, gevectoriseerd):
          1. R_yy(omega) -> eigendecompositie
          2. noise subspace E_n = eigenvecs van M-Q kleinste eigenwaarden
          3. pseudospectrum P(theta) = 1 / ||E_n^H a_rir(theta)||^2
          4. combineer P over bins (eventueel SNR-gewogen via snr_weight=True)
          5. find peaks -> Q grootste = DOAs

        Returns
        -------
        doas_sorted : ndarray (Q,) gesorteerd op hoek (graden)
        spectrum_db : ndarray (len(angles),) in dB (alleen als return_spectrum=True)
        """
        if not self.initialized:
            return (np.full(self.Q, np.nan), None) if return_spectrum else np.full(self.Q, np.nan)

        R_batch = self.Ryy[self.valid_bins]
        trace_K = np.einsum('kii->k', R_batch).real  # (K,) -- per-bin signal power
        load = self.diag_load * trace_K / self.M
        eye = np.eye(self.M)
        R_loaded = R_batch + load[:, None, None] * eye[None, :, :]

        # Batch eigendecompositie -> noise subspace
        eigvals, eigvecs = np.linalg.eigh(R_loaded)  # ascending eigenvalues

        # Fase C2: Eigenvalue ratio check voor adaptive source counting (opt-in).
        # Per bin: lambda_2 / lambda_1 (top-2 / top-1). Als de mediaan over bins onder
        # de drempel ligt -> slechts 1 dominante bron -> behandel als single source.
        # Default uit (thresh=0): default returnen we altijd Q peaks. Alleen aan
        # zetten als je ook expliciet wilt dat MUSIC zegt "geen 2e bron actief nu".
        Q_effective = self.Q
        if self.eigvalue_ratio_thresh > 0 and self.M >= 2:
            top1 = eigvals[:, -1]
            top2 = eigvals[:, -2]
            ratios = top2 / (top1 + 1e-30)
            if np.median(ratios) < self.eigvalue_ratio_thresh:
                Q_effective = 1
        En = eigvecs[:, :, : self.M - Q_effective]   # (K, M, M-Q_effective)

        # Pseudospectrum P_k(theta) = 1 / || E_n^H @ a_rir(theta) ||^2
        proj = np.einsum('kmn,kma->kna', En.conj(), self.A_stacked)
        denom = np.sum(np.abs(proj) ** 2, axis=1)   # (K, n_angles)
        pseudospectra = 1.0 / (denom + 1e-30)

        # Combineren over bins (Fase B1 SNR-weighted optie)
        if self.combine == "geometric":
            log_p = np.log(pseudospectra + 1e-30)
            if self.snr_weight:
                w = trace_K / (trace_K.sum() + 1e-30)        # (K,)
                p_combined = np.exp(np.einsum('k,ka->a', w, log_p))
            else:
                p_combined = np.exp(np.mean(log_p, axis=0))
        else:  # arithmetic
            ps_norm = pseudospectra / (np.max(pseudospectra, axis=1, keepdims=True) + 1e-30)
            if self.snr_weight:
                w = trace_K / (trace_K.sum() + 1e-30)
                p_combined = np.einsum('k,ka->a', w, ps_norm)
            else:
                p_combined = np.mean(ps_norm, axis=0)

        spectrum_db = 10 * np.log10(p_combined / (np.max(p_combined) + 1e-30) + 1e-30)

        # Fase C3: Spectrum quality gate (opt-in). Als de hoogste piek niet duidelijk
        # uitsteekt boven de mediaan, vertrouwen we de schatting niet.
        # Default uit (drempel 0): we returnen altijd de top peaks. Aan zetten >0 dB
        # voor noise-only chunks expliciet als NaN te markeren (handig voor metering).
        if self.quality_gate_dB > 0:
            peak_dB = float(np.max(spectrum_db))
            median_dB = float(np.median(spectrum_db))
            if (peak_dB - median_dB) < self.quality_gate_dB:
                doas = np.full(self.Q, np.nan)
                if return_spectrum:
                    return doas, spectrum_db
                return doas

        # Pieken zoeken
        peaks_idx, _ = signal.find_peaks(spectrum_db)
        if len(peaks_idx) == 0:
            doas = np.full(self.Q, np.nan)
            if return_spectrum:
                return doas, spectrum_db
            return doas

        # Selecteer top-Q_effective pieken
        n_pieken = min(Q_effective, len(peaks_idx))
        top = peaks_idx[np.argsort(spectrum_db[peaks_idx])][-n_pieken:]

        # Fase C1: Parabolic peak refinement voor sub-grid resolutie.
        # Voor elke piek bij index i, fit kwadratisch door (i-1, i, i+1):
        #   delta = 0.5 * (P[i-1] - P[i+1]) / (P[i-1] - 2*P[i] + P[i+1])
        # geclipt op [-0.5, +0.5]. Het verschuift de hoek met delta * lokale grid-stap.
        # Voor onze RIR-grid is de stap niet uniform; we gebruiken (angle[i+1]-angle[i-1])/2.
        refined_angles = []
        n_angles = len(self.angles)
        for i in top:
            if self.peak_refine and 0 < i < n_angles - 1:
                p_l = spectrum_db[i - 1]
                p_c = spectrum_db[i]
                p_r = spectrum_db[i + 1]
                denom_pf = (p_l - 2.0 * p_c + p_r)
                if abs(denom_pf) > 1e-12:
                    delta = 0.5 * (p_l - p_r) / denom_pf
                    delta = float(np.clip(delta, -0.5, 0.5))
                    # Lokale halve grid-stap (niet-uniform veilig)
                    half_step = 0.5 * (self.angles[i + 1] - self.angles[i - 1])
                    angle_refined = float(self.angles[i] + delta * half_step)
                else:
                    angle_refined = float(self.angles[i])
            else:
                angle_refined = float(self.angles[i])
            refined_angles.append(angle_refined)

        refined_angles = np.sort(np.asarray(refined_angles, dtype=float))

        # Pad naar self.Q lengte met NaN voor downstream split_left_right compatibiliteit
        doas = np.full(self.Q, np.nan)
        doas[: len(refined_angles)] = refined_angles

        if return_spectrum:
            return doas, spectrum_db
        return doas


class DOATracker:
    """Outlier-rejecting smoother voor MUSIC DOA-schattingen.

    Doel: Onderdruk uitschieters tijdens stilte/warmup zonder echte sprekerverplaatsingen
    te onderdrukken. Houdt per zijde (links/rechts) een aparte historie aan.

    Pipeline per zijde:
      1. NaN input -> behoud last_smooth (geen update)
      2. Outlier clip: als |raw - last_smooth| > outlier_thresh -> tel als 'twijfelachtig'.
         Pas pas accepteren als 2 opeenvolgende metingen consistent zijn (vermijdt
         dat 1 spurious peak de tracker laat springen).
      3. Mediaan over laatste N schattingen -> robuust tegen single-frame outliers.
      4. EMA op de mediaan -> smooth tijdsverloop. alpha=0.3 = 3-frame tijdconstante.

    Parameters
    ----------
    window         : aantal samples voor mediaan (default 5).
    alpha          : EMA-smoothing (0.0 = geen smoothing, 1.0 = alleen huidige).
                     Default 0.3 balanceert latency vs ruisonderdrukking.
    outlier_thresh : graden. Als nieuwe raw meer dan dit afwijkt van last_smooth,
                     wordt het als "kandidaat" gemarkeerd en pas geaccepteerd na
                     2e bevestiging. Default 30° -> staat reele beweging tot 30°/chunk
                     toe maar onderdrukt single-spike noise.
    """

    def __init__(self, window=5, alpha=0.3, outlier_thresh=30.0):
        self.window = int(window)
        self.alpha = float(alpha)
        self.outlier_thresh = float(outlier_thresh)
        # Per-zijde state: buffer van geaccepteerde raw, smoothed (laatste output),
        # en pending kandidaat (nog te bevestigen).
        self._buf = {"left": [], "right": []}
        self._smooth = {"left": float("nan"), "right": float("nan")}
        self._pending = {"left": None, "right": None}

    def reset(self):
        self._buf = {"left": [], "right": []}
        self._smooth = {"left": float("nan"), "right": float("nan")}
        self._pending = {"left": None, "right": None}

    def update(self, raw_left, raw_right):
        """Update tracker met 1 nieuwe raw MUSIC-schatting per zijde.

        Returns
        -------
        (left_smooth, right_smooth) : tuple van floats (kunnen NaN zijn pre-warmup)
        """
        return self._update_side("left", raw_left), self._update_side("right", raw_right)

    def _update_side(self, side, raw):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return self._smooth[side]

        last = self._smooth[side]

        # Stap 1: outlier-detectie -- vraag bevestiging als raw ver van last
        if not np.isnan(last) and abs(raw - last) > self.outlier_thresh:
            pending = self._pending[side]
            if pending is None or abs(raw - pending) > self.outlier_thresh:
                # Eerste verdachte meting: park als pending, behoud current smoothed
                self._pending[side] = float(raw)
                return self._smooth[side]
            # Tweede consistent verdachte meting -> echte sprekerverplaatsing.
            # Reset buffer naar deze nieuwe positie en accept.
            self._buf[side] = [pending, float(raw)]
            self._pending[side] = None
            self._smooth[side] = float(np.median(self._buf[side]))
            return self._smooth[side]

        # Geen outlier (of last is NaN) -> reset pending en update
        self._pending[side] = None
        self._buf[side].append(float(raw))
        if len(self._buf[side]) > self.window:
            self._buf[side] = self._buf[side][-self.window :]

        # Stap 2: mediaan-filter
        med = float(np.median(self._buf[side]))
        # Stap 3: EMA smoothing
        if np.isnan(last):
            self._smooth[side] = med
        else:
            self._smooth[side] = self.alpha * med + (1.0 - self.alpha) * last
        return self._smooth[side]


def split_left_right(doas):
    """Verdeel een lijst geschatte DOAs in (left_angle, right_angle).

    Conventie uit phase_3 data: linker spreker zit per definitie in (90, 180],
    rechter in [0, 90]. Dit komt omdat de mic-array in 't midden van de kamer staat
    en de sprekers L/R zijn gekruist t.o.v. het array-front.

    Als MUSIC slechts 1 zijde vindt (typisch bij gelijktijdig stille bron of
    overlappende DOAs nabij broadside), retourneert die zijde de waarde en de andere
    np.nan. De aanroeper houdt dan typisch de VORIGE schatting aan (zie processor.py).
    """
    if doas is None or len(doas) == 0:
        return np.nan, np.nan
    arr = np.asarray(doas, dtype=float)
    arr = arr[~np.isnan(arr)]
    left_cands = arr[arr > 90.0]
    right_cands = arr[arr <= 90.0]
    if len(left_cands) > 0:
        # Bij meerdere kandidaten op dezelfde zijde: kies de hoek het verst van 90°.
        # Heuristiek -- typisch is dat de echte spreker, terwijl iets vlakbij 90°
        # vaak een spurious-piek is door de andere spreker of reflectie.
        left = float(left_cands[np.argmax(np.abs(left_cands - 90.0))])
    else:
        left = np.nan
    if len(right_cands) > 0:
        right = float(right_cands[np.argmax(np.abs(right_cands - 90.0))])
    else:
        right = np.nan
    return left, right
