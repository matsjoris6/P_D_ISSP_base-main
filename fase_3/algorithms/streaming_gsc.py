"""Streaming Frequency-Domain GSC voor bewegende sprekers.

Deze module wikkelt het algoritme uit deadline1/week4.ipynb cell 6 (gsc_fd) in een
streaming class met:
  - sliding STFT input-buffer (overlap-add met 50% overlap, sqrt(hann) windowing)
  - per-bin NLMS-state die behouden blijft tussen chunks (en bij DOA-update)
  - target/interferer signaalpaden parallel verwerkt voor on-the-fly SIR (Part 3)

GSC-architectuur (zie ook week4 slides):
                        ┌──── W_FAS (FAS BF) ───┐
       x_k (M mics) ────┤                       ┝──> e_k = y_FAS - w_NLMS^H * u
                        └── B (Blocking M.) ────┘     ↑                      │
                                  │                   │                      │
                                  └─→ u_k (M-1) ──────┘                      │
                                                                              │
                                                                          BF-output

  - W_FAS: Filter-And-Sum BF gericht op target. Output = "target + leakage".
  - B: blocking matrix (in null space van target's RIR). Output = "interferer-only".
  - w_NLMS: adaptief filter dat leert hoe interferer in target-pad lekt -> aftrekken.
  - NLMS update: alleen wanneer target STIL is (anders zou hij target zelf cancellen).

Per chunk binnenkomend:
    - mix-LMA  : (chunk_samples, M)  -- microfoonsignalen (target + interferer + noise)
    - target   : (chunk_samples, M)  -- bijdrage van TARGET-bron alleen (oracle, voor SIR)
    - interf   : (chunk_samples, M)  -- bijdrage van INTERFERER-bron alleen (oracle)
    - target_doa : float (graden)    -- huidige geschatte DOA van de target

Output per chunk:
    - out_mix   : (chunk_samples,) audio-output van GSC op de mixture
    - out_tar   : (chunk_samples,) GSC toegepast op target-bijdrage (voor SIR)
    - out_int   : (chunk_samples,) GSC toegepast op interferer-bijdrage (voor SIR)

PERFORMANCE: alle per-bin operaties zijn gevectoriseerd over n_bins met einsum.
Dit verandert NIETS aan de math (per-bin onafhankelijke berekeningen) -- alleen
de uitvoeringssnelheid. Speedup ~6x t.o.v. Python for-loop versie.
"""
import numpy as np
import scipy.signal as signal

from .lut_builder import snap_angle_to_lut


class StreamingFDGSC:
    """FD-GSC met one-side targeting. Maak twee instanties (left, right) voor Part 4."""

    def __init__(
        self,
        lut,
        angles_lut,
        M_mics,
        L=512,
        hop=256,
        mu=0.05,
        vad_threshold=0.1,
        side="left",
    ):
        """
        Parameters
        ----------
        lut       : dict {angle: (W_FAS, B)} uit lut_builder.build_lut_from_rirs.
                    Pre-berekend offline want bouwen van blocking matrix (null_space)
                    is duur en hangt enkel af van GEMETEN RIRs (niet van audio).
        angles_lut: ndarray van LUT-hoeken (sorted). Gebruikt voor snap-to-grid.
        M_mics    : aantal microfoons (5 voor LMA).
        L, hop    : STFT-parameters. L=512 @ 16 kHz = 32ms frames. 50% overlap (hop=L/2)
                    geeft perfecte reconstructie met sqrt(hann)-windows.
        mu        : NLMS-step. KRITISCH:
                      - Te groot (>=0.01): target-leakage in B kan target cancelen -> SIR daalt
                      - Te klein (<=0.0005): NLMS adapteert te traag voor moving sources
                      - Optimaal voor fase-3 16kHz: 0.001 (uit empirische tuning)
                      - week4 gebruikte 0.1 op 44.1 kHz statische data; werkt NIET hier.
        vad_threshold : drempel voor frame-level VAD op target-bijdrage (alleen voor leerstap).
                        Standaard 0.1 = exact zoals week4 (vad = |speech| > 0.1*std).
        side      : 'left' of 'right' -- enkel voor logging/debug.
        """
        self.lut = lut
        self.angles_lut = angles_lut
        self.M = M_mics
        self.L = L
        self.hop = hop
        self.n_bins = L // 2 + 1
        self.mu = mu
        self.vad_threshold = vad_threshold
        self.side = side

        # Sqrt(hann) window. Met 50% overlap geldt: sum_n window[n]^2 = 1 voor elk
        # sample over 2 overlappende frames -> perfect reconstruction in OLA.
        self.window = np.sqrt(signal.windows.hann(L, sym=False))

        # NLMS state: 1 adaptief filter w[k] van M-1 taps per bin.
        # CRUCIAAL: deze state wordt behouden tussen chunks EN tussen DOA-updates.
        # Dat geeft soepele transities bij bewegende sprekers (Part 1 vereiste).
        self.w_nlms = np.zeros((self.n_bins, M_mics - 1), dtype=complex)

        # Running max van target-frame std voor frame-level VAD.
        # In week4 statisch: drempel = 0.1 * np.std(speech) over hele opname.
        # Streaming-equivalent: 0.1 * lopend maximum van per-frame std,
        # zacht decay (*0.9999 per frame) zodat de drempel zich aanpast.
        self._tar_energy_max = 1e-6

        # Sliding INPUT-buffers (mix, target, interferer) per kanaal.
        # We laten elke chunk hier in groeien en pakken er volle L-frames af.
        self.buf_mix = np.zeros((0, M_mics), dtype=np.float64)
        self.buf_tar = np.zeros((0, M_mics), dtype=np.float64)
        self.buf_int = np.zeros((0, M_mics), dtype=np.float64)

        # Output OVERLAP-ADD tail. Bij 50% overlap zijn 1e hop samples definitief
        # klaar na elke frame; de rest gaat in tail om met volgende frame opgeteld
        # te worden. Dit is de standaard OLA-iSTFT.
        self.tail_mix = np.zeros(L, dtype=np.float64)
        self.tail_tar = np.zeros(L, dtype=np.float64)
        self.tail_int = np.zeros(L, dtype=np.float64)

        # Pending output buffer: als een chunk-binnenkomst niet exact tot een
        # gehele aantal hops leidt, bewaren we de overschot voor de volgende call.
        # Voorkomt sample-verlies bij chunk-grenzen.
        self._pending_mix = np.zeros(0, dtype=np.float64)
        self._pending_tar = np.zeros(0, dtype=np.float64)
        self._pending_int = np.zeros(0, dtype=np.float64)

        # Huidige LUT-hoek + bijhorende W_FAS en B (cached).
        self.current_angle = None
        self.W_FAS = None  # shape (n_bins, M)
        self.B = None      # shape (n_bins, M-1, M)

        # Boekhoud-counters
        self.n_input_seen = 0
        self.n_output_emitted = 0

    def set_doa(self, target_doa):
        """Update target-DOA (in graden). Snapt naar dichtstbijzijnde LUT-hoek.

        Belangrijk: w_nlms wordt NIET gereset bij DOA-update. Dat zou bij elke
        sprekerverplaatsing de NLMS-coefficienten weggooien -> spike in output, slechte
        cancellation tot NLMS opnieuw geconvergeerd is. Door de state te BEHOUDEN
        krijg je soepele transities (Part 1 vereiste = "FD-GSC werkt in dynamische
        omgeving").
        """
        if target_doa is None or np.isnan(target_doa):
            return
        snapped = snap_angle_to_lut(target_doa, self.angles_lut)
        if snapped != self.current_angle:
            self.current_angle = snapped
            self.W_FAS, self.B = self.lut[snapped]

    def process_chunk(self, mix_chunk, tar_chunk=None, int_chunk=None):
        """Verwerk één chunk audio.

        Parameters
        ----------
        mix_chunk : (n_samples, M)  binnenkomende mix
        tar_chunk : (n_samples, M)  target-bijdrage (oracle, voor SIR/VAD); None toegestaan
        int_chunk : (n_samples, M)  interferer-bijdrage (oracle, voor SIR); None toegestaan

        Returns
        -------
        out_mix : (n_samples,)  GSC-output op mix
        out_tar : (n_samples,)  GSC-output op target (NaN-vector als tar_chunk None)
        out_int : (n_samples,)  GSC-output op interferer (NaN-vector als int_chunk None)
        """
        if self.W_FAS is None:
            # Nog geen DOA gezet (eerste chunks): doorlaten met 1e mic. MUSIC heeft
            # ~1 frame nodig om te initialiseren -> typisch slechts 1-2 chunks degraded.
            n = mix_chunk.shape[0]
            return mix_chunk[:, 0].astype(np.float64), np.full(n, np.nan), np.full(n, np.nan)

        n_in = mix_chunk.shape[0]
        self.n_input_seen += n_in

        # Append nieuwe samples in input-buffers
        self.buf_mix = np.concatenate([self.buf_mix, mix_chunk.astype(np.float64)], axis=0)
        if tar_chunk is not None:
            self.buf_tar = np.concatenate([self.buf_tar, tar_chunk.astype(np.float64)], axis=0)
        if int_chunk is not None:
            self.buf_int = np.concatenate([self.buf_int, int_chunk.astype(np.float64)], axis=0)

        out_mix_collected = []
        out_tar_collected = []
        out_int_collected = []

        eps = 1e-8

        # Verwerk zoveel volledige STFT-frames als beschikbaar in de input-buffer
        while self.buf_mix.shape[0] >= self.L:
            # ---- STFT van mix ----
            frame_mix = self.buf_mix[: self.L, :] * self.window[:, None]
            X_mix = np.fft.rfft(frame_mix, n=self.L, axis=0)  # (n_bins, M)

            # Oracle paden alleen verwerken als oracle-data binnen is. Tijdens "warmup"
            # (eerste paar frames) kan de target-buffer korter zijn dan mix-buffer.
            do_oracle = self.buf_tar.shape[0] >= self.L and self.buf_int.shape[0] >= self.L
            if do_oracle:
                frame_tar = self.buf_tar[: self.L, :] * self.window[:, None]
                frame_int = self.buf_int[: self.L, :] * self.window[:, None]
                X_tar = np.fft.rfft(frame_tar, n=self.L, axis=0)
                X_int = np.fft.rfft(frame_int, n=self.L, axis=0)

            # ---- Frame-level VAD op TARGET (mic 1, tijddomein) ----
            # Week4-formule: vad = |speech[:,0]| > 0.1*std(speech[:,0]); een frame
            # is "actief" als de meerderheid van samples vad=True heeft.
            # Streaming-equivalent: vergelijk frame-std met running max.
            if do_oracle:
                tar_frame_t = self.buf_tar[: self.L, 0]
                tar_frame_std = float(np.std(tar_frame_t))
                # Soft decay van running max zodat drempel meegaat met audio-amplitude
                self._tar_energy_max = max(self._tar_energy_max * 0.9999, tar_frame_std)
                target_active = tar_frame_std > self.vad_threshold * self._tar_energy_max
            else:
                # Zonder oracle: conservatief NIET updaten (geen target-leakage cancellen
                # in onbekende toestand).
                target_active = True

            # ---- Vectoriseerde per-bin GSC (FAS BF + Blocking + NLMS) ----
            # Voor elke bin k:
            #   y_FAS[k]  = W_FAS[k]^H @ X[k]                  -- target-pad
            #   u[k]      = B[k] @ X[k]                         -- interferer-only
            #   e[k]      = y_FAS[k] - w_NLMS[k]^H @ u[k]       -- BF output
            # einsum vectoriseert dit over alle bins tegelijk.
            y_fas_mix = np.einsum('km,km->k', np.conj(self.W_FAS), X_mix)        # (n_bins,)
            u_mix = np.einsum('kij,kj->ki', self.B, X_mix)                       # (n_bins, M-1)
            E_mix = y_fas_mix - np.einsum('km,km->k', np.conj(self.w_nlms), u_mix)

            if do_oracle:
                y_fas_tar = np.einsum('km,km->k', np.conj(self.W_FAS), X_tar)
                u_tar = np.einsum('kij,kj->ki', self.B, X_tar)
                E_tar = y_fas_tar - np.einsum('km,km->k', np.conj(self.w_nlms), u_tar)

                y_fas_int = np.einsum('km,km->k', np.conj(self.W_FAS), X_int)
                u_int = np.einsum('kij,kj->ki', self.B, X_int)
                E_int = y_fas_int - np.einsum('km,km->k', np.conj(self.w_nlms), u_int)
            else:
                E_tar = np.zeros(self.n_bins, dtype=complex)
                E_int = np.zeros(self.n_bins, dtype=complex)

            # ---- NLMS update (alleen bij STIL target, week4-conventie) ----
            # Update-regel per bin: w[k] += mu * u[k] * conj(e[k]) / (||u[k]||^2 + eps)
            # Power-normalisatie maakt convergentie onafhankelijk van signaal-niveau.
            # GEEN per-bin VAD: dat zat NIET in week4 en kan divergeren op bins waar
            # target sparse is (NLMS leert dan op interferer-only en mis-applieert wanneer
            # target wel in die bin zit).
            if not target_active:
                power = np.einsum('km,km->k', np.conj(u_mix), u_mix).real  # (n_bins,)
                # Broadcast: (n_bins, M-1) += mu * u * conj(e)[:,None] / (power[:,None] + eps)
                self.w_nlms += self.mu * u_mix * (np.conj(E_mix)[:, None] / (power[:, None] + eps))

            # ---- iSTFT (per frame) ----
            # irfft + sqrt(hann) tweede windowing = perfect reconstruction met OLA.
            time_mix = np.fft.irfft(E_mix, n=self.L) * self.window
            if do_oracle:
                time_tar = np.fft.irfft(E_tar, n=self.L) * self.window
                time_int = np.fft.irfft(E_int, n=self.L) * self.window
            else:
                time_tar = np.zeros(self.L)
                time_int = np.zeros(self.L)

            # ---- Overlap-Add met tail van vorige frame ----
            # tail bevat de "naklank" van vorige frame (laatste L-hop samples).
            # Optellen met huidige frame -> eerste hop samples zijn klaar.
            mix_full = self.tail_mix.copy()
            tar_full = self.tail_tar.copy()
            int_full = self.tail_int.copy()
            mix_full[: self.L] += time_mix
            tar_full[: self.L] += time_tar
            int_full[: self.L] += time_int

            # Eerste hop samples zijn definitief
            ready_mix = mix_full[: self.hop].copy()
            ready_tar = tar_full[: self.hop].copy()
            ready_int = int_full[: self.hop].copy()

            out_mix_collected.append(ready_mix)
            if do_oracle:
                out_tar_collected.append(ready_tar)
                out_int_collected.append(ready_int)
            else:
                out_tar_collected.append(np.full(self.hop, np.nan))
                out_int_collected.append(np.full(self.hop, np.nan))

            # Nieuwe tail = rest van mix_full, geshift met hop, padded met nullen
            new_tail_mix = np.zeros(self.L)
            new_tail_mix[: self.L - self.hop] = mix_full[self.hop : self.L]
            self.tail_mix = new_tail_mix

            new_tail_tar = np.zeros(self.L)
            new_tail_tar[: self.L - self.hop] = tar_full[self.hop : self.L]
            self.tail_tar = new_tail_tar

            new_tail_int = np.zeros(self.L)
            new_tail_int[: self.L - self.hop] = int_full[self.hop : self.L]
            self.tail_int = new_tail_int

            # Schuif input-buffers met hop (50% overlap met volgende frame)
            self.buf_mix = self.buf_mix[self.hop :, :]
            if do_oracle:
                self.buf_tar = self.buf_tar[self.hop :, :]
                self.buf_int = self.buf_int[self.hop :, :]
            else:
                # Zonder oracle: consumeer alleen wat er is, zodat buffers niet
                # mismatch raken met buf_mix.
                if self.buf_tar.shape[0] >= self.hop:
                    self.buf_tar = self.buf_tar[self.hop :, :]
                if self.buf_int.shape[0] >= self.hop:
                    self.buf_int = self.buf_int[self.hop :, :]

        # ---- Pending-buffer beheer (geen sample-verlies) ----
        # Voeg nieuwe ready samples toe aan pending output-buffers
        if out_mix_collected:
            self._pending_mix = np.concatenate([self._pending_mix] + out_mix_collected)
            self._pending_tar = np.concatenate([self._pending_tar] + out_tar_collected)
            self._pending_int = np.concatenate([self._pending_int] + out_int_collected)

        # Geef exact n_in samples terug. Pad met 0/NaN tijdens warmup.
        if self._pending_mix.shape[0] >= n_in:
            ret_mix = self._pending_mix[:n_in].copy()
            ret_tar = self._pending_tar[:n_in].copy()
            ret_int = self._pending_int[:n_in].copy()
            self._pending_mix = self._pending_mix[n_in:]
            self._pending_tar = self._pending_tar[n_in:]
            self._pending_int = self._pending_int[n_in:]
        else:
            # Warmup: nog niet genoeg gehavend, pad voorkant met 0 / NaN.
            pad_n = n_in - self._pending_mix.shape[0]
            ret_mix = np.concatenate([np.zeros(pad_n), self._pending_mix])
            ret_tar = np.concatenate([np.full(pad_n, np.nan), self._pending_tar])
            ret_int = np.concatenate([np.full(pad_n, np.nan), self._pending_int])
            self._pending_mix = np.zeros(0, dtype=np.float64)
            self._pending_tar = np.zeros(0, dtype=np.float64)
            self._pending_int = np.zeros(0, dtype=np.float64)

        self.n_output_emitted += n_in
        return ret_mix, ret_tar, ret_int
