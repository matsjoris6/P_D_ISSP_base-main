"""Streaming Frequency-Domain GSC voor bewegende sprekers.

Deze module wikkelt het algoritme uit deadline1/week4.ipynb cell 6 (gsc_fd) in een
streaming class met:
- sliding STFT input-buffer (overlap-add)
- per-bin NLMS-state die behouden blijft tussen chunks (ook bij DOA-update)
- target/interferer signaalpaden parallel verwerkt voor on-the-fly SIR

Per chunk binnenkomend:
    - mix-LMA  : (chunk_samples, M)  -- microfoonsignalen (target + interferer + noise)
    - target   : (chunk_samples, M)  -- bijdrage van TARGET-bron alleen (oracle, voor SIR)
    - interf   : (chunk_samples, M)  -- bijdrage van INTERFERER-bron alleen (oracle)
    - target_doa : float (graden)    -- huidige geschatte DOA van de target

Output per chunk:
    - out_mix   : (chunk_samples,) audio-output van GSC op de mixture
    - out_tar   : (chunk_samples,) GSC toegepast op target-bijdrage (voor SIR)
    - out_int   : (chunk_samples,) GSC toegepast op interferer-bijdrage (voor SIR)
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
        lut       : dict {angle: (W_FAS, B)} uit lut_builder.build_lut_from_rirs
        angles_lut: ndarray van LUT-hoeken (sorted)
        M_mics    : aantal microfoons (5 voor LMA)
        L, hop    : STFT-parameters
        mu        : NLMS-step
        vad_threshold : drempel voor frequentiebin-VAD op target-bijdrage (alleen voor leerstap)
        side      : 'left' of 'right' -- enkel voor logging/debug
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

        self.window = np.sqrt(signal.windows.hann(L, sym=False))

        # NLMS state per bin (behouden tussen chunks; ook bij DOA-update)
        self.w_nlms = np.zeros((self.n_bins, M_mics - 1), dtype=complex)

        # Running max van target-frame energie (mic 1) voor adaptive VAD
        self._tar_energy_max = 1e-6
        # Per-bin running gem. van |X_tar| voor subband-VAD (zoals week4)
        self._tar_bin_running = np.zeros(self.n_bins, dtype=np.float64)
        self._tar_bin_count = 0

        # Sliding input-buffers (mix, target, interferer) per kanaal
        self.buf_mix = np.zeros((0, M_mics), dtype=np.float64)
        self.buf_tar = np.zeros((0, M_mics), dtype=np.float64)
        self.buf_int = np.zeros((0, M_mics), dtype=np.float64)

        # Output overlap-add tail (1 channel each)
        self.tail_mix = np.zeros(L, dtype=np.float64)
        self.tail_tar = np.zeros(L, dtype=np.float64)
        self.tail_int = np.zeros(L, dtype=np.float64)

        # Huidige LUT-hoek
        self.current_angle = None
        self.W_FAS = None
        self.B = None

        # totale input-samples gezien (nodig om frame-grenzen te berekenen)
        self.n_input_seen = 0
        self.n_output_emitted = 0

    def set_doa(self, target_doa):
        """Update target-DOA (in graden). Snapt naar dichtstbijzijnde LUT-hoek.

        Belangrijk: w_nlms wordt NIET gereset bij DOA-update -- dit zorgt voor
        soepele transities bij bewegende sprekers (Part 1 van fase 3 week 1).
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
            # nog geen DOA gezet -> doorlaten met gemiddelde van 1e mic (degraded mode)
            n = mix_chunk.shape[0]
            return mix_chunk[:, 0].astype(np.float64), np.full(n, np.nan), np.full(n, np.nan)

        n_in = mix_chunk.shape[0]
        self.n_input_seen += n_in

        # Append in buffers
        self.buf_mix = np.concatenate([self.buf_mix, mix_chunk.astype(np.float64)], axis=0)
        if tar_chunk is not None:
            self.buf_tar = np.concatenate([self.buf_tar, tar_chunk.astype(np.float64)], axis=0)
        if int_chunk is not None:
            self.buf_int = np.concatenate([self.buf_int, int_chunk.astype(np.float64)], axis=0)

        # Verzamel uitgaande samples voor deze call
        out_mix_collected = []
        out_tar_collected = []
        out_int_collected = []

        eps = 1e-8

        # Verwerk zoveel STFT-frames als beschikbaar
        while self.buf_mix.shape[0] >= self.L:
            frame_mix = self.buf_mix[: self.L, :] * self.window[:, None]
            X_mix = np.fft.rfft(frame_mix, n=self.L, axis=0)  # (n_bins, M)

            do_oracle = self.buf_tar.shape[0] >= self.L and self.buf_int.shape[0] >= self.L
            if do_oracle:
                frame_tar = self.buf_tar[: self.L, :] * self.window[:, None]
                frame_int = self.buf_int[: self.L, :] * self.window[:, None]
                X_tar = np.fft.rfft(frame_tar, n=self.L, axis=0)
                X_int = np.fft.rfft(frame_int, n=self.L, axis=0)
            else:
                X_tar = None
                X_int = None

            # Per-bin GSC
            E_mix = np.zeros(self.n_bins, dtype=complex)
            E_tar = np.zeros(self.n_bins, dtype=complex)
            E_int = np.zeros(self.n_bins, dtype=complex)

            # Frame-level VAD op target-mic1 (tijddomein): "is target actief?"
            # Adaptieve drempel: vergelijk frame-energie met running max.
            if do_oracle:
                tar_frame_t = self.buf_tar[: self.L, 0]
                tar_frame_energy = float(np.std(tar_frame_t))
                # decay running max zodat we ons aanpassen aan stillere segmenten
                self._tar_energy_max = max(self._tar_energy_max * 0.9999, tar_frame_energy)
                target_active = tar_frame_energy > self.vad_threshold * self._tar_energy_max

                # Per-bin running gem voor subband-VAD (week4 stijl)
                tar_bin_mag = np.abs(X_tar[:, 0])  # mic 1, shape (n_bins,)
                self._tar_bin_count += 1
                self._tar_bin_running += (tar_bin_mag - self._tar_bin_running) / self._tar_bin_count
            else:
                target_active = True  # zonder oracle: conservatief geen update
                tar_bin_mag = None

            for k in range(self.n_bins):
                w_fas_k = self.W_FAS[k, :]
                B_k = self.B[k, :, :]

                x_mix_k = X_mix[k, :]
                y_fas_mix = np.vdot(w_fas_k, x_mix_k)
                u_mix = B_k @ x_mix_k
                e_mix = y_fas_mix - np.vdot(self.w_nlms[k], u_mix)
                E_mix[k] = e_mix

                if do_oracle:
                    x_tar_k = X_tar[k, :]
                    x_int_k = X_int[k, :]

                    y_fas_tar = np.vdot(w_fas_k, x_tar_k)
                    u_tar = B_k @ x_tar_k
                    E_tar[k] = y_fas_tar - np.vdot(self.w_nlms[k], u_tar)

                    y_fas_int = np.vdot(w_fas_k, x_int_k)
                    u_int = B_k @ x_int_k
                    E_int[k] = y_fas_int - np.vdot(self.w_nlms[k], u_int)

                    # Per-bin VAD: deze bin is "stil" als target-magnitude << gemiddelde
                    bin_silent = tar_bin_mag[k] < self.vad_threshold * (self._tar_bin_running[k] + 1e-6)
                    # Update alleen als (frame stil) OF (bin stil binnen actief frame)
                    if (not target_active) or bin_silent:
                        power = np.vdot(u_mix, u_mix).real
                        self.w_nlms[k] += self.mu * u_mix * np.conj(e_mix) / (power + eps)
                else:
                    # Geen oracle beschikbaar -> conservatief: geen NLMS-update deze frame
                    pass

            # iSTFT van deze frame -> tijddomein L samples (gewindowed)
            time_mix = np.fft.irfft(E_mix, n=self.L) * self.window
            time_tar = np.fft.irfft(E_tar, n=self.L) * self.window if do_oracle else np.zeros(self.L)
            time_int = np.fft.irfft(E_int, n=self.L) * self.window if do_oracle else np.zeros(self.L)

            # Overlap-add met tail
            mix_full = self.tail_mix.copy()
            tar_full = self.tail_tar.copy()
            int_full = self.tail_int.copy()
            mix_full[: self.L] += time_mix
            tar_full[: self.L] += time_tar
            int_full[: self.L] += time_int

            # eerste hop samples zijn nu definitief klaar
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

            # tail = rest van mix_full geshift met hop, padded met L-hop nullen
            new_tail_mix = np.zeros(self.L)
            new_tail_mix[: self.L - self.hop] = mix_full[self.hop : self.L]
            self.tail_mix = new_tail_mix

            new_tail_tar = np.zeros(self.L)
            new_tail_tar[: self.L - self.hop] = tar_full[self.hop : self.L]
            self.tail_tar = new_tail_tar

            new_tail_int = np.zeros(self.L)
            new_tail_int[: self.L - self.hop] = int_full[self.hop : self.L]
            self.tail_int = new_tail_int

            # advance input buffers met hop
            self.buf_mix = self.buf_mix[self.hop :, :]
            if do_oracle:
                self.buf_tar = self.buf_tar[self.hop :, :]
                self.buf_int = self.buf_int[self.hop :, :]
            else:
                # consumeer in mix-tempo zodat buffers niet blijven groeien
                if self.buf_tar.shape[0] >= self.hop:
                    self.buf_tar = self.buf_tar[self.hop :, :]
                if self.buf_int.shape[0] >= self.hop:
                    self.buf_int = self.buf_int[self.hop :, :]

        # plak alle ready samples aaneen
        out_mix = np.concatenate(out_mix_collected) if out_mix_collected else np.zeros(0)
        out_tar = np.concatenate(out_tar_collected) if out_tar_collected else np.zeros(0)
        out_int = np.concatenate(out_int_collected) if out_int_collected else np.zeros(0)

        # We willen exact n_in samples teruggeven. Trim of pad met stilte (warmup).
        if out_mix.shape[0] >= n_in:
            ret_mix = out_mix[:n_in]
            ret_tar = out_tar[:n_in]
            ret_int = out_int[:n_in]
            # restant blijft niet bewaard -- volgende call genereert nieuwe samples
            # (algoritmisch delay = L samples warmup, daarna 1-op-1)
        else:
            pad_n = n_in - out_mix.shape[0]
            ret_mix = np.concatenate([np.zeros(pad_n), out_mix])
            ret_tar = np.concatenate([np.full(pad_n, np.nan), out_tar])
            ret_int = np.concatenate([np.full(pad_n, np.nan), out_int])

        self.n_output_emitted += n_in
        return ret_mix, ret_tar, ret_int
