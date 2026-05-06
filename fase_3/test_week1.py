"""End-to-end standalone test voor fase 3 week 1.

Draait de volledige streaming pipeline op één pair (default: pair1, anechoic) zonder
het server/worker skeleton, zodat we de 4 onderdelen van week 1 kunnen valideren:

  Part 1: FD-GSC werkt in dynamische omgeving (bewegende sprekers).
  Part 2: Dynamische DOA-schatting met exp. middeling van R_yy (beta-tunable).
  Part 3: SIR per chunk, geplot tegen tijd.
  Part 4: Twee beamformed streams (left-target en right-target) -- demo van outputselectie.

Output:
  - Geluidsbestanden in fase_3/output/<pair>_<scenario>/
  - DOA + SIR plots als PNG
  - Console-samenvatting met DOA-error en eind-SIR

Gebruik:
    python fase_3/test_week1.py                      # default: pair1, anechoic
    python fase_3/test_week1.py --pair 5 --reverberant
    python fase_3/test_week1.py --beta 0.99 --mu 0.05
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import wavfile

# Maak het mogelijk om als script of module te draaien
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.dirname(THIS_DIR))

from algorithms.lut_builder import build_lut_from_rirs
from algorithms.streaming_doa import StreamingMUSIC, RIRSteeringMUSIC, DOATracker, split_left_right
from algorithms.streaming_gsc import StreamingFDGSC
from algorithms.streaming_sir import StreamingSIR, compute_sir_full


# Pad naar de phase 3 audio data (op de Desktop, niet in deze repo)
DATA_ROOT = "/Users/macbookmats/Desktop/P_D_ISSP_base-main/fase_3/data/phase3_audioData/audiodata_batch_1"


def load_pair(pair_no, scenario="anechoic"):
    """Laad alle bestanden van één pair + scenario-level params en RIRs."""
    base = os.path.join(DATA_ROOT, scenario)
    pair_dir = os.path.join(base, f"pair{pair_no}")

    with open(os.path.join(base, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    rir_filename = "lma_16kHz.npz" if scenario == "anechoic" else "lma_16kHz_200ms.npz"
    rirs_data = np.load(os.path.join(base, rir_filename))

    fs_mix, mix = wavfile.read(os.path.join(pair_dir, "mixture_LMA.wav"))
    _, lf = wavfile.read(os.path.join(pair_dir, "leftSpeaker_LMA.wav"))
    _, rt = wavfile.read(os.path.join(pair_dir, "rightSpeaker_LMA.wav"))

    gt = np.load(os.path.join(pair_dir, "gt.npz"))
    # Bouw per-sample DOA ground truth (piecewise constant)
    doa_l_gt = np.concatenate([np.repeat(a, n) for a, n in zip(gt["angles_l"], np.diff(np.concatenate([[0], gt["endSamples_l"]])))])
    doa_r_gt = np.concatenate([np.repeat(a, n) for a, n in zip(gt["angles_r"], np.diff(np.concatenate([[0], gt["endSamples_r"]])))])
    # pad naar audio-lengte
    n_audio = mix.shape[0]
    if doa_l_gt.shape[0] < n_audio:
        doa_l_gt = np.concatenate([doa_l_gt, np.full(n_audio - doa_l_gt.shape[0], doa_l_gt[-1])])
    else:
        doa_l_gt = doa_l_gt[:n_audio]
    if doa_r_gt.shape[0] < n_audio:
        doa_r_gt = np.concatenate([doa_r_gt, np.full(n_audio - doa_r_gt.shape[0], doa_r_gt[-1])])
    else:
        doa_r_gt = doa_r_gt[:n_audio]

    return {
        "fs": fs_mix,
        "mix": mix,
        "left": lf,
        "right": rt,
        "doa_l_gt": doa_l_gt,
        "doa_r_gt": doa_r_gt,
        "params": params,
        "rirs": rirs_data["rirs"],
        "rir_thetas": rirs_data["thetas"],
        "mic_pos": np.array(params["LMAcoords"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, default=1)
    parser.add_argument("--scenario", type=str, default="anechoic", choices=["anechoic", "reverberant"])
    parser.add_argument("--duration", type=float, default=60.0, help="Aantal seconden te verwerken (default: 60s)")
    parser.add_argument("--L", type=int, default=512, help="STFT-lengte")
    parser.add_argument("--update_rate", type=int, default=32, help="Chunks per seconde (skeleton-default: 32)")
    parser.add_argument("--beta", type=float, default=0.95, help="Exponentiele middelingsconstante voor R_yy")
    parser.add_argument("--mu", type=float, default=0.001, help="NLMS step")
    parser.add_argument("--doa_update_every", type=int, default=4, help="Update DOA elke N chunks (1/8 sec bij 32 chunks/s)")
    parser.add_argument("--bin_range", type=str, default="auto",
                        help="MUSIC bin range 'k_min,k_max' of 'auto' (= onder aliasing-limiet) of 'full' (1..L/2 zoals week4)")
    parser.add_argument("--combine", type=str, default="geometric", choices=["geometric", "arithmetic"],
                        help="MUSIC pseudospectrum combiner (week4 default = geometric)")
    parser.add_argument("--sv_model", type=str, default="rir", choices=["rir", "planewave"],
                        help="MUSIC steering vector model: 'rir' (Fase A, gemeten RIRs) of 'planewave' (oude versie)")
    parser.add_argument("--snr_weight", action="store_true", default=False,
                        help="Fase B1: per-bin power-weighted pseudospectrum (alleen sv_model=rir)")
    parser.add_argument("--use_fb", type=str, default="auto", choices=["auto", "on", "off"],
                        help="Fase B2: Forward-Backward averaging in R_yy. 'auto' = aan voor reverberant")
    parser.add_argument("--tracker", action="store_true", default=False,
                        help="Fase D: DOATracker (mediaan + EMA + outlier-rejectie) toepassen")
    parser.add_argument("--out_dir", type=str, default=os.path.join(THIS_DIR, "output"))
    args = parser.parse_args()

    print(f"=== Fase 3 Week 1 standalone test ===")
    print(f"Pair: {args.pair}, scenario: {args.scenario}, duration: {args.duration}s")
    print(f"L={args.L}, update_rate={args.update_rate}, beta={args.beta}, mu={args.mu}")
    print(f"sv_model={args.sv_model}, snr_weight={args.snr_weight}, use_fb={args.use_fb}")

    print("\n[1/5] Loading data...")
    data = load_pair(args.pair, args.scenario)
    fs = int(data["fs"])
    print(f"  fs={fs}, mix shape={data['mix'].shape}, mics={data['mic_pos'].shape[0]}")

    # Trim naar gewenste duratie
    n_total = min(data["mix"].shape[0], int(args.duration * fs))
    mix = data["mix"][:n_total].astype(np.float32)
    sigL = data["left"][:n_total].astype(np.float32)
    sigR = data["right"][:n_total].astype(np.float32)

    # Chunk-grootte exact zoals het skeleton (chunk per 1/update_rate seconde)
    chunk_audio = fs // args.update_rate
    n_chunks = n_total // chunk_audio
    print(f"  chunk_audio={chunk_audio} samples, n_chunks={n_chunks}")

    print("\n[2/5] Building LUT (FAS BF + Blocking Matrix per RIR angle)...")
    t0 = time.time()
    lut, angles_lut = build_lut_from_rirs(data["rirs"], data["rir_thetas"], L=args.L)
    print(f"  LUT-hoeken: {len(angles_lut)} (van {angles_lut[0]:.1f} tot {angles_lut[-1]:.1f} graden)")
    print(f"  LUT bouwen duurde {time.time()-t0:.1f}s")

    print("\n[3/5] Initialiseren streaming componenten...")
    M = data["mic_pos"].shape[0]

    # Bin-range bepalen (pas tunen, NIET het algoritme veranderen)
    if args.bin_range == "full":
        bin_range = (1, args.L // 2)
    elif args.bin_range == "auto":
        # spatial aliasing limiet: f = c / (2 * d_min)
        mic_pos_a = data["mic_pos"]
        dists = [np.linalg.norm(mic_pos_a[i] - mic_pos_a[j])
                 for i in range(M) for j in range(i + 1, M)]
        d_min = min(dists) if dists else 0.1
        f_alias = 343.0 / (2.0 * d_min)
        k_alias = int(round(f_alias / (fs / args.L)))
        bin_range = (2, max(8, min(args.L // 2, k_alias)))
        print(f"  Auto bin_range: ({bin_range[0]}, {bin_range[1]}) [<= aliasing limiet {f_alias:.0f} Hz]")
    else:
        a, b = args.bin_range.split(",")
        bin_range = (int(a), int(b))

    # Bepaal use_fb (auto = aan voor reverberant, uit voor anechoic)
    if args.use_fb == "auto":
        use_fb = (args.scenario == "reverberant")
    else:
        use_fb = (args.use_fb == "on")

    if args.sv_model == "rir":
        music = RIRSteeringMUSIC(
            rirs=data["rirs"],
            thetas=data["rir_thetas"],
            fs=fs,
            L=args.L,
            num_sources=2,
            beta=args.beta,
            bin_range=bin_range,
            combine=args.combine,
            snr_weight=args.snr_weight,
            use_fb=use_fb,
        )
        print(f"  MUSIC: RIR-derived steering vectors ({len(music.angles)} hoeken), "
              f"snr_weight={args.snr_weight}, use_fb={use_fb}")
    else:
        music = StreamingMUSIC(
            mic_pos=data["mic_pos"],
            fs=fs,
            L=args.L,
            num_sources=2,
            beta=args.beta,
            bin_range=bin_range,
            combine=args.combine,
        )
        print(f"  MUSIC: plane-wave steering vectors (361 hoeken op 0.5° grid)")
    gsc_left = StreamingFDGSC(lut, angles_lut, M, L=args.L, hop=args.L // 2, mu=args.mu, side="left")
    gsc_right = StreamingFDGSC(lut, angles_lut, M, L=args.L, hop=args.L // 2, mu=args.mu, side="right")
    tracker = DOATracker(window=5, alpha=0.3, outlier_thresh=30.0) if args.tracker else None
    sir_left = StreamingSIR(fs=fs, window_seconds=2.0)
    sir_right = StreamingSIR(fs=fs, window_seconds=2.0)

    # Pre-compute STFT-frame-window voor MUSIC update
    music_window = np.sqrt(signal.windows.hann(args.L, sym=False))
    music_buf = np.zeros((0, M), dtype=np.float64)

    # Output buffers
    out_left_full = np.zeros(n_chunks * chunk_audio, dtype=np.float64)
    out_right_full = np.zeros(n_chunks * chunk_audio, dtype=np.float64)
    out_tar_left_full = np.zeros(n_chunks * chunk_audio, dtype=np.float64)
    out_int_left_full = np.zeros(n_chunks * chunk_audio, dtype=np.float64)
    out_tar_right_full = np.zeros(n_chunks * chunk_audio, dtype=np.float64)
    out_int_right_full = np.zeros(n_chunks * chunk_audio, dtype=np.float64)

    # Logs
    log_t = []
    log_doa_left_est = []
    log_doa_right_est = []
    log_doa_left_gt = []
    log_doa_right_gt = []
    log_sir_left = []
    log_sir_right = []

    # Persistente raw MUSIC DOA-schatting (niet de LUT-snapped!)
    raw_doa_left = np.nan
    raw_doa_right = np.nan
    log_doa_left_snapped = []
    log_doa_right_snapped = []

    print("\n[4/5] Streaming-verwerking...")
    t_start = time.time()
    for ci in range(n_chunks):
        s = ci * chunk_audio
        e = s + chunk_audio
        mix_c = mix[s:e]            # (chunk_audio, M)
        # target voor LEFT-GSC = leftSpeaker; voor RIGHT-GSC = rightSpeaker
        sigL_c = sigL[s:e]
        sigR_c = sigR[s:e]

        # ---- DOA update (MUSIC) ----
        music_buf = np.concatenate([music_buf, mix_c.astype(np.float64)], axis=0)
        # extract zoveel frames als beschikbaar (50% overlap)
        hop_music = args.L // 2
        while music_buf.shape[0] >= args.L:
            frame = music_buf[: args.L, :] * music_window[:, None]
            Y = np.fft.rfft(frame, n=args.L, axis=0)  # (n_bins, M)
            music.update(Y)
            music_buf = music_buf[hop_music:, :]

        # Schat DOAs niet elke chunk (te zwaar) -- elke args.doa_update_every chunks
        if ci % args.doa_update_every == 0 and music.initialized:
            doas = music.estimate_doas()
            l, r = split_left_right(doas)
            if not np.isnan(l):
                raw_doa_left = l
            if not np.isnan(r):
                raw_doa_right = r
            if tracker is not None:
                sm_l, sm_r = tracker.update(raw_doa_left, raw_doa_right)
                gsc_in_l = sm_l if not np.isnan(sm_l) else raw_doa_left
                gsc_in_r = sm_r if not np.isnan(sm_r) else raw_doa_right
            else:
                gsc_in_l = raw_doa_left
                gsc_in_r = raw_doa_right
            gsc_left.set_doa(gsc_in_l)
            gsc_right.set_doa(gsc_in_r)
        # log altijd dezelfde persistente raw MUSIC schatting (NIET smoothed -> eerlijke benchmark)
        doa_left_est = raw_doa_left
        doa_right_est = raw_doa_right

        # ---- GSC links: target = sigL, interferer = sigR ----
        out_mix_L, out_tar_L, out_int_L = gsc_left.process_chunk(mix_c, sigL_c, sigR_c)
        # ---- GSC rechts: target = sigR, interferer = sigL ----
        out_mix_R, out_tar_R, out_int_R = gsc_right.process_chunk(mix_c, sigR_c, sigL_c)

        out_left_full[s:e] = out_mix_L
        out_right_full[s:e] = out_mix_R
        out_tar_left_full[s:e] = out_tar_L
        out_int_left_full[s:e] = out_int_L
        out_tar_right_full[s:e] = out_tar_R
        out_int_right_full[s:e] = out_int_R

        # ---- SIR ----
        sir_l = sir_left.update(out_tar_L, out_int_L)
        sir_r = sir_right.update(out_tar_R, out_int_R)

        # Logging
        log_t.append((s + chunk_audio / 2) / fs)
        log_doa_left_est.append(doa_left_est)
        log_doa_right_est.append(doa_right_est)
        log_doa_left_snapped.append(gsc_left.current_angle if gsc_left.current_angle is not None else np.nan)
        log_doa_right_snapped.append(gsc_right.current_angle if gsc_right.current_angle is not None else np.nan)
        log_doa_left_gt.append(float(data["doa_l_gt"][min(s + chunk_audio // 2, len(data["doa_l_gt"]) - 1)]))
        log_doa_right_gt.append(float(data["doa_r_gt"][min(s + chunk_audio // 2, len(data["doa_r_gt"]) - 1)]))
        log_sir_left.append(sir_l)
        log_sir_right.append(sir_r)

        if (ci + 1) % args.update_rate == 0:
            print(f"  chunk {ci+1}/{n_chunks} ({(ci+1)/args.update_rate:.1f}s audio) -- "
                  f"DOA est L/R = {log_doa_left_est[-1]:.1f}/{log_doa_right_est[-1]:.1f} -- "
                  f"SIR L/R = {sir_l:.2f}/{sir_r:.2f} dB")

    elapsed = time.time() - t_start
    rt_factor = (n_total / fs) / elapsed if elapsed > 0 else float("inf")
    print(f"  Klaar: {elapsed:.1f}s wall-time voor {n_total/fs:.1f}s audio (RT factor {rt_factor:.2f}x)")

    print("\n[5/5] Opslaan resultaten...")
    out_dir = os.path.join(args.out_dir, f"pair{args.pair}_{args.scenario}")
    os.makedirs(out_dir, exist_ok=True)

    # Audio (genormaliseerd, int16)
    def to_int16(x):
        x = np.nan_to_num(x)
        peak = np.max(np.abs(x))
        if peak < 1e-12:
            return np.zeros_like(x, dtype=np.int16)
        return (x / peak * 0.95 * 32767).astype(np.int16)

    wavfile.write(os.path.join(out_dir, "gsc_left.wav"), fs, to_int16(out_left_full))
    wavfile.write(os.path.join(out_dir, "gsc_right.wav"), fs, to_int16(out_right_full))
    wavfile.write(os.path.join(out_dir, "mix_mic1.wav"), fs, to_int16(mix[:, 0]))

    # Globale SIR over de eerste 60s (zoals evaluatiemetriek voorschrijft)
    n_eval = min(60 * fs, out_left_full.shape[0])
    valid = ~np.isnan(out_tar_left_full[:n_eval]) & ~np.isnan(out_int_left_full[:n_eval])
    sir_global_left = compute_sir_full(
        y=out_tar_left_full[:n_eval][valid] + out_int_left_full[:n_eval][valid],
        x_target=out_tar_left_full[:n_eval][valid],
        x_interferer=out_int_left_full[:n_eval][valid],
    )
    valid = ~np.isnan(out_tar_right_full[:n_eval]) & ~np.isnan(out_int_right_full[:n_eval])
    sir_global_right = compute_sir_full(
        y=out_tar_right_full[:n_eval][valid] + out_int_right_full[:n_eval][valid],
        x_target=out_tar_right_full[:n_eval][valid],
        x_interferer=out_int_right_full[:n_eval][valid],
    )

    # DOA-fout
    log_doa_left_est_arr = np.array(log_doa_left_est, dtype=float)
    log_doa_right_est_arr = np.array(log_doa_right_est, dtype=float)
    log_doa_left_gt_arr = np.array(log_doa_left_gt, dtype=float)
    log_doa_right_gt_arr = np.array(log_doa_right_gt, dtype=float)
    valid_doa = ~np.isnan(log_doa_left_est_arr)
    err_left = np.abs(log_doa_left_est_arr[valid_doa] - log_doa_left_gt_arr[valid_doa])
    err_right = np.abs(log_doa_right_est_arr[valid_doa] - log_doa_right_gt_arr[valid_doa])

    log_doa_left_snapped_arr = np.array(log_doa_left_snapped, dtype=float)
    log_doa_right_snapped_arr = np.array(log_doa_right_snapped, dtype=float)
    valid_snap = ~np.isnan(log_doa_left_snapped_arr)
    err_left_snap = np.abs(log_doa_left_snapped_arr[valid_snap] - log_doa_left_gt_arr[valid_snap])
    err_right_snap = np.abs(log_doa_right_snapped_arr[valid_snap] - log_doa_right_gt_arr[valid_snap])

    print(f"\n=== Resultaten (eerste 60 s) ===")
    print(f"SIR globaal (target alleen vs interferer alleen, na BF):")
    print(f"  links-target  : {sir_global_left:.2f} dB")
    print(f"  rechts-target : {sir_global_right:.2f} dB")
    print(f"\nGemiddelde DOA-fout (raw MUSIC vs gt, 0.5-graden grid):")
    print(f"  links-spreker : {np.mean(err_left):.2f} graden (median {np.median(err_left):.2f})")
    print(f"  rechts-spreker: {np.mean(err_right):.2f} graden (median {np.median(err_right):.2f})")
    print(f"\nGemiddelde DOA-fout (LUT-snapped = gebruikt door GSC):")
    print(f"  links-spreker : {np.mean(err_left_snap):.2f} graden (median {np.median(err_left_snap):.2f})")
    print(f"  rechts-spreker: {np.mean(err_right_snap):.2f} graden (median {np.median(err_right_snap):.2f})")

    # Plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(log_t, log_doa_left_est_arr, label="DOA links est", color="C0")
    axes[0].plot(log_t, log_doa_left_gt_arr, "--", label="DOA links gt", color="C0", alpha=0.5)
    axes[0].plot(log_t, log_doa_right_est_arr, label="DOA rechts est", color="C1")
    axes[0].plot(log_t, log_doa_right_gt_arr, "--", label="DOA rechts gt", color="C1", alpha=0.5)
    axes[0].set_xlabel("tijd (s)")
    axes[0].set_ylabel("DOA (graden)")
    axes[0].set_title(f"Dynamische DOA-schatting (beta={args.beta})")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(log_t, log_sir_left, label="SIR links-targeting GSC", color="C0")
    axes[1].plot(log_t, log_sir_right, label="SIR rechts-targeting GSC", color="C1")
    axes[1].set_xlabel("tijd (s)")
    axes[1].set_ylabel("SIR (dB)")
    axes[1].set_title("SIR per 0.5s-venster")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "doa_sir.png")
    fig.savefig(plot_path, dpi=120)
    print(f"\nPlot opgeslagen: {plot_path}")
    print(f"WAV-bestanden in: {out_dir}")


if __name__ == "__main__":
    main()
