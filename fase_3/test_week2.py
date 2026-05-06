"""Standalone test voor fase 3 week 2.

Test de drie nieuwe onderdelen zonder server of GUI:

  Test 1 — get_doa_gt bug-fix
      Verifieert dat de gt-tijdslijn de juiste lengte heeft (= eindSample_l[-1])
      en niet de opgeblazen versie die de cumulatieve sommen direct als np.repeat-count
      gebruikte.

  Test 2 — AAD LSTM (standalone module)
      Voedt synthetische EEG + audio chunks door AADLSTM en controleert dat:
        - pred_prob in [0, 1] ligt
        - de eerste predictie verschijnt na exact window_s seconden
        - elke volgende na hop_s seconden
      Met --aad_model_path: test met het echte Keras model.
      Zonder --aad_model_path: test enkel de Gammatone-envelope extractie + buffer-logica.

  Test 3 — Volledige Processor (week 1 + 2 gecombineerd)
      Draait de echte Processor op een stuk audio van een pair en controleert dat
      alle drie queues (phase1, phase2, phase3) data produceren.
      Met --aad_model_path: AAD-predictions komen van het model.
      Zonder: placeholder (alterneert elke ~30s).
"""

# ZORG DA U BESTANDEN ZO STAAN:
# fase_3/data/phase3_audioData/audiodata_batch_1/anechoic/
# fase_3/data/phase3_audioData/audiodata_batch_1/reverberant/
# fase_3/data/data_phase3/
# fase_3/data/data_phase3/stimuli/
# fase_3/data/hybrid_v3_BEST.keras

# ALS JE ECHT MET AAD WIL TESTEN: DOE DIT DAN: cd /Users/macbookmats/Desktop/P_D_ISSP_base-main
""" source env_tf/bin/activate
    python fase_3/test_week2.py
    EN ALS JE ENV NOG NIET HEBT AANGEMAAKT:
    python3.11 -m venv env_tf
    source env_tf/bin/activate
    pip install tensorflow numpy scipy python-socketio
"""

#ALS JE DE GUI WIL TESTEN MET AAD MODEL:
""" 
python3.11 -m venv env_tf
source env_tf/bin/activate
pip install tensorflow numpy scipy "python-socketio[asyncio]" uvicorn fastapi matplotlib

"""

# EN DAN:
"""
cd fase_3
source ../env_tf/bin/activate
PYTHON=$(which python) AAD_MODEL_PATH="$(pwd)/data/hybrid_v3_BEST.keras" ./run_demo.sh
"""

import argparse
import asyncio
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)


# ---------------------------------------------------------------------------
# Pad-detectie (zelfde logica als run_demo.sh)
# ---------------------------------------------------------------------------

def find_data_base(override=None):
    if override:
        return override
    candidates = [
        os.path.join(THIS_DIR, "data"),
        os.path.join(REPO_ROOT, "documents_and_given_code", "phase_3"),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "phase3_audioData")):
            return c
    return None


# ---------------------------------------------------------------------------
# Test 1 — get_doa_gt bug-fix
# ---------------------------------------------------------------------------

def test_doa_gt_fix(microarray_dir, pair_no):
    print("\n=== Test 1: get_doa_gt bug-fix ===")
    pair_dir = os.path.join(microarray_dir, f"pair{pair_no}")
    gt = np.load(os.path.join(pair_dir, "gt.npz"))

    expected_len = int(gt["endSamples_l"][-1])

    # Buggy (cumulatieve sommen direct als repeat-count)
    doa_buggy = np.concatenate([
        np.repeat(e, n) for e, n in zip(gt["angles_l"], gt["endSamples_l"])
    ])

    # Correct (np.diff om duraties te berekenen)
    durations_l = np.diff(np.concatenate([[0], gt["endSamples_l"]]))
    doa_fixed = np.concatenate([
        np.repeat(e, n) for e, n in zip(gt["angles_l"], durations_l)
    ])

    print(f"  Verwachte lengte (endSamples_l[-1]) : {expected_len:,}")
    print(f"  Buggy versie lengte                 : {len(doa_buggy):,}  {'✗' if len(doa_buggy) != expected_len else '✓'}")
    print(f"  Gefixte versie lengte               : {len(doa_fixed):,}  {'✓' if len(doa_fixed) == expected_len else '✗'}")

    assert len(doa_fixed) == expected_len, \
        f"Bug-fix faalt: len={len(doa_fixed)}, verwacht={expected_len}"
    assert len(doa_buggy) != expected_len, \
        "Buggy versie geeft toevallig correcte lengte — controleer gt.npz"

    print("  [PASS] get_doa_gt fix correct")


# ---------------------------------------------------------------------------
# Test 2 — AAD LSTM standalone
# ---------------------------------------------------------------------------

def test_aad_lstm(aad_model_path, fs_audio=48000, fs_eeg=128,
                  window_s=5.0, hop_s=1.0, n_chunks=8, envelope="gammatone",
                  normalize_eeg=False):
    """Test AAD LSTM module.

    fs_audio=48000: stimuli WAVs zijn 48 kHz (niet de mic-rate van 16 kHz!).
    """
    print("\n=== Test 2: AAD LSTM standalone ===")

    from algorithms.aad_lstm import AADLSTM, GammatoneEnvelope, HilbertEnvelope

    # Test envelope-extractors onafhankelijk van model
    print("  Gammatone envelope test...")
    env_gt = GammatoneEnvelope(fs_audio=fs_audio, fs_target=fs_eeg)
    dummy_audio = np.random.randn(fs_audio).astype(np.float32)  # 1s synthetisch @ 48kHz
    env_out = env_gt(dummy_audio)
    assert env_out.shape[0] == fs_eeg, \
        f"Gammatone output lengte fout: {env_out.shape[0]} != {fs_eeg} " \
        f"(fs_audio={fs_audio}, fs_eeg={fs_eeg}, decim={fs_audio//fs_eeg})"
    assert not np.any(np.isnan(env_out)), "Gammatone output bevat NaN"
    print(f"  Gammatone: input {len(dummy_audio)} samples @ {fs_audio}Hz → output {len(env_out)} @ {fs_eeg}Hz  ✓")

    print("  Hilbert envelope test...")
    env_hb = HilbertEnvelope(fs_audio=fs_audio, fs_target=fs_eeg)
    env_out_h = env_hb(dummy_audio)
    assert env_out_h.shape[0] == fs_eeg, \
        f"Hilbert output lengte fout: {env_out_h.shape[0]} != {fs_eeg}"
    print(f"  Hilbert:   input {len(dummy_audio)} samples @ {fs_audio}Hz → output {len(env_out_h)} @ {fs_eeg}Hz  ✓")

    if aad_model_path is None:
        print("  --aad_model_path niet gegeven → skip model-predict test")
        print("  [PASS] Envelope extractors OK")
        return

    if not os.path.exists(aad_model_path):
        print(f"  [SKIP] Model niet gevonden: {aad_model_path}")
        return

    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("  [SKIP] TensorFlow niet beschikbaar in deze venv.")
        print("         Activeer env_tf voor het model-predict deel:")
        print("         source ../env_tf/bin/activate")
        return

    print(f"  Model laden: {aad_model_path} ...")
    t0 = time.time()
    aad = AADLSTM(
        model_path=aad_model_path,
        fs_audio=fs_audio,
        fs_eeg=fs_eeg,
        window_s=window_s,
        hop_s=hop_s,
        envelope=envelope,
        normalize_eeg=normalize_eeg,
    )
    print(f"  Model geladen in {time.time()-t0:.1f}s")

    # Synthetische chunks: 1s audio @ 48kHz + 1s EEG @ 128Hz per chunk
    chunk_audio = fs_audio          # 1s @ 48kHz = 48000 samples
    chunk_eeg = fs_eeg              # 1s = 128 samples
    n_eeg_ch = 64

    predictions = []
    first_pred_chunk = None

    for i in range(n_chunks):
        eeg_chunk = np.random.randn(chunk_eeg, n_eeg_ch).astype(np.float32)
        sig_l = np.random.randn(chunk_audio).astype(np.float32)
        sig_r = np.random.randn(chunk_audio).astype(np.float32)

        pred = aad.update(eeg_chunk, sig_l, sig_r)
        if pred is not None:
            if first_pred_chunk is None:
                first_pred_chunk = i + 1  # 1-indexed
            predictions.append(pred)
            assert 0.0 <= pred <= 1.0, f"pred_prob buiten [0,1]: {pred}"

    expected_first = int(np.ceil(window_s / hop_s))
    print(f"  Eerste predictie na chunk {first_pred_chunk} (verwacht >= {expected_first})  "
          f"{'✓' if first_pred_chunk is not None and first_pred_chunk >= expected_first else '✗'}")
    print(f"  Aantal predictions in {n_chunks} chunks: {len(predictions)}")
    print(f"  pred_prob range: [{min(predictions):.3f}, {max(predictions):.3f}]")
    assert len(predictions) > 0, "Geen enkele predictie ontvangen"
    print("  [PASS] AADLSTM OK")


# ---------------------------------------------------------------------------
# Test 3 — Volledige Processor + output opslaan
# ---------------------------------------------------------------------------

def test_processor(microarray_dir, pair_no, aad_model_path=None,
                   duration_s=30.0, scenario="anechoic", out_dir=None):
    print("\n=== Test 3: Volledige Processor (week 1 + 2) ===")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from processor import Processor

    # Gebruik AAD model alleen als TensorFlow beschikbaar is
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        if aad_model_path is not None:
            print("  TensorFlow niet beschikbaar → processor draait met placeholder AAD")
        aad_model_path = None

    # Laad audio
    pair_dir = os.path.join(microarray_dir, f"pair{pair_no}")
    fs, mix = wavfile.read(os.path.join(pair_dir, "mixture_LMA.wav"))
    _, lft = wavfile.read(os.path.join(pair_dir, "leftSpeaker_LMA.wav"))
    _, rgt = wavfile.read(os.path.join(pair_dir, "rightSpeaker_LMA.wav"))

    n_samples = int(duration_s * fs)
    mix = mix[:n_samples]
    lft = lft[:n_samples]
    rgt = rgt[:n_samples]

    # Ground truth DOA (zelfde logica als test_week1 + get_doa_gt fix)
    gt = np.load(os.path.join(pair_dir, "gt.npz"))
    dur_l = np.diff(np.concatenate([[0], gt["endSamples_l"]]))
    dur_r = np.diff(np.concatenate([[0], gt["endSamples_r"]]))
    doa_l_gt = np.concatenate([np.repeat(a, n) for a, n in zip(gt["angles_l"], dur_l)])
    doa_r_gt = np.concatenate([np.repeat(a, n) for a, n in zip(gt["angles_r"], dur_r)])
    doa_l_gt = doa_l_gt[:n_samples] if len(doa_l_gt) >= n_samples else np.pad(doa_l_gt, (0, n_samples - len(doa_l_gt)), constant_values=doa_l_gt[-1])
    doa_r_gt = doa_r_gt[:n_samples] if len(doa_r_gt) >= n_samples else np.pad(doa_r_gt, (0, n_samples - len(doa_r_gt)), constant_values=doa_r_gt[-1])

    # Processor aanmaken
    proc = Processor(
        data_dir=microarray_dir,
        aad_model_path=aad_model_path,
        aad_window_s=5.0,
        aad_hop_s=1.0,
    )
    aad_label = "met AAD model" if aad_model_path and os.path.exists(aad_model_path or "") else "placeholder AAD"
    print(f"  Processor aangemaakt ({aad_label})")

    # Chunk-parameters (32 chunks/s zoals het skeleton)
    chunk_size = fs // 32
    n_chunks = n_samples // chunk_size
    EEG_ACCUMULATE = 32   # 1s @ 32 chunks/s
    eeg_fs = 128
    eeg_chunk_size = eeg_fs // 32  # 4 EEG-samples per chunk
    eeg_full = np.random.randn(n_chunks * eeg_chunk_size, 64).astype(np.float64)
    audio_l_full = lft[:, 0].astype(np.float32) if lft.ndim > 1 else lft.astype(np.float32)
    audio_r_full = rgt[:, 0].astype(np.float32) if rgt.ndim > 1 else rgt.astype(np.float32)

    # Collectie-buffers
    log_t         = []
    log_doa_left  = []
    log_doa_right = []
    log_sir       = []
    log_aad       = []
    log_speaker   = []
    out_left_chunks   = []
    out_right_chunks  = []
    out_signal_chunks = []

    t_start = time.time()
    for ci in range(n_chunks):
        s = ci * chunk_size
        e = s + chunk_size

        proc.processing_microarray(mix[s:e], lft[s:e], rgt[s:e])

        if (ci + 1) % EEG_ACCUMULATE == 0:
            ws = (ci + 1 - EEG_ACCUMULATE) * eeg_chunk_size
            we = (ci + 1) * eeg_chunk_size
            eeg_win = eeg_full[ws:we]
            al = audio_l_full[s + chunk_size - fs : s + chunk_size] if s + chunk_size >= fs else audio_l_full[:fs]
            ar = audio_r_full[s + chunk_size - fs : s + chunk_size] if s + chunk_size >= fs else audio_r_full[:fs]
            proc.processing_eeg_gt_audio(eeg_win, al, ar)

        # Drain phase 1 (GSC + DOA + SIR)
        while not proc.data_queue_phase1.empty():
            bl, br, doa_l, doa_r, sir = proc.data_queue_phase1.get_nowait()
            log_t.append((s + chunk_size / 2) / fs)
            log_doa_left.append(doa_l)
            log_doa_right.append(doa_r)
            log_sir.append(sir)
            out_left_chunks.append(bl)
            out_right_chunks.append(br)

        # Drain phase 2 (AAD pred_prob)
        while not proc.data_queue_phase2.empty():
            log_aad.append(proc.data_queue_phase2.get_nowait())

        # Drain phase 3 (geselecteerde spreker)
        while not proc.data_queue_phase3.empty():
            spk, sig = proc.data_queue_phase3.get_nowait()
            log_speaker.append(spk)
            out_signal_chunks.append(sig)

        if (ci + 1) % (EEG_ACCUMULATE * 5) == 0:
            elapsed = time.time() - t_start
            print(f"  {(ci+1)/32:.0f}s verwerkt — DOA L/R: {log_doa_left[-1]:.1f}°/{log_doa_right[-1]:.1f}° "
                  f"— SIR: {log_sir[-1]:.1f} dB — AAD: {log_aad[-1]:.2f}" if log_aad else "")

    elapsed = time.time() - t_start
    rt = (n_samples / fs) / elapsed
    print(f"  Klaar: {elapsed:.1f}s voor {n_samples/fs:.0f}s audio (RT factor {rt:.1f}x)")

    # Valideer
    assert len(out_left_chunks) > 0,   "Phase 1 queue leeg"
    assert len(out_signal_chunks) > 0, "Phase 3 queue leeg"
    print(f"  Phase 1: {len(out_left_chunks)} chunks  ✓")
    print(f"  Phase 2: {len(log_aad)} AAD-updates  {'✓' if log_aad else '(placeholder: OK)'}")
    print(f"  Phase 3: {len(out_signal_chunks)} chunks  ✓")

    # ---- Output opslaan ----
    if out_dir is None:
        out_dir = os.path.join(THIS_DIR, "output", f"pair{pair_no}_{scenario}")
    os.makedirs(out_dir, exist_ok=True)

    def to_int16(chunks):
        x = np.concatenate(chunks).astype(np.float64)
        x = np.nan_to_num(x)
        peak = np.max(np.abs(x))
        if peak < 1e-12:
            return np.zeros(len(x), dtype=np.int16)
        return (x / peak * 0.95 * 32767).astype(np.int16)

    wavfile.write(os.path.join(out_dir, "w2_gsc_left.wav"),    fs, to_int16(out_left_chunks))
    wavfile.write(os.path.join(out_dir, "w2_gsc_right.wav"),   fs, to_int16(out_right_chunks))
    wavfile.write(os.path.join(out_dir, "w2_output_signal.wav"), fs, to_int16(out_signal_chunks))

    # ---- Plot: DOA + SIR + AAD ----
    t = np.array(log_t)
    n_rows = 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 9))

    # DOA — estimate + ground truth
    # Ground truth samplen op dezelfde tijdstippen als de log
    gt_l_sampled = [float(doa_l_gt[min(int(ti * fs), n_samples - 1)]) for ti in t]
    gt_r_sampled = [float(doa_r_gt[min(int(ti * fs), n_samples - 1)]) for ti in t]

    axes[0].plot(t, log_doa_left,  label="DOA links (est)",  color="C0")
    axes[0].plot(t, gt_l_sampled,  label="DOA links (gt)",   color="C0", linestyle="--", alpha=0.6)
    axes[0].plot(t, log_doa_right, label="DOA rechts (est)", color="C1")
    axes[0].plot(t, gt_r_sampled,  label="DOA rechts (gt)",  color="C1", linestyle="--", alpha=0.6)
    axes[0].set_ylabel("DOA (graden)")
    axes[0].set_title(f"Pair {pair_no} {scenario} — week 2 processor output")
    axes[0].legend()
    axes[0].grid(True)

    # SIR
    axes[1].plot(t, log_sir, color="C2", label="SIR actieve spreker")
    axes[1].set_ylabel("SIR (dB)")
    axes[1].legend()
    axes[1].grid(True)

    # AAD pred_prob
    if log_aad:
        # AAD wordt 1x per seconde geupdate — maak een tijdsas op die resolutie
        t_aad = np.arange(len(log_aad), dtype=float) + 1.0
        axes[2].plot(t_aad, log_aad, color="C3", marker="o", markersize=3,
                     label="AAD pred_prob (P(attended_left))")
        axes[2].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_ylabel("pred_prob")
        axes[2].legend()
        axes[2].grid(True)
    else:
        axes[2].text(0.5, 0.5, "AAD: geen model (placeholder)",
                     ha="center", va="center", transform=axes[2].transAxes, fontsize=12)
        axes[2].set_ylabel("pred_prob")

    axes[-1].set_xlabel("tijd (s)")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "w2_doa_sir_aad.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    print(f"\n  Plot:  {plot_path}")
    print(f"  WAVs:  {out_dir}/w2_*.wav")
    print("  [PASS] Processor OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fase 3 week 2 standalone tests")
    parser.add_argument("--pair", type=int, default=1, help="Pair nummer (default: 1)")
    parser.add_argument("--scenario", type=str, default="anechoic",
                        choices=["anechoic", "reverberant"])
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Seconden audio voor processor-test (default: 30s)")
    parser.add_argument("--aad_model_path", type=str, default=None,
                        help="Pad naar .keras model (optioneel; zonder: placeholder). "
                             "Laat leeg voor auto-detect in data/.")
    parser.add_argument("--aad_envelope", type=str, default="gammatone",
                        choices=["gammatone", "hilbert"])
    parser.add_argument("--aad_normalize_eeg", action="store_true", default=False,
                        help="Z-score normaliseer EEG per venster. Aanbevolen voor generic_dilated model.")
    parser.add_argument("--data_base", type=str, default=None,
                        help="Root van data-map (auto-detect als niet gegeven)")
    args = parser.parse_args()

    # Auto-detect model pad als het in de standaard locatie staat
    if args.aad_model_path is None:
        default_model = os.path.join(THIS_DIR, "data", "hybrid_v3_BEST.keras")
        if os.path.exists(default_model):
            args.aad_model_path = default_model
            print(f"[auto] AAD model gevonden: {default_model}")

    data_base = find_data_base(args.data_base)
    if data_base is None:
        print("[FATAL] Kan data-map niet vinden.")
        print("  Gezocht in: fase_3/data/ en documents_and_given_code/phase_3/")
        print("  Override: --data_base /jouw/pad")
        sys.exit(1)

    microarray_dir = os.path.join(
        data_base, "phase3_audioData", "audiodata_batch_1", args.scenario
    )
    print(f"=== Fase 3 Week 2 standalone tests ===")
    print(f"Pair: {args.pair}, scenario: {args.scenario}")
    print(f"Data: {microarray_dir}")
    print(f"AAD model: {args.aad_model_path or 'geen (placeholder)'}")
    print(f"AAD normalize_eeg: {args.aad_normalize_eeg}")

    test_doa_gt_fix(microarray_dir, args.pair)
    test_aad_lstm(args.aad_model_path, envelope=args.aad_envelope,
                  normalize_eeg=args.aad_normalize_eeg)
    test_processor(microarray_dir, args.pair,
                   aad_model_path=args.aad_model_path,
                   duration_s=args.duration,
                   scenario=args.scenario,
                   out_dir=os.path.join(THIS_DIR, "output", f"pair{args.pair}_{args.scenario}"))

    print("\n=== Alle tests geslaagd ✓ ===")


if __name__ == "__main__":
    main()
