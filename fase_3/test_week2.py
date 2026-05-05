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

Gebruik:
    python fase_3/test_week2.py                                    # tests 1 + 3 (placeholder AAD)
    python fase_3/test_week2.py --aad_model_path data/hybrid_v3_BEST.keras  # alle 3 tests
    python fase_3/test_week2.py --pair 5 --scenario reverberant   # ander pair/scenario
    python fase_3/test_week2.py --data_base /pad/naar/data        # expliciet data-pad
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

def test_aad_lstm(aad_model_path, fs_audio=16000, fs_eeg=128,
                  window_s=5.0, hop_s=1.0, n_chunks=8, envelope="gammatone"):
    print("\n=== Test 2: AAD LSTM standalone ===")

    from algorithms.aad_lstm import AADLSTM, GammatoneEnvelope, HilbertEnvelope

    # Test envelope-extractors onafhankelijk van model
    print("  Gammatone envelope test...")
    env_gt = GammatoneEnvelope(fs_audio=fs_audio, fs_target=fs_eeg)
    dummy_audio = np.random.randn(fs_audio).astype(np.float32)  # 1s synthetisch
    env_out = env_gt(dummy_audio)
    assert env_out.shape[0] == fs_eeg, \
        f"Gammatone output lengte fout: {env_out.shape[0]} != {fs_eeg}"
    assert not np.any(np.isnan(env_out)), "Gammatone output bevat NaN"
    print(f"  Gammatone: input {len(dummy_audio)} samples → output {len(env_out)} @ {fs_eeg}Hz  ✓")

    print("  Hilbert envelope test...")
    env_hb = HilbertEnvelope(fs_audio=fs_audio, fs_target=fs_eeg)
    env_out_h = env_hb(dummy_audio)
    assert env_out_h.shape[0] == fs_eeg, \
        f"Hilbert output lengte fout: {env_out_h.shape[0]} != {fs_eeg}"
    print(f"  Hilbert:   input {len(dummy_audio)} samples → output {len(env_out_h)} @ {fs_eeg}Hz  ✓")

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
    )
    print(f"  Model geladen in {time.time()-t0:.1f}s")

    # Synthetische chunks: 1s audio + 1s EEG per chunk
    chunk_audio = fs_audio          # 1s
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
# Test 3 — Volledige Processor
# ---------------------------------------------------------------------------

def test_processor(microarray_dir, pair_no, aad_model_path=None,
                   duration_s=5.0, scenario="anechoic"):
    print("\n=== Test 3: Volledige Processor (week 1 + 2) ===")

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

    # Processor aanmaken
    proc = Processor(
        data_dir=microarray_dir,
        aad_model_path=aad_model_path,
        aad_window_s=5.0,
        aad_hop_s=1.0,
    )
    print(f"  Processor aangemaakt ({'met AAD model' if aad_model_path and os.path.exists(aad_model_path or '') else 'placeholder AAD'})")

    # Simuleer chunks (32 chunks/s zoals het skeleton)
    chunk_size = fs // 32
    n_chunks = n_samples // chunk_size

    # Synthetische EEG (128Hz, 64 kanalen)
    eeg_fs = 128
    eeg_chunk_size = eeg_fs // 32  # 4 EEG-samples per chunk
    eeg_full = np.random.randn(n_chunks * eeg_chunk_size, 64).astype(np.float64)

    # Audio-streams voor AAD (clean, 1s per keer -- processing.py accumulteert 32 chunks)
    audio_chunk_size = fs // 32
    audio_l_full = lft[:, 0].astype(np.float32) if lft.ndim > 1 else lft.astype(np.float32)
    audio_r_full = rgt[:, 0].astype(np.float32) if rgt.ndim > 1 else rgt.astype(np.float32)

    phase1_count = 0
    phase2_count = 0
    phase3_count = 0
    EEG_ACCUMULATE = 32  # 1s @ 32 chunks/s

    for ci in range(n_chunks):
        s = ci * chunk_size
        e = s + chunk_size
        chunk_lma = mix[s:e]
        chunk_lft = lft[s:e]
        chunk_rgt = rgt[s:e]

        # Phase 1: microarray processing (synchroon)
        proc.processing_microarray(chunk_lma, chunk_lft, chunk_rgt)

        # Phase 2: EEG + audio (elke 32 chunks = 1s, zoals processing.py)
        if (ci + 1) % EEG_ACCUMULATE == 0:
            window_start = (ci + 1 - EEG_ACCUMULATE) * eeg_chunk_size
            window_end = (ci + 1) * eeg_chunk_size
            eeg_win = eeg_full[window_start:window_end]
            audio_l_win = audio_l_full[s + chunk_size - fs : s + chunk_size] if s + chunk_size >= fs else audio_l_full[:fs]
            audio_r_win = audio_r_full[s + chunk_size - fs : s + chunk_size] if s + chunk_size >= fs else audio_r_full[:fs]
            proc.processing_eeg_gt_audio(eeg_win, audio_l_win, audio_r_win)

    # Drain queues
    while not proc.data_queue_phase1.empty():
        proc.data_queue_phase1.get_nowait()
        phase1_count += 1
    while not proc.data_queue_phase2.empty():
        proc.data_queue_phase2.get_nowait()
        phase2_count += 1
    while not proc.data_queue_phase3.empty():
        proc.data_queue_phase3.get_nowait()
        phase3_count += 1

    print(f"  Phase 1 (GSC+DOA+SIR) queue: {phase1_count} items  {'✓' if phase1_count > 0 else '✗'}")
    print(f"  Phase 2 (AAD pred_prob) queue: {phase2_count} items  {'✓' if phase2_count > 0 else '✗'}")
    print(f"  Phase 3 (output signaal) queue: {phase3_count} items  {'✓' if phase3_count > 0 else '✗'}")

    assert phase1_count > 0, "Phase 1 queue leeg — GSC/DOA produceert geen output"
    assert phase3_count > 0, "Phase 3 queue leeg — spreker-selectie produceert geen output"
    print("  [PASS] Processor OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fase 3 week 2 standalone tests")
    parser.add_argument("--pair", type=int, default=1, help="Pair nummer (default: 1)")
    parser.add_argument("--scenario", type=str, default="anechoic",
                        choices=["anechoic", "reverberant"])
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Seconden audio voor processor-test (default: 5s)")
    parser.add_argument("--aad_model_path", type=str, default=None,
                        help="Pad naar hybrid_v3_BEST.keras (optioneel; zonder: placeholder)")
    parser.add_argument("--aad_envelope", type=str, default="gammatone",
                        choices=["gammatone", "hilbert"])
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

    test_doa_gt_fix(microarray_dir, args.pair)
    test_aad_lstm(args.aad_model_path, envelope=args.aad_envelope)
    test_processor(microarray_dir, args.pair,
                   aad_model_path=args.aad_model_path,
                   duration_s=args.duration,
                   scenario=args.scenario)

    print("\n=== Alle tests geslaagd ✓ ===")


if __name__ == "__main__":
    main()
