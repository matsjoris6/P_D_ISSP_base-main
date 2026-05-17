"""
Pre-compute gammatone envelopes voor alle clean stimuli.
Slaat de envelopes op in een npz cache file zodat test scripts snel kunnen draaien.

Run dit script één keer. Daarna gebruikt test_aad_phase3test_fast.py de cache.
"""
import os
import time
import numpy as np
from scipy.io import wavfile
from processor import compute_audio_envelope

# === CONFIG ===
STIMULI_DIR = "data/phase3_test/audio_data/clean_stimuli"
CACHE_FILE  = "data/phase3_test/envelope_cache.npz"

AUDIO_FS = 48000  # input sample rate
AAD_FS   = 64     # output sample rate (matcht model input)

# === SCAN STIMULI ===
wav_files = sorted([f for f in os.listdir(STIMULI_DIR) if f.endswith(".wav")])
print(f"Gevonden {len(wav_files)} wav files in {STIMULI_DIR}\n")

# === BEREKEN ENVELOPES ===
envelopes = {}
durations = {}
t_total_start = time.time()

for i, wav in enumerate(wav_files, 1):
    path = os.path.join(STIMULI_DIR, wav)
    fs, audio = wavfile.read(path)
    
    if fs != AUDIO_FS:
        print(f"  [SKIP] {wav}: sample rate {fs} Hz ≠ {AUDIO_FS} Hz")
        continue
    
    duration_sec = len(audio) / fs
    print(f"[{i}/{len(wav_files)}] {wav} ({duration_sec:.1f}s) ... ", end="", flush=True)
    
    t0 = time.time()
    env = compute_audio_envelope(audio.astype(np.float32), sr_in=AUDIO_FS, sr_out=AAD_FS)
    elapsed = time.time() - t0
    
    # Key zonder .wav extensie
    key = wav.replace(".wav", "")
    envelopes[key] = env.astype(np.float32)
    durations[key] = duration_sec
    
    print(f"env shape={env.shape}, {elapsed:.1f}s")

t_total = time.time() - t_total_start
print(f"\nTotale tijd: {t_total:.1f}s")
print(f"Aantal envelopes berekend: {len(envelopes)}")

# === OPSLAAN ===
print(f"\nOpslaan in {CACHE_FILE} ...")
np.savez_compressed(CACHE_FILE, **envelopes)
print(f"Cache size: {os.path.getsize(CACHE_FILE) / 1024 / 1024:.1f} MB")

# === SAMENVATTING ===
print("\n" + "=" * 60)
print("CACHE KLAAR. Beschikbare keys:")
for key in sorted(envelopes.keys()):
    print(f"  {key}: {envelopes[key].shape[0]} samples ({durations[key]:.1f}s)")
print("=" * 60)
print("\nJe kan nu test_aad_phase3test_fast.py runnen.")