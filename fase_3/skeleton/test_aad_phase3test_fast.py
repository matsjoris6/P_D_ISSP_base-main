"""
SNELLE versie van test_aad_phase3test.py.
Gebruikt vooraf berekende gammatone envelopes uit envelope_cache.npz.

Run eerst precompute_envelopes.py om de cache aan te maken.

Voordeel: ~12× sneller dan de originele versie (geen brian2 in de hot loop).
Identieke resultaten qua accuracy — alleen de timing verschilt.
"""
import os
import numpy as np
import tensorflow as tf
import time

from processor import preprocess_eeg

# === CONFIG ===
EEG_BASE       = "data/phase3_test/eeg_data"
STIMULI_DIR    = "data/phase3_test/audio_data/clean_stimuli"
CACHE_FILE     = "data/phase3_test/envelope_cache.npz"
MODEL_PATH     = "models/generic_dilated_alle_proefpersonen_beste_pieter_3laag_5sec_VERVOLG.keras"

SUBJECT        = "sub-005"       # Pas aan voor andere subjects
PAIR_NO        = 1

WINDOW_SECONDS = 5
HOP_SECONDS    = 1
EEG_FS         = 128
AAD_FS         = 64

# Pair → (LEFT_FILE, RIGHT_FILE) mapping
LEFTRIGHT_MAPPING = {
    1: ("audiobook_1_part3.wav", "audiobook_2_2_part3.wav"),
}

# === CHECK CACHE ===
if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(
        f"Cache file niet gevonden: {CACHE_FILE}\n"
        f"Run eerst: python precompute_envelopes.py"
    )

# === LAAD MODEL ===
print(f"Laden model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# === LAAD ENVELOPE CACHE ===
print(f"Laden envelope cache: {CACHE_FILE}")
env_cache = np.load(CACHE_FILE)
print(f"  Beschikbare envelopes: {list(env_cache.keys())[:5]}... ({len(env_cache.files)} totaal)")

# === LAAD EEG NPZ ===
eeg_dir = os.path.join(EEG_BASE, SUBJECT)
npz_files = [f for f in os.listdir(eeg_dir) if f.endswith(".npz")]
if not npz_files:
    raise FileNotFoundError(f"Geen npz files in {eeg_dir}")
eeg_path = os.path.join(eeg_dir, npz_files[0])
print(f"Laden EEG: {eeg_path}")

data = np.load(eeg_path)
eeg          = data["eeg"]
attended_raw = data["attended_speaker"].astype(int)
stim_0_name  = str(data["stimulus_0"])
stim_1_name  = str(data["stimulus_1"])

print(f"  EEG shape: {eeg.shape} @ {int(data['fs'])} Hz")
print(f"  stimulus_0: {stim_0_name}")
print(f"  stimulus_1: {stim_1_name}")

# === SWAP-LOGIC ===
LEFT_FILE, RIGHT_FILE = LEFTRIGHT_MAPPING[PAIR_NO]
swap = (LEFT_FILE != stim_0_name)
print(f"Swap = {swap}")

if swap:
    attended = 1 - attended_raw
else:
    attended = attended_raw.copy()
print(f"Na swap: attended = 1 betekent LEFT ({LEFT_FILE})")

# === PRE-COMPUTE EEG voor de hele opname (eenmalig) ===
# Dit is veel sneller dan elke window apart preprocessen
print(f"\nPre-processing volledige EEG (eenmalig)...")
t0 = time.time()
eeg_full = preprocess_eeg(eeg, fs_in=EEG_FS, fs_out=AAD_FS)  # (N_aad, 64)
print(f"  EEG na preprocessing: {eeg_full.shape}, {(time.time()-t0)*1000:.0f} ms")

# === HAAL ENVELOPES UIT CACHE ===
left_key  = LEFT_FILE.replace(".wav", "")
right_key = RIGHT_FILE.replace(".wav", "")

if left_key not in env_cache.files:
    raise KeyError(f"'{left_key}' niet in cache. Aanwezig: {list(env_cache.files)}")
if right_key not in env_cache.files:
    raise KeyError(f"'{right_key}' niet in cache. Aanwezig: {list(env_cache.files)}")

env_left_full  = env_cache[left_key]      # (N_aad,) @ 64 Hz
env_right_full = env_cache[right_key]
print(f"\nLEFT envelope:  {env_left_full.shape}")
print(f"RIGHT envelope: {env_right_full.shape}")

# === SLIDING WINDOW INFERENCE ===
window_aad = WINDOW_SECONDS * AAD_FS    # 5 × 64 = 320 samples
hop_aad    = HOP_SECONDS * AAD_FS       # 1 × 64 = 64 samples
hop_eeg    = HOP_SECONDS * EEG_FS       # voor ground truth slicing

n_max = min(eeg_full.shape[0], env_left_full.shape[0], env_right_full.shape[0])
n_windows = (n_max - window_aad) // hop_aad + 1
print(f"\nAantal sliding windows: {n_windows}")
print(f"Totale duur: {n_max / AAD_FS:.1f}s\n")

correct      = 0
total        = 0
pred_history = []
gt_history   = []
prob_history = []

t_start = time.time()

for w in range(n_windows):
    aad_start = w * hop_aad
    aad_end   = aad_start + window_aad

    # Direct slicen uit pre-computed arrays — geen preprocessing meer
    eeg_proc  = eeg_full[aad_start:aad_end, :]
    env_left  = env_left_full[aad_start:aad_end]
    env_right = env_right_full[aad_start:aad_end]

    # Veiligheidscheck (in geval ronding iets afwijkt)
    if eeg_proc.shape[0] < window_aad or len(env_left) < window_aad or len(env_right) < window_aad:
        break

    # Model input
    eeg_in  = eeg_proc[np.newaxis, :, :].astype(np.float32)
    env1_in = env_left[np.newaxis, :, np.newaxis].astype(np.float32)
    env2_in = env_right[np.newaxis, :, np.newaxis].astype(np.float32)

    pred       = model([eeg_in, env1_in, env2_in], training=False)
    pred_prob  = 1.0 -float(pred[0, 0])
    pred_label = int(round(pred_prob))

    # Ground truth voor deze laatste seconde
    # eeg_end (in originele 128 Hz) = aad_end × 2
    # Ground truth voor de hele 5s window (zoals het model het zag)
    eeg_end_orig   = aad_end * (EEG_FS // AAD_FS)
    eeg_start_orig = eeg_end_orig - (WINDOW_SECONDS * EEG_FS)
    gt_slice = attended[eeg_start_orig:eeg_end_orig]
    gt_label = int(np.round(np.mean(gt_slice)))
    is_correct = (pred_label == gt_label)
    correct   += int(is_correct)
    total     += 1

    pred_history.append(pred_label)
    gt_history.append(gt_label)
    prob_history.append(pred_prob)

    if w % 20 == 0 or w < 5:
        sec = aad_end / AAD_FS
        status = "OK" if is_correct else "FOUT"
        print(f"  Win {w:3d} (t={sec:6.1f}s): pred={pred_label} (prob={pred_prob:.3f}), gt={gt_label}  [{status}]")

elapsed = time.time() - t_start
acc = correct / total if total else 0

# === RESULTATEN ===
print("\n" + "=" * 70)
print("RESULTATEN")
print("=" * 70)
print(f"Subject: {SUBJECT}  Pair: {PAIR_NO}  Swap: {swap}")
print(f"Aantal windows: {total}")
print(f"Correct: {correct}")
print(f"ACCURACY: {acc * 100:.2f}%")
print(f"\nTotale tijd: {elapsed:.1f}s ({elapsed/total*1000:.0f} ms per window)")

pred_arr = np.array(pred_history)
gt_arr   = np.array(gt_history)
left_mask  = gt_arr == 1
right_mask = gt_arr == 0
if left_mask.sum() > 0:
    acc_l = (pred_arr[left_mask] == gt_arr[left_mask]).mean()
    print(f"\nAccuracy als GT=LEFT  (n={left_mask.sum()}): {acc_l*100:.2f}%")
if right_mask.sum() > 0:
    acc_r = (pred_arr[right_mask] == gt_arr[right_mask]).mean()
    print(f"Accuracy als GT=RIGHT (n={right_mask.sum()}): {acc_r*100:.2f}%")

prob_arr = np.array(prob_history)
print(f"\npred_prob stats: min={prob_arr.min():.3f}, max={prob_arr.max():.3f}, mean={prob_arr.mean():.3f}")

switches = np.diff(gt_arr)
n_switches = (switches != 0).sum()
print(f"Aantal GT switches: {n_switches}")

print("\n" + "=" * 70)
if acc < 0.20:
    print(">>> WAARSCHUWING: accuracy heel laag → labels mogelijk geflipt!")
elif acc > 0.80:
    print(">>> Goed resultaat, labels lijken correct te kloppen.")
else:
    print(">>> Matig resultaat — onderzoek nodig.")
print("=" * 70)