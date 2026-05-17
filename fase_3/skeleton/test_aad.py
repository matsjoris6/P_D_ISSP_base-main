"""
Test AAD accuracy op phase3_test data met sliding window van 5s en hop 1s.

Verifieert of de labeling klopt voor de test data set.
Verwacht: ~85-92% accuracy als alles correct is, ~8-15% als labels geflipt zijn.
"""
import os
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import time

from processor import compute_audio_envelope, preprocess_eeg

# === CONFIG ===
EEG_BASE       = "data/phase3_test/eeg_data"
STIMULI_DIR    = "data/phase3_test/audio_data/clean_stimuli"
MODEL_PATH     = "models/generic_dilated_alle_proefpersonen_beste_pieter_3laag_5sec_VERVOLG.keras"

SUBJECT        = "sub-002"       # Pas aan voor andere subjects
PAIR_NO        = 1               # Pair 1 = audiobook_1 / audiobook_2_2

WINDOW_SECONDS = 5
HOP_SECONDS    = 1
EEG_FS         = 128
AUDIO_FS       = 48000           # clean stimuli zijn 48 kHz
AAD_FS         = 64              # interne sample rate voor model

# Pair → (LEFT_FILE, RIGHT_FILE) mapping uit issp_data.py
# Pas aan voor andere pairs
LEFTRIGHT_MAPPING = {
    1: ("audiobook_1_part3.wav", "audiobook_2_2_part3.wav"),
}

# === LAAD MODEL ===
print(f"Laden model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# === LAAD EEG NPZ ===
eeg_dir = os.path.join(EEG_BASE, SUBJECT)
npz_files = [f for f in os.listdir(eeg_dir) if f.endswith(".npz")]
if not npz_files:
    raise FileNotFoundError(f"Geen npz files in {eeg_dir}")
eeg_path = os.path.join(eeg_dir, npz_files[0])
print(f"Laden EEG: {eeg_path}")

data = np.load(eeg_path)
eeg          = data["eeg"]                            # (N, 64) @ 128 Hz
attended_raw = data["attended_speaker"].astype(int)   # (N,) met 0/1
stim_0_name  = str(data["stimulus_0"])
stim_1_name  = str(data["stimulus_1"])

print(f"  EEG shape: {eeg.shape} @ {int(data['fs'])} Hz")
print(f"  stimulus_0: {stim_0_name}")
print(f"  stimulus_1: {stim_1_name}")
print(f"  attended_raw unique: {np.unique(attended_raw)}")

# === SWAP-LOGIC ===
LEFT_FILE, RIGHT_FILE = LEFTRIGHT_MAPPING[PAIR_NO]
print(f"\nLEFT  voor pair{PAIR_NO}: {LEFT_FILE}")
print(f"RIGHT voor pair{PAIR_NO}: {RIGHT_FILE}")

# Originele conventie: attended=0 -> stimulus_0, attended=1 -> stimulus_1
# Na swap willen we: attended=1 -> LEFT, attended=0 -> RIGHT (consistent met model: pred_prob≈1 = LEFT)
swap = (LEFT_FILE != stim_0_name)
print(f"Swap = {swap}  (LEFT_FILE matcht{' niet' if swap else ''} stim_0_name)")

if swap:
    attended = 1 - attended_raw   # flip
else:
    attended = attended_raw.copy()
print(f"  Na swap: attended = 1 betekent LEFT ({LEFT_FILE})")
print(f"  Eerste 30 attended labels: {attended[:30]}")

# === LAAD STIMULI ===
fs_left, audio_left = wavfile.read(os.path.join(STIMULI_DIR, LEFT_FILE))
fs_right, audio_right = wavfile.read(os.path.join(STIMULI_DIR, RIGHT_FILE))
print(f"\nLEFT audio:  {audio_left.shape} @ {fs_left} Hz")
print(f"RIGHT audio: {audio_right.shape} @ {fs_right} Hz")
assert fs_left == fs_right == AUDIO_FS, f"Verwacht {AUDIO_FS} Hz, kreeg {fs_left}/{fs_right}"

# === SLIDING WINDOW INFERENCE ===
n_eeg_total      = eeg.shape[0]
eeg_window       = WINDOW_SECONDS * EEG_FS                 # 5 × 128 = 640 EEG samples
eeg_hop          = HOP_SECONDS * EEG_FS                    # 1 × 128 = 128 EEG samples
audio_window     = WINDOW_SECONDS * AUDIO_FS               # 5 × 48000 = 240000 audio samples
audio_hop        = HOP_SECONDS * AUDIO_FS                  # 1 × 48000 = 48000 audio samples
aad_window_samps = WINDOW_SECONDS * AAD_FS                 # 5 × 64 = 320 samples

n_windows = (n_eeg_total - eeg_window) // eeg_hop + 1
print(f"\nAantal sliding windows (5s, hop 1s): {n_windows}")
print(f"Totale duur: {n_eeg_total / EEG_FS:.1f}s\n")

correct          = 0
total            = 0
correct_per_sec  = []
pred_history     = []
gt_history       = []
prob_history     = []

t_start = time.time()

for w in range(n_windows):
    eeg_start = w * eeg_hop
    eeg_end   = eeg_start + eeg_window
    aud_start = w * audio_hop
    aud_end   = aud_start + audio_window

    if aud_end > len(audio_left) or aud_end > len(audio_right):
        break

    # Slice de windows
    eeg_win   = eeg[eeg_start:eeg_end, :]
    audio_l   = audio_left[aud_start:aud_end]
    audio_r   = audio_right[aud_start:aud_end]

    # Preprocessing
    eeg_proc  = preprocess_eeg(eeg_win, fs_in=EEG_FS, fs_out=AAD_FS)
    env_left  = compute_audio_envelope(audio_l.astype(np.float32), sr_in=AUDIO_FS, sr_out=AAD_FS)
    env_right = compute_audio_envelope(audio_r.astype(np.float32), sr_in=AUDIO_FS, sr_out=AAD_FS)

    # Truncate
    eeg_proc  = eeg_proc[:aad_window_samps]
    env_left  = env_left[:aad_window_samps]
    env_right = env_right[:aad_window_samps]

    # Model input
    eeg_in  = eeg_proc[np.newaxis, :, :].astype(np.float32)
    env1_in = env_left[np.newaxis, :, np.newaxis].astype(np.float32)
    env2_in = env_right[np.newaxis, :, np.newaxis].astype(np.float32)

    pred      = model([eeg_in, env1_in, env2_in], training=False)
    pred_prob = float(pred[0, 0])               # GEEN flip, want we hebben swap al toegepast op labels
    pred_label = int(round(pred_prob))          # 1 = LEFT, 0 = RIGHT

    # Ground truth voor deze seconde = attended in [eeg_end-128, eeg_end]
    gt_slice = attended[eeg_end - eeg_hop:eeg_end]
    # Meerderheid stem (in geval er een switch in de seconde zit)
    gt_label = int(np.round(np.mean(gt_slice)))

    is_correct = (pred_label == gt_label)
    correct += int(is_correct)
    total   += 1

    pred_history.append(pred_label)
    gt_history.append(gt_label)
    prob_history.append(pred_prob)

    # Print om de 10 windows
    if w % 10 == 0 or w < 5:
        sec = (eeg_end / EEG_FS)
        status = "OK" if is_correct else "FOUT"
        print(f"  Win {w:3d} (t={sec:6.1f}s): pred={pred_label} (prob={pred_prob:.3f}), gt={gt_label}  [{status}]")

elapsed = time.time() - t_start
acc = correct / total if total else 0

# === RESULTATEN ===
print("\n" + "=" * 70)
print("RESULTATEN")
print("=" * 70)
print(f"Subject: {SUBJECT}  Pair: {PAIR_NO}  Swap: {swap}")
print(f"Aantal windows verwerkt: {total}")
print(f"Correct: {correct}")
print(f"ACCURACY: {acc * 100:.2f}%")
print(f"\nTotale verwerkingstijd: {elapsed:.1f}s ({elapsed/total*1000:.0f} ms per window)")

# Detail per label
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

# Switches
switches = np.diff(gt_arr)
n_switches = (switches != 0).sum()
print(f"Aantal ground truth switches: {n_switches}")

print("\n" + "=" * 70)
if acc < 0.20:
    print(">>> WAARSCHUWING: accuracy heel laag → labels mogelijk geflipt!")
    print(">>> Controleer of de swap-logic en LEFTRIGHT_MAPPING correct zijn.")
elif acc > 0.80:
    print(">>> Resultaat ziet er goed uit, labels lijken correct te kloppen.")
else:
    print(">>> Resultaat is matig — mogelijk model-issue of partial label issue.")
print("=" * 70)