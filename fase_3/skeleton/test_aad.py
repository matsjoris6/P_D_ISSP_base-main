"""
AAD-only test: meet hoe goed het Phase 2 model presteert op streaming data.
Geen GSC, geen MUSIC — alleen de AAD pipeline (EEG + clean stimuli → pred_prob).

Gebruikt dezelfde label-mapping als issp_data.py (swap-logic), zodat de accuracy
hier exact overeenkomt met wat de browser-visualisatie toont.

Pair-1 mapping (uit issp_data.py):
    LEFT  speaker = audiobook_1_part2.wav
    RIGHT speaker = audiobook_2_2_part2.wav

Label-conventie na swap (consistent met model en server):
    attended_speaker == 1  →  LEFT attended  →  pred_prob hoog (≈1)
    attended_speaker == 0  →  RIGHT attended →  pred_prob laag (≈0)
"""
import time
import numpy as np
import librosa
from processor import Processor

# === CONFIG ===
EEG_FILE = "data/data_phase3/sub-002/sub-002_-_audiobook_2_2.npz"
STIMULI_DIR = "data/data_phase3/stimuli"
WINDOW_SEC = 5
EEG_FS = 128
AUDIO_FS = 48000  # originele sample rate van de stimuli
LEFT_FILE = "audiobook_1_part2.wav"    # pair1 LEFT (uit issp_data.py)
RIGHT_FILE = "audiobook_2_2_part2.wav" # pair1 RIGHT

# === DATA LADEN ===
print(f"Laden EEG: {EEG_FILE}")
data = np.load(EEG_FILE)
eeg = data['eeg']                            # (30720, 64) bij 128 Hz
attended_speaker = data['attended_speaker']  # (30720,) — rauwe labels uit npz
stim0_name = str(data['stimulus_0'])
stim1_name = str(data['stimulus_1'])

duration_sec = eeg.shape[0] / EEG_FS
print(f"Subject:    {data['subject']}")
print(f"SNR:        {data['snr']:.2f}")
print(f"Duur:       {duration_sec:.1f}s")
print(f"stimulus_0: {stim0_name}")
print(f"stimulus_1: {stim1_name}")

# === SWAP-LOGIC (identiek aan issp_data._cache_eeg) ===
# Als stimulus_0 niet de LEFT speaker is, flip de labels
swap = LEFT_FILE != stim0_name
if swap:
    attended_speaker = 1 - attended_speaker
    print(f"\nSwap toegepast: stimulus_0 ({stim0_name}) is RIGHT, labels geflipt.")
else:
    print(f"\nGeen swap: stimulus_0 ({stim0_name}) is al LEFT.")

print(f"Na swap: % LEFT attended: {np.mean(attended_speaker == 1):.2%} | "
      f"% RIGHT attended: {np.mean(attended_speaker == 0):.2%} | "
      f"switches: {int(np.sum(np.diff(attended_speaker) != 0))}")

# === STIMULI LADEN ===
print("\nLaden stimuli...")
audio_LEFT, sr_l = librosa.load(f"{STIMULI_DIR}/{LEFT_FILE}", sr=None, mono=True)
audio_RIGHT, sr_r = librosa.load(f"{STIMULI_DIR}/{RIGHT_FILE}", sr=None, mono=True)
assert sr_l == AUDIO_FS and sr_r == AUDIO_FS, \
    f"Verwachte {AUDIO_FS} Hz maar kreeg {sr_l}/{sr_r} Hz"
print(f"LEFT  audio: {audio_LEFT.shape[0]/sr_l:.1f}s @ {sr_l} Hz")
print(f"RIGHT audio: {audio_RIGHT.shape[0]/sr_r:.1f}s @ {sr_r} Hz")

# === PROCESSOR ===
print("\nInitialiseren Processor (laadt AAD model)...")
proc = Processor()

# === LOOP OVER 5-SEC WINDOWS ===
n_windows = int(duration_sec // WINDOW_SEC)
eeg_window_size = WINDOW_SEC * EEG_FS       # 640 EEG-samples
audio_window_size = WINDOW_SEC * AUDIO_FS   # 240000 audio-samples bij 48 kHz

print(f"\nVerwerken van {n_windows} windows van {WINDOW_SEC}s...\n")

predictions = []
ground_truths = []
inference_times = []

for w in range(n_windows):
    eeg_start = w * eeg_window_size
    eeg_end = eeg_start + eeg_window_size
    audio_start = w * audio_window_size
    audio_end = audio_start + audio_window_size

    if audio_end > len(audio_LEFT) or audio_end > len(audio_RIGHT):
        break

    eeg_window = eeg[eeg_start:eeg_end, :]
    audio_LEFT_window = audio_LEFT[audio_start:audio_end]
    audio_RIGHT_window = audio_RIGHT[audio_start:audio_end]

    # env1 = LEFT, env2 = RIGHT → pred_prob ≈ 1 betekent "LEFT attended"
    t0 = time.time()
    proc.processing_eeg_gt_audio(eeg_window, audio_LEFT_window, audio_RIGHT_window)
    t1 = time.time()
    inference_times.append((t1 - t0) * 1000)

    pred_prob = proc.data_queue_phase2.get_nowait()

    # Ground truth voor dit window: meerderheidsstem over de 640 EEG-samples
    gt_window = attended_speaker[eeg_start:eeg_end]
    gt_label = int(np.round(np.mean(gt_window)))  # 1 = LEFT attended

    predictions.append(pred_prob)
    ground_truths.append(gt_label)

    pred_label = int(round(pred_prob))
    correct = "✓" if pred_label == gt_label else "✗"
    gt_str = "LEFT " if gt_label == 1 else "RIGHT"
    print(f"Window {w:2d} ({w*WINDOW_SEC:3d}-{(w+1)*WINDOW_SEC:3d}s): "
          f"pred_prob={pred_prob:.3f} → {'LEFT ' if pred_label==1 else 'RIGHT'} | "
          f"gt={gt_str} {correct} | ({inference_times[-1]:.0f} ms)")

# === STATISTIEKEN ===
predictions = np.array(predictions)
ground_truths = np.array(ground_truths)
pred_labels = np.round(predictions).astype(int)

n_total = len(predictions)
n_correct = int(np.sum(pred_labels == ground_truths))
accuracy = n_correct / n_total

print(f"\n===== AAD Resultaten =====")
print(f"Aantal windows:          {n_total}")
print(f"Correct geclassificeerd: {n_correct}")
print(f"Accuracy:                {accuracy:.2%}")

mask_left = ground_truths == 1
mask_right = ground_truths == 0
if mask_left.sum() > 0:
    acc_left = np.mean(pred_labels[mask_left] == 1)
    print(f"Accuracy als LEFT attended:  {acc_left:.2%} ({mask_left.sum()} windows)")
if mask_right.sum() > 0:
    acc_right = np.mean(pred_labels[mask_right] == 0)
    print(f"Accuracy als RIGHT attended: {acc_right:.2%} ({mask_right.sum()} windows)")

confidence = np.abs(predictions - 0.5) * 2
print(f"\nGem. confidence:      {np.mean(confidence):.3f}")
print(f"Mediaan confidence:   {np.median(confidence):.3f}")
print(f"Pred_prob range:      [{predictions.min():.3f}, {predictions.max():.3f}], "
      f"mean={predictions.mean():.3f}")

gt_switches = int(np.sum(np.diff(ground_truths) != 0))
print(f"\nGround truth switches (per window): {gt_switches}")

print(f"\n===== Timing =====")
print(f"Gem. inference:       {np.mean(inference_times):.1f} ms")
print(f"Mediaan inference:    {np.median(inference_times):.1f} ms")
print(f"Max inference:        {np.max(inference_times):.1f} ms")
print(f"Real-time deadline:   5000 ms (1 predictie per 5s window)")
print(f"Marge:                factor {5000/np.mean(inference_times):.1f}x sneller dan real-time")