import numpy as np
import pickle
from scipy.io import wavfile
import matplotlib.pyplot as plt
from processor import Processor

BASE_PATH = "data/phase3_audioData/audiodata_batch_1/anechoic"
PAIR = 5
DURATION_SECONDS = 120   

# Laad data
fs, lma_audio = wavfile.read(f"{BASE_PATH}/pair{PAIR}/mixture_LMA.wav")

gt = np.load(f"{BASE_PATH}/pair{PAIR}/gt.npz")
# Bereken de lengte (in samples) van elk interval, beginnend vanaf sample 0
durations_l = np.diff(np.insert(gt["endSamples_l"], 0, 0))
durations_r = np.diff(np.insert(gt["endSamples_r"], 0, 0))

# Bouw de arrays nu met de juiste duraties
doa_left_gt = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_l"], durations_l)])
doa_right_gt = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_r"], durations_r)])

n_samples = int(DURATION_SECONDS * fs)
lma_audio = lma_audio[:n_samples]

print(f"Verwerken: {DURATION_SECONDS}s audio = {n_samples} samples\n")

proc = Processor()
chunk_size = fs // 32  # zelfde als server (500 samples)

all_t, all_est_l, all_est_r, all_gt_l, all_gt_r = [], [], [], [], []
n_frames = n_samples // chunk_size

import time
t_start = time.time()

for i in range(n_frames):
    chunk = lma_audio[i * chunk_size : (i + 1) * chunk_size, :]
    proc.processing_microarray(chunk)

    while not proc.data_queue_phase1.empty():
        _, _, angle_left, angle_right, _ = proc.data_queue_phase1.get_nowait()
        mid = i * chunk_size + chunk_size // 2
        all_t.append(mid / fs)
        all_est_l.append(angle_left)
        all_est_r.append(angle_right)
        all_gt_l.append(doa_left_gt[mid])
        all_gt_r.append(doa_right_gt[mid])

print(f"Verwerkt in {time.time()-t_start:.1f}s\n")

all_t = np.array(all_t)
all_est_l = np.array(all_est_l)
all_est_r = np.array(all_est_r)
all_gt_l = np.array(all_gt_l)
all_gt_r = np.array(all_gt_r)

mask = all_t > 1.0
err_l = np.abs(all_est_l[mask] - all_gt_l[mask])
err_r = np.abs(all_est_r[mask] - all_gt_r[mask])

print(f"=== DOA Statistieken (na 1s convergentie) ===")
print(f"Links:  gem={np.mean(err_l):5.2f}°, mediaan={np.median(err_l):5.2f}°, max={np.max(err_l):5.2f}°")
print(f"Rechts: gem={np.mean(err_r):5.2f}°, mediaan={np.median(err_r):5.2f}°, max={np.max(err_r):5.2f}°")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
ax1.plot(all_t, all_gt_l, 'g-', label='gt links', linewidth=2)
ax1.plot(all_t, all_est_l, 'b-', label='est links', alpha=0.7)
ax1.set_ylabel('DOA (°)'); ax1.legend(); ax1.grid()
ax1.set_title(f'Links — gem fout: {np.mean(err_l):.1f}°')

ax2.plot(all_t, all_gt_r, 'g-', label='gt rechts', linewidth=2)
ax2.plot(all_t, all_est_r, 'b-', label='est rechts', alpha=0.7)
ax2.set_ylabel('DOA (°)'); ax2.legend(); ax2.grid()
ax2.set_title(f'Rechts — gem fout: {np.mean(err_r):.1f}°')

ax3.plot(all_t, np.abs(all_est_l - all_gt_l), 'r-', label='fout links')
ax3.plot(all_t, np.abs(all_est_r - all_gt_r), 'b-', label='fout rechts')
ax3.set_xlabel('Tijd (s)'); ax3.set_ylabel('|fout| (°)'); ax3.legend(); ax3.grid()
ax3.set_ylim(0, 60)

plt.tight_layout()
plt.savefig(f'doa_fast_pair{PAIR}.png', dpi=100)
plt.show()