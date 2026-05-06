import time  
import numpy as np
import pickle
from scipy.io import wavfile
from scipy.io.wavfile import write
from processor import Processor

BASE_PATH = "data/phase3_audioData/audiodata_batch_1/anechoic"

with open(f"{BASE_PATH}/params.pkl", "rb") as f:
    params = pickle.load(f)

fs, lma_audio = wavfile.read(f"{BASE_PATH}/pair1/mixture_LMA.wav")
_, lma_gt0 = wavfile.read(f"{BASE_PATH}/pair1/leftSpeaker_LMA.wav")
_, lma_gt1 = wavfile.read(f"{BASE_PATH}/pair1/rightSpeaker_LMA.wav")

gt = np.load(f"{BASE_PATH}/pair1/gt.npz")
durations_l = np.diff(np.insert(gt["endSamples_l"], 0, 0))
durations_r = np.diff(np.insert(gt["endSamples_r"], 0, 0))
doa_left_gt = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_l"], durations_l)])
doa_right_gt = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_r"], durations_r)])

print(f"Sample rate: {fs} Hz, audio shape: {lma_audio.shape}, duur: {lma_audio.shape[0]/fs:.1f}s\n")

proc = Processor()
chunk_size = fs // 32

all_sig0 = []
all_sig1 = []
all_doa_l = []
all_doa_r = []
all_gt_l = []
all_gt_r = []
processing_times = []  # Lijstje voor de timer
backlog_history = []
current_backlog_ms = 0.0

n_frames = 400  
print(f"Verwerken van {n_frames} frames ({n_frames*chunk_size/fs:.1f}s audio)...\n")
#begin aad toevoeging
# --- DUMMY AAD DATA VOOR STRESS TEST ---
# 5 seconden aan data:
# EEG (128 Hz) = 640 samples, 64 kanalen
dummy_eeg = np.random.randn(640, 64) 
# Audio (16000 Hz) = 80000 samples
dummy_audio_L = np.random.randn(80000) 
dummy_audio_R = np.random.randn(80000) 

aad_uitvoeringen = 0
#einde aad toevoeging

for i in range(n_frames):
    chunk = lma_audio[i * chunk_size : (i + 1) * chunk_size, :]
    chunk_gt0 = lma_gt0[i * chunk_size : (i + 1) * chunk_size, :]
    chunk_gt1 = lma_gt1[i * chunk_size : (i + 1) * chunk_size, :]
    
   
    start_time = time.time()
    
    proc.processing_microarray(chunk, chunk_gt0, chunk_gt1)
    #begin aad toevoeging
    if i % 32 == 0:
        proc.processing_eeg_gt_audio(dummy_eeg, dummy_audio_L, dummy_audio_R)
        aad_uitvoeringen += 1
    #einde aad toevoeging
    end_time = time.time()

    processing_times.append((end_time - start_time) * 1000) # In milliseconden

    # --- DE BACKLOG TEST ---
    proc_t_ms = processing_times[-1]
    
    # + (vertraging) als we te traag zijn, - (inhaalslag) als we snel zijn
    current_backlog_ms += (proc_t_ms - 31.25)
    
    # Achterstand kan nooit minder dan 0 zijn (als we te snel zijn, wachten we gewoon)
    current_backlog_ms = max(0.0, current_backlog_ms)
    
    backlog_history.append(current_backlog_ms)

    
    
    while not proc.data_queue_phase1.empty():
        sig0, sig1, angle_left, angle_right, sir = proc.data_queue_phase1.get_nowait()
        all_sig0.append(sig0)
        all_sig1.append(sig1)
        all_doa_l.append(angle_left)
        all_doa_r.append(angle_right)
        
        mid_sample = i * chunk_size + chunk_size // 2
        if mid_sample < len(doa_left_gt):
            all_gt_l.append(doa_left_gt[mid_sample])
            all_gt_r.append(doa_right_gt[mid_sample])

#  TIMING STATISTIEKEN 
gemiddelde_tijd = np.mean(processing_times)
max_tijd = np.max(processing_times)
print(f"\n===== Real-Time Prestaties =====")
print(f"Deadline per blokje:  31.25 ms")
print(f"Jouw gem. proc. tijd: {gemiddelde_tijd:.2f} ms")
print(f"Jouw max. proc. tijd: {max_tijd:.2f} ms")
if max_tijd < 31.25:
    print(" Je algoritme is 100% Real-Time!")
else:
    print(" Let op: Sommige frames missen de real-time deadline.")

print(f"\n===== VAD Statistieken =====")
if proc.vad_evals > 0:
    upd_l_pct = (proc.vad_update_count_left / proc.vad_evals) * 100
    upd_r_pct = (proc.vad_update_count_right / proc.vad_evals) * 100
    print(f"Totaal evaluaties: {proc.vad_evals}")
    print(f"NLMS update LEFT (target stil):  {proc.vad_update_count_left}/{proc.vad_evals} ({upd_l_pct:.1f}%)")
    print(f"NLMS update RIGHT (target stil): {proc.vad_update_count_right}/{proc.vad_evals} ({upd_r_pct:.1f}%)")

# --- DOA Statistieken ---
all_doa_l = np.array(all_doa_l)
all_doa_r = np.array(all_doa_r)
all_gt_l = np.array(all_gt_l)
all_gt_r = np.array(all_gt_r)

print(f"\n===== DOA Statistieken (laatste helft) =====")
half = len(all_doa_l) // 2
if half > 0 and half < len(all_gt_l):
    print(f"Gem. fout links:  {np.mean(np.abs(all_doa_l[half:] - all_gt_l[half:])):.2f}°")
    print(f"Gem. fout rechts: {np.mean(np.abs(all_doa_r[half:] - all_gt_r[half:])):.2f}°")

print(f"\n===== Real-Time Buffer / Backlog Test =====")
max_backlog = np.max(backlog_history)
eind_backlog = backlog_history[-1]
print(f"Grootste achterstand ooit gemeten: {max_backlog:.2f} ms")
print(f"Achterstand aan het eind (Frame {n_frames}): {eind_backlog:.2f} ms")

if eind_backlog == 0.0:
    print(" ingehaald")
else:
    print("niet ingehaald")

# Save
sig0_concat = np.concatenate(all_sig0)
sig1_concat = np.concatenate(all_sig1)
sig0_norm = (sig0_concat / (np.abs(sig0_concat).max() + 1e-9) * 20000).astype(np.int16)
sig1_norm = (sig1_concat / (np.abs(sig1_concat).max() + 1e-9) * 20000).astype(np.int16)
write("test_output_left_beam.wav", fs, sig0_norm)
write("test_output_right_beam.wav", fs, sig1_norm)