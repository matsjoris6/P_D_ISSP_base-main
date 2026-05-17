"""
Batch AAD test over alle subjects in phase3_test.
Detecteert automatisch het pair op basis van stimulus filenames in EEG npz.
Gebruikt de officiële leftright_mapping uit issp_data.py (pairs 16-30).
"""
import os
import csv
import time
import numpy as np
import tensorflow as tf
from collections import defaultdict

from processor import preprocess_eeg

# === CONFIG ===
EEG_BASE       = "data/phase3_test/eeg_data"
CACHE_FILE     = "data/phase3_test/envelope_cache.npz"
MODEL_PATH     = "models/generic_dilated_alle_proefpersonen_beste_pieter_3laag_5sec_VERVOLG.keras"
OUTPUT_CSV     = "aad_results_phase3test.csv"

WINDOW_SECONDS = 5
HOP_SECONDS    = 1
EEG_FS         = 128
AAD_FS         = 64

# Officiële leftright_mapping voor pairs 16-30 (uit issp_data.py)
LEFTRIGHT_MAPPING = {
    16: ("audiobook_2_2_part3.wav",  "audiobook_1_part3.wav"),
    17: ("podcast_4_part3.wav",      "podcast_3_part3.wav"),
    18: ("audiobook_8_1_part3.wav",  "audiobook_8_2_part3.wav"),
    19: ("audiobook_9_1_part3.wav",  "audiobook_9_2_part3.wav"),
    20: ("audiobook_10_1_part3.wav", "audiobook_10_2_part3.wav"),
    21: ("audiobook_11_1_part3.wav", "audiobook_11_2_part3.wav"),
    22: ("podcast_22_part3.wav",     "podcast_21_part3.wav"),
    23: ("podcast_24_part3.wav",     "podcast_25_part3.wav"),
    24: ("podcast_30_part3.wav",     "podcast_31_part3.wav"),
    25: ("audiobook_14_2_part3.wav", "podcast_32_part3.wav"),
    26: ("audiobook_14_1_part3.wav", "podcast_33_part3.wav"),
    27: ("audiobook_1_part3.wav",    "podcast_34_part3.wav"),
    28: ("podcast_35_part3.wav",     "audiobook_14_2_part3.wav"),
    29: ("audiobook_14_1_part3.wav", "podcast_36_part3.wav"),
    30: ("podcast_37_part3.wav",     "audiobook_1_part3.wav"),
}

# Reverse lookup: gesorteerde tuple → pair_no
PAIR_LOOKUP = {tuple(sorted([l, r])): p for p, (l, r) in LEFTRIGHT_MAPPING.items()}

# Subjects om te skippen (mixed pairs etc.)
SKIP_SUBJECTS = {"sub-081"}

# === LADEN ===
print(f"Laden model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Laden envelope cache: {CACHE_FILE}")
env_cache = np.load(CACHE_FILE)
print(f"  {len(env_cache.files)} envelopes in cache\n")


def detect_pair(stim_0_name, stim_1_name):
    key = tuple(sorted([stim_0_name, stim_1_name]))
    return PAIR_LOOKUP.get(key, None)


def test_one_subject(subject, eeg_path):
    data = np.load(eeg_path)
    eeg          = data["eeg"]
    attended_raw = data["attended_speaker"].astype(int)
    stim_0_name  = str(data["stimulus_0"])
    stim_1_name  = str(data["stimulus_1"])

    pair_no = detect_pair(stim_0_name, stim_1_name)
    if pair_no is None:
        return {"subject": subject, "status": "SKIP",
                "reason": f"onbekende combinatie: {stim_0_name} + {stim_1_name}"}

    LEFT_FILE, RIGHT_FILE = LEFTRIGHT_MAPPING[pair_no]
    swap = (LEFT_FILE != stim_0_name)
    attended = (1 - attended_raw) if swap else attended_raw.copy()

    eeg_full = preprocess_eeg(eeg, fs_in=EEG_FS, fs_out=AAD_FS)

    left_key  = LEFT_FILE.replace(".wav", "")
    right_key = RIGHT_FILE.replace(".wav", "")
    if left_key not in env_cache.files or right_key not in env_cache.files:
        return {"subject": subject, "status": "SKIP",
                "reason": f"envelope niet in cache: {left_key}/{right_key}"}

    env_left_full  = env_cache[left_key]
    env_right_full = env_cache[right_key]

    window_aad = WINDOW_SECONDS * AAD_FS
    hop_aad    = HOP_SECONDS * AAD_FS
    hop_eeg    = HOP_SECONDS * EEG_FS

    n_max = min(eeg_full.shape[0], env_left_full.shape[0], env_right_full.shape[0])
    n_windows = (n_max - window_aad) // hop_aad + 1

    correct = 0
    total   = 0
    preds, gts = [], []

    ema_alpha = 0.3
    ema_filtered = 0.5

    for w in range(n_windows):
        aad_start = w * hop_aad
        aad_end   = aad_start + window_aad

        eeg_proc  = eeg_full[aad_start:aad_end, :]
        env_left  = env_left_full[aad_start:aad_end]
        env_right = env_right_full[aad_start:aad_end]

        if eeg_proc.shape[0] < window_aad or len(env_left) < window_aad or len(env_right) < window_aad:
            break

        eeg_in  = eeg_proc[np.newaxis, :, :].astype(np.float32)
        env1_in = env_left[np.newaxis, :, np.newaxis].astype(np.float32)
        env2_in = env_right[np.newaxis, :, np.newaxis].astype(np.float32)

        pred       = model([eeg_in, env1_in, env2_in], training=False)
        pred_prob  = 1.0 - float(pred[0, 0])  # FLIP zoals in processor.py
        ema_filtered = (ema_alpha * pred_prob) + ((1.0 - ema_alpha) * ema_filtered)
        pred_label = int(round(ema_filtered))


        eeg_end_orig = aad_end * (EEG_FS // AAD_FS)
        gt_slice = attended[eeg_end_orig - hop_eeg:eeg_end_orig]
        gt_label = int(np.round(np.mean(gt_slice)))

        if pred_label == gt_label:
            correct += 1
        total += 1
        preds.append(pred_label)
        gts.append(gt_label)

    if total == 0:
        return {"subject": subject, "status": "SKIP", "reason": "geen windows"}

    preds_arr = np.array(preds)
    gts_arr   = np.array(gts)
    left_mask  = gts_arr == 1
    right_mask = gts_arr == 0
    acc_l = (preds_arr[left_mask]  == gts_arr[left_mask]).mean()  if left_mask.sum()  > 0 else float("nan")
    acc_r = (preds_arr[right_mask] == gts_arr[right_mask]).mean() if right_mask.sum() > 0 else float("nan")
    switches = int((np.diff(gts_arr) != 0).sum())

    return {
        "subject":   subject,
        "status":    "OK",
        "pair":      pair_no,
        "swap":      swap,
        "n_windows": total,
        "accuracy":  correct / total,
        "acc_left":  acc_l,
        "acc_right": acc_r,
        "n_left":    int(left_mask.sum()),
        "n_right":   int(right_mask.sum()),
        "switches":  switches,
    }


# === LOOP OVER ALLE SUBJECTS ===
subjects = sorted([s for s in os.listdir(EEG_BASE)
                   if s.startswith("sub-") and os.path.isdir(os.path.join(EEG_BASE, s))])
print(f"Gevonden {len(subjects)} subjects, skip: {SKIP_SUBJECTS}\n")

results = []
t_start = time.time()

for i, sub in enumerate(subjects, 1):
    if sub in SKIP_SUBJECTS:
        print(f"[{i:3d}/{len(subjects)}] {sub}: SKIP (uitgesloten)")
        continue

    sub_dir = os.path.join(EEG_BASE, sub)
    npz_files = [f for f in os.listdir(sub_dir) if f.endswith(".npz")]
    if not npz_files:
        print(f"[{i:3d}/{len(subjects)}] {sub}: SKIP (geen npz)")
        continue

    eeg_path = os.path.join(sub_dir, npz_files[0])
    result = test_one_subject(sub, eeg_path)

    if result["status"] == "OK":
        print(f"[{i:3d}/{len(subjects)}] {sub} (pair{result['pair']:2d}): "
              f"{result['accuracy']*100:5.2f}%  "
              f"(L={result['acc_left']*100:5.1f}% n={result['n_left']:3d}, "
              f"R={result['acc_right']*100:5.1f}% n={result['n_right']:3d})")
    else:
        print(f"[{i:3d}/{len(subjects)}] {sub}: SKIP — {result.get('reason', '?')}")

    results.append(result)

elapsed = time.time() - t_start
ok_results = [r for r in results if r["status"] == "OK"]
print(f"\nTotale tijd: {elapsed:.0f}s, {len(ok_results)}/{len(results)} subjects verwerkt")

# === SAMENVATTING ===
if ok_results:
    accs = np.array([r["accuracy"] for r in ok_results])

    print("\n" + "=" * 70)
    print("ALGEMENE SAMENVATTING")
    print("=" * 70)
    print(f"Aantal subjects:              {len(ok_results)}")
    print(f"Gemiddelde accuracy:          {accs.mean()*100:.2f}%")
    print(f"Mediaan:                      {np.median(accs)*100:.2f}%")
    print(f"Standaarddeviatie:            {accs.std()*100:.2f}%")
    print(f"Min / Max:                    {accs.min()*100:.2f}% / {accs.max()*100:.2f}%")
    print(f"\nVerdeling:")
    print(f"  > 80%:  {(accs > 0.80).sum():3d} subjects ({(accs > 0.80).mean()*100:.0f}%)")
    print(f"  60-80%: {((accs >= 0.60) & (accs <= 0.80)).sum():3d} subjects ({((accs >= 0.60) & (accs <= 0.80)).mean()*100:.0f}%)")
    print(f"  40-60%: {((accs >= 0.40) & (accs < 0.60)).sum():3d} subjects ({((accs >= 0.40) & (accs < 0.60)).mean()*100:.0f}%)")
    print(f"  < 40%:  {(accs < 0.40).sum():3d} subjects ({(accs < 0.40).mean()*100:.0f}%)")

    sorted_r = sorted(ok_results, key=lambda r: r["accuracy"])
    print(f"\n5 slechtste:")
    for r in sorted_r[:5]:
        print(f"  {r['subject']} (pair{r['pair']}): {r['accuracy']*100:.2f}%")
    print(f"\n5 beste:")
    for r in sorted_r[-5:]:
        print(f"  {r['subject']} (pair{r['pair']}): {r['accuracy']*100:.2f}%")

    # Per-pair
    print("\n" + "=" * 70)
    print("PER-PAIR SAMENVATTING")
    print("=" * 70)
    per_pair = defaultdict(list)
    for r in ok_results:
        per_pair[r["pair"]].append(r["accuracy"])
    print(f"\n{'Pair':>5} | {'N':>3} | {'Mean':>7} | {'Median':>7} | {'Min':>6} | {'Max':>6}")
    print("-" * 55)
    for pair in sorted(per_pair.keys()):
        p = np.array(per_pair[pair])
        print(f"{pair:>5d} | {len(p):>3d} | {p.mean()*100:>6.2f}% | "
              f"{np.median(p)*100:>6.2f}% | {p.min()*100:>5.1f}% | {p.max()*100:>5.1f}%")

# === CSV EXPORT ===
if results:
    keys = sorted({k for r in results for k in r.keys()})
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResultaten opgeslagen in {OUTPUT_CSV}")