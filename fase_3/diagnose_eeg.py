# diagnose_eeg.py
import numpy as np
import os

eeg_dir = "fase_3/skeleton/data/data_phase3"


# Loop alle sub-XXX folders door
for sub_folder in sorted(os.listdir(eeg_dir)):
    if not sub_folder.startswith("sub-"):
        continue
    sub_path = os.path.join(eeg_dir, sub_folder)
    if not os.path.isdir(sub_path):
        continue
    for f in os.listdir(sub_path):
        if f.endswith(".npz"):
            data = np.load(os.path.join(sub_path, f))
            print(f"{f}: stim_0={data['stimulus_0']}, stim_1={data['stimulus_1']}")