import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from package import load_rirs
from package.utils import create_micsigs_modified




rirs_folder = os.path.join("rirs")
files = [os.path.join(rirs_folder, f) for f in os.listdir(rirs_folder) if f.endswith('.pkl')]
latest_rir = max(files, key=os.path.getmtime)
acoustic_scenario = load_rirs(latest_rir)


speech_files = [
    os.path.join("sound_files", "speech1.wav"),
    os.path.join("sound_files", "speech2.wav")
]
noise_files = [os.path.join("sound_files", "Babble_noise1.wav")] 


duration = 5.0
micsigs, speech_comp, noise_comp = create_micsigs_modified(acoustic_scenario, speech_files, noise_files, duration)


