from scipy.io import wavfile

base = "data/phase3_audioData/audiodata_batch_1/anechoic"

# Microfoon-array signalen
fs_lma, _ = wavfile.read(f"{base}/pair1/mixture_LMA.wav")
fs_gt_l, _ = wavfile.read(f"{base}/pair1/leftSpeaker_LMA.wav")
fs_gt_r, _ = wavfile.read(f"{base}/pair1/rightSpeaker_LMA.wav")

# Stimuli (clean speech)
fs_stim, _ = wavfile.read("data/data_phase3/stimuli/audiobook_1_part2.wav")

print(f"LMA mixture:        {fs_lma} Hz")
print(f"LMA leftSpeaker GT: {fs_gt_l} Hz")
print(f"LMA rightSpeaker GT: {fs_gt_r} Hz")
print(f"Stimuli (clean):    {fs_stim} Hz")
