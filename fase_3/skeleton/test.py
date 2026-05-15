from server.issp_data import ISSPData

issp = ISSPData(
    microarray_dir="data/phase3_audioData/audiodata_batch_1/anechoic",
    eeg_dir="data/data_phase3",
    gt_audio_dir="data/data_phase3/stimuli",
    num_pairs=15
)

print("Pair 7 subjects:", issp.pair_subject_mapping["pair7"])