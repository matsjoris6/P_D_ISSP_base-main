import os
import pickle
import datetime
from functools import cached_property

import numpy as np
from scipy.io import wavfile


class ISSPData:
    def __init__(self, microarray_dir, eeg_dir, gt_audio_dir, chunk_size_num_eeg_samples=128, num_pairs=15):
        self.microarray_dir = microarray_dir
        self.eeg_dir = eeg_dir
        self.gt_audio_dir = gt_audio_dir
        self.chunk_size = chunk_size_num_eeg_samples
        self.num_pairs = num_pairs

        self.microarray_signals = {}
        self.eeg_signals = {}
        self.gt_audio_signals = {}

        self.leftright_mapping = {
            "pair1": {"left": "audiobook_1_part2.wav", "right": "audiobook_2_2_part2.wav"},
            "pair2": {"left": "podcast_3_part2.wav", "right": "podcast_4_part2.wav"},
            "pair3": {"left": "audiobook_8_2_part2.wav", "right": "audiobook_8_1_part2.wav"},
            "pair4": {"left": "audiobook_9_1_part2.wav", "right": "audiobook_9_2_part2.wav"},
            "pair5": {"left": "audiobook_10_1_part2.wav", "right": "audiobook_10_2_part2.wav"},
            "pair6": {"left": "audiobook_11_2_part2.wav", "right": "audiobook_11_1_part2.wav"},
            "pair7": {"left": "podcast_22_part2.wav", "right": "podcast_21_part2.wav"},
            "pair8": {"left": "podcast_24_part2.wav", "right": "podcast_25_part2.wav"},
            "pair9": {"left": "podcast_30_part2.wav", "right": "podcast_31_part2.wav"},
            "pair10": {"left": "audiobook_14_2_part2.wav", "right": "podcast_32_part2.wav"},
            "pair11": {"left": "podcast_33_part2.wav", "right": "audiobook_14_1_part2.wav"},
            "pair12": {"left": "audiobook_1_part2.wav", "right": "podcast_34_part2.wav"},
            "pair13": {"left": "audiobook_14_2_part2.wav", "right": "podcast_35_part2.wav"},
            "pair14": {"left": "podcast_36_part2.wav", "right": "audiobook_14_1_part2.wav"},
            "pair15": {"left": "audiobook_1_part2.wav", "right": "podcast_37_part2.wav"},
            "pair16": {"left": "audiobook_2_2_part3.wav", "right": "audiobook_1_part3.wav"},
            "pair17": {"left": "podcast_4_part3.wav", "right": "podcast_3_part3.wav"},
            "pair18": {"left": "audiobook_8_1_part3.wav", "right": "audiobook_8_2_part3.wav"},
            "pair19": {"left": "audiobook_9_1_part3.wav", "right": "audiobook_9_2_part3.wav"},
            "pair20": {"left": "audiobook_10_1_part3.wav", "right": "audiobook_10_2_part3.wav"},
            "pair21": {"left": "audiobook_11_1_part3.wav", "right": "audiobook_11_2_part3.wav"},
            "pair22": {"left": "podcast_22_part3.wav", "right": "podcast_21_part3.wav"},
            "pair23": {"left": "podcast_24_part3.wav", "right": "podcast_25_part3.wav"},
            "pair24": {"left": "podcast_30_part3.wav", "right": "podcast_31_part3.wav"},
            "pair25": {"left": "audiobook_14_2_part3.wav", "right": "podcast_32_part3.wav"},
            "pair26": {"left": "audiobook_14_1_part3.wav", "right": "podcast_33_part3.wav"},
            "pair27": {"left": "audiobook_1_part3.wav", "right": "podcast_34_part3.wav"},
            "pair28": {"left": "podcast_35_part3.wav", "right": "audiobook_14_2_part3.wav"},
            "pair29": {"left": "audiobook_14_1_part3.wav", "right": "podcast_36_part3.wav"},
            "pair30": {"left": "podcast_37_part3.wav", "right": "audiobook_1_part3.wav"},
        }

        # In case you want to know which subjects to select for every pair of audios:
        # print(self.pair_subject_mapping)

    @cached_property
    def pair_subject_mapping(self):
        pair_subject_mapping = {f"pair{i+1}": [] for i in range(self.num_pairs)}
        for root, _, files in os.walk(self.eeg_dir):
            for filename in files:
                if filename.endswith(".npz"):
                    data = np.load(os.path.join(root, filename))
                    stimuli = {str(data["stimulus_0"]), str(data["stimulus_1"])}
                    for pair in [f"pair{i+1}" for i in range(self.num_pairs)]:
                        if stimuli == set(self.leftright_mapping[pair].values()):
                            pair_subject_mapping[pair].append(filename)
        return pair_subject_mapping

    def load_data(self):
        for pair in self.pair_subject_mapping:
            self.load_pair(pair)

    def load_pair(self, pair_no, subject_no=None):
        pair = f"pair{pair_no}"
        self._cache_micro(pair)

        subject = f"sub-{subject_no:03}" if subject_no is not None else None
        for subject_filename in self.pair_subject_mapping[pair]:
            if subject is None or subject in subject_filename:
                self._cache_eeg(subject_filename, self.leftright_mapping[pair]["left"])

    def mic_setup(self, pair_no):
        pair = f"pair{pair_no}"
        with open(os.path.join(self.microarray_dir, pair, "params.pkl"), "rb") as file:
            params = pickle.load(file)
        return params

    def get_doa_gt(self, pair_no):
        pair = f"pair{pair_no}"
        gt = np.load(os.path.join(self.microarray_dir, pair, "gt.npz"))
        doa_0 = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_l"], gt["endSamples_l"])])
        doa_1 = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_r"], gt["endSamples_r"])])
        return doa_0.tolist(), doa_1.tolist()

    def _cache_micro(self, pair):
        if pair not in self.microarray_signals:
            microphone_signals = {}
            microphone_signals["LMA"] = wavfile.read(os.path.join(self.microarray_dir, pair, "mixture_LMA.wav"))
            microphone_signals["LMA_gt_0"] = wavfile.read(os.path.join(self.microarray_dir, pair, "leftSpeaker_LMA.wav"))
            microphone_signals["LMA_gt_1"] = wavfile.read(os.path.join(self.microarray_dir, pair, "rightSpeaker_LMA.wav"))
            microphone_signals["HMA"] = wavfile.read(os.path.join(self.microarray_dir, pair, "mixture_HMA.wav"))
            microphone_signals["HMA_gt_0"] = wavfile.read(os.path.join(self.microarray_dir, pair, "leftSpeaker_HMA.wav"))
            microphone_signals["HMA_gt_1"] = wavfile.read(os.path.join(self.microarray_dir, pair, "rightSpeaker_HMA.wav"))
            self.microarray_signals[pair] = microphone_signals

    def _cache_eeg(self, subject_filename, left_audioname):
        if subject_filename not in self.eeg_signals:
            subject_dirname = subject_filename[:7]
            eeg_data = np.load(os.path.join(self.eeg_dir, subject_dirname, subject_filename))

            stimulus_0 = str(eeg_data["stimulus_0"])
            stimulus_1 = str(eeg_data["stimulus_1"])
            swap = left_audioname != stimulus_0
            attended_speaker = eeg_data["attended_speaker"] if not swap else 1 - eeg_data["attended_speaker"]
            stimulus_left = stimulus_0 if not swap else stimulus_1
            stimulus_right = stimulus_1 if not swap else stimulus_0

            self.eeg_signals[subject_filename] = eeg_data["fs"], eeg_data["eeg"], stimulus_left, stimulus_right, attended_speaker

            self._cache_gt_audio(stimulus_left)
            self._cache_gt_audio(stimulus_right)

    def _cache_gt_audio(self, gt_audio_name):
        if gt_audio_name not in self.gt_audio_signals:
            audio_fs, audio = wavfile.read(os.path.join(self.gt_audio_dir, gt_audio_name))
            self.gt_audio_signals[gt_audio_name] = audio_fs, audio
#chuncks the data and yiels a chuck every time its being asked
    def generate_chunks(self, pair_no, subject_no):
        self.load_pair(pair_no, subject_no)
        pair = f"pair{pair_no}"
        subject = self.pair_subject_mapping[pair][0] if subject_no is None else next(s for s in self.pair_subject_mapping[pair] if f"sub-{subject_no:03}" in s)
        eeg_fs, eeg, audio1_name, audio2_name, attended_speaker = self.eeg_signals[subject]
        audio1_fs, audio1 = self.gt_audio_signals[audio1_name]
        audio2_fs, audio2 = self.gt_audio_signals[audio2_name]
        microphone_signals = self.microarray_signals[pair]

        audio1_chunk_len = audio1_fs * self.chunk_size // eeg_fs
        audio2_chunk_len = audio2_fs * self.chunk_size // eeg_fs
        micro_chunk_len = microphone_signals["LMA"][0] * self.chunk_size // eeg_fs
        for i in range(eeg.shape[0] // self.chunk_size):
            yield (
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "chunk_no": i,
                    "eeg": eeg[i * self.chunk_size : (i + 1) * self.chunk_size].tobytes(),
                    "audio1": audio1[i * audio1_chunk_len : (i + 1) * audio1_chunk_len].tobytes(),
                    "audio2": audio2[i * audio2_chunk_len : (i + 1) * audio2_chunk_len].tobytes(),
                    "LMA": microphone_signals["LMA"][1][i * micro_chunk_len : (i + 1) * micro_chunk_len, :].tobytes(),
                    "HMA": microphone_signals["HMA"][1][i * micro_chunk_len : (i + 1) * micro_chunk_len, :].tobytes(),
                    "LMA_gt_0": microphone_signals["LMA_gt_0"][1][i * micro_chunk_len : (i + 1) * micro_chunk_len, :].tobytes(),
                    "LMA_gt_1": microphone_signals["LMA_gt_1"][1][i * micro_chunk_len : (i + 1) * micro_chunk_len, :].tobytes(),
                    "HMA_gt_0": microphone_signals["HMA_gt_0"][1][i * micro_chunk_len : (i + 1) * micro_chunk_len, :].tobytes(),
                    "HMA_gt_1": microphone_signals["HMA_gt_1"][1][i * micro_chunk_len : (i + 1) * micro_chunk_len, :].tobytes(),
                },
                {
                    "attended_speaker": attended_speaker[i * self.chunk_size : (i + 1) * self.chunk_size].tolist(),
                },
            )
