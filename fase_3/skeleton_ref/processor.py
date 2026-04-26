import asyncio
import numpy as np


class Processor:
    def __init__(self):
        self.attended_left = 1

        # Output 'pipes'
        self.data_queue_phase1 = asyncio.Queue()
        self.data_queue_phase2 = asyncio.Queue()
        self.data_queue_phase3 = asyncio.Queue()

    def processing_microarray(self, lma):
        # Your processing goes here. You can use instance variables to persist state.

        # Those are some placeholders which are put into the queue, and subsequently indirectly send to the frontend.
        sig0 = np.random.randint(-32768, 32767, size=500, dtype=np.int16) * 0.5
        sig1 = np.random.randint(-32768, 32767, size=500, dtype=np.int16) * 0.5
        angle_0 = 10
        angle_1 = 170
        sir = np.random.random() * 0.6 + 0.2
        self.data_queue_phase1.put_nowait((sig0, sig1, angle_0, angle_1, sir))

        # It makes sense to compile the final output here.
        sig_out = sig0 if self.attended_left else sig1
        speaker = round(np.random.random())
        self.data_queue_phase3.put_nowait((speaker, sig_out))

    def processing_eeg_gt_audio(self, eeg, sig_left_clean, sig_right_clean):
        # Preprocessing, Prediction, etc...
        # Using your own gsc out vs oracle?

        pred_prob = np.random.random() * 0.8 + 0.1
        self.attended_left = round(pred_prob)

        self.data_queue_phase2.put_nowait(pred_prob)
