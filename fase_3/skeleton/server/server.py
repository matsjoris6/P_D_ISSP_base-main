import time
import datetime
import argparse

import socketio
import uvicorn

from issp_data import ISSPData

X_SCALE_FS = 48000


class Emitter:
    def __init__(self, update_rate, eeg_fs, aad_window_size):
        self.update_rate = update_rate
        self.eeg_fs = eeg_fs
        self.aad_window_size = aad_window_size

        self.phase1_tick = 0
        #self.phase2_tick = aad_window_size-1
        self.phase3_tick = 0

        self.doa_gt = None
        self.attended_speaker = []

        self.correct_total = 0
        self.total = 0

        #Evaluatie-accumulatoren (eindscore)
        self.doa_err_sum = 0.0      # som van absolute DOA-fouten (beide kanten samen)
        self.doa_err_count = 0      # aantal DOA-foutmetingen
        self.sir_60s_values = []    # SIR-waarden binnen de eerste 60 seconden

    async def emit_data(self, issp_data, sio, pair_no, subject_no):
        print(datetime.datetime.now().isoformat(), "Emitting data...")
        time_start = time.time()

        issp_data.load_pair(pair_no, subject_no)
        self.doa_gt = issp_data.get_doa_gt(pair_no)

        for i, (chunk, gt) in enumerate(issp_data.generate_chunks(pair_no, subject_no)):
            await sio.emit("data_event", data=chunk, namespace="/worker")
            self.attended_speaker.extend(gt["attended_speaker"])

            await sio.sleep(1 / (self.update_rate + 1))

            if (i + 1) % (22 * self.update_rate) == 0:
                print(datetime.datetime.now().isoformat(), f"Emitted {i+1} chunks in", time.time() - time_start)

        await sio.emit("end_data", datetime.datetime.now().isoformat(), namespace="/worker")
        print(datetime.datetime.now().isoformat(), "End of data")
        self.print_final_score()

    async def handle_intermediate_result1(self, sio, data):
        num_mini_ticks = X_SCALE_FS / self.update_rate
        num_labels = len(data["gsc_left"])
        data["timestamps"] = [(self.phase1_tick * num_mini_ticks) + (i * num_mini_ticks / num_labels) for i in range(num_labels)]
        data["doa_gt_0"] = self.doa_gt[0][(self.phase1_tick + 1) * num_labels]
        data["doa_gt_1"] = self.doa_gt[1][(self.phase1_tick + 1) * num_labels]

        # Evaluatie: DOA-fout accumuleren (absolute fout per kant)
        err_left = abs(data["doa_left"] - data["doa_gt_0"])
        err_right = abs(data["doa_right"] - data["doa_gt_1"])
        self.doa_err_sum += err_left + err_right
        self.doa_err_count += 2

        # Evaluatie: SIR binnen de eerste 60 seconden verzamelen
        current_time_sec = self.phase1_tick / self.update_rate
        sir_val = data.get("sir", 0.0)
        if current_time_sec < 60 and sir_val is not None and sir_val != 0.0:
            self.sir_60s_values.append(sir_val)


        self.phase1_tick += 1

        await sio.emit("gsc_data", data, namespace="/frontend")

    async def handle_intermediate_result2(self, sio, data):
        HOP_SIZE = 4 

        # DE ABSOLUTE MASTER KLOK: We kijken hoe ver de audio (phase1) al is!
        current_time_sec = self.phase1_tick / self.update_rate
        start_time_sec = current_time_sec - HOP_SIZE

        # Beveiliging voor de allereerste window
        if start_time_sec < 0:
            start_time_sec = 0

        num_mini_ticks = X_SCALE_FS * HOP_SIZE 
        num_labels = self.eeg_fs * HOP_SIZE    

        # 1. Teken de AAD-grafiek EXACT op de huidige audiotijd!
        base_timestamp = start_time_sec * X_SCALE_FS
        data["timestamps"] = [base_timestamp + (i * num_mini_ticks / num_labels) for i in range(num_labels)]

        # 2. Haal de bijbehorende Ground Truth op basis van absolute tijd 
        # (We gooien de lijst niet meer leeg met 'del', we pakken gewoon het juiste stukje)
        start_idx = int(start_time_sec * self.eeg_fs)
        end_idx = int(current_time_sec * self.eeg_fs)
        
        data["attended_speaker"] = self.attended_speaker[start_idx:end_idx]

        # 3. Accuracy berekenen
        num_samples_window = len(data["attended_speaker"])
        correct_samples_window = sum(1 for i in range(num_samples_window) if data["attended_speaker"][i] == round(data["pred_prob"]))
        
        # Voeg alleen toe aan het totaal als we daadwerkelijk data hebben
        if num_samples_window > 0:
            self.total += num_samples_window
            self.correct_total += correct_samples_window
            data["accuracy"] = correct_samples_window / num_samples_window
        else:
            data["accuracy"] = 0

        if self.total > 0:
            data["avg_accuracy"] = self.correct_total / self.total
        else:
            data["avg_accuracy"] = 0

        await sio.emit("aad_data", data, namespace="/frontend")

    async def handle_intermediate_result3(self, sio, data):
        num_mini_ticks = X_SCALE_FS / self.update_rate
        num_labels = len(data["output_signal"])

        data["timestamps"] = [(self.phase3_tick * num_mini_ticks) + (i * num_mini_ticks / num_labels) for i in range(num_labels)]
        self.phase3_tick += 1

        await sio.emit("out_data", data, namespace="/frontend")

    def print_final_score(self):
        print("\n" + "=" * 50)
        print("        FINAL EVALUATION SCORE")
        print("=" * 50)

        #  Gemiddelde DOA-hoekfout
        if self.doa_err_count > 0:
            avg_doa_err = self.doa_err_sum / self.doa_err_count
            print(f"  Average DOA error      : {avg_doa_err:.2f} deg")
        else:
            print("  Average DOA error      : n/a")

        #  Gemiddelde AAD-accuracy
        if self.total > 0:
            avg_acc = self.correct_total / self.total
            print(f"  Average AAD accuracy   : {avg_acc * 100:.1f} %")
        else:
            print("  Average AAD accuracy   : n/a")

        #  SIR over de eerste 60 seconden
        if len(self.sir_60s_values) > 0:
            avg_sir = sum(self.sir_60s_values) / len(self.sir_60s_values)
            print(f"  SIR (first 60s)        : {avg_sir:+.2f} dB")
        else:
            print("  SIR (first 60s)        : n/a")

        print("=" * 50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--microarray_path", type=str, default="./data_anechoic_16kHz", help="Path to microarray data")
    parser.add_argument("--eeg_data_path", type=str, default="./data_test_convolved_1_with_switches", help="Path to subject dirs")
    parser.add_argument("--stimuli_path", type=str, default="./data_test_convolved_1_with_switches/stimuli", help="Path to stimuli")
    parser.add_argument("--num_pairs", type=int, default=15, help="Number of pairs to load")
    parser.add_argument("--update_rate", type=int, default=32, help="Data chunks per second emitted to the processor")
    parser.add_argument("--aad_window_size", type=int, default=5, help="Window length in seconds that is used for AAD processing")
    parser.add_argument("--eeg_fs", type=int, default=128, help="Sampling frequency of the EEG data")
    args = parser.parse_args()

    sio = socketio.AsyncServer(async_mode="asgi")
    app = socketio.ASGIApp(sio, static_files={"/": "index.html", "/static": "static"})

    issp_data = ISSPData(args.microarray_path, args.eeg_data_path, args.stimuli_path, chunk_size_num_eeg_samples=args.eeg_fs // args.update_rate, num_pairs=args.num_pairs)
    data_emitter = Emitter(args.update_rate, args.eeg_fs, args.aad_window_size)

    @sio.on("connect")
    async def handle_connect():
        print("Client connected")

    @sio.on("disconnect request")
    async def handle_disconnect(sid):
        await sio.disconnect(sid)
        print("Client disconnected")

    @sio.on("get_data", namespace="/worker")
    async def handle_get_data(sid, msg):
        print(datetime.datetime.now().isoformat(), "Received get_data request")
        current_pair_no = msg["pair_no"]
        current_subject_no = msg["subject_no"] if "subject_no" in msg else None
        await data_emitter.emit_data(issp_data, sio, current_pair_no, current_subject_no)

    @sio.on("phase1_out", namespace="/worker")
    async def handle_intermediate_result1(sid, data):
        await data_emitter.handle_intermediate_result1(sio, data)

    @sio.on("phase2_out", namespace="/worker")
    async def handle_intermediate_result2(sid, data):
        await data_emitter.handle_intermediate_result2(sio, data)

    @sio.on("phase3_out", namespace="/worker")
    async def handle_intermediate_result3(sid, data):
        await data_emitter.handle_intermediate_result3(sio, data)

    @sio.on("register frontend", namespace="/frontend")
    async def register_frontend(sid):
        pass

    uvicorn.run(app, port=8000)  # , log_level="debug")
