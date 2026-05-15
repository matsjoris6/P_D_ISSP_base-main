import time
import datetime
import argparse
import os
import sys

import socketio
import uvicorn

from issp_data import ISSPData

# Config staat één map hoger (fase_3/skeleton/config.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import MODELS

X_SCALE_FS = 48000


class Emitter:
    def __init__(self, update_rate, eeg_fs, aad_window_size, aad_hop_size=None, session_config=None):
        self.update_rate = update_rate
        self.eeg_fs = eeg_fs
        self.aad_window_size = aad_window_size
        # aad_hop_size = hoe veel seconden elke phase2_out vertegenwoordigt op de x-as.
        # Bij een sliding window met stap 1s: aad_hop_size=1.
        # Standaard gelijk aan aad_window_size (= oud blok-venster gedrag).
        self.aad_hop_size = aad_hop_size if aad_hop_size is not None else aad_window_size
        # Alle sessie-instellingen om naar de frontend te sturen
        self.session_config = session_config or {}

        self.phase1_tick = 0
        self.phase2_tick = 0
        self.phase3_tick = 0

        self.doa_gt = None
        self.attended_speaker = []

        self.correct_total = 0
        self.total = 0

    async def emit_data(self, issp_data, sio, pair_no, subject_no):
        print(datetime.datetime.now().isoformat(), "Emitting data...")
        time_start = time.time()

        issp_data.load_pair(pair_no, subject_no)
        self.doa_gt = issp_data.get_doa_gt(pair_no)

        for i, (chunk, gt) in enumerate(issp_data.generate_chunks(pair_no, subject_no)):
            chunk["attended_speaker"] = gt["attended_speaker"]
            await sio.emit("data_event", data=chunk, namespace="/worker")
            self.attended_speaker.extend(gt["attended_speaker"])

            await sio.sleep(1 / (self.update_rate + 1))

            if (i + 1) % (22 * self.update_rate) == 0:
                print(datetime.datetime.now().isoformat(), f"Emitted {i+1} chunks in", time.time() - time_start)

        await sio.emit("end_data", datetime.datetime.now().isoformat(), namespace="/worker")
        print(datetime.datetime.now().isoformat(), "End of data")

    async def handle_intermediate_result1(self, sio, data):
        num_mini_ticks = X_SCALE_FS / self.update_rate
        num_labels = len(data["gsc_left"])
        data["timestamps"] = [(self.phase1_tick * num_mini_ticks) + (i * num_mini_ticks / num_labels) for i in range(num_labels)]
        data["doa_gt_0"] = self.doa_gt[0][(self.phase1_tick + 1) * num_labels]
        data["doa_gt_1"] = self.doa_gt[1][(self.phase1_tick + 1) * num_labels]
        self.phase1_tick += 1

        await sio.emit("gsc_data", data, namespace="/frontend")

    async def handle_intermediate_result2(self, sio, data):
        # num_mini_ticks en num_labels zijn gebaseerd op de HOP (niet het venster).
        # Bij sliding window (hop=1s): elke phase2_out schuift de x-as 1s op.
        # Bij blok-venster (hop=window=5s): oud gedrag behouden.
        num_mini_ticks = X_SCALE_FS * self.aad_hop_size
        num_labels = self.eeg_fs * self.aad_hop_size

        # Compenseer voor de aanlooptijd van de rolling buffer:
        # de eerste inferentie arriveert pas na aad_window_size seconden,
        # maar moet op de x-as getoond worden op de juiste positie t.o.v. de andere plots.
        # tick_offset = aantal hop-stappen dat overeenkomt met (window - hop).
        # Voorbeeld: 5s venster, 1s hop → offset = 4 → eerste segment op x=[4s, 5s]. ✓
        # Oud blok-venster (hop=window): offset = 0 → achterwaarts compatibel. ✓
        tick_offset = (self.aad_window_size - self.aad_hop_size) // self.aad_hop_size
        display_tick = self.phase2_tick + tick_offset

        data["timestamps"] = [(display_tick * num_mini_ticks) + (i * num_mini_ticks / num_labels) for i in range(num_labels)]
        self.phase2_tick += 1

        num_samples_window = self.eeg_fs * self.aad_hop_size
        data["attended_speaker"] = self.attended_speaker[:num_samples_window]
        del self.attended_speaker[:num_samples_window]
        num_samples_window = min(num_samples_window, len(data["attended_speaker"]))

        # Accuracy berekenen op de binaire beslissing (0/1), niet op de kans
        correct_samples_window = sum(1 for i in range(num_samples_window) if data["attended_speaker"][i] == round(data["pred_prob"]))
        self.total += num_samples_window
        self.correct_total += correct_samples_window

        data["accuracy"] = correct_samples_window / num_samples_window
        data["avg_accuracy"] = self.correct_total / self.total

        # Vervang pred_prob door de EMA-kans voor de UI-plot (0.0–1.0 i.p.v. 0 of 1)
        # De binaire beslissing is al verwerkt voor accuracy hierboven.
        if "prob" in data:
            data["pred_prob"] = data.pop("prob")

        await sio.emit("aad_data", data, namespace="/frontend")

    async def handle_intermediate_result3(self, sio, data):
        num_mini_ticks = X_SCALE_FS / self.update_rate
        num_labels = len(data["output_signal"])

        data["timestamps"] = [(self.phase3_tick * num_mini_ticks) + (i * num_mini_ticks / num_labels) for i in range(num_labels)]
        self.phase3_tick += 1

        await sio.emit("out_data", data, namespace="/frontend")


if __name__ == "__main__":
    _model_keys = list(MODELS.keys())

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "ISSP fase-3 server\n\n"
            "Snelste manier om te starten:\n"
            "  python server.py --model dilated_5s  --microarray_path ...\n\n"
            "Beschikbare modellen:\n" +
            "".join(f"  {k:<14} {MODELS[k]['description']}\n" for k in _model_keys)
        ),
    )
    parser.add_argument("--model", type=str, required=True, choices=_model_keys,
                        help=("Modelsleutel uit config.py — stelt venster en hop automatisch in. "
                              "Kies uit: " + ", ".join(_model_keys)))
    parser.add_argument("--scenario", type=str, required=True,
                        choices=["anechoic", "reverberant"],
                        help="Akoestisch scenario — wordt getoond in de GUI")
    parser.add_argument("--gsc_audio", action="store_true", default=False,
                        help="AAD gebruikt GSC-beamformer output i.p.v. clean speech — wordt getoond in de GUI")
    parser.add_argument("--microarray_path", type=str, default="./data_anechoic_16kHz",
                        help="Pad naar de microarray-data (anechoic of reverberant map)")
    parser.add_argument("--eeg_data_path",   type=str, default="./data_test_convolved_1_with_switches",
                        help="Pad naar de EEG-data (map met subject-mappen)")
    parser.add_argument("--stimuli_path",    type=str, default="./data_test_convolved_1_with_switches/stimuli",
                        help="Pad naar de stimuli-map")
    parser.add_argument("--num_pairs",       type=int, default=15,
                        help="Aantal pairs om te laden")
    parser.add_argument("--update_rate",     type=int, default=32,
                        help="Data chunks per seconde naar de processor")
    parser.add_argument("--eeg_fs",          type=int, default=128,
                        help="Samplefrequentie van de EEG-data")
    parser.add_argument("--aad_window_size", type=int, default=None,
                        help="Venstergrootte in seconden (overschrijft --model). "
                             "Standaard: bepaald door --model, of 5 als geen model opgegeven.")
    parser.add_argument("--aad_hop_size",    type=int, default=None,
                        help="Hop-grootte in seconden (overschrijft --model). "
                             "Standaard: bepaald door --model, of gelijk aan venster.")
    args = parser.parse_args()

    # ── Model-gebaseerde instellingen ─────────────────────────────────────────
    _m = MODELS[args.model]
    # Expliciete --aad_window_size / --aad_hop_size hebben altijd voorrang op model-defaults
    if args.aad_window_size is None:
        args.aad_window_size = _m["window_sec"]
    if args.aad_hop_size is None:
        args.aad_hop_size = _m["hop_sec"]

    print(f"[INFO] Model      : {args.model}  —  {_m['description']}")
    print(f"[INFO] Scenario   : {args.scenario}")
    print(f"[INFO] Venster    : {args.aad_window_size}s  |  Hop: {args.aad_hop_size}s")
    print(f"[INFO] AAD-audio  : {'GSC-output (16 kHz)' if args.gsc_audio else 'Clean speech (48 kHz)'}")

    sio = socketio.AsyncServer(async_mode="asgi")
    app = socketio.ASGIApp(sio, static_files={"/": "index.html", "/static": "static"})

    issp_data = ISSPData(args.microarray_path, args.eeg_data_path, args.stimuli_path, chunk_size_num_eeg_samples=args.eeg_fs // args.update_rate, num_pairs=args.num_pairs)

    _session_config = {
        "model":       args.model,
        "description": _m["description"],
        "scenario":    args.scenario,
        "window_sec":  args.aad_window_size,
        "hop_sec":     args.aad_hop_size,
        "gsc_audio":   args.gsc_audio,
    }
    data_emitter = Emitter(args.update_rate, args.eeg_fs, args.aad_window_size, args.aad_hop_size,
                           session_config=_session_config)

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
        # Stuur sessie-configuratie naar de frontend zodra die verbindt
        await sio.emit("session_config", data_emitter.session_config, namespace="/frontend", to=sid)

    uvicorn.run(app, port=8000)  # , log_level="debug")
