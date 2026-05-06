"""Ingevulde versie van het skeleton's processing.py voor fase 3 week 1.

Verschillen met skeleton_ref/processing.py:
- decodeert ook LMA_gt_0 en LMA_gt_1 (per-spreker bijdragen) zodat de processor
  de SIR per frame kan berekenen (Part 3 van fase 3 week 1).
- ondersteunt --data_dir CLI-arg om de Processor te koppelen aan het juiste
  RIR/scenario (anechoic vs reverberant) voor LUT-opbouw.

Het overige asynchrone schema (Socket.IO client, queues, sender-coroutines) is identiek
aan het skeleton -- niet aangepast.
"""
import argparse
import asyncio
import os

import socketio
import numpy as np

# Importeer onze ingevulde Processor
from processor import Processor

# 1s window-accumulatie zodat AADLSTM elke seconde een hop kan doen op zijn
# interne 5s sliding-window. Het model verwacht (640, 64) EEG @ 128Hz = 5s.
# Het AADLSTM-object doet zelf de buffer + window-management.
WINDOW_SIZE_SECONDS = 1
UPDATE_RATE = 32

sio = socketio.AsyncClient()
stop_event = asyncio.Event()

# Processor wordt later in main() geinitialiseerd zodra we args hebben
data_processor: Processor = None

data_queue = asyncio.Queue()


async def process_phase1(data):
    # 5 mics LMA = standaard voor onze data; HMA heeft 4 mics maar gebruiken we
    # niet voor week 1 (de fase 1 GSC werkte op LMA).
    lma = np.frombuffer(data["LMA"], dtype=np.int16).reshape(-1, 5)
    # Per-spreker bijdragen (oracle, voor SIR-berekening Part 3)
    lma_gt_0 = np.frombuffer(data["LMA_gt_0"], dtype=np.int16).reshape(-1, 5) if "LMA_gt_0" in data else None
    lma_gt_1 = np.frombuffer(data["LMA_gt_1"], dtype=np.int16).reshape(-1, 5) if "LMA_gt_1" in data else None

    data_processor.processing_microarray(lma, lma_gt_0, lma_gt_1)


async def process_phase2():
    window = {key: b"" for key in ["eeg", "audio1", "audio2"]}
    for _ in range(WINDOW_SIZE_SECONDS * UPDATE_RATE):
        data = await data_queue.get()
        for key in window:
            window[key] += data[key]

    eeg = np.frombuffer(window["eeg"], dtype=np.float64).reshape(-1, 64)
    audio1 = np.frombuffer(window["audio1"], dtype=np.float32)
    audio2 = np.frombuffer(window["audio2"], dtype=np.float32)

    data_processor.processing_eeg_gt_audio(eeg, audio1, audio2)


async def send_processed_data_phase1():
    while not stop_event.is_set():
        beam_left, beam_right, doa_left, doa_right, sir = await data_processor.data_queue_phase1.get()

        await sio.emit(
            "phase1_out",
            data={
                "gsc_left": beam_left.tolist(),
                "gsc_right": beam_right.tolist(),
                "doa_left": doa_left,
                "doa_right": doa_right,
                "sir": sir,
            },
            namespace="/worker",
        )


async def send_processed_data_phase2():
    while not stop_event.is_set():
        pred_prob = await data_processor.data_queue_phase2.get()
        await sio.emit("phase2_out", data={"pred_prob": pred_prob}, namespace="/worker")


async def send_processed_data_phase3():
    while not stop_event.is_set():
        predicted_speaker, output_signal = await data_processor.data_queue_phase3.get()
        await sio.emit(
            "phase3_out",
            data={"predicted_speaker": predicted_speaker, "output_signal": output_signal.tolist()},
            namespace="/worker",
        )


@sio.on("connect")
async def connect():
    print("Connected to server")


@sio.on("disconnect")
async def disconnect():
    print("Disconnected from server")


@sio.on("data_event", namespace="/worker")
async def on_data(data):
    await data_queue.put(data)
    await process_phase1(data)

    if data_queue.qsize() >= WINDOW_SIZE_SECONDS * UPDATE_RATE:
        await process_phase2()


@sio.on("end_data", namespace="/worker")
async def on_end_data(data):
    pass


async def main(pair_no, subject_no):
    await sio.connect("http://localhost:8000", transports=["websocket"], namespaces=["/worker"])
    await sio.emit("get_data", data={"pair_no": pair_no, "subject_no": subject_no}, namespace="/worker")

    await asyncio.gather(
        send_processed_data_phase1(),
        send_processed_data_phase2(),
        send_processed_data_phase3(),
    )

    await sio.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_no", type=int, default=1)
    parser.add_argument("--subject_no", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Pad naar audio-data scenario (bv .../anechoic). Bepaalt welke RIRs voor LUT.")
    parser.add_argument("--beta", type=float, default=0.92, help="Exp. R_yy averaging voor MUSIC")
    parser.add_argument("--mu", type=float, default=0.001, help="NLMS step voor FD-GSC")
    parser.add_argument("--bin_range", type=str, default="auto",
                        help="MUSIC bin range: 'auto'|'full'|'k_min,k_max'")
    # AAD LSTM (optioneel)
    parser.add_argument("--aad_model_path", type=str, default=None,
                        help="Pad naar dilated+LSTM .keras of .h5 model voor AAD. Als niet gegeven: placeholder.")
    parser.add_argument("--aad_window_s", type=float, default=5.0,
                        help="AAD predictie-venster in seconden (default 5)")
    parser.add_argument("--aad_hop_s", type=float, default=1.0,
                        help="AAD predictie-hop in seconden (default 1)")
    parser.add_argument("--aad_envelope", type=str, default="gammatone",
                        choices=["gammatone", "hilbert"],
                        help="Audio-envelope methode voor AAD")
    parser.add_argument("--aad_normalize_eeg", action="store_true", default=False,
                        help="Z-score normaliseer EEG per venster per kanaal voor AAD. "
                             "Aanbevolen voor modellen zonder interne BatchNorm (bv. generic_dilated).")
    parser.add_argument("--aad_fs_audio", type=int, default=48000,
                        help="Sample rate van de audio-stimuli die de server stuurt (default 48000 Hz). "
                             "Niet de mic-rate! Fase-3 stimuli zijn 48 kHz WAVs.")
    args = parser.parse_args()

    if args.data_dir is not None:
        os.environ["PHASE3_DATA_DIR"] = args.data_dir

    bin_range = args.bin_range
    if "," in bin_range:
        a, b = bin_range.split(",")
        bin_range = (int(a), int(b))

    data_processor = Processor(
        data_dir=args.data_dir,
        beta=args.beta,
        mu=args.mu,
        bin_range=bin_range,
        aad_model_path=args.aad_model_path,
        aad_window_s=args.aad_window_s,
        aad_hop_s=args.aad_hop_s,
        aad_envelope=args.aad_envelope,
        aad_normalize_eeg=args.aad_normalize_eeg,
        aad_fs_audio=args.aad_fs_audio,
    )

    asyncio.run(main(args.pair_no, args.subject_no))
