import argparse
import asyncio
import socketio
import numpy as np
from collections import deque
from processor import Processor

WINDOW_SIZE_SECONDS = 5  # Adjust your window size.
UPDATE_RATE = 32  # Corresponding to the server's update rate.
HOP_SIZE_SECONDS = 4
WIN_CHUNKS = WINDOW_SIZE_SECONDS * UPDATE_RATE  # 160 chunks voor een vol venster
HOP_CHUNKS = HOP_SIZE_SECONDS * UPDATE_RATE     # 32 chunks wachten voor de volgende update

aad_buffer = deque(maxlen=WIN_CHUNKS)
aad_hop_counter = 0
sio = socketio.AsyncClient()
stop_event = asyncio.Event()



async def process_phase1(data):
    lma = np.frombuffer(data["LMA"], dtype=np.int16).reshape(-1, 5)
    # get all other relevant keys, care about encoding and shapes.
    lma_gt0 = np.frombuffer(data["LMA_gt_0"], dtype=np.int16).reshape(-1, 5) #toegevoegd voor sir
    lma_gt1 = np.frombuffer(data["LMA_gt_1"], dtype=np.int16).reshape(-1, 5) #toegevoegd voor sir

    # HEAD MOUNTED ARRAY
    #lma = np.frombuffer(data["HMA"], dtype=np.int16).reshape(-1, 4)
    #lma_gt0 = np.frombuffer(data["HMA_gt_0"], dtype=np.int16).reshape(-1, 4) #toegevoegd voor sir
    #lma_gt1 = np.frombuffer(data["HMA_gt_1"], dtype=np.int16).reshape(-1, 4) #toegevoegd voor sir
    data_processor.processing_microarray(lma, lma_gt0, lma_gt1) #lma_gt0 en lma_gt1 toegevoegd voor sir


async def process_phase2():
    # Bouw het 5-seconde venster op uit de huidige stand van de rolling buffer
    window = {key: b"" for key in ["eeg", "audio1", "audio2"]}
    for chunk in aad_buffer:
        for key in window:
            window[key] += chunk[key]

    eeg = np.frombuffer(window["eeg"], dtype=np.float64).reshape(-1, 64)
    audio1 = np.frombuffer(window["audio1"], dtype=np.float32)
    audio2 = np.frombuffer(window["audio2"], dtype=np.float32)

    await asyncio.to_thread(data_processor.processing_eeg_gt_audio, eeg, audio1, audio2)
    #data_processor.processing_eeg_gt_audio(eeg, audio1, audio2) #asyncio wait to thread

   # 1. Stuur de zware taak naar de achtergrond en WACHT op het antwoord (return pred_prob)
    #berekende_prob = await asyncio.to_thread(data_processor.processing_eeg_gt_audio, eeg, audio1, audio2)
    
    # 2. Nu we weer veilig in de hoofd-thread zijn, stoppen we het netjes in de wachtrij!
    #data_processor.data_queue_phase2.put_nowait(berekende_prob)

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

        await sio.emit(
            "phase2_out",
            data={
                "pred_prob": pred_prob,
            },
            namespace="/worker",
        )


async def send_processed_data_phase3():
    while not stop_event.is_set():
        predicted_speaker, output_signal = await data_processor.data_queue_phase3.get()

        await sio.emit(
            "phase3_out",
            data={
                "predicted_speaker": predicted_speaker,
                "output_signal": output_signal.tolist(),
            },
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
    global aad_hop_counter
    
    # 1. Microfoons direct verwerken (GSC beamformer blijft real-time)
    await process_phase1(data)

    # 2. AAD Data verzamelen in de rolling buffer
    aad_buffer.append(data)
    aad_hop_counter += 1

    # 3. Bereken AAD als het 5s-venster vol is, én we weer 1 seconde (32 hops) verder zijn
    if len(aad_buffer) == WIN_CHUNKS and aad_hop_counter >= HOP_CHUNKS:
        aad_hop_counter = 0  # Reset de hop teller
        await process_phase2()


@sio.on("end_data", namespace="/worker")
async def on_end_data(data):
    # Request new data or end
    #pass
    data_processor.save_output_audio("output_attended.wav")
    stop_event.set()

async def main(pair_no, subject_no):
    await sio.connect("http://localhost:8000", transports=["websocket"], namespaces=["/worker"])

    # Request data from the server
    await sio.emit("get_data", data={"pair_no": pair_no, "subject_no": subject_no}, namespace="/worker")

    # Send response
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
    parser.add_argument("--rir_path", type=str,
    default="data/phase3_audioData/audiodata_batch_1/anechoic/lma_16kHz.npz")
    args = parser.parse_args()

    data_processor = Processor(rir_path=args.rir_path)
    #HEAD MOUNTED ARRAY
    #data_processor = Processor(rir_path="data/phase3_audioData/audiodata_batch_1/anechoic/hma_16kHz.npz")

    asyncio.run(main(args.pair_no, args.subject_no))