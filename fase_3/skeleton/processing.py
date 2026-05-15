import argparse
import asyncio
import os
import signal
import socketio
import numpy as np
from collections import deque

# ── Vroeg parsen: --model en --scenario (vóór Processor import) ──────────────
# Beide argumenten moeten config.py overschrijven vóór Processor wordt geïmporteerd,
# want Processor laadt het keras-model en de RIR bij __init__ op basis van config.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--scenario",   type=str,  default=None,
                   choices=["anechoic", "reverberant"])
_pre.add_argument("--model",      type=str,  default=None)
_pre.add_argument("--gsc_audio",     action="store_true", default=False)
_pre.add_argument("--gsc_swap",      action="store_true", default=False,
                  help="DEBUG: swap GSC L/R kanalen voor AAD (test DOA-swap hypothese)")
_pre.add_argument("--no_aad_filter", action="store_true", default=False,
                  help="DEBUG: AAD-filter uitschakelen (raw model output, geen EMA/Schmitt)")
_pre_args, _ = _pre.parse_known_args()

import config as _cfg_mod

if _pre_args.scenario is not None:
    # Overschrijf RIR en microarray-pad op basis van scenario
    _cfg_mod.SCENARIO        = _pre_args.scenario
    _cfg_mod.RIR_PATH        = _cfg_mod._RIR_PATHS[_pre_args.scenario]
    _cfg_mod.MICROARRAY_PATH = _cfg_mod._MICROARRAY_PATHS[_pre_args.scenario]

if _pre_args.model is not None:
    if _pre_args.model not in _cfg_mod.MODELS:
        raise SystemExit(f"[FOUT] Onbekend model '{_pre_args.model}'. "
                         f"Kies uit: {list(_cfg_mod.MODELS.keys())}")
    _m = _cfg_mod.MODELS[_pre_args.model]
    # Overschrijf alle model-afhankelijke config-waarden vóór Processor import
    _cfg_mod.ACTIVE_MODEL       = _pre_args.model
    _cfg_mod.MODEL_PATH         = _m["model_path"]
    _cfg_mod.WINDOW_SEC         = _m["window_sec"]
    _cfg_mod.HOP_SEC            = _m["hop_sec"]
    _cfg_mod.EEG_WINDOW_SAMPLES = _m["eeg_window_samples"]
    _cfg_mod.EMA_ALPHA          = _m["ema_alpha"]
    _cfg_mod.SCHMITT_THRESHOLD  = _m["schmitt_threshold"]
    _cfg_mod.SCHMITT_HYSTERESIS = _m["schmitt_hysteresis"]
    _cfg_mod.AAD_WIN_CHUNKS     = _m["window_sec"] * _cfg_mod.UPDATE_RATE
    _cfg_mod.AAD_HOP_CHUNKS     = _m["hop_sec"]    * _cfg_mod.UPDATE_RATE

# --gsc_audio vlag overschrijft USE_GSC_AUDIO_FOR_AAD in config.py
if _pre_args.gsc_audio:
    _cfg_mod.USE_GSC_AUDIO_FOR_AAD = True

from processor import Processor
from config import WINDOW_SEC, HOP_SEC, AAD_WIN_CHUNKS, AAD_HOP_CHUNKS, ACTIVE_MODEL, SCENARIO, RIR_PATH, MODELS

# ── AAD sliding-window parameters (uit config.py) ────────────────────────────
AAD_HOP_SECONDS = HOP_SEC
UPDATE_RATE     = 32

print(f"[INFO] processing.py gestart met model '{ACTIVE_MODEL}'  |  scenario='{SCENARIO}'")
print(f"[INFO] RIR: {RIR_PATH}")
print(f"[INFO] Venster={WINDOW_SEC}s  Hop={HOP_SEC}s  "
      f"Win-chunks={AAD_WIN_CHUNKS}  Hop-chunks={AAD_HOP_CHUNKS}")

sio            = socketio.AsyncClient()
stop_event     = asyncio.Event()
data_processor = Processor()   # pikt RIR_PATH automatisch op via config
data_processor._aad_hop_seconds = AAD_HOP_SECONDS  # voor eindstatistiek

# Debug-vlaggen vanuit de CLI (--gsc_swap, --no_aad_filter)
data_processor.gsc_swap = _pre_args.gsc_swap
if _pre_args.no_aad_filter:
    data_processor.use_aad_filter = False
    print("[INFO] AAD-filter UIT (raw model output, drempel 0.5)")
if _pre_args.gsc_swap:
    print("[INFO] GSC L/R SWAP actief — gebruikt voor swap-diagnose")

# Rolling buffer: bevat de laatste AAD_WIN_CHUNKS chunks (= WINDOW_SEC aan data).
# Automatisch oudste chunk weggegooid zodra hij vol is.
_aad_buf           = deque(maxlen=AAD_WIN_CHUNKS)
_aad_chunk_counter = 0      # telt nieuwe chunks sinds laatste inferentie
_aad_busy          = False  # voorkomt gelijktijdige inferentie-aanroepen


async def process_phase1(data):
    lma      = np.frombuffer(data["LMA"],      dtype=np.int16).reshape(-1, 5)
    lma_gt0  = np.frombuffer(data["LMA_gt_0"], dtype=np.int16).reshape(-1, 5)
    lma_gt1  = np.frombuffer(data["LMA_gt_1"], dtype=np.int16).reshape(-1, 5)
    data_processor.processing_microarray(lma, lma_gt0, lma_gt1)


async def process_phase2_sliding():
    """
    Sliding-window AAD inferentie.

    Wordt elke AAD_HOP_SECONDS aangeroepen zodra het rolling buffer vol is (5s data).
    Loopt NIET gelijktijdig: als de vorige inferentie nog bezig is, slaan we deze stap over.

    Voordeel t.o.v. het oude blok-venster (elke 5s):
      - (WINDOW_SIZE / HOP) = 5× meer AAD-updates per tijdseenheid.
      - Snellere detectie van aandachtswisselingen.
      - De EMA + Schmitt filter heeft meer input → soepelere beslissing.
      - We gebruiken de beschikbare rekenmarge (marge ≈ 16×) zinvol.
    """
    global _aad_busy
    if _aad_busy:
        return   # vorige inferentie nog bezig — sla dit stap over
    _aad_busy = True
    try:
        # Bouw de 5s-arrays op uit de rolling buffer (snapshot, thread-safe genoeg
        # want we bevinden ons in de hoofd asyncio event loop)
        window_eeg    = b""
        window_audio1 = b""
        window_audio2 = b""
        attended_gt_samples = []
        for chunk in _aad_buf:
            window_eeg    += chunk["eeg"]
            window_audio1 += chunk["audio1"]
            window_audio2 += chunk["audio2"]
            if "attended_speaker" in chunk:
                attended_gt_samples.extend(chunk["attended_speaker"])

        eeg    = np.frombuffer(window_eeg,    dtype=np.float64).reshape(-1, 64)
        audio1 = np.frombuffer(window_audio1, dtype=np.float32)
        audio2 = np.frombuffer(window_audio2, dtype=np.float32)

        # Zware berekening in achtergrond-thread (event loop blijft vrij)
        await asyncio.to_thread(data_processor.processing_eeg_gt_audio, eeg, audio1, audio2)

        # GT-label voor dit venster (majority vote over de 5s)
        if attended_gt_samples:
            window_gt = 1.0 if np.mean(attended_gt_samples) >= 0.5 else 0.0
            data_processor.record_aad_gt(window_gt)

        # Stuur gefilterd besluit + EMA-kans naar frontend
        # prob     = EMA-gefilterde kans (0.0–1.0) → voor UI-plot
        # decision = binaire beslissing (0.0=LEFT, 1.0=RIGHT) → voor accuracy
        data_processor.data_queue_phase2.put_nowait({
            "prob":     data_processor.ema_filtered,
            "decision": 0.0 if data_processor.attended_left else 1.0,
        })
    finally:
        _aad_busy = False


async def send_processed_data_phase1():
    while not stop_event.is_set():
        beam_left, beam_right, doa_left, doa_right, sir = \
            await data_processor.data_queue_phase1.get()
        await sio.emit(
            "phase1_out",
            data={
                "gsc_left":  beam_left.tolist(),
                "gsc_right": beam_right.tolist(),
                "doa_left":  doa_left,
                "doa_right": doa_right,
                "sir":       sir,
            },
            namespace="/worker",
        )


async def send_processed_data_phase2():
    while not stop_event.is_set():
        aad_out = await data_processor.data_queue_phase2.get()
        await sio.emit(
            "phase2_out",
            data={
                "pred_prob": aad_out["decision"],  # binair (0/1) → accuracy server
                "prob":      aad_out["prob"],       # EMA-kans (0.0–1.0) → UI-plot
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
                "output_signal":     output_signal.tolist(),
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
    global _aad_chunk_counter

    # Fase 1: directe microarray-verwerking (DOA + GSC)
    await process_phase1(data)

    # Voeg chunk toe aan rolling buffer
    _aad_buf.append(data)
    _aad_chunk_counter += 1

    # Fase 2: sliding-window AAD — elke AAD_HOP_CHUNKS nieuwe chunks,
    # zodra het buffer volledig gevuld is (eerste 5s aanlooptijd)
    if len(_aad_buf) == AAD_WIN_CHUNKS and _aad_chunk_counter >= AAD_HOP_CHUNKS:
        _aad_chunk_counter = 0
        await process_phase2_sliding()


@sio.on("end_data", namespace="/worker")
async def on_end_data(data):
    print("\n[INFO] Server heeft alle data verstuurd — run afgerond.")
    data_processor.print_statistics()
    stop_event.set()


async def main(pair_no, subject_no):
    # Ctrl+C → statistieken printen en netjes afsluiten
    loop = asyncio.get_running_loop()
    def _on_sigint():
        print("\n[INFO] Onderbroken door gebruiker (Ctrl+C).")
        data_processor.print_statistics()
        stop_event.set()
    loop.add_signal_handler(signal.SIGINT, _on_sigint)

    # DOA ground truth laden — scenario-first, daarna fallback naar het andere
    _gt_first   = SCENARIO
    _gt_second  = "anechoic" if SCENARIO == "reverberant" else "reverberant"
    for _gt_path in [
        os.path.join("data", "phase3_audioData", "audiodata_batch_1",
                     _gt_first,  f"pair{pair_no}", "gt.npz"),
        os.path.join("data", "phase3_audioData", "audiodata_batch_1",
                     _gt_second, f"pair{pair_no}", "gt.npz"),
    ]:
        if os.path.exists(_gt_path):
            data_processor.set_doa_gt_raw(_gt_path)
            break

    await sio.connect("http://localhost:8000", transports=["websocket"],
                      namespaces=["/worker"])
    await sio.emit("get_data",
                   data={"pair_no": pair_no, "subject_no": subject_no},
                   namespace="/worker")

    await asyncio.gather(
        send_processed_data_phase1(),
        send_processed_data_phase2(),
        send_processed_data_phase3(),
    )

    await sio.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_no",    type=int, default=1)
    parser.add_argument("--subject_no", type=int, default=None)
    parser.add_argument("--model",      type=str, required=True,
                        choices=list(MODELS.keys()),
                        help=("Modelsleutel uit config.py. "
                              "Kies uit: " + ", ".join(MODELS.keys())))
    parser.add_argument("--scenario",   type=str, required=True,
                        choices=["anechoic", "reverberant"],
                        help="Akoestisch scenario: anechoic of reverberant")
    parser.add_argument("--gsc_audio",      action="store_true", default=False,
                        help=("Gebruik GSC-output van de beamformer (16 kHz) als audio-invoer "
                              "voor het AAD-model in plaats van clean speech (48 kHz). "
                              "Zonder vlag: clean speech (standaard)."))
    parser.add_argument("--gsc_swap",       action="store_true", default=False,
                        help="DEBUG: swap GSC L/R kanalen voor AAD (test DOA-swap hypothese)")
    parser.add_argument("--no_aad_filter",  action="store_true", default=False,
                        help="DEBUG: AAD-filter uitschakelen (raw model output, geen EMA/Schmitt)")
    args = parser.parse_args()
    asyncio.run(main(args.pair_no, args.subject_no))
