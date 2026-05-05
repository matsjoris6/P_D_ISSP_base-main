"""End-to-end GUI roundtrip validatie voor fase 3 week 1.

Doel: bewijzen dat de complete pipeline (server <-> worker <-> /frontend) werkt
zodat de demo aan de prof zonder verrassingen draait.

Wat dit script doet:
  1. Spawnt skeleton_ref/server/server.py als subprocess
  2. Wacht tot uvicorn websocket up is
  3. Spawnt processing.py als subprocess (worker)
  4. Connect een test-client op /frontend namespace
  5. Verzamelt gsc_data, aad_data, out_data events gedurende --duration sec
  6. Valideert: minimum aantal events + alle vereiste keys aanwezig
  7. Cleanup (kill subprocs)

Bij succes: exit 0 + "GUI pipeline OK"
Bij falen : exit 1 + exact welke key/event ontbreekt
"""
import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time

import socketio


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(THIS_DIR, "skeleton_ref", "server")
DEFAULT_MICROARRAY = "/Users/macbookmats/Desktop/P_D_ISSP_base-main/documents_and_given_code/phase_3/phase3_audioData/audiodata_batch_1/anechoic"
DEFAULT_EEG = "/Users/macbookmats/Desktop/P_D_ISSP_base-main/documents_and_given_code/phase_3/data_phase3"
DEFAULT_STIMULI = "/Users/macbookmats/Desktop/P_D_ISSP_base-main/documents_and_given_code/phase_3/data_phase3/stimuli"


# Verwachte keys per event-type (zie skeleton_ref/server/server.py Emitter)
EXPECTED_KEYS = {
    "gsc_data": {"gsc_left", "gsc_right", "doa_left", "doa_right", "sir", "timestamps", "doa_gt_0", "doa_gt_1"},
    "aad_data": {"pred_prob", "timestamps", "attended_speaker", "accuracy", "avg_accuracy"},
    "out_data": {"predicted_speaker", "output_signal", "timestamps"},
}


async def wait_for_server_ready(log_path, timeout=20):
    """Poll het server-logbestand tot 'Uvicorn running' verschijnt."""
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(log_path):
            with open(log_path) as f:
                if "Uvicorn running" in f.read():
                    return True
        await asyncio.sleep(0.3)
    return False


async def collect_events(duration_sec, server_url="http://localhost:8000"):
    """Connect als frontend-client en verzamel events."""
    sio = socketio.AsyncClient()
    received = {"gsc_data": [], "aad_data": [], "out_data": []}

    @sio.on("gsc_data", namespace="/frontend")
    async def on_gsc(data):
        received["gsc_data"].append(data)

    @sio.on("aad_data", namespace="/frontend")
    async def on_aad(data):
        received["aad_data"].append(data)

    @sio.on("out_data", namespace="/frontend")
    async def on_out(data):
        received["out_data"].append(data)

    await sio.connect(server_url, transports=["websocket"], namespaces=["/frontend"])
    # Skeleton expecteert dit event om frontend te registreren
    await sio.emit("register frontend", namespace="/frontend")

    print(f"[test] Frontend connected, verzamel events {duration_sec}s...")
    await asyncio.sleep(duration_sec)

    await sio.disconnect()
    return received


def validate(received, expected_min_counts):
    """Check: tellingen >= minimums, alle keys aanwezig."""
    errors = []

    for event_name, min_count in expected_min_counts.items():
        events = received[event_name]
        n = len(events)
        if n < min_count:
            errors.append(
                f"  [FAIL] {event_name}: {n} events ontvangen, verwacht >= {min_count}"
            )
            continue
        # Check eerste, midden en laatste event op missing keys
        idxs = [0, n // 2, n - 1]
        for idx in idxs:
            ev = events[idx]
            missing = EXPECTED_KEYS[event_name] - set(ev.keys())
            if missing:
                errors.append(
                    f"  [FAIL] {event_name}[{idx}]: missing keys {missing}"
                )
                break

    return errors


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Hoe lang events verzamelen (sec). Default 30.")
    parser.add_argument("--pair_no", type=int, default=1)
    parser.add_argument("--subject_no", type=int, default=None)
    parser.add_argument("--microarray", type=str, default=DEFAULT_MICROARRAY)
    parser.add_argument("--eeg_data", type=str, default=DEFAULT_EEG)
    parser.add_argument("--stimuli", type=str, default=DEFAULT_STIMULI)
    parser.add_argument("--python", type=str, default="python3.11",
                        help="Python interpreter (server gebruikt cached_property -> python>=3.8)")
    args = parser.parse_args()

    # Validatie van paden voor we iets spawn
    for path, name in [(args.microarray, "microarray"), (args.eeg_data, "eeg_data"), (args.stimuli, "stimuli")]:
        if not os.path.isdir(path):
            print(f"[FATAL] Path bestaat niet: --{name} = {path}")
            sys.exit(2)

    # Drempelwaarden voor validatie. Server emit 32 chunks/sec.
    # gsc_data: ~1 per chunk = 32/sec. We accepteren 90% (vertraging tijdens warmup).
    # aad_data: 1 per WINDOW_SIZE_SECONDS=3 sec. Dus ~duration/3 events.
    # out_data: idem als gsc_data (1 per chunk).
    expected_min = {
        "gsc_data": int(args.duration * 32 * 0.85),
        "aad_data": int(args.duration / 3 * 0.5),  # zachter want grote window
        "out_data": int(args.duration * 32 * 0.85),
    }

    server_log = "/tmp/test_gui_server.log"
    worker_log = "/tmp/test_gui_worker.log"

    # Cleanup oude logs
    for p in [server_log, worker_log]:
        if os.path.exists(p):
            os.remove(p)

    server_proc = None
    worker_proc = None
    try:
        # ---- 1. Spawn server ----
        print(f"[test] Spawning server (log: {server_log})...")
        server_cmd = [
            args.python, "server.py",
            "--microarray_path", args.microarray,
            "--eeg_data_path", args.eeg_data,
            "--stimuli_path", args.stimuli,
            "--num_pairs", "15",
        ]
        with open(server_log, "w") as flog:
            server_proc = subprocess.Popen(
                server_cmd,
                cwd=SERVER_DIR,
                stdout=flog,
                stderr=subprocess.STDOUT,
            )

        if not await wait_for_server_ready(server_log, timeout=30):
            print("[FAIL] Server kwam niet op binnen 30s. Server log:")
            with open(server_log) as f:
                print(f.read())
            sys.exit(1)
        print("[test] Server is up ✓")

        # ---- 2. Spawn worker ----
        print(f"[test] Spawning worker (log: {worker_log})...")
        worker_cmd = [
            args.python, "processing.py",
            "--pair_no", str(args.pair_no),
            "--data_dir", args.microarray,
        ]
        if args.subject_no is not None:
            worker_cmd += ["--subject_no", str(args.subject_no)]
        with open(worker_log, "w") as flog:
            worker_proc = subprocess.Popen(
                worker_cmd,
                cwd=THIS_DIR,
                stdout=flog,
                stderr=subprocess.STDOUT,
            )

        await asyncio.sleep(2.0)  # geef worker tijd om te connecten
        if worker_proc.poll() is not None:
            print(f"[FAIL] Worker crashte voor we kunnen testen. Worker log:")
            with open(worker_log) as f:
                print(f.read())
            sys.exit(1)

        # ---- 3. Verzamel events ----
        received = await collect_events(args.duration)

        # ---- 4. Valideer ----
        print(f"\n[test] Ontvangen events:")
        for k, v in received.items():
            print(f"  {k}: {len(v)} events (min vereist: {expected_min[k]})")

        errors = validate(received, expected_min)
        if errors:
            print("\n[FAIL] Validatie-errors:")
            for e in errors:
                print(e)
            print(f"\nServer log tail (last 30 lines):")
            with open(server_log) as f:
                lines = f.readlines()
                print("".join(lines[-30:]))
            print(f"\nWorker log tail (last 30 lines):")
            with open(worker_log) as f:
                lines = f.readlines()
                print("".join(lines[-30:]))
            sys.exit(1)

        # ---- 5. Toon sample-data van eerste event van elk type ----
        print("\n[test] Voorbeeld eerste event per type:")
        for k, v in received.items():
            ev = v[0]
            preview = {key: (f"<list len={len(val)}>" if isinstance(val, list) else val)
                       for key, val in ev.items()}
            print(f"  {k}: {preview}")

        print("\n=== GUI pipeline OK ===")
        return 0

    finally:
        # Cleanup
        if worker_proc and worker_proc.poll() is None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker_proc.kill()
        if server_proc and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        print("[test] Cleanup voltooid.")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
