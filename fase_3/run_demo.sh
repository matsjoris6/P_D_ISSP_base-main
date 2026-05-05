#!/usr/bin/env bash
# run_demo.sh -- One-command launcher voor de fase 3 week 1 demo.
#
# Spawnt:
#   1. server.py (skeleton_ref/server) als background process
#   2. processing.py (deze map) als background process
#   3. Opens browser op http://localhost:8000
#
# Usage:
#   ./run_demo.sh                    # default: pair 1, anechoic
#   ./run_demo.sh 5 anechoic         # pair 5, anechoic
#   ./run_demo.sh 1 reverberant      # pair 1, reverberant
#
# Stop met Ctrl-C; alle subprocessen worden netjes opgekuist.

set -e

PAIR_NO="${1:-1}"
SCENARIO="${2:-anechoic}"

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$THIS_DIR/skeleton_ref/server"
PYTHON="${PYTHON:-python3.11}"

# Locaties van de data (aanpassen indien nodig)
DATA_BASE="/Users/macbookmats/Desktop/P_D_ISSP_base-main/documents_and_given_code/phase_3"
MICROARRAY_DIR="$DATA_BASE/phase3_audioData/audiodata_batch_1/$SCENARIO"
EEG_DIR="$DATA_BASE/data_phase3"
STIMULI_DIR="$DATA_BASE/data_phase3/stimuli"

SERVER_LOG="/tmp/fase3_server.log"
WORKER_LOG="/tmp/fase3_worker.log"

# ---- Path-validatie ----
for p in "$MICROARRAY_DIR" "$EEG_DIR" "$STIMULI_DIR"; do
    if [[ ! -d "$p" ]]; then
        echo "[FATAL] Data-pad bestaat niet: $p"
        echo "       Pas DATA_BASE aan in run_demo.sh of zorg dat het pad correct is."
        exit 2
    fi
done

# ---- Cleanup-functie ----
SERVER_PID=""
WORKER_PID=""
cleanup() {
    echo
    echo "[demo] Cleaning up..."
    if [[ -n "$WORKER_PID" ]]; then
        kill "$WORKER_PID" 2>/dev/null || true
    fi
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    sleep 1
    if [[ -n "$WORKER_PID" ]]; then
        kill -9 "$WORKER_PID" 2>/dev/null || true
    fi
    if [[ -n "$SERVER_PID" ]]; then
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    echo "[demo] Klaar."
}
trap cleanup INT TERM EXIT

# ---- 1. Spawn server ----
echo "[demo] Pair $PAIR_NO, scenario $SCENARIO"
echo "[demo] Spawn server (log: $SERVER_LOG)..."
cd "$SERVER_DIR"
"$PYTHON" server.py \
    --microarray_path "$MICROARRAY_DIR" \
    --eeg_data_path "$EEG_DIR" \
    --stimuli_path "$STIMULI_DIR" \
    --num_pairs 15 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# ---- 2. Wacht op server bootup ----
echo "[demo] Wachten op server bootup..."
for i in $(seq 1 60); do
    if grep -q "Uvicorn running" "$SERVER_LOG" 2>/dev/null; then
        echo "[demo] Server is up ✓"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[FATAL] Server crashte. Log:"
        cat "$SERVER_LOG"
        exit 1
    fi
    sleep 0.5
done

# ---- 3. Open browser ----
URL="http://localhost:8000"
echo "[demo] Open browser op $URL ..."
if command -v open >/dev/null 2>&1; then
    open "$URL" || true
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" || true
else
    echo "[demo] Geen browser-launcher gevonden -- open zelf $URL"
fi

# ---- 4. Spawn worker ----
echo "[demo] Spawn worker (log: $WORKER_LOG)..."
cd "$THIS_DIR"
"$PYTHON" processing.py \
    --pair_no "$PAIR_NO" \
    --data_dir "$MICROARRAY_DIR" \
    > "$WORKER_LOG" 2>&1 &
WORKER_PID=$!

echo "[demo] Server PID=$SERVER_PID, Worker PID=$WORKER_PID"
echo "[demo] Demo loopt. Druk Ctrl-C om te stoppen."
echo

# ---- 5. Monitor: tail beide logs naast elkaar ----
tail -f "$SERVER_LOG" "$WORKER_LOG"
