#!/usr/bin/env bash
# run_demo.sh -- One-command launcher voor de fase 3 demo (week 1 + 2).
#
# Spawnt:
#   1. server.py (skeleton_ref/server) als background process
#   2. processing.py (deze map) als background process
#   3. Opens browser op http://localhost:8000
#
# Usage:
#   ./run_demo.sh                                    # default: pair 1, anechoic
#   ./run_demo.sh 5 anechoic                         # pair 5, anechoic
#   ./run_demo.sh 1 reverberant                      # pair 1, reverberant
#   AAD_MODEL_PATH=/pad/model.keras ./run_demo.sh    # met AAD LSTM
#
# Env vars:
#   AAD_MODEL_PATH  -- pad naar dilated+LSTM .keras model (optioneel)
#   AAD_WINDOW_S    -- AAD venster in seconden (default 5)
#   AAD_HOP_S       -- AAD hop in seconden (default 1)
#   PYTHON          -- override Python interpreter (default: auto via venv)
#   DATA_BASE       -- override data root (default: autodetect)
#
# Stop met Ctrl-C; alle subprocessen worden netjes opgekuist.

set -e

PAIR_NO="${1:-1}"
SCENARIO="${2:-anechoic}"

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"
SERVER_DIR="$THIS_DIR/skeleton_ref/server"

# ---- Auto-activeer venv (env/ of venv/) ----
if [[ -z "$PYTHON" ]]; then
    if [[ -f "$REPO_ROOT/env/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$REPO_ROOT/env/bin/activate"
        echo "[demo] Venv geactiveerd: $REPO_ROOT/env"
    elif [[ -f "$REPO_ROOT/venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$REPO_ROOT/venv/bin/activate"
        echo "[demo] Venv geactiveerd: $REPO_ROOT/venv"
    fi
    PYTHON="python"
fi

# ---- Auto-detect data-locatie ----
if [[ -z "$DATA_BASE" ]]; then
    # Prioriteit: lokale fase_3/data, dan documents_and_given_code/phase_3
    if [[ -d "$THIS_DIR/data/phase3_audioData" ]]; then
        DATA_BASE="$THIS_DIR/data"
    elif [[ -d "$REPO_ROOT/documents_and_given_code/phase_3/phase3_audioData" ]]; then
        DATA_BASE="$REPO_ROOT/documents_and_given_code/phase_3"
    else
        echo "[FATAL] Kan data-locatie niet vinden. Gezocht in:"
        echo "  - $THIS_DIR/data/phase3_audioData"
        echo "  - $REPO_ROOT/documents_and_given_code/phase_3/phase3_audioData"
        echo "  Override met env: DATA_BASE=/jouw/pad ./run_demo.sh"
        exit 2
    fi
fi

MICROARRAY_DIR="$DATA_BASE/phase3_audioData/audiodata_batch_1/$SCENARIO"
EEG_DIR="$DATA_BASE/data_phase3"
STIMULI_DIR="$DATA_BASE/data_phase3/stimuli"

SERVER_LOG="/tmp/fase3_server.log"
WORKER_LOG="/tmp/fase3_worker.log"

# ---- Path-validatie ----
for p in "$MICROARRAY_DIR" "$EEG_DIR" "$STIMULI_DIR"; do
    if [[ ! -d "$p" ]]; then
        echo "[FATAL] Data-pad bestaat niet: $p"
        echo "       Override met DATA_BASE env-var of pas DATA_BASE in script aan."
        exit 2
    fi
done

# ---- Beschikbare AAD-modellen tonen en selecteren ----
#
# Gebruik:
#   AAD_MODEL_PATH=/volledig/pad/naar/model.keras ./run_demo.sh   # expliciet pad
#   AAD_MODEL_NAME=hybrid ./run_demo.sh                            # zoekt hybrid_*.keras in data/
#   AAD_MODEL_NAME=generic ./run_demo.sh                           # zoekt generic_*.keras in data/
#   (geen env vars) ./run_demo.sh                                  # placeholder (geen TF nodig)
#
# AAD_NORMALIZE_EEG=1  -> z-score normaliseer EEG per venster (aanbevolen voor generic model)
# AAD_ENVELOPE=hilbert -> snellere envelope (default: gammatone)

DATA_MODEL_DIR="$THIS_DIR/data"

# Lijst beschikbare .keras modellen
echo "[demo] Beschikbare AAD-modellen in $DATA_MODEL_DIR:"
AVAILABLE_MODELS=()
while IFS= read -r -d '' f; do
    AVAILABLE_MODELS+=("$f")
    echo "  - $(basename "$f")"
done < <(find "$DATA_MODEL_DIR" -maxdepth 1 -name "*.keras" -print0 2>/dev/null)
if [[ ${#AVAILABLE_MODELS[@]} -eq 0 ]]; then
    echo "  (geen .keras bestanden gevonden)"
fi

# Resolve AAD_MODEL_NAME -> AAD_MODEL_PATH
if [[ -z "$AAD_MODEL_PATH" && -n "$AAD_MODEL_NAME" ]]; then
    for f in "${AVAILABLE_MODELS[@]}"; do
        if [[ "$(basename "$f")" == *"$AAD_MODEL_NAME"* ]]; then
            AAD_MODEL_PATH="$f"
            echo "[demo] AAD_MODEL_NAME='$AAD_MODEL_NAME' → $AAD_MODEL_PATH"
            break
        fi
    done
    if [[ -z "$AAD_MODEL_PATH" ]]; then
        echo "[FATAL] Geen model gevonden met naam '$AAD_MODEL_NAME' in $DATA_MODEL_DIR"
        exit 2
    fi
fi

# Bouw AAD-argumenten
AAD_ARGS=""
if [[ -n "$AAD_MODEL_PATH" ]]; then
    if [[ ! -f "$AAD_MODEL_PATH" ]]; then
        echo "[FATAL] AAD_MODEL_PATH bestaat niet: $AAD_MODEL_PATH"
        exit 2
    fi
    AAD_WINDOW_S="${AAD_WINDOW_S:-5}"
    AAD_HOP_S="${AAD_HOP_S:-1}"
    AAD_ENVELOPE="${AAD_ENVELOPE:-hilbert}"
    AAD_NORMALIZE_EEG="${AAD_NORMALIZE_EEG:-0}"
    AAD_FS_AUDIO="${AAD_FS_AUDIO:-48000}"

    AAD_ARGS="--aad_model_path $AAD_MODEL_PATH --aad_window_s $AAD_WINDOW_S --aad_hop_s $AAD_HOP_S"
    AAD_ARGS="$AAD_ARGS --aad_envelope $AAD_ENVELOPE"
    AAD_ARGS="$AAD_ARGS --aad_fs_audio $AAD_FS_AUDIO"
    if [[ "$AAD_NORMALIZE_EEG" == "1" || "$AAD_NORMALIZE_EEG" == "true" ]]; then
        AAD_ARGS="$AAD_ARGS --aad_normalize_eeg"
    fi
    echo "[demo] AAD model  : $(basename "$AAD_MODEL_PATH")"
    echo "[demo] AAD window : ${AAD_WINDOW_S}s / hop=${AAD_HOP_S}s"
    echo "[demo] AAD envelope: $AAD_ENVELOPE, fs_audio=${AAD_FS_AUDIO}Hz, normalize_eeg=${AAD_NORMALIZE_EEG}"
else
    echo "[demo] Geen AAD model → placeholder modus (geen TF nodig)"
    echo "[demo] Tip: gebruik AAD_MODEL_NAME=hybrid of AAD_MODEL_NAME=generic om een model te selecteren"
fi

# ---- Poort 8000 vrijmaken indien bezet ----
if lsof -i :8000 >/dev/null 2>&1; then
    echo "[demo] Poort 8000 in gebruik, kill bestaand proces..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null || true
    sleep 0.5
fi

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
echo "[demo] DATA_BASE: $DATA_BASE"
echo "[demo] Spawn server (log: $SERVER_LOG)..."
cd "$SERVER_DIR"
"$PYTHON" server.py \
    --microarray_path "$MICROARRAY_DIR" \
    --eeg_data_path "$EEG_DIR" \
    --stimuli_path "$STIMULI_DIR" \
    --num_pairs 15 \
    --aad_window_size "${AAD_HOP_S:-1}" \
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
    $AAD_ARGS \
    > "$WORKER_LOG" 2>&1 &
WORKER_PID=$!

echo "[demo] Server PID=$SERVER_PID, Worker PID=$WORKER_PID"
echo "[demo] Demo loopt. Druk Ctrl-C om te stoppen."
echo

# ---- 5. Monitor: tail beide logs naast elkaar ----
tail -f "$SERVER_LOG" "$WORKER_LOG"
