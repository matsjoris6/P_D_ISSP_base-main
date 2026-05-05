# Fase 3

## Vereisten

Zet de data-map in `fase_3/data/` met deze structuur:
```
fase_3/data/
├── phase3_audioData/audiodata_batch_1/anechoic/
├── phase3_audioData/audiodata_batch_1/reverberant/
├── data_phase3/
└── data_phase3/stimuli/
```

Voor AAD: plaats het model `hybrid_v3_BEST.keras` ook in `fase_3/data/`.

---

## GUI draaien

### Zonder AAD (placeholder, geen TensorFlow nodig)
```bash
cd fase_3
source ../env/bin/activate
./run_demo.sh                    # pair 1, anechoic
./run_demo.sh 5 reverberant      # pair 5, reverberant
```

### Met AAD LSTM model (vereist Python 3.11 + TensorFlow)
```bash
cd fase_3
source ../env_tf/bin/activate
PYTHON=$(which python) AAD_MODEL_PATH="$(pwd)/data/hybrid_v3_BEST.keras" ./run_demo.sh
```

De browser opent automatisch op `http://localhost:8000`. Stop met **Ctrl-C**.

---

## Standalone test (geen server nodig)

```bash
cd fase_3
source ../env/bin/activate
python test_week1.py --pair 1 --duration 60
```

Output in `fase_3/output/pair1_anechoic/`:
- `gsc_left.wav` + `gsc_right.wav` — beamformed audio
- `doa_sir.png` — DOA-tracking + SIR over tijd

---

## GUI roundtrip test

```bash
cd fase_3
source ../env/bin/activate
python test_gui_roundtrip.py
```

Valideert dat server, worker en frontend correct communiceren via Socket.IO.

---

## Troubleshooting

| Probleem | Oplossing |
|----------|-----------|
| `ModuleNotFoundError: tensorflow` | Gebruik `env_tf` (Python 3.11): `source ../env_tf/bin/activate` |
| Poort 8000 bezet | `lsof -ti :8000 \| xargs kill -9` |
| Geen data in GUI | Check `/tmp/fase3_worker.log` op errors |
| `[FATAL] Data-pad bestaat niet` | Zet `DATA_BASE=/jouw/pad` voor `./run_demo.sh` |
