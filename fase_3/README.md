# Fase 3

## Structuur

```
fase_3/
├── processor.py              # Centrale processor: koppelt alle algoritmes aan het skeleton
├── processing.py             # Worker entry point: Socket.IO client, stuurt data door
├── run_demo.sh               # One-command launcher: start server + worker + browser
├── test_week1.py             # Standalone test: verwerkt audio zonder server (genereer plots + WAVs)
├── test_gui_roundtrip.py     # Roundtrip test: valideert server ↔ worker ↔ frontend communicatie
│
├── algorithms/
│   ├── streaming_doa.py      # Wideband MUSIC met exp. R_yy middeling + DOATracker (outlier-rejectie)
│   ├── streaming_gsc.py      # FD-GSC met sliding STFT, per-bin NLMS-state persistent over chunks
│   ├── streaming_sir.py      # SIR-berekening via schuifvenster (2s default)
│   ├── lut_builder.py        # Offline LUT: FAS-beamformer + Blocking Matrix per RIR-hoek
│   └── aad_lstm.py           # AAD wrapper: Gammatone envelope + 5s/1s sliding window rond Keras model
│
├── skeleton/                 # Origineel universiteits-skeleton (ongewijzigd, ter referentie)
└── skeleton_ref/             # Skeleton met onze bug-fix in issp_data.py (get_doa_gt: cum-sum → diff)
```

**Pipeline per binnenkomende chunk:**
1. `processing.py` ontvangt LMA-chunk via Socket.IO → roept `processor.py` aan
2. `processor.py` → MUSIC update R_yy → DOA schatten → GSC links + rechts → SIR berekenen → output naar queues
3. Elke 1s: EEG + audio → `aad_lstm.py` → `pred_prob` → geselecteerde spreker

---

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

Twee modellen zijn beschikbaar in `fase_3/data/`:

| Model | Bestand | Intern | Aanbevolen |
|-------|---------|--------|-----------|
| `hybrid` | `hybrid_v3_BEST.keras` | Heeft BatchNorm op EEG-input | Standaard keuze |
| `generic` | `generic_dilated_alle_proefpersonen_beste_pieter_3laag.keras` | Geen BatchNorm | Gebruik met `AAD_NORMALIZE_EEG=1` |

```bash
cd fase_3
source ../env_tf/bin/activate

# hybrid model (standaard, geen extra normalisatie nodig)
PYTHON=$(which python) AAD_MODEL_NAME=hybrid ./run_demo.sh

# generic model (z-score normalisatie aan)
PYTHON=$(which python) AAD_MODEL_NAME=generic AAD_NORMALIZE_EEG=1 ./run_demo.sh

# of expliciet pad
PYTHON=$(which python) AAD_MODEL_PATH="$(pwd)/data/hybrid_v3_BEST.keras" ./run_demo.sh
```

**Env vars voor AAD:**
- `AAD_MODEL_NAME=hybrid` — zoekt `hybrid_*.keras` in `data/`
- `AAD_MODEL_NAME=generic` — zoekt `generic_*.keras` in `data/`
- `AAD_MODEL_PATH=/volledig/pad` — overschrijft naam-selectie
- `AAD_NORMALIZE_EEG=1` — z-score normalisatie EEG per venster (aanbevolen voor `generic`)
- `AAD_ENVELOPE=hilbert` — snellere envelope (default: `gammatone`)
- `AAD_WINDOW_S=5` — venstergrootte in seconden (default: 5)
- `AAD_HOP_S=1` — hop in seconden (default: 1)

De browser opent automatisch op `http://localhost:8000`. Stop met **Ctrl-C**.

---

## Standalone tests (geen server nodig)

### Week 1 — DOA, GSC, SIR
```bash
cd fase_3
source ../env/bin/activate
python test_week1.py --pair 1 --duration 60
```
Output in `fase_3/output/pair1_anechoic/`:
- `gsc_left.wav` + `gsc_right.wav` — beamformed audio
- `doa_sir.png` — DOA-tracking + SIR over tijd

### Week 2 — get_doa_gt fix, AAD LSTM, Processor
```bash
# Zonder AAD model (gebruikt placeholder, geen TensorFlow nodig)
cd fase_3
source ../env/bin/activate
python test_week2.py --pair 1

# Met AAD model (vereist env_tf + TensorFlow)
source ../env_tf/bin/activate
python test_week2.py --pair 1 --aad_model_path data/hybrid_v3_BEST.keras
```
Wat getest wordt:
- **Test 1** — `get_doa_gt` bug-fix: verifieert dat de GT-tijdslijn de correcte lengte heeft
- **Test 2** — AAD LSTM module: envelope-extractie + model-predictions in [0,1]
- **Test 3** — Volledige Processor: alle drie queues (phase1, phase2, phase3) produceren output

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
