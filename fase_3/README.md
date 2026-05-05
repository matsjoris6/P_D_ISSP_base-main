# Fase 3 – Week 1 + 2

Streaming-implementatie van de FD-GSC + dynamische MUSIC + SIR uit
[deadline1/week4.ipynb](../deadline1/week4.ipynb), aangepast aan de fase 3 audio
data (16 kHz, 4 minuten, bewegende sprekers). **Week 2** voegt toe:
- Live AAD via dilated+LSTM model (`hybrid_v3_BEST.keras`) met 5s/1s sliding window
- Bug-fix `get_doa_gt` in skeleton server (cum-sum → diff)
- Robuuste `run_demo.sh` met venv-autodetect en data-pad-autodetect

## TL;DR — Demo aan collega's of prof (3 minuten setup)

**Zonder AAD model** (placeholder, alterneert sprekers):
```bash
cd fase_3
./run_demo.sh                            # default: pair 1, anechoic
./run_demo.sh 5 reverberant              # pair 5, reverberant
```

**Met AAD LSTM model** (live EEG-driven spreker-selectie):
```bash
cd fase_3
AAD_MODEL_PATH="$(pwd)/data/hybrid_v3_BEST.keras" ./run_demo.sh
# Optioneel: AAD_WINDOW_S=5 AAD_HOP_S=1 ./run_demo.sh
```

Dit start automatisch:
1. `server.py` op `http://localhost:8000`
2. `processing.py` (worker met onze algoritmes)
3. Browser-tab op de live GUI

Stop met **Ctrl-C** -- alle subprocessen worden netjes opgekuist.

Wat je in de browser zou moeten zien:
- **DOA tracking** (links/rechts spreker) -- mediaan fout 0° op de gemeten LUT-hoeken
- **Twee beamformed waveforms** (gsc_left + gsc_right)
- **SIR over tijd** -- typisch +6 tot +12 dB
- **AAD probability** -- met model: live EEG-driven; zonder model: alterneert elke ~30s
- **Output signal** (geselecteerde spreker volgens AAD)

Volledig **trouw aan de algoritmes** uit fase 1 week 4 — alleen ingepakt in
streaming wrappers met de aanpassingen die de universiteit voor week 1 vraagt:

| Onderdeel | Universiteits-vereiste week 1 | Implementatie |
|-----------|-------------------------------|---------------|
| Part 1    | FD-GSC werkt in dynamische omgeving | [`algorithms/streaming_gsc.py`](algorithms/streaming_gsc.py) — sliding STFT + per-bin NLMS-state behouden tussen chunks |
| Part 2    | Dynamische MUSIC met exp. middeling R_yy(ω) | [`algorithms/streaming_doa.py`](algorithms/streaming_doa.py) — `RIRSteeringMUSIC`: steering vectors uit gemeten RIRs + `DOATracker` voor outlier-rejectie; `R_yy(ω,k) = β R_yy(ω,k-1) + (1-β) y yᴴ` |
| Part 3    | SIR per frame | [`algorithms/streaming_sir.py`](algorithms/streaming_sir.py) — schuifvenster (default 2s) |
| Part 4    | Twee beamformed streams | Twee `StreamingFDGSC`-instanties (links/rechts) in [`processor.py`](processor.py) |

## Folderstructuur

```
fase_3/
├── algorithms/                  -- streaming kernel (faithful aan week4 algos)
│   ├── lut_builder.py           -- FAS BF + Blocking Matrix per RIR-hoek
│   ├── streaming_doa.py         -- RIRSteeringMUSIC + DOATracker (v2), StreamingMUSIC (Part 2)
│   ├── streaming_gsc.py         -- StreamingFDGSC (sliding window, Part 1)
│   ├── streaming_sir.py         -- StreamingSIR (Part 3)
│   └── aad_lstm.py              -- [WEEK 2] AADLSTM wrapper + Gammatone envelope
├── data/
│   ├── hybrid_v3_BEST.keras     -- [WEEK 2] dilated+LSTM AAD model
│   ├── phase3_audioData/...     -- microarray + RIRs (anechoic / reverberant)
│   └── data_phase3/             -- EEG data + stimuli
├── processor.py                 -- ingevulde universiteits-skeleton processor
├── processing.py                -- ingevulde universiteits-skeleton worker
├── run_demo.sh                  -- one-command demo launcher (week 1+2)
├── test_week1.py                -- standalone end-to-end test (geen server nodig)
├── test_gui_roundtrip.py        -- GUI Socket.IO roundtrip validatie
├── skeleton_ref/                -- universiteits-skeleton (issp_data.py: bug-fix)
└── output/                      -- gegenereerde WAVs + plots per pair
```

## Snel testen (zonder server)

```bash
python fase_3/test_week1.py --pair 1 --duration 60
```

Genereert in `fase_3/output/pair1_anechoic/`:
- `gsc_left.wav` + `gsc_right.wav` (twee beamformed streams)
- `mix_mic1.wav` (referentie: ruwe mic 1)
- `doa_sir.png` (DOA-tracking + SIR over tijd)

## Volledige skeleton draaien (server + worker)

Verwacht data-structuur onder een `DATA_BASE`-map (bijv. `fase_3/data/`):
```
DATA_BASE/
├── phase3_audioData/audiodata_batch_1/anechoic/   # microarray + RIRs
├── data_phase3/                                   # EEG data
└── data_phase3/stimuli/                           # audio stimuli
```

```bash
# Terminal 1: server
cd fase_3/skeleton_ref/server
python server.py \
    --microarray_path $DATA_BASE/phase3_audioData/audiodata_batch_1/anechoic \
    --eeg_data_path $DATA_BASE/data_phase3 \
    --stimuli_path $DATA_BASE/data_phase3/stimuli

# Terminal 2: worker (onze ingevulde versie)
cd fase_3
python processing.py \
    --pair_no 1 \
    --subject_no 2 \
    --data_dir $DATA_BASE/phase3_audioData/audiodata_batch_1/anechoic
```

## Belangrijke parameters

| Parameter | Default | Toelichting |
|-----------|---------|-------------|
| `L`       | 512     | STFT-lengte (≈ 32ms bij 16 kHz, vergelijkbaar met week4 op 44.1 kHz) |
| `hop`     | 256     | 50% overlap, zoals week4 |
| `beta`    | 0.92    | exp. middeling R_yy. Zie beta-analyse hieronder. |
| `mu`      | 0.001   | NLMS step. Kleiner dan week4 (0.1) wegens target-leakage bij 16 kHz fase-3 data. |
| `bin_range` | `auto` | MUSIC bins gelimiteerd onder spatial-aliasing. Voor 16 kHz / 10cm spacing: bins 2-55 (= < 1715 Hz). Met `--bin_range full` exact week4 (1..L/2). |
| `combine` | `geometric` | pseudospectrum-combiner over bins (= week4) |

## Part 2: Beta-analyse (effect van exponentiële middelingsconstante)

Resultaten pair1, anechoic, 30s (input SIR = ±0.5 dB):

| β    | SIR links (dB) | SIR rechts (dB) | DOA mediaan fout L/R |
|------|---------------|-----------------|----------------------|
| 0.80 | +8.10         | +8.70           | ~12° / ~16°         |
| 0.90 | +8.07         | +8.66           | ~12° / ~15°         |
| **0.92** | **+8.03** | **+8.78**   | **~12° / ~16°**     |
| 0.95 | +7.95         | +8.76           | ~12° / ~15°         |
| 0.99 | +7.91         | +8.89           | ~12° / ~12°         |

- **Laag β (< 0.85)**: snel adapterend bij positieveranderingen, maar R_yy is ruisiger → instabiele DOA-schatting.
- **Hoog β (> 0.97)**: stabiele R_yy, maar traag bij snelle positieveranderingen (>2s vertraging).
- **β = 0.92 (default)**: goede balans voor fase-3 data met ~16s segmenten per positie.

## Gemeten week-1 resultaten (v2 — RIR steering + DOATracker)

### DOA-precisie: voor vs. na (mediaan absolute fout)

| Scenario | Zijde | Planewave (oud) | RIR + tracker (nieuw) |
|----------|-------|-----------------|----------------------|
| pair1 anechoic | links | ~15° | **0°** |
| pair1 anechoic | rechts | ~15° | **0°** |
| pair5 reverberant | links | ~5° | **0°** |
| pair5 reverberant | rechts | ~18° | **0°** |

### SIR (pair1, anechoic, 60s, mu=0.001, beta=0.92)

| Metriek | Waarde |
|---------|--------|
| Input SIR mic1 | ±0.5 dB (meting voor BF) |
| Globale SIR links-target | **+6.9 dB** (+6.4 dB verbetering) |
| Globale SIR rechts-target | **+10.8 dB** (+11.3 dB verbetering) |
| DOA-fout links (mediaan) | **0°** (was ~15° met planewave) |
| DOA-fout rechts (mediaan) | **0°** (was ~15° met planewave) |
| Real-time factor | ~2.4× (sneller dan real-time) |

> **Noot μ vs week4**: week4 gebruikte μ=0.1 op 44.1 kHz anechoïsche kamer-data met statische sprekers.
> Bij fase-3 data (bewegende sprekers, 16 kHz) zorgt μ=0.05 voor target-leakage in de blocking-matrix
> waardoor NLMS de target mee-cancelt. μ=0.001 minimaliseert dit: SIR gaat van −4.6 → +6.9 dB.

## DOA-verbeteringen v2 (root-cause analyse)

De oorspronkelijke `StreamingMUSIC` gebruikte een **far-field plane-wave model**
(`e^{-jωτ}` met τ gebaseerd op microfoonposities + vrije-ruimte voortplantingssnelheid).
Dit geeft een **systematische fout van 5–15°** die niet via smoothing weg te werken is.

### Root cause 1 — Steering vector mismatch (grootste bijdrage)

**Nieuw**: `RIRSteeringMUSIC` berekent per-bin complex transfer-functies uit de gemeten
RIRs in `lma_16kHz.npz`:

```
H[k, m, θ] = rfft(RIR[:, m, θ], n=L)   # per-bin, per-mic, per-LUT-hoek
A[k, m, θ] = H[k, m, θ] / H[k, 0, θ]  # normaliseer t.o.v. mic 0
```

MUSIC zoekt nu over de ~20 gemeten LUT-hoeken (4.49°–176.89°) in plaats van 361
synthetische plane-wave-hoeken. De steering vectors matchen perfect met de data die
ook via dezelfde RIRs geconvolveerd is → mediaan fout 0° in plaats van 12–15°.

### Root cause 2 — "Soms volledig naast" tijdens overgangen

Bij positieveranderingen (sprekers bewegen) convergeert R_yy traag. In die frames kan
MUSIC tijdelijk beide pieken aan één kant toewijzen. De nieuwe `DOATracker` lost dit op:

- Mediaan-buffer (N=5 schattingen) per zijde
- EMA (α=0.3) voor vloeiende tracking
- Outlier-rejectie: `|nieuw − vorig| > 30°` → vereist 2 opeenvolgende bevestigingen

### Overige verbeteringen (opt-in)

| Optie | Flag | Default | Wanneer nuttig |
|-------|------|---------|----------------|
| Forward-Backward averaging | `use_fb=True` | auto (aan bij reverberant) | Coherente multipath |
| SNR-gewogen bin-combinatie | `snr_weight=True` | uit | Sterk variabele SNR per bin |
| Parabola peak-refinement | `peak_refine=True` | uit | Sub-grid precisie (alleen indien GT ≠ LUT-hoeken) |
| Spectrum quality gate | `quality_gate_dB=3.0` | 0 (uit) | Rejecteer lage SNR-frames |

> **Noot peak-refinement**: de fase-3 ground-truth hoeken zijn identiek aan de
> LUT-meetposities, dus parabola-interpolatie schuift de schatting weg van het
> correcte antwoord. Default staat dit uit.

## Wat is NIET gewijzigd t.o.v. fase 1 week 4

- Pseudospectrum-combiner (geometrisch gemiddelde over bins)
- FAS-beamformer + Blocking Matrix berekening (`build_lut_for_target`)
- NLMS-update regel (per-bin, tijdens niet-target-spraak)
- VAD-strategie (frame-level: std(frame) > 0.1 × running_max van std)
- SIR-formule (`compute_sir`, identiek aan `computeSIR.py`)

De aanpassingen voor streaming + fase-3 data zijn:
1. State-management voor streaming (sliding STFT-buffer, per-bin w_nlms persistent)
2. Exponentiële middeling R_yy (= expliciete vereiste van Part 2)
3. LUT bouwen uit `lma_16kHz.npz` i.p.v. uit week4's scenario.RIRs_audio (nieuwe data-format)
4. μ = 0.001 i.p.v. 0.1 (week4) — fase-3 data met bewegende sprekers heeft kleinere stap nodig
   om target-leakage via blocking matrix te beheersen (zie beta-analyse hierboven)

## Week 2 — AAD LSTM + GUI bug-fix

### AAD LSTM+dilated CNN integratie

Het Colab-getrainde model `fase_3/data/hybrid_v3_BEST.keras` is geïntegreerd in
de live pipeline:

| Parameter | Waarde | Toelichting |
|-----------|--------|-------------|
| Input shape | `(640, 64) + (640, 1) + (640, 1)` | EEG + env links + env rechts @ 128Hz |
| Window | 5s sliding | overeenkomstig 640 EEG-samples @ 128Hz |
| Hop | 1s | nieuwe predictie elke 1 seconde |
| Envelope | Gammatone-bank (default) | 28 banden ERB-spaced 80–6000Hz |
| Output | 1 sigmoid | P(attended_left) ∈ [0, 1] |
| Latency | ~30ms per predict | well within 1s budget (real-time) |

**Pipeline** (in `fase_3/algorithms/aad_lstm.py`):
1. Audio → Gammatone-bank (28 banden) → |Hilbert| per band → ^0.6 compressie → som → 8Hz lowpass → downsample naar 128Hz
2. EEG (al @ 128Hz, 64 kanalen) → buffer 5s
3. Sliding window: predict elke 1s, gebruik laatste 5s; gooi oudste 1s weg
4. Output `pred_prob` → `attended_left = (pred_prob ≥ 0.5)`

**Alternatief envelope** (sneller, ~10× minder rekentijd, vergelijkbare AAD-prestatie):
```bash
AAD_MODEL_PATH=… ./run_demo.sh   # default: gammatone
# of expliciet:
python processing.py --aad_model_path … --aad_envelope hilbert --aad_window_s 5 --aad_hop_s 1
```

### GUI bug-fix: `get_doa_gt` cumulatieve-sum bug

In `skeleton_ref/server/issp_data.py` gebruikt de oorspronkelijke `get_doa_gt`
de cumulatieve `endSamples_l/_r` direct als `np.repeat`-count:

```python
# OUD (BUGGY):
doa_0 = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_l"], gt["endSamples_l"])])
```

`endSamples_l` is `[262144, 502239, 742334, ...]` (cumulatief in samples), niet duraties.
Resultaat: GT wordt 8× te lang gerekt (33M samples i.p.v. 3.86M voor 4-minuten audio),
GT-tijdslijn loopt voor op werkelijke audio in de live plot → schijnbare DOA-mismatches
ook al is mediaan fout 0°.

**Fix** (zoals `test_week1.py` regel 66 al correct doet):
```python
durations_l = np.diff(np.concatenate([[0], gt["endSamples_l"]]))
doa_0 = np.concatenate([np.repeat(e, n) for e, n in zip(gt["angles_l"], durations_l)])
```

### Verbeterde `run_demo.sh`

| Feature | Beschrijving |
|---------|-------------|
| Venv autodetect | Probeert `env/` dan `venv/` (geen handmatige `source` nodig) |
| Data-pad autodetect | Probeert `fase_3/data/` dan `documents_and_given_code/phase_3/` |
| AAD env-vars | `AAD_MODEL_PATH`, `AAD_WINDOW_S`, `AAD_HOP_S` |
| Poort 8000 cleanup | Auto-kill bestaand proces als poort bezet |
| `DATA_BASE` override | `DATA_BASE=/jouw/pad ./run_demo.sh` |

## Troubleshooting

| Probleem | Oorzaak | Oplossing |
|----------|---------|-----------|
| `[FATAL] Data-pad bestaat niet` | Data-pad detectie faalt | `DATA_BASE=/jouw/pad ./run_demo.sh` of leg data in `fase_3/data/` |
| `ModuleNotFoundError: algorithms` | Niet vanuit `fase_3/` gestart | `cd fase_3` vóór `python ...` |
| Server start niet op poort 8000 | Poort al in gebruik | `run_demo.sh` doet auto-kill; anders `lsof -ti :8000 \| xargs kill -9` |
| DOA altijd NaN | `lma_16kHz.npz` niet gevonden | Controleer `data_dir` pad; bestand zit in audiodata-map |
| SIR negatief | μ te groot → target-leakage | Gebruik `--mu 0.001` (default) |
| GUI toont geen data | Worker niet verbonden | Check `/tmp/fase3_worker.log` op Socket.IO errors |
| `RIRSteeringMUSIC` valt terug op planewave | Geen RIRs geladen | Zorg dat `lma_16kHz.npz` beschikbaar is; anders `--sv_model planewave` |
| `ModuleNotFoundError: tensorflow` | TF niet in actieve venv | TF vereist Python 3.10–3.12; maak venv met `python3.11 -m venv env_tf && source env_tf/bin/activate && pip install tensorflow` |
| AAD predictions altijd 0.5 | Buffer nog niet vol (eerste 5s) | Normaal; eerste 5 chunks geven `last_pred=0.5` default |
| GT-DOA loopt voor op werkelijkheid | Oude `issp_data.py` zonder bug-fix | Pull laatste mats branch (commit met `np.diff` fix) |

## Status

| Onderdeel | Status |
|-----------|--------|
| FD-GSC streaming (Part 1) | ✅ |
| Dynamische MUSIC + DOATracker (Part 2) | ✅ |
| SIR per frame (Part 3) | ✅ |
| Twee beamformed streams (Part 4) | ✅ |
| Live GUI via `run_demo.sh` | ✅ |
| AAD LSTM+dilated integratie (Week 2) | ✅ |
| `get_doa_gt` bug-fix (cum-sum → diff) | ✅ |
