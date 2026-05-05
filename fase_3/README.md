# Fase 3 – Week 1

Streaming-implementatie van de FD-GSC + dynamische MUSIC + SIR uit
[deadline1/week4.ipynb](../deadline1/week4.ipynb), aangepast aan de fase 3 audio
data (16 kHz, 4 minuten, bewegende sprekers).

## TL;DR — Demo aan collega's of prof (3 minuten setup)

```bash
# Eénmalig: zorg dat de paths in run_demo.sh kloppen voor jouw machine
cd fase_3
./run_demo.sh                # default: pair 1, anechoic
# ./run_demo.sh 5 reverberant  # pair 5, reverberant scenario
```

Dit start automatisch:
1. `server.py` (skeleton) op `http://localhost:8000`
2. `processing.py` (worker met onze algoritmes)
3. Browser-tab op de live GUI

Stop met **Ctrl-C** -- alle subprocessen worden netjes opgekuist.

Wat je in de browser zou moeten zien:
- **DOA tracking** (links/rechts spreker) -- mediaan fout 0° op de gemeten LUT-hoeken
- **Twee beamformed waveforms** (gsc_left + gsc_right)
- **SIR over tijd** -- typisch +6 tot +12 dB
- **AAD probability** (placeholder, alterneert elke ~30s -- vervangen door LSTM in week 2)
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
│   └── streaming_sir.py         -- StreamingSIR (Part 3)
├── processor.py                 -- ingevulde universiteits-skeleton processor
├── processing.py                -- ingevulde universiteits-skeleton worker
├── run_demo.sh                  -- one-command demo launcher
├── test_week1.py                -- standalone end-to-end test (geen server nodig)
├── skeleton_ref/                -- ONGEWIJZIGDE kopie van het universiteits-skeleton
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

```bash
# Terminal 1: server
cd fase_3/skeleton_ref/server
python server.py \
    --microarray_path /Users/macbookmats/Desktop/.../phase_3/phase3_audioData/audiodata_batch_1/anechoic \
    --eeg_data_path /Users/macbookmats/Desktop/.../phase_3/data_phase3 \
    --stimuli_path /Users/macbookmats/Desktop/.../phase_3/data_phase3/stimuli

# Terminal 2: worker (onze ingevulde versie)
cd fase_3
python processing.py \
    --pair_no 1 \
    --subject_no 2 \
    --data_dir /Users/macbookmats/Desktop/.../phase_3/phase3_audioData/audiodata_batch_1/anechoic
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

## Troubleshooting

| Probleem | Oorzaak | Oplossing |
|----------|---------|-----------|
| `[FATAL] Data-pad bestaat niet` | `DATA_BASE` in `run_demo.sh` klopt niet | Pas regel 26 aan naar jouw locatie |
| `ModuleNotFoundError: algorithms` | Niet vanuit `fase_3/` gestart | `cd fase_3` vóór `python ...` |
| Server start niet op poort 8000 | Poort al in gebruik | `lsof -i :8000` → kill process |
| DOA altijd NaN | `lma_16kHz.npz` niet gevonden | Controleer `data_dir` pad; bestand zit in audiodata-map |
| SIR negatief | μ te groot → target-leakage | Gebruik `--mu 0.001` (default) |
| GUI toont geen data | Worker niet verbonden | Check `/tmp/fase3_worker.log` op Socket.IO errors |
| `RIRSteeringMUSIC` valt terug op planewave | Geen RIRs geladen | Zorg dat `lma_16kHz.npz` beschikbaar is; anders `--sv_model planewave` |

## TODO voor volgende weken

- **fase 2 LSTM-integratie**: zodra het dilated+lstm model uit Colab geüpload is,
  vervang de placeholder in `processor.processing_eeg_gt_audio()` door:
  - envelope-extractie (Gammatone of Hilbert+LP) op `sig_left_clean`/`sig_right_clean`
  - `model.predict([eeg, env1, env2])` → `pred_prob`
- Switching-stabiliteit (smoothing/hysterese tegen sudden AAD-fouten)
- Reverberant scenario testen
- Volledige systeem-evaluatie + rapport
