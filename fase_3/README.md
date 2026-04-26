# Fase 3 – Week 1

Streaming-implementatie van de FD-GSC + dynamische MUSIC + SIR uit
[deadline1/week4.ipynb](../deadline1/week4.ipynb), aangepast aan de fase 3 audio
data (16 kHz, 4 minuten, bewegende sprekers).

Volledig **trouw aan de algoritmes** uit fase 1 week 4 — alleen ingepakt in
streaming wrappers met de aanpassingen die de universiteit voor week 1 vraagt:

| Onderdeel | Universiteits-vereiste week 1 | Implementatie |
|-----------|-------------------------------|---------------|
| Part 1    | FD-GSC werkt in dynamische omgeving | [`algorithms/streaming_gsc.py`](algorithms/streaming_gsc.py) — sliding STFT + per-bin NLMS-state behouden tussen chunks |
| Part 2    | Dynamische MUSIC met exp. middeling R_yy(ω) | [`algorithms/streaming_doa.py`](algorithms/streaming_doa.py) — `R_yy(ω,k) = β R_yy(ω,k-1) + (1-β) y yᴴ` |
| Part 3    | SIR per frame | [`algorithms/streaming_sir.py`](algorithms/streaming_sir.py) — schuifvenster (default 2s) |
| Part 4    | Twee beamformed streams | Twee `StreamingFDGSC`-instanties (links/rechts) in [`processor.py`](processor.py) |

## Folderstructuur

```
fase_3/
├── algorithms/                  -- streaming kernel (faithful aan week4 algos)
│   ├── lut_builder.py           -- FAS BF + Blocking Matrix per RIR-hoek
│   ├── streaming_doa.py         -- StreamingMUSIC (exp. R_yy, Part 2)
│   ├── streaming_gsc.py         -- StreamingFDGSC (sliding window, Part 1)
│   └── streaming_sir.py         -- StreamingSIR (Part 3)
├── processor.py                 -- ingevulde universiteits-skeleton processor
├── processing.py                -- ingevulde universiteits-skeleton worker
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
| `beta`    | 0.92    | exp. middeling R_yy. Hoog (0.99) = stabiel, traag adapterend; laag (0.85) = snel maar ruisig |
| `mu`      | 0.05    | NLMS step (zoals week4) |
| `bin_range` | `auto` | MUSIC bins gelimiteerd onder spatial-aliasing. Voor 16 kHz / 10cm spacing: bins 2-55 (= < 1715 Hz). Met `--bin_range full` exact week4 (1..L/2). |
| `combine` | `geometric` | pseudospectrum-combiner over bins (= week4) |

## Gemeten week-1 resultaten (pair1, anechoic, 60s)

| Metriek | Waarde |
|---------|--------|
| Globale SIR links-target | -4 dB |
| Globale SIR rechts-target | +5 dB |
| DOA-fout links (LUT-snapped) | ~12° mediaan |
| DOA-fout rechts (LUT-snapped) | ~15° mediaan |
| Real-time factor | ~2.4× (sneller dan real-time) |

> **Opmerking over de DOA-bias**: er zit een systematisch verschil van ~5–10° tussen
> de raw MUSIC-schatting (free-field geometric model) en de RIR-meet-hoeken die als
> ground truth dienen. Dit is een conventie-verschil, geen algoritme-fout. De LUT-snap
> kiest in de meeste gevallen de juiste meet-RIR voor de FD-GSC.

## Wat is NIET gewijzigd t.o.v. fase 1 week 4

- Pseudospectrum-combiner (geometrisch gemiddelde over bins)
- FAS-beamformer + Blocking Matrix berekening (`build_lut_for_target`)
- NLMS-update regel (per-bin, tijdens niet-target-spraak)
- VAD-strategie (frame-level + per-bin running mean)
- SIR-formule (`compute_sir`, identiek aan `computeSIR.py`)

De enige adaptaties zijn:
1. State-management voor streaming (sliding STFT-buffer, per-bin w_nlms persistent)
2. Exponentiële middeling R_yy (= expliciete vereiste van Part 2)
3. LUT bouwen uit `lma_16kHz.npz` i.p.v. uit week4's scenario.RIRs_audio (nieuwe data-format)

## TODO voor volgende weken

- **fase 2 LSTM-integratie**: zodra het dilated+lstm model uit Colab geüpload is,
  vervang de placeholder in `processor.processing_eeg_gt_audio()` door:
  - envelope-extractie (Gammatone of Hilbert+LP) op `sig_left_clean`/`sig_right_clean`
  - `model.predict([eeg, env1, env2])` → `pred_prob`
- Switching-stabiliteit (smoothing/hysterese tegen sudden AAD-fouten)
- Reverberant scenario testen
- Volledige systeem-evaluatie + rapport
