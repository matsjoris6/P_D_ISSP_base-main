"""
config.py  —  Centrale modelconfiguratie voor fase 3

Verander enkel ACTIVE_MODEL om te switchen tussen modellen.
Alle andere bestanden (processor.py, processing.py) lezen hier automatisch uit.

Start de server altijd met de bijbehorende server_cmd hieronder,
of run:  python config.py  om de juiste commando's te zien.
"""

# ════════════════════════════════════════════════════════════════════
#  BESCHIKBARE MODELLEN
# ════════════════════════════════════════════════════════════════════

MODELS = {

    # ── Model A: origineel dilated CNN (geen overlap in training) ────
    # 3-laags Dilated CNN, getraind op niet-overlappende 5s blokken.
    # Inference met 1s sliding window → buiten training-distributie, maar werkt goed.
    "dilated_5s": {
        "model_path":         "models/generic_dilated_alle_proefpersonen_beste_pieter_3laag_5sec_VERVOLG.keras",
        "window_sec":         5,
        "hop_sec":            1,
        "eeg_window_samples": 5 * 64,   # 320 samples @ 64 Hz
        "description":        "Dilated CNN  |  5s venster  |  geen overlap in training",
        "ema_alpha":          0.3,
        "schmitt_threshold":  0.5,
        "schmitt_hysteresis": 0.0,
        "server_args":        "--aad_window_size 5 --aad_hop_size 1",
    },

    # ── Model B: hybrid 3s (Dilated CNN + LSTM, 50% overlap) ────────
    # Getraind met 50% overlap op 3s vensters.
    # Kleinste venster → snelste reactietijd, minder context.
    "hybrid_3s": {
        "model_path":         "models/hybrid/hybrid_3sec_50overlap_BEST.keras",
        "window_sec":         3,
        "hop_sec":            1,
        "eeg_window_samples": 3 * 64,   # 192 samples @ 64 Hz
        "description":        "Dilated CNN + LSTM  |  3s venster  |  50% overlap in training",
        "ema_alpha":          0.3,
        "schmitt_threshold":  0.5,
        "schmitt_hysteresis": 0.0,
        "server_args":        "--aad_window_size 3 --aad_hop_size 1",
    },

    # ── Model C: hybrid 5s (Dilated CNN + LSTM, 50% overlap) ────────
    # Getraind met 50% overlap op 5s vensters.
    # Zelfde venstergrootte als Model A, maar met LSTM en overlap-training.
    "hybrid_5s": {
        "model_path":         "models/hybrid/hybrid_5sec_50overlap_BEST.keras",
        "window_sec":         5,
        "hop_sec":            1,
        "eeg_window_samples": 5 * 64,   # 320 samples @ 64 Hz
        "description":        "Dilated CNN + LSTM  |  5s venster  |  50% overlap in training",
        "ema_alpha":          0.3,
        "schmitt_threshold":  0.5,
        "schmitt_hysteresis": 0.0,
        "server_args":        "--aad_window_size 5 --aad_hop_size 1",
    },

    # ── Model D: hybrid 10s (Dilated CNN + LSTM, 50% overlap) ───────
    # Getraind met 50% overlap op 10s vensters → hop=5s matcht de training exact.
    # Meeste context → hoogste verwachte accuracy, maar 10s aanlooptijd.
    # hop=5s i.p.v. 1s om twee redenen:
    #   1. Rekentijd: gammatone op 10s audio (~480k samples) duurt >1s → 1s hop is te krap.
    #   2. Training-distributie: model zag 50% overlap (hop=5s); 1s hop = 90% overlap
    #      → consecutive vensters zijn bijna identiek → instabiele/slechte beslissingen.
    "hybrid_10s": {
        "model_path":         "models/hybrid/hybrid_10sec_50overlap_BEST.keras",
        "window_sec":         10,
        "hop_sec":            5,        # matcht 50% overlap uit training
        "eeg_window_samples": 10 * 64,  # 640 samples @ 64 Hz
        "description":        "Dilated CNN + LSTM  |  10s venster  |  50% overlap in training  |  5s hop",
        "ema_alpha":          0.3,
        "schmitt_threshold":  0.5,
        "schmitt_hysteresis": 0.0,
        "server_args":        "--aad_window_size 10 --aad_hop_size 5",
    },

}

# ════════════════════════════════════════════════════════════════════
#  ← VERANDER DIT OM TE SWITCHEN
# ════════════════════════════════════════════════════════════════════
#
#   "dilated_5s"   →  origineel model, geen overlap, 5s venster
#   "hybrid_3s"    →  hybrid model, 50% overlap, 3s venster
#   "hybrid_5s"    →  hybrid model, 50% overlap, 5s venster
#   "hybrid_10s"   →  hybrid model, 50% overlap, 10s venster
#
ACTIVE_MODEL = "dilated_5s"

# "anechoic" of "reverberant" — bepaalt welke RIR geladen wordt in
# processor.py (beamformer + MUSIC) én welk pad de server gebruikt.
SCENARIO = "reverberant"

# ── Audio-bron voor AAD ──────────────────────────────────────────────
# False (standaard): AAD gebruikt de clean speech van de stimuli (48 kHz).
#   → Ideale omstandigheid: het model krijgt perfecte envelops zonder reverb of ruis.
#   → Basis-implementatie die de universiteit vereist.
#
# True: AAD gebruikt de GSC-output van de LMA beamformer (16 kHz).
#   → Realistisch: envelops berekend op echte microfoon-audio na ruimtelijk filteren.
#   → Bevat resterende reverb, sprekerlekkage en microfoonruis.
#   → Levert extra punten als de AAD-accuracy nog steeds aanvaardbaar is.
USE_GSC_AUDIO_FOR_AAD = False

# ════════════════════════════════════════════════════════════════════
#  Afgeleid — niet aanpassen
# ════════════════════════════════════════════════════════════════════

# RIR-paden per scenario (beamformer + MUSIC steering vectors)
_RIR_PATHS = {
    "anechoic":    "data/phase3_audioData/audiodata_batch_1/anechoic/lma_16kHz.npz",
    "reverberant": "data/phase3_audioData/audiodata_batch_1/reverberant/lma_16kHz_200ms.npz",
}
# Microarray-data pad per scenario (voor server --microarray_path)
_MICROARRAY_PATHS = {
    "anechoic":    "data/phase3_audioData/audiodata_batch_1/anechoic",
    "reverberant": "data/phase3_audioData/audiodata_batch_1/reverberant",
}

RIR_PATH        = _RIR_PATHS[SCENARIO]
MICROARRAY_PATH = _MICROARRAY_PATHS[SCENARIO]

_cfg = MODELS[ACTIVE_MODEL]

MODEL_PATH          = _cfg["model_path"]
WINDOW_SEC          = _cfg["window_sec"]
HOP_SEC             = _cfg["hop_sec"]
EEG_WINDOW_SAMPLES  = _cfg["eeg_window_samples"]
EMA_ALPHA           = _cfg["ema_alpha"]
SCHMITT_THRESHOLD   = _cfg["schmitt_threshold"]
SCHMITT_HYSTERESIS  = _cfg["schmitt_hysteresis"]

# Niet aanpassen — constanten die niet model-afhankelijk zijn
UPDATE_RATE = 32   # server chunks per seconde (moet overeenkomen met server)

# Afgeleid voor processing.py
AAD_WIN_CHUNKS = WINDOW_SEC * UPDATE_RATE   # rolling buffer grootte
AAD_HOP_CHUNKS = HOP_SEC    * UPDATE_RATE   # stap tussen inferenties

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  ACTIEF MODEL  :  {ACTIVE_MODEL}")
    print(f"  SCENARIO      :  {SCENARIO}")
    print(f"  AAD-audio     :  {'GSC-output (16 kHz)' if USE_GSC_AUDIO_FOR_AAD else 'Clean speech (48 kHz)'}")
    print(f"{'='*60}")
    print(f"  {_cfg['description']}")
    print(f"  Venster       :  {WINDOW_SEC}s  ({EEG_WINDOW_SAMPLES} samples @ 64 Hz)")
    print(f"  Hop           :  {HOP_SEC}s")
    print(f"  EMA α         :  {EMA_ALPHA}")
    print(f"  Schmitt       :  {SCHMITT_THRESHOLD} ± {SCHMITT_HYSTERESIS}")
    print(f"  RIR           :  {RIR_PATH}")
    print(f"\n  Server starten (vanuit fase_3/skeleton/):")
    print(f"    python server/server.py \\")
    print(f"      --microarray_path \"{MICROARRAY_PATH}\" \\")
    print(f"      --eeg_data_path \"data/data_phase3\" \\")
    print(f"      --stimuli_path \"data/data_phase3/stimuli\" \\")
    print(f"      {_cfg['server_args']}")
    print(f"\n  Processor starten (vanuit fase_3/skeleton/):")
    print(f"    python processing.py --pair_no 1 --subject_no 3")
    print(f"{'='*60}")
    print(f"\n  Beschikbare modellen:")
    for naam, m in MODELS.items():
        marker = "  ← ACTIEF" if naam == ACTIVE_MODEL else ""
        print(f"    \"{naam:<12}\"  {m['description']}{marker}")
    print(f"\n  Beschikbare scenario's: anechoic / reverberant")
    print(f"  Switchen: pas ACTIVE_MODEL en/of SCENARIO aan in config.py")
    print(f"{'='*60}\n")
