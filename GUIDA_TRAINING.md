# 🚀 Guida Completa — LaTeX-OCR Training + Inferenza C++

## Panoramica

```
Dataset scaricati (~10M campioni)
        │
        ▼
[STEP 1] Installa dipendenze Python
        │
        ▼
[STEP 2] Estrai immagini fusion dataset (root.rar → cartella)
        │
        ▼
[STEP 3] Estrai HME100K (zip → cartella)
        │
        ▼
[STEP 4] Prepara dataset unificato (genera train.pkl + val.pkl)
        │                                    ⏱ ~2-4 ore
        ▼
[STEP 5] Lancia training PyTorch
        │                                    ⏱ ~24-72 ore
        ▼
[STEP 6] Export modello → GGUF
        │
        ▼
[STEP 7] Compila engine C++ (macOS o Linux)
        │
        ▼
[STEP 8] Inferenza C++ 🎉
```

---

## STEP 1 — Installa dipendenze Python

```bash
cd ~/Desktop/LaTeX-OCR

# Dipendenze base del progetto
pip install -e .

# Dipendenze extra per il training potenziato
pip install datasets transformers tokenizers
pip install opencv-python-headless
pip install imagesize Levenshtein
pip install munch pyyaml tqdm wandb
pip install einops timm==0.5.4
pip install x_transformers==0.15.0
pip install torchtext
```

---

## STEP 2 — Estrai le immagini del fusion dataset

Il fusion dataset ha le immagini dentro `root.rar` (27GB). Devi estrarlo:

```bash
# Installa unrar se non ce l'hai
brew install rar     # macOS
# oppure: sudo apt install unrar   # Linux

# Trova il file rar
FUSION_DIR="$HOME/.cache/huggingface/hub/datasets--hoang-quoc-trung--fusion-image-to-latex-datasets/snapshots/82906d1f80b4bd36d6e05fa40ee051fb391effe3"

# Estrai (ci vorrà un po')
cd /tmp
unrar x "$FUSION_DIR/root.rar" fusion_images/

# Controlla che ci siano le immagini
ls fusion_images/ | head -10
```

⚠️ **Servono ~30GB di spazio libero** per l'estrazione.

Se non vuoi estrarre tutto (o non hai spazio), puoi **saltare il fusion** e usare solo gli altri dataset (comunque ~2M campioni).

---

## STEP 3 — Estrai HME100K

```bash
# Estrai lo zip
mkdir -p ~/Desktop/LaTeX-OCR/data/hme100k
unzip ~/Downloads/hme100k.zip -d ~/Desktop/LaTeX-OCR/data/hme100k/

# Verifica
ls ~/Desktop/LaTeX-OCR/data/hme100k/images/ | wc -l
# Dovrebbe mostrare ~99000
```

---

## STEP 4 — Prepara il dataset unificato (⏱ ~2-4 ore)

Questo script scarica i dataset HuggingFace (già in cache), carica HME100K e fusion,
deduplica, genera il tokenizer BPE, e crea i .pkl per il training.

### Opzione A: TUTTI i dataset (con fusion — serve root.rar estratto)

```bash
cd ~/Desktop/LaTeX-OCR

FUSION_DIR="$HOME/.cache/huggingface/hub/datasets--hoang-quoc-trung--fusion-image-to-latex-datasets/snapshots/82906d1f80b4bd36d6e05fa40ee051fb391effe3"

python scripts/prepare_unified_dataset.py \
    --output data/unified \
    --fusion-dir "$FUSION_DIR" \
    --fusion-images /tmp/fusion_images \
    --hme100k-zip ~/Downloads/hme100k.zip \
    --hme100k-extract data/hme100k \
    --vocab-size 8000 \
    --hw-ratio 0.2
```

### Opzione B: SENZA fusion (se non vuoi estrarre 27GB)

```bash
cd ~/Desktop/LaTeX-OCR

python scripts/prepare_unified_dataset.py \
    --output data/unified \
    --hme100k-zip ~/Downloads/hme100k.zip \
    --hme100k-extract data/hme100k \
    --skip-datasets fusion \
    --vocab-size 8000 \
    --hw-ratio 0.2
```

### Output atteso:
```
DATASET PREPARATION COMPLETE
════════════════════════════════════════════════════════════
  Train samples:  XXXXXX (printed + handwritten)
  Val samples:    XXXXX
  Tokenizer:      data/unified/tokenizer.json
  Sample weights: data/unified/sample_weights.json
  Train pkl:      data/unified/train.pkl
  Val pkl:        data/unified/val.pkl
```

---

## STEP 5 — Lancia il Training (⏱ ~24-72 ore)

### Su macOS M4 (MPS):

```bash
cd ~/Desktop/LaTeX-OCR

python -m pix2tex.train --config pix2tex/model/settings/config_large.yaml --no_cuda --debug
```

> `--no_cuda` forza il fallback a MPS su Mac.
> `--debug` disabilita wandb (se non l'hai configurato).

### Su Linux con CUDA:

```bash
cd ~/Desktop/LaTeX-OCR

python -m pix2tex.train --config pix2tex/model/settings/config_large.yaml
```

### Cosa aspettarsi:
- Vedrà automaticamente MPS (Mac) o CUDA (Linux)
- Stampa un riepilogo configurazione
- Ogni 2000 step valuta BLEU/ACC sul validation set
- Salva checkpoint quando migliora
- Puoi interromperlo con Ctrl+C (salva automatico dopo epoch 2)
- I checkpoint vanno in `checkpoints/latex_ocr_large/`

### Riprendere il training (se interrotto):

```bash
python -m pix2tex.train \
    --config pix2tex/model/settings/config_large.yaml \
    --no_cuda --debug \
    --resume
```

E nel config, imposta:
```yaml
load_chkpt: checkpoints/latex_ocr_large/latex_ocr_large_e05_step12345.pth
```

---

## STEP 6 — Export del modello a GGUF

Dopo il training, converti il miglior checkpoint in formato GGUF per il C++:

```bash
cd ~/Desktop/LaTeX-OCR

# FP16 (raccomandato — buon bilanciamento velocità/precisione)
python scripts/export_gguf.py \
    --checkpoint checkpoints/latex_ocr_large/NOME_MIGLIOR_CHECKPOINT.pth \
    --config pix2tex/model/settings/config_large.yaml \
    --output latex_ocr_model.gguf \
    --dtype fp16

# Oppure Q8_0 (più veloce, leggermente meno preciso)
python scripts/export_gguf.py \
    --checkpoint checkpoints/latex_ocr_large/NOME_MIGLIOR_CHECKPOINT.pth \
    --config pix2tex/model/settings/config_large.yaml \
    --output latex_ocr_model_q8.gguf \
    --dtype q8_0
```

---

## STEP 7 — Compila l'engine C++

### macOS:

```bash
cd ~/Desktop/LaTeX-OCR/latex-ocr-cpp
chmod +x setup.sh
./setup.sh
```

### Linux:

```bash
# Installa OpenBLAS
sudo apt install libopenblas-dev cmake g++

cd ~/Desktop/LaTeX-OCR/latex-ocr-cpp
./setup.sh
```

Se ggml non si clona (firewall/proxy), scaricalo manualmente:
```bash
git clone --depth 1 https://github.com/ggml-org/ggml.git third_party/ggml
./setup.sh
```

---

## STEP 8 — Inferenza C++

```bash
cd ~/Desktop/LaTeX-OCR/latex-ocr-cpp/build

# Riconosci una formula da immagine
./latex_ocr \
    -m ../../latex_ocr_model.gguf \
    -t ../../data/unified/tokenizer.json \
    -i /path/to/formula.png

# Con parametri custom
./latex_ocr \
    -m ../../latex_ocr_model.gguf \
    -t ../../data/unified/tokenizer.json \
    -i formula.png \
    --temperature 0.1 \
    --max-tokens 256

# Solo CPU (no Metal GPU)
./latex_ocr \
    -m ../../latex_ocr_model.gguf \
    -t ../../data/unified/tokenizer.json \
    -i formula.png \
    --cpu --threads 8
```

---

## Riepilogo file creati/modificati

### Nuovi script Python:
| File | Scopo |
|------|-------|
| `scripts/prepare_unified_dataset.py` | Unifica tutti i dataset (HF + fusion + HME100K) |
| `scripts/convert_inkml.py` | Converte CROHME InkML → PNG con OpenCV |
| `scripts/export_gguf.py` | Esporta modello PyTorch → GGUF per C++ |

### File Python modificati:
| File | Modifiche |
|------|-----------|
| `pix2tex/train.py` | AMP, AdamW, cosine annealing, warmup, MPS, early stopping, weighted sampling |
| `pix2tex/model/settings/config_large.yaml` | Config training potenziato |

### Progetto C++ (inferenza):
```
latex-ocr-cpp/
├── CMakeLists.txt          # Build cross-platform (Metal/OpenBLAS)
├── setup.sh                # Script build automatico
├── include/
│   ├── latex_ocr.h         # API pubblica
│   ├── tokenizer.h         # BPE tokenizer
│   ├── image_preprocess.h  # Preprocessing immagini
│   ├── encoder.h           # ResNet + ViT encoder (ggml)
│   ├── decoder.h           # Decoder autoregressivo (ggml)
│   └── model.h             # Caricamento GGUF
├── src/
│   ├── tokenizer.cpp
│   ├── image_preprocess.cpp
│   ├── encoder.cpp
│   ├── decoder.cpp
│   ├── model.cpp
│   ├── latex_ocr.cpp
│   └── main.cpp            # CLI entry point
└── third_party/
    ├── ggml/               # (clonato da setup.sh)
    ├── stb_image.h
    ├── stb_image_write.h
    ├── cJSON.h
    └── cJSON.c
```

---

## Dataset utilizzati

| Dataset | Campioni | Tipo | Fonte |
|---------|----------|------|-------|
| fusion-image-to-latex | 3.4M train | Stampato + Manoscritto | HF (locale) |
| OleehyO/latex-formulas | 552k | Stampato | HF |
| UniMER-1M | 1.06M | Misto | HF |
| im2latex-100k | 68k | Stampato | HF |
| lukbl/LaTeX-OCR-dataset | 158k | Stampato | HF |
| HME100K | 99k | Manoscritto | Kaggle (locale) |
| **TOTALE** | **~5.3M** | | |

Il Weighted Random Sampler bilancia: **~20% manoscritto** per batch.
