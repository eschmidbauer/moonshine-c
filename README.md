# moonshine-c

A zero-dependency C implementation of [Moonshine](https://github.com/usefulsensors/moonshine), a fast and accurate automatic speech recognition (ASR) model.

All weights are exported as **float32** from HuggingFace.

## Getting Started

### 1. Install Python dependencies

```bash
pip install -r scripts/requirements.txt
```

### 2. Export float32 model weights

Downloads weights from HuggingFace and converts them to `.bin` files.

```bash
# Export both tiny and base (default)
python scripts/export-weights.py

# Export only tiny
python scripts/export-weights.py --model tiny
```

Output goes to `models/<model>/` (e.g. `models/tiny/`, `models/base/`), each containing:

- `encoder.bin` — float32 encoder weights
- `decoder.bin` — float32 decoder weights
- `tokenizer.bin` — BPE tokenizer (downloaded and converted from HuggingFace)

### 3. Build and run

```bash
make
./test_process models/tiny jfk.wav
```
