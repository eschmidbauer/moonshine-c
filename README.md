# moonshine-c

A zero-dependency C implementation of [Moonshine](https://github.com/usefulsensors/moonshine), a fast and accurate automatic speech recognition (ASR) model.

Supports both the original **batch** models (tiny, base) and the second-generation **streaming** models (tiny-streaming, small-streaming, medium-streaming). All weights are exported as float32 from HuggingFace.

## Models

| Model | Params | WER | Architecture |
|---|---|---|---|
| tiny | 26M | 12.66% | Batch — global self-attention encoder |
| base | 58M | 10.07% | Batch — global self-attention encoder |
| tiny-streaming | 34M | 12.00% | Streaming — sliding-window encoder |
| small-streaming | 123M | 7.84% | Streaming — sliding-window encoder |
| medium-streaming | 245M | 6.65% | Streaming — sliding-window encoder |

WER measured on LibriSpeech test-clean. Streaming models are the second-generation architecture and outperform their batch counterparts at equivalent parameter counts.

## Getting Started

### 1. Install Python dependencies

```bash
pip install -r scripts/requirements.txt
```

### 2. Export model weights

Downloads weights from HuggingFace and converts them to `.bin` files.

```bash
# Batch models
python scripts/export-weights.py --model tiny
python scripts/export-weights.py --model base

# Streaming models
python scripts/export-weights.py --model tiny-streaming
python scripts/export-weights.py --model small-streaming
python scripts/export-weights.py --model medium-streaming
```

Output goes to `models/<model>/`, each containing:

- `encoder.bin` — float32 encoder weights
- `decoder.bin` — float32 decoder weights
- `tokenizer.bin` — BPE tokenizer

### 3. Build

```bash
make
```

This builds two binaries: `test_process` (batch) and `test_streaming` (streaming).

### 4. Run

**Batch model:**
```bash
./test_process models/tiny jfk.wav
./test_process models/base jfk.wav
```

**Streaming model:**
```bash
./test_streaming models/tiny-streaming jfk.wav
./test_streaming models/small-streaming jfk.wav
```

`test_streaming` automatically chunks audio longer than 30 seconds into non-overlapping windows, transcribes each, and concatenates the results.

## API

### Batch

```c
#include "moonshine.h"

moonshine_model *model = moonshine_model_load("models/tiny");
moonshine_state *state = moonshine_state_create(model);

const char *text = moonshine_transcribe(model, state, pcm_f32, num_samples);

moonshine_state_free(state);
moonshine_model_free(model);
```

### Streaming

```c
#include "moonshine.h"

moonshine_streaming_model *model = moonshine_streaming_model_load("models/tiny-streaming");
moonshine_state *state = moonshine_state_create(NULL);

const char *text = moonshine_streaming_transcribe(model, state, pcm_f32, num_samples);

moonshine_state_free(state);
moonshine_streaming_model_free(model);
```

Both APIs accept 16 kHz mono float32 PCM. `moonshine_state` is the same type for both and holds only the last result string — create one per concurrent request, share the model freely across threads.

## Concurrency

`moonshine_model` and `moonshine_streaming_model` are immutable after load and safe to share across threads with no synchronization. Each concurrent request needs its own `moonshine_state`.

```c
// Load once
moonshine_streaming_model *model = moonshine_streaming_model_load("models/tiny-streaming");

// One state per thread / per request
moonshine_state *state = moonshine_state_create(NULL);
const char *text = moonshine_streaming_transcribe(model, state, pcm, n);
```

## Performance

Encoder compute vs audio length (GFLOPs, tiny models):

| Audio | batch-tiny | stream-tiny |
|---|---|---|
| 5s | 2.8 | 4.0 |
| 30s | 25.7 | 24.1 |
| 60s | 73.0 | 48.2 |
| 120s | 232.4 | 96.4 |

The batch encoder uses global O(T²) self-attention; the streaming encoder uses sliding-window O(T×W) attention (W≈20). They cross over around 28 seconds — streaming is cheaper for anything longer.

Peak encoder work-buffer memory follows the same pattern: at 60s, batch-tiny allocates a 33 MB T×T attention matrix; streaming-tiny uses 10 MB (no attention matrix stored).

## Long Audio

Audio longer than ~69 seconds hits a decoder token cap (`MAX_TOKENS=448`). `test_streaming` handles this by splitting into 30-second chunks. For the batch `test_process`, chunking must be handled by the caller.

Neither model includes VAD or silence detection. Silent passages are normalized by the streaming encoder's per-frame CMVN and can cause the decoder to hallucinate words. Trim silence before passing audio if this is a concern.
