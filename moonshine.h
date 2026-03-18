/*
 * moonshine.h — Zero-dependency Moonshine speech-to-text library.
 *
 * Supports moonshine-tiny and moonshine-base.  Model architecture is
 * auto-detected from the config embedded in the weight files.
 *
 * Concurrency model:
 *   moonshine_model is immutable after creation — share it freely across
 *   threads with no synchronization.  Each concurrent request gets its own
 *   moonshine_state which holds the mutable KV cache and result buffer.
 *
 * Usage:
 *   moonshine_model *model = moonshine_model_load("path/to/model_dir");
 *   moonshine_state *state = moonshine_state_create(model);
 *
 *   const char *text = moonshine_transcribe(model, state, pcm_f32, n);
 *   printf("%s\n", text);
 *
 *   moonshine_state_free(state);
 *   moonshine_model_free(model);
 */

#ifndef MOONSHINE_H
#define MOONSHINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque types. */
typedef struct moonshine_model moonshine_model;
typedef struct moonshine_state moonshine_state;

/* ── Model (immutable, thread-safe, load once) ────────────────────── */

/* Load model files from a directory.
 * The directory must contain: encoder.bin, decoder.bin, tokenizer.bin
 * Returns NULL on failure. */
moonshine_model *moonshine_model_load(const char *model_dir);

/* Free model and all weight data. */
void moonshine_model_free(moonshine_model *model);

/* Return a string describing the model (e.g. "dim=288 heads=8 ...").
 * Valid until moonshine_model_free(). */
const char *moonshine_model_info(const moonshine_model *model);

/* ── State (mutable, one per concurrent request) ──────────────────── */

/* Create inference state for the given model.
 * Lightweight — just allocates scratch buffers.  Create one per thread
 * or per request; do not share across concurrent calls. */
moonshine_state *moonshine_state_create(const moonshine_model *model);

/* Free state and scratch buffers. */
void moonshine_state_free(moonshine_state *state);

/* ── Transcription ────────────────────────────────────────────────── */

/* Transcribe 16 kHz mono float32 PCM audio.
 * model:  shared, read-only — safe to use from multiple threads.
 * state:  per-request scratch — must not be shared across concurrent calls.
 * Returns a pointer to the transcribed text.  The pointer is valid until
 * the next call to moonshine_transcribe() on the same state, or until
 * moonshine_state_free(). Returns NULL on failure. */
const char *moonshine_transcribe(const moonshine_model *model,
                                 moonshine_state *state,
                                 const float *pcm_f32,
                                 int num_samples);

#ifdef __cplusplus
}
#endif

#endif /* MOONSHINE_H */
