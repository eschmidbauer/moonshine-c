/*
 * test_streaming.c — Test the moonshine streaming model API.
 *
 * Audio longer than CHUNK_SECS is split into non-overlapping chunks and
 * transcribed sequentially; results are concatenated with a space.
 *
 * Usage:
 *   ./test_streaming <model_dir> <audio1.wav> [audio2.wav] ...
 */

#include "moonshine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Safe upper bound: well below the ~69s token cap and ~82s pos_emb limit. */
#define CHUNK_SECS   30
#define SAMPLE_RATE  16000
#define CHUNK_SAMPLES (CHUNK_SECS * SAMPLE_RATE)

static float *load_wav(const char *path, int *out_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    char riff[4]; fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(f); return NULL; }
    fseek(f, 4, SEEK_CUR);
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(f); return NULL; }

    int16_t channels = 0, bits_per_sample = 0;
    int32_t sample_rate = 0, data_size = 0;

    while (1) {
        char chunk_id[4]; uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            int16_t fmt; fread(&fmt, 2, 1, f);
            fread(&channels, 2, 1, f);
            fread(&sample_rate, 4, 1, f);
            fseek(f, 6, SEEK_CUR);
            fread(&bits_per_sample, 2, 1, f);
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size; break;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (bits_per_sample != 16 || channels != 1) {
        fprintf(stderr, "WAV must be 16-bit mono 16kHz PCM (got %d-bit, %d ch)\n",
                bits_per_sample, channels);
        fclose(f); return NULL;
    }

    int n = data_size / 2;
    int16_t *raw = (int16_t *)malloc(data_size);
    fread(raw, 2, n, f); fclose(f);

    float *audio = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) audio[i] = raw[i] / 32768.0f;
    free(raw);

    *out_samples = n;
    return audio;
}

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Append src to *buf (growing it), inserting a space if buf is non-empty. */
static void append_text(char **buf, size_t *cap, size_t *len, const char *src) {
    if (!src || src[0] == '\0') return;
    size_t add = strlen(src) + 2; /* space + NUL */
    if (*len + add > *cap) {
        *cap = (*len + add) * 2;
        *buf = (char *)realloc(*buf, *cap);
    }
    if (*len > 0) (*buf)[(*len)++] = ' ';
    size_t sl = strlen(src);
    memcpy(*buf + *len, src, sl + 1);
    *len += sl;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_dir> <audio1.wav> [audio2.wav] ...\n", argv[0]);
        return 1;
    }

    const char *model_dir = argv[1];

    printf("Loading streaming model from %s ...\n", model_dir);
    double t0 = now_ms();
    moonshine_streaming_model *model = moonshine_streaming_model_load(model_dir);
    if (!model) {
        fprintf(stderr, "Failed to load streaming model\n");
        return 1;
    }
    printf("Model: %s (loaded in %.0f ms)\n\n",
           moonshine_streaming_model_info(model), now_ms() - t0);

    moonshine_state *state = moonshine_state_create(NULL);
    if (!state) {
        fprintf(stderr, "Failed to create state\n");
        moonshine_streaming_model_free(model);
        return 1;
    }

    char   *result_buf = NULL;
    size_t  result_cap = 0, result_len = 0;

    for (int i = 2; i < argc; i++) {
        int num_samples;
        float *audio = load_wav(argv[i], &num_samples);
        if (!audio) continue;

        float duration = num_samples / (float)SAMPLE_RATE;
        int num_chunks = (num_samples + CHUNK_SAMPLES - 1) / CHUNK_SAMPLES;

        if (num_chunks > 1)
            printf("[%s] %.1fs → %d chunks of %ds\n",
                   argv[i], duration, num_chunks, CHUNK_SECS);

        result_len = 0;
        double t1 = now_ms();

        for (int c = 0; c < num_chunks; c++) {
            int offset  = c * CHUNK_SAMPLES;
            int samples = num_samples - offset;
            if (samples > CHUNK_SAMPLES) samples = CHUNK_SAMPLES;

            const char *text = moonshine_streaming_transcribe(
                model, state, audio + offset, samples);

            append_text(&result_buf, &result_cap, &result_len, text);
        }

        double elapsed = now_ms() - t1;

        printf("[%s] (%.0f ms, audio=%.1fs, RTF=%.2fx)\n  %s\n\n",
               argv[i], elapsed,
               duration,
               elapsed / (num_samples / 16.0f),
               result_len > 0 ? result_buf : "(no output)");

        free(audio);
    }

    free(result_buf);
    moonshine_state_free(state);
    moonshine_streaming_model_free(model);
    return 0;
}
