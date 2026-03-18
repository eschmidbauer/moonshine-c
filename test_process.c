/*
 * test_process.c — Concurrent transcription using the moonshine library.
 *
 * Loads a model once, then transcribes multiple WAV files in parallel
 * using one thread per file.
 *
 * Usage:
 *   ./test_process <model_dir> <audio1.wav> [audio2.wav] ...
 */

#include "moonshine.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* ───────────────── Simple WAV loader ───────────────── */

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
            data_size = chunk_size;
            break;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (bits_per_sample != 16 || channels != 1) {
        fprintf(stderr, "WAV must be 16-bit mono PCM (got %d-bit, %d ch)\n",
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

/* ───────────────── Per-thread work ───────────────── */

typedef struct {
    const moonshine_model *model;
    const char *wav_path;
    int index;
    char *result;
    double elapsed_ms;
} WorkItem;

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void *transcribe_thread(void *arg) {
    WorkItem *item = (WorkItem *)arg;
    double t0 = now_ms();

    int num_samples;
    float *audio = load_wav(item->wav_path, &num_samples);
    if (!audio) {
        item->result = strdup("(failed to load audio)");
        item->elapsed_ms = now_ms() - t0;
        return NULL;
    }

    moonshine_state *state = moonshine_state_create(item->model);
    const char *text = moonshine_transcribe(item->model, state, audio, num_samples);
    item->result = strdup(text ? text : "(transcription failed)");
    item->elapsed_ms = now_ms() - t0;

    moonshine_state_free(state);
    free(audio);
    return NULL;
}

/* ───────────────── Main ───────────────── */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_dir> <audio1.wav> [audio2.wav] ...\n", argv[0]);
        return 1;
    }

    const char *model_dir = argv[1];
    int num_files = argc - 2;

    /* Load model once (shared across all threads) */
    printf("Loading model from %s ...\n", model_dir);
    double t0 = now_ms();
    moonshine_model *model = moonshine_model_load(model_dir);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model: %s (loaded in %.0f ms)\n\n", moonshine_model_info(model), now_ms() - t0);

    /* Set up work items */
    WorkItem *items = (WorkItem *)calloc(num_files, sizeof(WorkItem));
    for (int i = 0; i < num_files; i++) {
        items[i].model = model;
        items[i].wav_path = argv[i + 2];
        items[i].index = i;
    }

    /* Launch threads */
    printf("Transcribing %d file%s concurrently...\n\n", num_files, num_files > 1 ? "s" : "");
    double wall_start = now_ms();

    pthread_t *threads = (pthread_t *)malloc(num_files * sizeof(pthread_t));
    for (int i = 0; i < num_files; i++)
        pthread_create(&threads[i], NULL, transcribe_thread, &items[i]);

    /* Wait for all to finish */
    for (int i = 0; i < num_files; i++)
        pthread_join(threads[i], NULL);

    double wall_elapsed = now_ms() - wall_start;

    /* Print results */
    for (int i = 0; i < num_files; i++) {
        printf("[%d] %s (%.0f ms)\n    %s\n\n",
               items[i].index, items[i].wav_path,
               items[i].elapsed_ms, items[i].result);
        free(items[i].result);
    }

    printf("Wall time: %.0f ms (%d files)\n", wall_elapsed, num_files);

    /* Cleanup */
    free(threads);
    free(items);
    moonshine_model_free(model);
    return 0;
}
