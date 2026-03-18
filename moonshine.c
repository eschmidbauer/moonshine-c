/*
 * moonshine.c — Zero-dependency Moonshine speech-to-text inference library.
 *
 * Supports moonshine-tiny and moonshine-base.  Model architecture is
 * auto-detected from the config embedded in the weight files.
 */

#include "moonshine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ───────────────── SIMD support ───────────────── */

#if defined(__aarch64__)
  #include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
#endif

/* ───────────────── Constants ───────────────── */

#define EOS_TOKEN   2
#define START_TOKEN 1
#define MAX_TOKENS  448
#define ROTARY_BASE 10000.0f

#define CONV1_K 127
#define CONV1_S 64
#define CONV2_K 7
#define CONV2_S 3
#define CONV3_K 3
#define CONV3_S 2

/* ───────────────── Internal types ───────────────── */

typedef struct {
    int dim, num_heads, head_dim, rotary_dim, mlp_dim, vocab_size, num_layers;
} ModelConfig;

typedef struct {
    char *name;
    int shape[8], ndim, offset, size;
} TensorInfo;

typedef struct {
    TensorInfo *tensors;
    int num_tensors;
    const uint8_t *data;
    uint8_t *file_buf;
    size_t file_size;
} WeightFile;

typedef struct {
    uint8_t **token_bytes;
    int *token_lens, num_tokens;
} Tokenizer;

typedef struct {
    const float *input_ln_w, *post_ln_w;
    const float *q_proj, *k_proj, *v_proj, *o_proj;
    const float *fc1, *fc1_bias, *fc2, *fc2_bias;
} EncoderLayer;

typedef struct {
    ModelConfig cfg;
    const float *conv1_w, *conv2_w, *conv2_b, *conv3_w, *conv3_b;
    const float *gn_w, *gn_b, *ln_w;
    EncoderLayer *layers;
} Encoder;

typedef struct {
    const float *input_ln_w, *post_attn_ln_w, *final_ln_w;
    const float *sa_q_proj, *sa_k_proj, *sa_v_proj, *sa_o_proj;
    const float *ca_q_proj, *ca_k_proj, *ca_v_proj, *ca_o_proj;
    const float *fc1, *fc1_bias, *fc2, *fc2_bias;
} DecoderLayer;

typedef struct {
    ModelConfig cfg;
    const float *embed_w, *norm_w, *proj_out;
    DecoderLayer *layers;
} Decoder;

typedef struct {
    float *self_k, *self_v, *cross_k, *cross_v;
    int self_len, cross_len;
} KVCache;

/* Model: immutable after creation, safe to share across threads. */
struct moonshine_model {
    Encoder enc;
    Decoder dec;
    Tokenizer *tok;
    WeightFile *enc_wf, *dec_wf;
    char info_str[256];
};

/* State: mutable per-request scratch, one per concurrent call. */
struct moonshine_state {
    char *last_result;
};

/* ───────────────── Weight file reader ───────────────── */

static const char *skip_ws(const char *p, const char *e) {
    while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
    return p;
}

static const char *parse_str(const char *p, const char *e, char **out) {
    if (p >= e || *p != '"') return NULL;
    p++;
    const char *s = p;
    while (p < e && *p != '"') p++;
    if (p >= e) return NULL;
    size_t len = (size_t)(p - s);
    *out = (char *)malloc(len + 1);
    memcpy(*out, s, len);
    (*out)[len] = '\0';
    return p + 1;
}

static const char *parse_int64(const char *p, const char *e, int64_t *out) {
    int64_t v = 0; int neg = 0;
    if (p < e && *p == '-') { neg = 1; p++; }
    while (p < e && *p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
    *out = neg ? -v : v;
    return p;
}

static WeightFile *load_weights(const char *path) {
#ifdef _WIN32
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "moonshine: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); size_t fsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc(fsz); fread(buf, 1, fsz, f); fclose(f);
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "moonshine: cannot open %s\n", path); return NULL; }
    struct stat st; fstat(fd, &st);
    size_t fsz = (size_t)st.st_size;
    uint8_t *buf = (uint8_t *)mmap(NULL, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (buf == MAP_FAILED) { fprintf(stderr, "moonshine: mmap failed %s\n", path); return NULL; }
#endif
    if (fsz < 12 || memcmp(buf, "MWTS", 4) != 0) {
        fprintf(stderr, "moonshine: bad magic in %s\n", path); return NULL;
    }
    uint32_t header_size;
    memcpy(&header_size, buf + 8, 4);
    const char *json = (const char *)(buf + 12);
    const char *json_end = json + header_size;
    const char *p = skip_ws(json, json_end);
    if (*p != '[') return NULL;
    p++;
    int count = 0;
    { const char *q = p; int d = 0;
      while (q < json_end) { if (*q == '{' && d++ == 0) count++; if (*q == '}') d--; q++; }
    }
    TensorInfo *tensors = (TensorInfo *)calloc(count, sizeof(TensorInfo));
    int idx = 0;
    while (idx < count) {
        p = skip_ws(p, json_end);
        if (*p == ',') p++;
        p = skip_ws(p, json_end);
        if (*p == ']') break;
        if (*p != '{') break;
        p++;
        TensorInfo *t = &tensors[idx];
        while (p < json_end && *p != '}') {
            p = skip_ws(p, json_end);
            if (*p == ',') { p++; p = skip_ws(p, json_end); }
            if (*p == '}') break;
            char *key = NULL;
            p = parse_str(p, json_end, &key);
            p = skip_ws(p, json_end);
            if (*p == ':') p++;
            p = skip_ws(p, json_end);
            if (strcmp(key, "name") == 0) {
                p = parse_str(p, json_end, &t->name);
            } else if (strcmp(key, "shape") == 0) {
                p++; t->ndim = 0;
                while (p < json_end && *p != ']') {
                    p = skip_ws(p, json_end);
                    if (*p == ',') { p++; p = skip_ws(p, json_end); }
                    if (*p == ']') break;
                    int64_t d; p = parse_int64(p, json_end, &d);
                    if (t->ndim < 8) t->shape[t->ndim++] = (int)d;
                }
                if (p < json_end) p++;
            } else if (strcmp(key, "dtype") == 0) {
                char *dt; p = parse_str(p, json_end, &dt); free(dt);
            } else if (strcmp(key, "offset") == 0) {
                int64_t v; p = parse_int64(p, json_end, &v); t->offset = (int)v;
            } else if (strcmp(key, "size") == 0) {
                int64_t v; p = parse_int64(p, json_end, &v); t->size = (int)v;
            }
            free(key);
        }
        if (p < json_end) p++;
        idx++;
    }
    WeightFile *wf = (WeightFile *)calloc(1, sizeof(WeightFile));
    wf->tensors = tensors; wf->num_tensors = idx;
    wf->data = buf + 12 + header_size;
    wf->file_buf = buf; wf->file_size = fsz;
    return wf;
}

static const TensorInfo *find_tensor(const WeightFile *wf, const char *name) {
    for (int i = 0; i < wf->num_tensors; i++)
        if (strcmp(wf->tensors[i].name, name) == 0) return &wf->tensors[i];
    return NULL;
}

static const float *get_weight(const WeightFile *wf, const char *name) {
    const TensorInfo *t = find_tensor(wf, name);
    if (!t) { fprintf(stderr, "moonshine: weight not found: %s\n", name); exit(1); }
    return (const float *)(wf->data + t->offset);
}

static int load_config(const WeightFile *wf, ModelConfig *cfg) {
    const TensorInfo *t = find_tensor(wf, "_config");
    if (!t) return -1;
    const char *js = (const char *)(wf->data + t->offset);
    const char *end = js + t->size;
    const char *p = skip_ws(js, end);
    if (*p != '{') return -1;
    p++;
    while (p < end && *p != '}') {
        p = skip_ws(p, end);
        if (*p == ',') { p++; p = skip_ws(p, end); }
        if (*p == '}') break;
        char *key = NULL;
        p = parse_str(p, end, &key);
        p = skip_ws(p, end);
        if (*p == ':') p++;
        p = skip_ws(p, end);
        int64_t val; p = parse_int64(p, end, &val);
        if      (strcmp(key, "dim") == 0)        cfg->dim = (int)val;
        else if (strcmp(key, "num_heads") == 0)   cfg->num_heads = (int)val;
        else if (strcmp(key, "head_dim") == 0)    cfg->head_dim = (int)val;
        else if (strcmp(key, "rotary_dim") == 0)  cfg->rotary_dim = (int)val;
        else if (strcmp(key, "mlp_dim") == 0)     cfg->mlp_dim = (int)val;
        else if (strcmp(key, "vocab_size") == 0)  cfg->vocab_size = (int)val;
        else if (strcmp(key, "num_layers") == 0)  cfg->num_layers = (int)val;
        free(key);
    }
    return 0;
}

static void free_weights(WeightFile *wf) {
    if (!wf) return;
    for (int i = 0; i < wf->num_tensors; i++) free(wf->tensors[i].name);
    free(wf->tensors);
#ifdef _WIN32
    free(wf->file_buf);
#else
    munmap(wf->file_buf, wf->file_size);
#endif
    free(wf);
}

/* ───────────────── Tokenizer ───────────────── */

static Tokenizer *load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "moonshine: cannot open tokenizer %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc(fsz);
    fread(data, 1, fsz, f); fclose(f);
    int count = 0; size_t off = 0;
    while (off < (size_t)fsz) {
        uint8_t b = data[off++];
        if (b == 0) { count++; continue; }
        size_t len = b < 128 ? b : (data[off++] * 128 + b - 128);
        off += len; count++;
    }
    Tokenizer *tok = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    tok->num_tokens = count;
    tok->token_bytes = (uint8_t **)calloc(count, sizeof(uint8_t *));
    tok->token_lens = (int *)calloc(count, sizeof(int));
    off = 0; int idx = 0;
    while (off < (size_t)fsz && idx < count) {
        uint8_t b = data[off++];
        if (b == 0) { tok->token_bytes[idx] = NULL; tok->token_lens[idx] = 0; idx++; continue; }
        size_t len = b < 128 ? b : (data[off++] * 128 + b - 128);
        tok->token_bytes[idx] = (uint8_t *)malloc(len);
        memcpy(tok->token_bytes[idx], data + off, len);
        tok->token_lens[idx] = (int)len; off += len; idx++;
    }
    free(data);
    return tok;
}

static char *tokens_to_text(const Tokenizer *tok, const int *tokens, int n) {
    size_t cap = 1024; char *buf = (char *)malloc(cap); size_t pos = 0;
    for (int i = 0; i < n; i++) {
        int t = tokens[i];
        if (t < 0 || t >= tok->num_tokens) continue;
        int len = tok->token_lens[t]; uint8_t *b = tok->token_bytes[t];
        if (!b || len == 0) continue;
        if (len > 2 && b[0] == '<' && b[len-1] == '>') continue;
        while (pos + len + 4 > cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
        memcpy(buf + pos, b, len); pos += len;
    }
    buf[pos] = '\0';
    char *out = (char *)malloc(pos + 1); size_t opos = 0;
    for (size_t i = 0; i < pos; ) {
        if (i+2 < pos && (uint8_t)buf[i]==0xE2 && (uint8_t)buf[i+1]==0x96 && (uint8_t)buf[i+2]==0x81)
            { out[opos++] = ' '; i += 3; } else { out[opos++] = buf[i++]; }
    }
    out[opos] = '\0'; free(buf);
    size_t s = 0; while (out[s] == ' ') s++;
    size_t e = opos; while (e > s && out[e-1] == ' ') e--;
    if (s > 0 || e < opos) { memmove(out, out+s, e-s); out[e-s] = '\0'; }
    return out;
}

static void free_tokenizer(Tokenizer *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->num_tokens; i++) free(tok->token_bytes[i]);
    free(tok->token_bytes); free(tok->token_lens); free(tok);
}

/* ───────────────── Math ops ───────────────── */

/* SIMD dot product: sum of a[i]*b[i] */
static inline float vec_dot(const float *a, const float *b, int n) {
    float sum = 0;
    int i = 0;
#if defined(__aarch64__)
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a+i),   vld1q_f32(b+i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a+i+4), vld1q_f32(b+i+4));
    }
    acc0 = vaddq_f32(acc0, acc1);
    for (; i + 4 <= n; i += 4)
        acc0 = vfmaq_f32(acc0, vld1q_f32(a+i), vld1q_f32(b+i));
    sum = vaddvq_f32(acc0);
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(2,3,0,1)));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(0,0,0,2)));
    sum = _mm_cvtss_f32(s);
#elif defined(__SSE2__)
    __m128 acc = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4)
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a+i), _mm_loadu_ps(b+i)));
    __m128 shuf = _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(2,3,0,1));
    __m128 sums = _mm_add_ps(acc, shuf);
    shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(0,0,0,2));
    sums = _mm_add_ss(sums, shuf);
    sum = _mm_cvtss_f32(sums);
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* SIMD fused multiply-add: y[i] += alpha * x[i] */
static inline void vec_fmadd(float *y, float alpha, const float *x, int n) {
    int i = 0;
#if defined(__aarch64__)
    float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y+i,   vfmaq_f32(vld1q_f32(y+i),   va, vld1q_f32(x+i)));
        vst1q_f32(y+i+4, vfmaq_f32(vld1q_f32(y+i+4), va, vld1q_f32(x+i+4)));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y+i, vfmaq_f32(vld1q_f32(y+i), va, vld1q_f32(x+i)));
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 va = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y+i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x+i), _mm256_loadu_ps(y+i)));
#elif defined(__SSE2__)
    __m128 va = _mm_set1_ps(alpha);
    for (; i + 4 <= n; i += 4) {
        __m128 vy = _mm_loadu_ps(y+i);
        __m128 vx = _mm_loadu_ps(x+i);
        _mm_storeu_ps(y+i, _mm_add_ps(vy, _mm_mul_ps(va, vx)));
    }
#endif
    for (; i < n; i++) y[i] += alpha * x[i];
}

static inline float gelu_f(float x) {
    float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

#define TILE 32

static void matmul(float *out, const float *a, const float *b, int M, int K, int N) {
    memset(out, 0, (size_t)M * N * sizeof(float));
    for (int i0 = 0; i0 < M; i0 += TILE)
      for (int k0 = 0; k0 < K; k0 += TILE)
        for (int j0 = 0; j0 < N; j0 += TILE) {
            int imax = i0+TILE < M ? i0+TILE : M;
            int kmax = k0+TILE < K ? k0+TILE : K;
            int jmax = j0+TILE < N ? j0+TILE : N;
            for (int i = i0; i < imax; i++) {
                const float *ar = a + i*K;
                float *cr = out + i*N + j0;
                int len = jmax - j0;
                for (int k = k0; k < kmax; k++)
                    vec_fmadd(cr, ar[k], b + k*N + j0, len);
            }
        }
}

static void linear(float *out, const float *x, const float *w, const float *bias,
                   int seq, int in_dim, int out_dim) {
    matmul(out, x, w, seq, in_dim, out_dim);
    if (bias)
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < out_dim; j++)
                out[i*out_dim+j] += bias[j];
}

static void layer_norm(float *out, const float *x, const float *w,
                       const float *bias, int seq, int dim) {
    for (int i = 0; i < seq; i++) {
        float mean = 0; for (int j = 0; j < dim; j++) mean += x[i*dim+j]; mean /= dim;
        float var = 0; for (int j = 0; j < dim; j++) { float d=x[i*dim+j]-mean; var+=d*d; } var /= dim;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int j = 0; j < dim; j++) {
            float v = (x[i*dim+j] - mean) * inv * w[j];
            out[i*dim+j] = bias ? v + bias[j] : v;
        }
    }
}

static void softmax(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void apply_rope(float *q, float *k, int seq, int start,
                       int H, int HD, int RD) {
    for (int s = 0; s < seq; s++) {
        int pos = start + s;
        for (int h = 0; h < H; h++) {
            float *qh = q + (s*H+h)*HD, *kh = k + (s*H+h)*HD;
            for (int i = 0; i < RD; i += 2) {
                float freq = 1.0f / powf(ROTARY_BASE, (float)i / (float)RD);
                float th = pos * freq, co = cosf(th), si = sinf(th);
                float q0=qh[i], q1=qh[i+1]; qh[i]=q0*co-q1*si; qh[i+1]=q0*si+q1*co;
                float k0=kh[i], k1=kh[i+1]; kh[i]=k0*co-k1*si; kh[i+1]=k0*si+k1*co;
            }
        }
    }
}

static int conv1d_outlen(int L, int K, int S) { return (L - K) / S + 1; }

static void conv1d(float *out, const float *in, const float *w,
                   const float *bias, int ic, int oc, int il, int k, int s) {
    int ol = conv1d_outlen(il, k, s);
    int patch = ic * k;
    /* im2col: unroll input patches into [patch, ol] matrix */
    float *col = (float *)malloc((size_t)patch * ol * sizeof(float));
    for (int c = 0; c < ic; c++)
        for (int ki = 0; ki < k; ki++) {
            float *dst = col + (c*k + ki) * ol;
            const float *src = in + c*il + ki;
            for (int p = 0; p < ol; p++) dst[p] = src[p * s];
        }
    /* out[oc, ol] = w[oc, patch] × col[patch, ol] */
    matmul(out, w, col, oc, patch, ol);
    free(col);
    if (bias)
        for (int o = 0; o < oc; o++) {
            float b = bias[o];
            float *row = out + o*ol;
            for (int p = 0; p < ol; p++) row[p] += b;
        }
}

static void groupnorm(float *out, const float *x, const float *w,
                      const float *b, int ch, int len) {
    int total = ch * len;
    double mean = 0; for (int i = 0; i < total; i++) mean += x[i]; mean /= total;
    double var = 0; for (int i = 0; i < total; i++) { double d=x[i]-mean; var+=d*d; } var /= total;
    float inv = 1.0f / sqrtf((float)var + 1e-5f);
    for (int c = 0; c < ch; c++)
        for (int l = 0; l < len; l++) {
            int i = c*len+l; out[i] = (x[i]-(float)mean)*inv*w[c]+b[c];
        }
}

/* ───────────────── Encoder ───────────────── */

static void encoder_load(Encoder *enc, const WeightFile *wf) {
    load_config(wf, &enc->cfg);
    enc->conv1_w = get_weight(wf, "conv1.weight");
    enc->conv2_w = get_weight(wf, "conv2.weight");
    enc->conv2_b = get_weight(wf, "conv2.bias");
    enc->conv3_w = get_weight(wf, "conv3.weight");
    enc->conv3_b = get_weight(wf, "conv3.bias");
    enc->gn_w = get_weight(wf, "groupnorm.weight");
    enc->gn_b = get_weight(wf, "groupnorm.bias");
    enc->ln_w = get_weight(wf, "layer_norm.weight");
    int nl = enc->cfg.num_layers;
    enc->layers = (EncoderLayer *)calloc(nl, sizeof(EncoderLayer));
    for (int i = 0; i < nl; i++) {
        char n[128]; EncoderLayer *l = &enc->layers[i];
        #define L(f,fmt) snprintf(n,sizeof(n),fmt,i); l->f = get_weight(wf,n)
        L(input_ln_w,"layers.%d.input_layernorm.weight");
        L(post_ln_w, "layers.%d.post_attention_layernorm.weight");
        L(q_proj,"layers.%d.self_attn.q_proj.weight"); L(k_proj,"layers.%d.self_attn.k_proj.weight");
        L(v_proj,"layers.%d.self_attn.v_proj.weight"); L(o_proj,"layers.%d.self_attn.o_proj.weight");
        L(fc1,"layers.%d.mlp.fc1.weight"); L(fc1_bias,"layers.%d.mlp.fc1.bias");
        L(fc2,"layers.%d.mlp.fc2.weight"); L(fc2_bias,"layers.%d.mlp.fc2.bias");
        #undef L
    }
}

static void enc_self_attn(float *out, const float *x, int seq,
                          const EncoderLayer *ly, const ModelConfig *c, float *work) {
    int D=c->dim, H=c->num_heads, HD=c->head_dim, tot=seq*D;
    float *q=work, *k=q+tot, *v=k+tot, *att=v+tot;
    linear(q,x,ly->q_proj,NULL,seq,D,D); linear(k,x,ly->k_proj,NULL,seq,D,D);
    linear(v,x,ly->v_proj,NULL,seq,D,D);
    apply_rope(q,k,seq,0,H,HD,c->rotary_dim);
    float sc = 1.0f/sqrtf((float)HD);
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < seq; i++) {
            const float *qi=q+(i*H+h)*HD;
            for (int j = 0; j < seq; j++)
                att[i*seq+j] = vec_dot(qi, k+(j*H+h)*HD, HD) * sc;
            softmax(att+i*seq,seq);
        }
        for (int i = 0; i < seq; i++) {
            float *oi=out+i*D+h*HD; memset(oi, 0, HD*sizeof(float));
            for (int j=0;j<seq;j++)
                vec_fmadd(oi, att[i*seq+j], v+(j*H+h)*HD, HD);
        }
    }
    memcpy(att,out,tot*sizeof(float));
    linear(out,att,ly->o_proj,NULL,seq,D,D);
}

static void enc_mlp(float *out, const float *x, int seq,
                    const EncoderLayer *ly, const ModelConfig *c, float *work) {
    int D=c->dim, MD=c->mlp_dim;
    linear(work,x,ly->fc1,ly->fc1_bias,seq,D,MD);
    for (int i=0;i<seq*MD;i++) work[i]=gelu_f(work[i]);
    linear(out,work,ly->fc2,ly->fc2_bias,seq,MD,D);
}

static float *encoder_forward(const Encoder *enc, const float *audio,
                              int alen, int *out_len) {
    const ModelConfig *c = &enc->cfg; int D = c->dim;
    int L1=conv1d_outlen(alen,CONV1_K,CONV1_S);
    float *c1=(float*)malloc(D*L1*sizeof(float));
    conv1d(c1,audio,enc->conv1_w,NULL,1,D,alen,CONV1_K,CONV1_S);
    for (int i=0;i<D*L1;i++) c1[i]=tanhf(c1[i]);
    float *gn=(float*)malloc(D*L1*sizeof(float));
    groupnorm(gn,c1,enc->gn_w,enc->gn_b,D,L1); free(c1);
    int L2=conv1d_outlen(L1,CONV2_K,CONV2_S);
    float *c2=(float*)malloc(2*D*L2*sizeof(float));
    conv1d(c2,gn,enc->conv2_w,enc->conv2_b,D,2*D,L1,CONV2_K,CONV2_S); free(gn);
    for (int i=0;i<2*D*L2;i++) c2[i]=gelu_f(c2[i]);
    int L3=conv1d_outlen(L2,CONV3_K,CONV3_S);
    float *c3=(float*)malloc(D*L3*sizeof(float));
    conv1d(c3,c2,enc->conv3_w,enc->conv3_b,2*D,D,L2,CONV3_K,CONV3_S); free(c2);
    for (int i=0;i<D*L3;i++) c3[i]=gelu_f(c3[i]);
    int elen=L3; float *x=(float*)malloc(elen*D*sizeof(float));
    for (int ch=0;ch<D;ch++) for (int l=0;l<elen;l++) x[l*D+ch]=c3[ch*elen+l];
    free(c3);
    size_t wsz=(size_t)elen*D*3+(size_t)elen*elen;
    size_t msz=(size_t)elen*c->mlp_dim; if (msz>wsz) wsz=msz;
    float *work=(float*)malloc(wsz*sizeof(float)), *tmp=(float*)malloc(elen*D*sizeof(float));
    for (int i=0;i<c->num_layers;i++) {
        const EncoderLayer *ly=&enc->layers[i];
        layer_norm(tmp,x,ly->input_ln_w,NULL,elen,D);
        float *ao=(float*)malloc(elen*D*sizeof(float));
        enc_self_attn(ao,tmp,elen,ly,c,work);
        for (int j=0;j<elen*D;j++) x[j]+=ao[j]; free(ao);
        layer_norm(tmp,x,ly->post_ln_w,NULL,elen,D);
        float *mo=(float*)malloc(elen*D*sizeof(float));
        enc_mlp(mo,tmp,elen,ly,c,work);
        for (int j=0;j<elen*D;j++) x[j]+=mo[j]; free(mo);
    }
    layer_norm(tmp,x,enc->ln_w,NULL,elen,D);
    memcpy(x,tmp,elen*D*sizeof(float));
    free(work); free(tmp); *out_len=elen; return x;
}

/* ───────────────── Decoder ───────────────── */

static void decoder_load(Decoder *dec, const WeightFile *wf) {
    load_config(wf, &dec->cfg);
    dec->embed_w  = get_weight(wf, "embed_tokens.weight");
    dec->norm_w   = get_weight(wf, "norm.weight");
    dec->proj_out = get_weight(wf, "proj_out.weight");
    int nl = dec->cfg.num_layers;
    dec->layers = (DecoderLayer *)calloc(nl, sizeof(DecoderLayer));
    for (int i = 0; i < nl; i++) {
        char n[128]; DecoderLayer *l = &dec->layers[i];
        #define L(f,fmt) snprintf(n,sizeof(n),fmt,i); l->f = get_weight(wf,n)
        L(input_ln_w,"layers.%d.input_layernorm.weight");
        L(post_attn_ln_w,"layers.%d.post_attention_layernorm.weight");
        L(final_ln_w,"layers.%d.final_layernorm.weight");
        L(sa_q_proj,"layers.%d.self_attn.q_proj.weight"); L(sa_k_proj,"layers.%d.self_attn.k_proj.weight");
        L(sa_v_proj,"layers.%d.self_attn.v_proj.weight"); L(sa_o_proj,"layers.%d.self_attn.o_proj.weight");
        L(ca_q_proj,"layers.%d.encoder_attn.q_proj.weight"); L(ca_k_proj,"layers.%d.encoder_attn.k_proj.weight");
        L(ca_v_proj,"layers.%d.encoder_attn.v_proj.weight"); L(ca_o_proj,"layers.%d.encoder_attn.o_proj.weight");
        L(fc1,"layers.%d.mlp.fc1.weight"); L(fc1_bias,"layers.%d.mlp.fc1.bias");
        L(fc2,"layers.%d.mlp.fc2.weight"); L(fc2_bias,"layers.%d.mlp.fc2.bias");
        #undef L
    }
}

static KVCache *kv_alloc(const ModelConfig *c, int elen) {
    int D=c->dim;
    KVCache *kv=(KVCache*)calloc(c->num_layers,sizeof(KVCache));
    for (int i=0;i<c->num_layers;i++) {
        kv[i].self_k=(float*)calloc(MAX_TOKENS*D,sizeof(float));
        kv[i].self_v=(float*)calloc(MAX_TOKENS*D,sizeof(float));
        kv[i].cross_k=(float*)calloc(elen*D,sizeof(float));
        kv[i].cross_v=(float*)calloc(elen*D,sizeof(float));
    }
    return kv;
}

static void kv_free(KVCache *kv, int nl) {
    for (int i=0;i<nl;i++) { free(kv[i].self_k); free(kv[i].self_v); free(kv[i].cross_k); free(kv[i].cross_v); }
    free(kv);
}

static void dec_self_attn(float *out, const float *x, int pos,
                          const DecoderLayer *ly, const ModelConfig *c, KVCache *kv) {
    int D=c->dim, H=c->num_heads, HD=c->head_dim;
    float *q=(float*)malloc(D*sizeof(float)), *k=(float*)malloc(D*sizeof(float)), *v=(float*)malloc(D*sizeof(float));
    linear(q,x,ly->sa_q_proj,NULL,1,D,D); linear(k,x,ly->sa_k_proj,NULL,1,D,D); linear(v,x,ly->sa_v_proj,NULL,1,D,D);
    apply_rope(q,k,1,pos,H,HD,c->rotary_dim);
    memcpy(kv->self_k+pos*D,k,D*sizeof(float)); memcpy(kv->self_v+pos*D,v,D*sizeof(float));
    kv->self_len=pos+1; int kvl=kv->self_len;
    float sc=1.0f/sqrtf((float)HD); float *ab=(float*)malloc(kvl*sizeof(float));
    for (int h=0;h<H;h++) {
        float *qh=q+h*HD;
        for (int j=0;j<kvl;j++)
            ab[j] = vec_dot(qh, kv->self_k+j*D+h*HD, HD) * sc;
        softmax(ab,kvl);
        float *oh=out+h*HD; memset(oh, 0, HD*sizeof(float));
        for (int j=0;j<kvl;j++)
            vec_fmadd(oh, ab[j], kv->self_v+j*D+h*HD, HD);
    }
    free(ab); float *tmp=(float*)malloc(D*sizeof(float)); memcpy(tmp,out,D*sizeof(float));
    linear(out,tmp,ly->sa_o_proj,NULL,1,D,D); free(tmp); free(q); free(k); free(v);
}

static void dec_cross_attn(float *out, const float *x, const float *enc, int elen,
                           const DecoderLayer *ly, const ModelConfig *c, KVCache *kv) {
    int D=c->dim, H=c->num_heads, HD=c->head_dim;
    if (kv->cross_len==0) {
        linear(kv->cross_k,enc,ly->ca_k_proj,NULL,elen,D,D);
        linear(kv->cross_v,enc,ly->ca_v_proj,NULL,elen,D,D);
        kv->cross_len=elen;
    }
    float *q=(float*)malloc(D*sizeof(float)); linear(q,x,ly->ca_q_proj,NULL,1,D,D);
    float sc=1.0f/sqrtf((float)HD); float *ab=(float*)malloc(elen*sizeof(float));
    for (int h=0;h<H;h++) {
        float *qh=q+h*HD;
        for (int j=0;j<elen;j++)
            ab[j] = vec_dot(qh, kv->cross_k+j*D+h*HD, HD) * sc;
        softmax(ab,elen);
        float *oh=out+h*HD; memset(oh, 0, HD*sizeof(float));
        for (int j=0;j<elen;j++)
            vec_fmadd(oh, ab[j], kv->cross_v+j*D+h*HD, HD);
    }
    free(ab); float *tmp=(float*)malloc(D*sizeof(float)); memcpy(tmp,out,D*sizeof(float));
    linear(out,tmp,ly->ca_o_proj,NULL,1,D,D); free(tmp); free(q);
}

static void dec_mlp(float *out, const float *x, const DecoderLayer *ly, const ModelConfig *c) {
    int D=c->dim, MD=c->mlp_dim;
    float *h=(float*)malloc(2*MD*sizeof(float));
    linear(h,x,ly->fc1,ly->fc1_bias,1,D,2*MD);
    float *mb=(float*)malloc(MD*sizeof(float));
    for (int i=0;i<MD;i++) mb[i]=gelu_f(h[MD+i])*h[i];
    linear(out,mb,ly->fc2,ly->fc2_bias,1,MD,D); free(h); free(mb);
}

static float *decoder_step(const Decoder *dec, int token, int pos,
                           const float *enc_h, int elen, KVCache *kv) {
    const ModelConfig *c=&dec->cfg; int D=c->dim;
    float *x=(float*)malloc(D*sizeof(float)), *tmp=(float*)malloc(D*sizeof(float)), *att=(float*)malloc(D*sizeof(float));
    memcpy(x, dec->embed_w+token*D, D*sizeof(float));
    for (int i=0;i<c->num_layers;i++) {
        const DecoderLayer *ly=&dec->layers[i];
        layer_norm(tmp,x,ly->input_ln_w,NULL,1,D);
        dec_self_attn(att,tmp,pos,ly,c,&kv[i]); for (int j=0;j<D;j++) x[j]+=att[j];
        layer_norm(tmp,x,ly->post_attn_ln_w,NULL,1,D);
        dec_cross_attn(att,tmp,enc_h,elen,ly,c,&kv[i]); for (int j=0;j<D;j++) x[j]+=att[j];
        layer_norm(tmp,x,ly->final_ln_w,NULL,1,D);
        dec_mlp(att,tmp,ly,c); for (int j=0;j<D;j++) x[j]+=att[j];
    }
    layer_norm(tmp,x,dec->norm_w,NULL,1,D);
    float *logits=(float*)malloc(c->vocab_size*sizeof(float));
    linear(logits,tmp,dec->proj_out,NULL,1,D,c->vocab_size);
    free(x); free(tmp); free(att); return logits;
}

/* ───────────────── Public API ───────────────── */

moonshine_model *moonshine_model_load(const char *model_dir) {
    char path[512];

    snprintf(path, sizeof(path), "%s/encoder.bin", model_dir);
    WeightFile *enc_wf = load_weights(path);
    if (!enc_wf) return NULL;

    snprintf(path, sizeof(path), "%s/decoder.bin", model_dir);
    WeightFile *dec_wf = load_weights(path);
    if (!dec_wf) { free_weights(enc_wf); return NULL; }

    snprintf(path, sizeof(path), "%s/tokenizer.bin", model_dir);
    Tokenizer *tok = load_tokenizer(path);
    if (!tok) { free_weights(enc_wf); free_weights(dec_wf); return NULL; }

    moonshine_model *m = (moonshine_model *)calloc(1, sizeof(moonshine_model));
    m->enc_wf = enc_wf;
    m->dec_wf = dec_wf;
    m->tok = tok;

    encoder_load(&m->enc, enc_wf);
    decoder_load(&m->dec, dec_wf);

    snprintf(m->info_str, sizeof(m->info_str),
             "dim=%d heads=%d layers=%d/%d mlp=%d vocab=%d",
             m->enc.cfg.dim, m->enc.cfg.num_heads,
             m->enc.cfg.num_layers, m->dec.cfg.num_layers,
             m->enc.cfg.mlp_dim, m->enc.cfg.vocab_size);

    return m;
}

void moonshine_model_free(moonshine_model *m) {
    if (!m) return;
    free(m->enc.layers);
    free(m->dec.layers);
    free_weights(m->enc_wf);
    free_weights(m->dec_wf);
    free_tokenizer(m->tok);
    free(m);
}

const char *moonshine_model_info(const moonshine_model *m) {
    return m ? m->info_str : NULL;
}

moonshine_state *moonshine_state_create(const moonshine_model *model) {
    (void)model;
    moonshine_state *s = (moonshine_state *)calloc(1, sizeof(moonshine_state));
    return s;
}

void moonshine_state_free(moonshine_state *s) {
    if (!s) return;
    free(s->last_result);
    free(s);
}

const char *moonshine_transcribe(const moonshine_model *model,
                                 moonshine_state *state,
                                 const float *pcm_f32,
                                 int num_samples) {
    if (!model || !state || !pcm_f32 || num_samples <= 0) return NULL;

    free(state->last_result);
    state->last_result = NULL;

    int enc_len;
    float *enc_hidden = encoder_forward(&model->enc, pcm_f32, num_samples, &enc_len);
    if (!enc_hidden) return NULL;

    KVCache *caches = kv_alloc(&model->dec.cfg, enc_len);

    int tokens[MAX_TOKENS];
    int num_tokens = 0;
    tokens[num_tokens++] = START_TOKEN;

    float duration = num_samples / 16000.0f;
    int max_len = (int)ceilf(duration * 6.5f);
    if (max_len > MAX_TOKENS - 1) max_len = MAX_TOKENS - 1;

    for (int step = 0; step < max_len; step++) {
        float *logits = decoder_step(&model->dec, tokens[num_tokens-1], step,
                                     enc_hidden, enc_len, caches);
        int best = 0;
        for (int i = 1; i < model->dec.cfg.vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        free(logits);
        tokens[num_tokens++] = best;
        if (best == EOS_TOKEN) break;
    }

    state->last_result = tokens_to_text(model->tok, tokens, num_tokens);

    free(enc_hidden);
    kv_free(caches, model->dec.cfg.num_layers);

    return state->last_result;
}
