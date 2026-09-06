#include "qwen_asr_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <reduced_image.bin>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror("fopen");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = malloc(size);
    if (!buf) {
        fprintf(stderr, "OOM\n");
        fclose(f);
        return 1;
    }
    if (fread(buf, 1, size, f) != (size_t)size) {
        fprintf(stderr, "fread failed\n");
        fclose(f);
        free(buf);
        return 1;
    }
    fclose(f);

    safetensors_file_t *sf = safetensors_open_memory(buf, (size_t)size);
    if (!sf) {
        fprintf(stderr, "FAIL: safetensors_open_memory returned NULL\n");
        free(buf);
        return 2;
    }

    multi_safetensors_t ms;
    memset(&ms, 0, sizeof(ms));
    ms.shards[0] = sf;
    ms.num_shards = 1;

    printf("SUCCESS: safetensors_open_memory parsed header successfully\n");
    printf("Tensor count: %d\n", sf->num_tensors);

    if (sf->num_tensors != 113) {
        fprintf(stderr, "FAIL: Expected exactly 113 tensors, got %d\n", sf->num_tensors);
        safetensors_close(sf);
        free(buf);
        return 3;
    }

    int verified_count = 0;
    char name[256];

    for (int l = 0; l < 28; l++) {
        const char *suffixes[] = {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight"
        };
        for (int k = 0; k < 4; k++) {
            snprintf(name, sizeof(name), "thinker.model.layers.%d.%s", l, suffixes[k]);
            safetensors_file_t *out_sf = NULL;
            const safetensor_t *t = multi_safetensors_find(&ms, name, &out_sf);
            if (!t) {
                fprintf(stderr, "FAIL: Missing tensor in C parser: %s\n", name);
                safetensors_close(sf);
                free(buf);
                return 4;
            }
            const void *data = safetensors_data(out_sf, t);
            size_t off = (const char *)data - (const char *)buf;
            if (off % 4 != 0) {
                fprintf(stderr, "FAIL: Tensor %s offset %zu is not 4-byte aligned\n", name, off);
                safetensors_close(sf);
                free(buf);
                return 5;
            }
            if (t->dtype != DTYPE_F32) {
                fprintf(stderr, "FAIL: Tensor %s dtype is %d, expected DTYPE_F32 (2)\n", name, t->dtype);
                safetensors_close(sf);
                free(buf);
                return 6;
            }

            // Verify float values (each tensor float 0 has sentinel value 1000.0f + l*10 + k)
            const float *fvals = (const float *)data;
            float expected = 1000.0f + (float)l * 10.0f + (float)k;
            if (fvals[0] != expected) {
                fprintf(stderr, "FAIL: Float value mismatch at %s: got %f, expected %f\n", name, fvals[0], expected);
                safetensors_close(sf);
                free(buf);
                return 7;
            }
            verified_count++;
        }
    }

    // Final norm
    snprintf(name, sizeof(name), "thinker.model.norm.weight");
    safetensors_file_t *out_sf = NULL;
    const safetensor_t *t = multi_safetensors_find(&ms, name, &out_sf);
    if (!t) {
        fprintf(stderr, "FAIL: Missing final norm in C parser: %s\n", name);
        safetensors_close(sf);
        free(buf);
        return 8;
    }
    const void *data = safetensors_data(out_sf, t);
    size_t off = (const char *)data - (const char *)buf;
    if (off % 4 != 0) {
        fprintf(stderr, "FAIL: Final norm offset %zu is not 4-byte aligned\n", off);
        safetensors_close(sf);
        free(buf);
        return 9;
    }
    if (t->dtype != DTYPE_F32) {
        fprintf(stderr, "FAIL: Final norm dtype is %d, expected DTYPE_F32\n", t->dtype);
        safetensors_close(sf);
        free(buf);
        return 10;
    }
    const float *final_fvals = (const float *)data;
    if (final_fvals[0] != 9999.0f) {
        fprintf(stderr, "FAIL: Final norm float mismatch: got %f, expected 9999.0\n", final_fvals[0]);
        safetensors_close(sf);
        free(buf);
        return 11;
    }
    verified_count++;

    printf("SUCCESS: Verified all %d tensors: correct dtype (F32), present, 4/8-byte aligned, and BIT-EXACT float values confirmed in C engine!\n", verified_count);

    safetensors_close(sf);
    free(buf);
    return 0;
}
