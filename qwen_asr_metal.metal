#include <metal_stdlib>
using namespace metal;

struct Shape { uint rows, cols, seq, pad; };

kernel void q8_dequant(device const char *w [[buffer(0)]],
                       device const float *sc [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       constant Shape &p [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < p.rows * p.cols) out[i] = float(w[i]) * sc[i / 64];
}
