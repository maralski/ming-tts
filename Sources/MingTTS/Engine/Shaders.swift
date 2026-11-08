// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Metal shader source, compiled at load time so the kernels work inside an .app bundle
// without a resource bundle. Weights are bf16, activations fp32.
enum ShaderSource {
    static let all = common + gemv + gemm + norm + rope + attention + dit + vae

    static let common = #"""
    #include <metal_stdlib>
    using namespace metal;

    inline float bf16_to_float(ushort v) { return as_type<float>(uint(v) << 16); }

    inline void unpack8(uint4 v, thread float* out) {
        out[0] = bf16_to_float(ushort(v.x & 0xFFFF)); out[1] = bf16_to_float(ushort(v.x >> 16));
        out[2] = bf16_to_float(ushort(v.y & 0xFFFF)); out[3] = bf16_to_float(ushort(v.y >> 16));
        out[4] = bf16_to_float(ushort(v.z & 0xFFFF)); out[5] = bf16_to_float(ushort(v.z >> 16));
        out[6] = bf16_to_float(ushort(v.w & 0xFFFF)); out[7] = bf16_to_float(ushort(v.w >> 16));
    }

    // Metal's tanh misbehaves past |u| ~ 44 (returns 0 or NaN); tanh(15) is already 1.0f, so the
    // clamp changes nothing representable.
    inline float gelu_tanh(float x) {
        const float k = 0.7978845608028654f;
        const float u = clamp(k * (x + 0.044715f * x * x * x), -15.0f, 15.0f);
        return 0.5f * x * (1.0f + tanh(u));
    }

    inline float silu(float x) { return x / (1.0f + exp(-x)); }

    """#

    // Y[M,N] = X[M,K] * W[N,K]^T with an epilogue, for M <= 2 rows (the LLM and aggregator decode).
    // One simdgroup per output column, lanes stride K by 8 bf16 (16 bytes), MR rows of X share every
    // weight load. The matrix-unit kernel below wastes 8/M of its multiply-accumulates at tiny M, which
    // costs more than these M scalar FMAs per weight. Grid y walks M in MR-row chunks; rows past M are
    // never read.
    static let gemv = #"""
    struct GemvParams { uint M; uint N; uint K; uint ldx; uint ldy; uint flags; float eps; };
    constant uint GEMV_BIAS = 1;
    constant uint GEMV_ACCUM = 2;
    constant uint GEMV_GELU = 4;
    constant uint GEMV_SILU_GATE = 8;
    constant uint GEMV_SILU = 16;
    constant int GEMV_SG = 4;

    template <int MR>
    kernel void gemv_kernel(device const float* X [[buffer(0)]],
                            device const ushort* W [[buffer(1)]],
                            device const ushort* W2 [[buffer(2)]],
                            device const ushort* bias [[buffer(3)]],
                            device float* Y [[buffer(4)]],
                            constant GemvParams& p [[buffer(5)]],
                            uint2 tgid [[threadgroup_position_in_grid]],
                            uint sgid [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
        const uint n = tgid.x * GEMV_SG + sgid;
        if (n >= p.N) return;
        const uint row0 = tgid.y * MR;
        const uint rows = min(uint(MR), p.M - row0);
        device const float* x = X + row0 * p.ldx;
        device const ushort* w = W + n * p.K;
        device const ushort* w2 = W2 + n * p.K;
        const bool gated = (p.flags & GEMV_SILU_GATE) != 0;
        float acc[MR], acc2[MR];
        #pragma clang loop unroll(full)
        for (int m = 0; m < MR; m++) { acc[m] = 0.0f; acc2[m] = 0.0f; }
        for (uint k = lane * 8; k < p.K; k += 256) {
            float wf[8];
            unpack8(*(device const uint4*)(w + k), wf);
            float wg[8];
            if (gated) unpack8(*(device const uint4*)(w2 + k), wg);
            #pragma clang loop unroll(full)
            for (int m = 0; m < MR; m++) {
                if (uint(m) >= rows) continue;
                const float4 a = *(device const float4*)(x + m * p.ldx + k);
                const float4 b = *(device const float4*)(x + m * p.ldx + k + 4);
                acc[m] += a.x * wf[0] + a.y * wf[1] + a.z * wf[2] + a.w * wf[3]
                        + b.x * wf[4] + b.y * wf[5] + b.z * wf[6] + b.w * wf[7];
                if (gated) {
                    acc2[m] += a.x * wg[0] + a.y * wg[1] + a.z * wg[2] + a.w * wg[3]
                             + b.x * wg[4] + b.y * wg[5] + b.z * wg[6] + b.w * wg[7];
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int m = 0; m < MR; m++) {
            acc[m] = simd_sum(acc[m]);
            if (gated) acc2[m] = simd_sum(acc2[m]);
        }
        if (lane != 0) return;
        const float bval = (p.flags & GEMV_BIAS) ? bf16_to_float(bias[n]) : 0.0f;
        #pragma clang loop unroll(full)
        for (int m = 0; m < MR; m++) {
            if (uint(m) < rows) {
                float v = acc[m] + bval;
                if (gated) v = silu(v) * acc2[m];
                if (p.flags & GEMV_GELU) v = gelu_tanh(v);
                if (p.flags & GEMV_SILU) v = silu(v);
                device float* y = Y + (row0 + m) * p.ldy + n;
                if (p.flags & GEMV_ACCUM) v += *y;
                *y = v;
            }
        }
    }

    template [[host_name("gemv_1")]] kernel void gemv_kernel<1>(device const float*, device const ushort*, device const ushort*, device const ushort*, device float*, constant GemvParams&, uint2, uint, uint);
    template [[host_name("gemv_2")]] kernel void gemv_kernel<2>(device const float*, device const ushort*, device const ushort*, device const ushort*, device float*, constant GemvParams&, uint2, uint, uint);

    // Matrix-unit GEMV for M <= 8 rows per chunk: Y^T[n, m] = sum_k W[n, k] X[m, k] as 8x8 simdgroup
    // matrix products, so a lane issues one 16-byte weight load and four multiply-accumulates per
    // 512-byte weight tile instead of M scalar FMAs per weight. A threadgroup is KS simdgroups over the
    // same 8 NC output columns, each owning K/KS of the reduction; partial tiles meet in threadgroup memory.
    // Inside a 32-wide k step the tiles use a permuted k order, tile u, tile column j <-> k = k0 + 4 j + u,
    // applied identically to A and B: a lane's two weight columns (c, c + 1) over the four tiles are
    // the 8 contiguous bf16 at k0 + 4 c, and its B row j over the four tiles is one float4 of X at k0 + 4 j.
    // thread_elements() lane layout: row = ((lane >> 1) & 3) | ((lane >> 4) << 2),
    // columns (c, c + 1) with c = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2); KernelTests pins it.
    // thread_elements() is typed as the whole matrix; only elements 0 and 1 exist per lane.
    inline void g8_set(thread simdgroup_float8x8& m, float2 v) {
        m.thread_elements()[0] = v.x;
        m.thread_elements()[1] = v.y;
    }

    inline float2 g8_bf16x2(uint v) { return float2(bf16_to_float(ushort(v & 0xFFFF)), bf16_to_float(ushort(v >> 16))); }

    // A's two elements for tile u out of the lane's 8 weights (k0 + 4 c .. + 7): offsets u and u + 4.
    inline float2 g8_a(uint4 wv, int u) {
        const uint lo = (u & 2) ? wv.y : wv.x, hi = (u & 2) ? wv.w : wv.z;
        const uint shift = (u & 1) * 16;
        return float2(bf16_to_float(ushort(lo >> shift)), bf16_to_float(ushort(hi >> shift)));
    }

    struct G8Lane {
        device const float* x0;
        device const float* x1;
        bool has0, has1, gated;
        uint r, c;
    };

    // One 32-k step over NC column tiles: the lane's B values, one float4 of X per row, are loaded once
    // and multiplied into every tile's weights. wrow is the lane's weight row in tile 0; tile t's row is
    // 8 t rows further on. Tile t accumulates into C[t][u % NA] so four chains stay in flight at any NC.
    template <int NC>
    inline void g8_step32(G8Lane L, device const ushort* wrow, device const ushort* w2row, uint K, uint k0,
                          thread simdgroup_float8x8 (&C)[NC][4 / NC], thread simdgroup_float8x8 (&C2)[NC][4 / NC]) {
        constexpr int NA = 4 / NC;
        const uint kx = k0 + 4 * L.r;
        float4 b0 = 0.0f, b1 = 0.0f;
        if (L.has0) b0 = *(device const float4*)(L.x0 + kx);
        if (L.has1) b1 = *(device const float4*)(L.x1 + kx);
        simdgroup_float8x8 B[4];
        #pragma clang loop unroll(full)
        for (int u = 0; u < 4; u++) g8_set(B[u], float2(b0[u], b1[u]));
        #pragma clang loop unroll(full)
        for (int t = 0; t < NC; t++) {
            const uint off = t * 8 * K + k0 + 4 * L.c;
            const uint4 wv = *(device const uint4*)(wrow + off);
            uint4 wv2 = 0;
            if (L.gated) wv2 = *(device const uint4*)(w2row + off);
            #pragma clang loop unroll(full)
            for (int u = 0; u < 4; u++) {
                simdgroup_float8x8 A;
                g8_set(A, g8_a(wv, u));
                simdgroup_multiply_accumulate(C[t][u % NA], A, B[u], C[t][u % NA]);
                if (L.gated) {
                    g8_set(A, g8_a(wv2, u));
                    simdgroup_multiply_accumulate(C2[t][u % NA], A, B[u], C2[t][u % NA]);
                }
            }
        }
    }

    // One tile in natural k order, weight rows guarded against N (the tail of K, and column tiles past N).
    inline void g8_tile8(G8Lane L, device const ushort* W, device const ushort* W2, uint n, uint N, uint K, uint k,
                         thread simdgroup_float8x8& C, thread simdgroup_float8x8& C2) {
        const float2 b = float2(L.has0 ? L.x0[k + L.r] : 0.0f, L.has1 ? L.x1[k + L.r] : 0.0f);
        simdgroup_float8x8 A, B;
        g8_set(B, b);
        g8_set(A, n < N ? g8_bf16x2(*(device const uint*)(W + n * K + k + L.c)) : float2(0.0f));
        simdgroup_multiply_accumulate(C, A, B, C);
        if (L.gated) {
            g8_set(A, n < N ? g8_bf16x2(*(device const uint*)(W2 + n * K + k + L.c)) : float2(0.0f));
            simdgroup_multiply_accumulate(C2, A, B, C2);
        }
    }

    // KS simdgroups split K; each walks NC column tiles (8 NC output columns per threadgroup) so one X
    // load feeds NC weight loads. The host passes NC > 1 only when N % (8 NC) == 0; the guarded tail
    // path below is for NC == 1. Simdgroup t sums the partials of tile t.
    template <int KS, int NC>
    kernel void gemm8_kernel(device const float* X [[buffer(0)]],
                             device const ushort* W [[buffer(1)]],
                             device const ushort* W2 [[buffer(2)]],
                             device const ushort* bias [[buffer(3)]],
                             device float* Y [[buffer(4)]],
                             constant GemvParams& p [[buffer(5)]],
                             uint2 tgid [[threadgroup_position_in_grid]],
                             uint sgid [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
        constexpr int NA = 4 / NC;
        threadgroup float part[KS][NC][2][64];
        const uint n0 = tgid.x * 8 * NC;
        const uint row0 = tgid.y * 8;
        const uint rows = min(8u, p.M - row0);
        G8Lane L;
        L.r = ((lane >> 1) & 3) | ((lane >> 4) << 2);
        L.c = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2);
        L.gated = (p.flags & GEMV_SILU_GATE) != 0;
        L.has0 = L.c < rows;
        L.has1 = L.c + 1 < rows;
        device const float* x = X + row0 * p.ldx;
        L.x0 = x + L.c * p.ldx;
        L.x1 = x + (L.c + 1) * p.ldx;
        const uint kLen = p.K / KS, kBegin = sgid * kLen, kEnd = kBegin + kLen;

        simdgroup_float8x8 C[NC][NA], C2[NC][NA];
        #pragma clang loop unroll(full)
        for (int t = 0; t < NC; t++) {
            #pragma clang loop unroll(full)
            for (int a = 0; a < NA; a++) { C[t][a] = simdgroup_float8x8(0.0f); C2[t][a] = simdgroup_float8x8(0.0f); }
        }
        const uint n = n0 + L.r;
        uint k = kBegin;
        if (n0 + 8 * NC <= p.N) {
            device const ushort* wrow = W + n * p.K;
            device const ushort* w2row = W2 + n * p.K;
            for (; k + 64 <= kEnd; k += 64) {
                g8_step32<NC>(L, wrow, w2row, p.K, k, C, C2);
                g8_step32<NC>(L, wrow, w2row, p.K, k + 32, C, C2);
            }
            for (; k + 32 <= kEnd; k += 32) g8_step32<NC>(L, wrow, w2row, p.K, k, C, C2);
        }
        for (; k < kEnd; k += 8) {
            #pragma clang loop unroll(full)
            for (int t = 0; t < NC; t++) g8_tile8(L, W, W2, n + 8 * t, p.N, p.K, k, C[t][0], C2[t][0]);
        }

        #pragma clang loop unroll(full)
        for (int t = 0; t < NC; t++) {
            float2 acc = float2(0.0f), acc2 = float2(0.0f);
            #pragma clang loop unroll(full)
            for (int a = 0; a < NA; a++) {
                acc += float2(C[t][a].thread_elements()[0], C[t][a].thread_elements()[1]);
                acc2 += float2(C2[t][a].thread_elements()[0], C2[t][a].thread_elements()[1]);
            }
            part[sgid][t][0][lane * 2] = acc.x;
            part[sgid][t][0][lane * 2 + 1] = acc.y;
            if (L.gated) {
                part[sgid][t][1][lane * 2] = acc2.x;
                part[sgid][t][1][lane * 2 + 1] = acc2.y;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid >= uint(NC)) return;
        const uint t = sgid, nt = n + 8 * t;
        if (nt >= p.N) return;

        float2 v = float2(0.0f), v2 = float2(0.0f);
        #pragma clang loop unroll(full)
        for (int s = 0; s < KS; s++) {
            v += float2(part[s][t][0][lane * 2], part[s][t][0][lane * 2 + 1]);
            if (L.gated) v2 += float2(part[s][t][1][lane * 2], part[s][t][1][lane * 2 + 1]);
        }
        v += (p.flags & GEMV_BIAS) ? bf16_to_float(bias[nt]) : 0.0f;
        if (L.gated) v = float2(silu(v.x), silu(v.y)) * v2;
        if (p.flags & GEMV_GELU) v = float2(gelu_tanh(v.x), gelu_tanh(v.y));
        if (p.flags & GEMV_SILU) v = float2(silu(v.x), silu(v.y));
        if (L.has0) {
            device float* y = Y + (row0 + L.c) * p.ldy + nt;
            *y = (p.flags & GEMV_ACCUM) ? *y + v.x : v.x;
        }
        if (L.has1) {
            device float* y = Y + (row0 + L.c + 1) * p.ldy + nt;
            *y = (p.flags & GEMV_ACCUM) ? *y + v.y : v.y;
        }
    }

    #define GEMM8(KS, NC) template [[host_name("gemm8_" #KS "_" #NC)]] kernel void gemm8_kernel<KS, NC>(device const float*, device const ushort*, device const ushort*, device const ushort*, device float*, constant GemvParams&, uint2, uint, uint);
    GEMM8(4, 1) GEMM8(4, 2) GEMM8(4, 4)
    GEMM8(8, 1) GEMM8(8, 2) GEMM8(8, 4)
    GEMM8(16, 1) GEMM8(16, 2)
    #undef GEMM8

    // Dumps which two matrix elements each lane of a simdgroup_float8x8 holds: out[lane*2 + i] = row*8 + col.
    kernel void simd_matrix_layout(device const float* M [[buffer(0)]],
                                   device float* out [[buffer(1)]],
                                   uint lane [[thread_index_in_simdgroup]]) {
        simdgroup_float8x8 A;
        simdgroup_load(A, M, 8);
        out[lane * 2] = A.thread_elements()[0];
        out[lane * 2 + 1] = A.thread_elements()[1];
    }

    """#

    // Tiled GEMM for prefill: Y[M,N] = X[M,K] W[N,K]^T with the GEMV's flags and buffer layout.
    // A BM x BN output tile per threadgroup, SG simdgroups each owning BN/SG columns, fp32
    // simdgroup matrices accumulating over K in BK slices staged through threadgroup memory.
    // Rows past M are clamped on load and skipped on store; the caller guarantees N % BN == 0
    // and K % BK == 0. With GEMV_SILU_GATE the W2 tile is accumulated too and the epilogue is
    // silu(x W^T) * (x W2^T), like the GEMV.
    static let gemm = #"""
    template <int BM, int BN, int BK, int SG>
    kernel void gemm_kernel(device const float* X [[buffer(0)]],
                            device const ushort* W [[buffer(1)]],
                            device const ushort* W2 [[buffer(2)]],
                            device const ushort* bias [[buffer(3)]],
                            device float* Y [[buffer(4)]],
                            constant GemvParams& p [[buffer(5)]],
                            uint2 tgid [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sgid [[simdgroup_index_in_threadgroup]]) {
        constexpr int LD = BK + 4;
        constexpr int THREADS = SG * 32;
        constexpr int SN = BN / SG;
        constexpr int TM = BM / 8, TN = SN / 8;
        constexpr int OUT_LD = BN + 4;
        threadgroup float stage[(BM + 2 * BN) * LD];
        threadgroup float* Xs = stage;
        threadgroup float* Ws = stage + BM * LD;
        threadgroup float* Ws2 = Ws + BN * LD;
        const uint m0 = tgid.y * BM, n0 = tgid.x * BN;
        const bool gated = (p.flags & GEMV_SILU_GATE) != 0;

        simdgroup_float8x8 acc[TM][TN];
        simdgroup_float8x8 acc2[TM][TN];
        #pragma clang loop unroll(full)
        for (int i = 0; i < TM; i++) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < TN; j++) { acc[i][j] = simdgroup_float8x8(0.0f); acc2[i][j] = simdgroup_float8x8(0.0f); }
        }

        for (uint k0 = 0; k0 < p.K; k0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint u = tid; u < BM * BK / 4; u += THREADS) {
                const uint row = u / (BK / 4), c4 = u % (BK / 4);
                const uint m = min(m0 + row, p.M - 1);
                const float4 v = *(device const float4*)(X + m * p.ldx + k0 + c4 * 4);
                threadgroup float* dst = Xs + row * LD + c4 * 4;
                dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
            }
            for (uint u = tid; u < BN * BK / 8; u += THREADS) {
                const uint row = u / (BK / 8), c8 = u % (BK / 8);
                const uint off = (n0 + row) * p.K + k0 + c8 * 8;
                float vals[8];
                unpack8(*(device const uint4*)(W + off), vals);
                threadgroup float* dst = Ws + row * LD + c8 * 8;
                for (int i = 0; i < 8; i++) dst[i] = vals[i];
                if (gated) {
                    unpack8(*(device const uint4*)(W2 + off), vals);
                    threadgroup float* dst2 = Ws2 + row * LD + c8 * 8;
                    for (int i = 0; i < 8; i++) dst2[i] = vals[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (int kk = 0; kk < BK; kk += 8) {
                simdgroup_float8x8 a[TM];
                simdgroup_float8x8 b[TN];
                #pragma clang loop unroll(full)
                for (int i = 0; i < TM; i++) simdgroup_load(a[i], Xs + i * 8 * LD + kk, LD);
                #pragma clang loop unroll(full)
                for (int j = 0; j < TN; j++) simdgroup_load(b[j], Ws + (sgid * SN + j * 8) * LD + kk, LD, ulong2(0, 0), true);
                #pragma clang loop unroll(full)
                for (int i = 0; i < TM; i++) {
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < TN; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
                if (gated) {
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < TN; j++) simdgroup_load(b[j], Ws2 + (sgid * SN + j * 8) * LD + kk, LD, ulong2(0, 0), true);
                    #pragma clang loop unroll(full)
                    for (int i = 0; i < TM; i++) {
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < TN; j++) simdgroup_multiply_accumulate(acc2[i][j], a[i], b[j], acc2[i][j]);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* out = stage;
        threadgroup float* out2 = stage + BM * OUT_LD;
        #pragma clang loop unroll(full)
        for (int i = 0; i < TM; i++) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < TN; j++) {
                simdgroup_store(acc[i][j], out + i * 8 * OUT_LD + sgid * SN + j * 8, OUT_LD);
                if (gated) simdgroup_store(acc2[i][j], out2 + i * 8 * OUT_LD + sgid * SN + j * 8, OUT_LD);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint u = tid; u < BM * BN; u += THREADS) {
            const uint row = u / BN, col = u % BN;
            const uint m = m0 + row;
            if (m >= p.M) continue;
            const uint n = n0 + col;
            float v = out[row * OUT_LD + col];
            if (p.flags & GEMV_BIAS) v += bf16_to_float(bias[n]);
            if (gated) v = silu(v) * out2[row * OUT_LD + col];
            if (p.flags & GEMV_GELU) v = gelu_tanh(v);
            if (p.flags & GEMV_SILU) v = silu(v);
            device float* y = Y + m * p.ldy + n;
            if (p.flags & GEMV_ACCUM) v += *y;
            *y = v;
        }
    }

    template [[host_name("gemm_32x64x32")]] kernel void gemm_kernel<32, 64, 32, 4>(device const float*, device const ushort*, device const ushort*, device const ushort*, device float*, constant GemvParams&, uint2, uint, uint);

    """#

    // One simdgroup per row. y = x * rsqrt(mean(x^2) + eps) * w, all in fp32 like the reference.
    static let norm = #"""
    struct NormParams { uint K; uint ldx; uint ldy; float eps; };

    kernel void rmsnorm(device const float* X [[buffer(0)]],
                        device const ushort* W [[buffer(1)]],
                        device float* Y [[buffer(2)]],
                        constant NormParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
        const uint row = gid / 32;
        device const float* x = X + row * p.ldx;
        float ss = 0.0f;
        for (uint k = lane * 4; k < p.K; k += 128) {
            const float4 v = *(device const float4*)(x + k);
            ss += dot(v, v);
        }
        ss = simd_sum(ss);
        const float scale = rsqrt(ss / float(p.K) + p.eps);
        device float* y = Y + row * p.ldy;
        for (uint k = lane * 4; k < p.K; k += 128) {
            const float4 v = *(device const float4*)(x + k);
            const uint2 w01 = *(device const uint2*)(W + k);
            const float4 w = float4(bf16_to_float(ushort(w01.x & 0xFFFF)), bf16_to_float(ushort(w01.x >> 16)),
                                    bf16_to_float(ushort(w01.y & 0xFFFF)), bf16_to_float(ushort(w01.y >> 16)));
            *(device float4*)(y + k) = v * scale * w;
        }
    }

    kernel void copy_rows(device const float* X [[buffer(0)]],
                          device float* Y [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
        Y[gid] = X[gid];
    }

    """#

    // Qwen2 (NeoX) RoPE on q in place and on k, which lands in the cache with v. Thread grid:
    // x = rotation pair (0..31), y = head slot over q heads, k heads, v heads, z = row.
    static let rope = #"""
    struct RopeParams { uint M; uint hq; uint hkv; uint pos0; uint ldq; uint cacheLen; };

    kernel void rope_cache(device float* QKV [[buffer(0)]],
                           device float* Kc [[buffer(1)]],
                           device float* Vc [[buffer(2)]],
                           device const float* invFreq [[buffer(3)]],
                           constant RopeParams& p [[buffer(4)]],
                           uint3 gid [[thread_position_in_grid]]) {
        const uint i = gid.x, slot = gid.y, m = gid.z;
        if (m >= p.M) return;
        const uint pos = p.pos0 + m;
        device float* row = QKV + m * p.ldq + slot * 64;
        if (slot >= p.hq + p.hkv) {
            const uint vh = slot - p.hq - p.hkv;
            device float* dst = Vc + (vh * p.cacheLen + pos) * 64;
            dst[i] = row[i];
            dst[i + 32] = row[i + 32];
            return;
        }
        const float angle = float(pos) * invFreq[i];
        const float c = cos(angle), s = sin(angle);
        const float x0 = row[i], x1 = row[i + 32];
        const float r0 = x0 * c - x1 * s;
        const float r1 = x1 * c + x0 * s;
        if (slot < p.hq) {
            row[i] = r0;
            row[i + 32] = r1;
        } else {
            device float* dst = Kc + ((slot - p.hq) * p.cacheLen + pos) * 64;
            dst[i] = r0;
            dst[i + 32] = r1;
        }
    }

    """#

    // Causal attention of M new queries against a KV cache, GQA, optional sliding window.
    // A (row, head) item is SPLIT simdgroups of a 4-simdgroup threadgroup: they split the keys
    // for the scores and again for the weighted sum of V, and meet in threadgroup memory for
    // the softmax max and sum and the 64 output dims. Decode (14 items at M=1) uses SPLIT 4 so
    // 56 simdgroups share the latency rounds; prefill has thousands of items and uses SPLIT 1.
    static let attention = #"""
    struct AttnParams { uint M; uint hq; uint hkv; uint pos0; uint window; uint ldq; uint ldo; uint cacheLen; float scale; };
    constant int ATT_SG = 4;
    constant int ATT_MAXKEYS = 512;

    template <int SPLIT>
    kernel void attention_cache(device const float* Q [[buffer(0)]],
                                device const float* Kc [[buffer(1)]],
                                device const float* Vc [[buffer(2)]],
                                device float* O [[buffer(3)]],
                                constant AttnParams& p [[buffer(4)]],
                                uint tgid [[threadgroup_position_in_grid]],
                                uint sgid [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
        constexpr int ITEMS = ATT_SG / SPLIT;
        threadgroup float scores[ITEMS * ATT_MAXKEYS];
        threadgroup float red[ATT_SG][2];
        threadgroup float4 part[ATT_SG][16];
        const uint item = sgid / SPLIT, sub = sgid % SPLIT;
        const uint work = tgid * ITEMS + item;
        const bool live = work < p.M * p.hq;
        const uint m = live ? work / p.hq : 0, h = live ? work % p.hq : 0;
        const uint kvh = h / (p.hq / p.hkv);
        const uint qpos = p.pos0 + m;
        const uint jEnd = qpos + 1;
        const uint jStart = (p.window > 0 && jEnd > p.window) ? jEnd - p.window : 0;
        const uint nKeys = live ? jEnd - jStart : 0;
        threadgroup float* sc = scores + item * ATT_MAXKEYS;
        device const float4* q = (device const float4*)(Q + m * p.ldq + h * 64);
        float4 qv[16];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++) qv[i] = q[i];
        device const float* kbase = Kc + kvh * p.cacheLen * 64;
        float mx = -INFINITY;
        for (uint t = sub * 32 + lane; t < nKeys; t += SPLIT * 32) {
            device const float4* kr = (device const float4*)(kbase + (jStart + t) * 64);
            float s = 0.0f;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 16; i++) s += dot(qv[i], kr[i]);
            s *= p.scale;
            sc[t] = s;
            mx = max(mx, s);
        }
        mx = simd_max(mx);
        if (SPLIT > 1) {
            if (lane == 0) red[sgid][0] = mx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (int j = 0; j < SPLIT; j++) mx = max(mx, red[item * SPLIT + j][0]);
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        float sum = 0.0f;
        for (uint t = sub * 32 + lane; t < nKeys; t += SPLIT * 32) {
            const float e = exp(sc[t] - mx);
            sc[t] = e;
            sum += e;
        }
        sum = simd_sum(sum);
        if (SPLIT > 1) {
            if (lane == 0) red[sgid][1] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sum = 0.0f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < SPLIT; j++) sum += red[item * SPLIT + j][1];
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Weighted sum of V: each half-simdgroup owns every (2 SPLIT)th key, a lane owns four
        // dims, and eight keys per half are loaded before any is used, so a round of memory
        // latency covers 16 SPLIT keys. One load per key per lane cost a latency per key;
        // whole V rows per lane (16 loads in flight) scattered each load instruction over 32
        // rows and halved prefill throughput.
        constexpr uint STEP = 2 * SPLIT;
        const uint kg = lane >> 4, dl = lane & 15;
        device const float* vb = Vc + kvh * p.cacheLen * 64 + 4 * dl;
        float4 acc = float4(0.0f);
        uint t = 2 * sub + kg;
        for (; t + 7 * STEP < nKeys; t += 8 * STEP) {
            float4 v[8];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; i++) v[i] = *(device const float4*)(vb + (jStart + t + i * STEP) * 64);
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; i++) acc += sc[t + i * STEP] * v[i];
        }
        for (; t < nKeys; t += STEP) acc += sc[t] * *(device const float4*)(vb + (jStart + t) * 64);
        acc += simd_shuffle_xor(acc, 16);
        if (SPLIT > 1) {
            if (lane < 16) part[sgid][dl] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sub != 0) return;
            acc = part[item * SPLIT][dl];
            #pragma clang loop unroll(full)
            for (int j = 1; j < SPLIT; j++) acc += part[item * SPLIT + j][dl];
        }
        if (live && lane < 16) *(device float4*)(O + m * p.ldo + h * 64 + 4 * dl) = acc / sum;
    }

    template [[host_name("attention_cache_1")]] kernel void attention_cache<1>(device const float*, device const float*, device const float*, device float*, constant AttnParams&, uint, uint, uint);
    template [[host_name("attention_cache_4")]] kernel void attention_cache<4>(device const float*, device const float*, device const float*, device float*, constant AttnParams&, uint, uint, uint);

    """#

    // DiT pieces: full attention over a tiny sequence with x_transformers interleaved RoPE, the
    // flow input assembly, and the Euler update of the CFM sampler.
    static let dit = #"""
    struct DitAttnParams { uint B; uint S; uint H; uint ld; uint ldo; float scale; };
    constant int DIT_SG = 4;
    constant int DIT_MAXS = 8;

    inline float2 rope_pair(float2 v, float angle) {
        const float c = cos(angle), s = sin(angle);
        return float2(v.x * c - v.y * s, v.y * c + v.x * s);
    }

    // One simdgroup per (sequence, query row, head). Lane l owns head dims 2l and 2l + 1, which is one
    // interleaved-RoPE pair, so each lane rotates with one cos/sin per position and a score is one
    // simd_sum. Sequences up to DIT_MAXS keys.
    kernel void dit_attention(device const float* QKV [[buffer(0)]],
                              device const float* invFreq [[buffer(1)]],
                              device float* O [[buffer(2)]],
                              constant DitAttnParams& p [[buffer(3)]],
                              uint tgid [[threadgroup_position_in_grid]],
                              uint sgid [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
        const uint work = tgid * DIT_SG + sgid;
        if (work >= p.B * p.S * p.H) return;
        const uint h = work % p.H, i = (work / p.H) % p.S, b = work / (p.H * p.S);
        const uint D = p.H * 64;
        const float freq = invFreq[lane];
        device const float* rows = QKV + b * p.S * p.ld + h * 64 + 2 * lane;
        const float2 q = rope_pair(*(device const float2*)(rows + i * p.ld), float(i) * freq);
        float sc[DIT_MAXS];
        float mx = -INFINITY;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < uint(DIT_MAXS); j++) {
            if (j < p.S) {
                const float2 k = rope_pair(*(device const float2*)(rows + j * p.ld + D), float(j) * freq);
                sc[j] = simd_sum(dot(q, k)) * p.scale;
                mx = max(mx, sc[j]);
            }
        }
        float sum = 0.0f;
        float2 acc = float2(0.0f);
        #pragma clang loop unroll(full)
        for (uint j = 0; j < uint(DIT_MAXS); j++) {
            if (j < p.S) {
                const float e = exp(sc[j] - mx);
                sum += e;
                acc += e * *(device const float2*)(rows + j * p.ld + 2 * D);
            }
        }
        *(device float2*)(O + (b * p.S + i) * p.ldo + h * 64 + 2 * lane) = acc / sum;
    }

    // Rows of the flow DiT input: [t + c, emb(history), emb(x)] for the conditional batch and
    // [t + c_bias, emb(history), emb(x)] for the unconditional one (c = 0 leaves the bias).
    struct FlowAssembleParams { uint hidden; uint latentDim; };

    kernel void flow_assemble(device const float* y [[buffer(0)]],
                              device const float* history [[buffer(1)]],
                              device const float* cemb [[buffer(2)]],
                              device const ushort* cbias [[buffer(3)]],
                              device const float* temb [[buffer(4)]],
                              device const ushort* xW [[buffer(5)]],
                              device const ushort* xB [[buffer(6)]],
                              device float* X [[buffer(7)]],
                              constant FlowAssembleParams& p [[buffer(8)]],
                              uint2 gid [[thread_position_in_grid]]) {
        const uint c = gid.x, r = gid.y;
        if (c >= p.hidden) return;
        const uint s = r % 3;
        float v;
        if (s == 0) {
            v = temb[c] + (r < 3 ? cemb[c] : bf16_to_float(cbias[c]));
        } else {
            device const float* src = (s == 1) ? history : y;
            device const ushort* w = xW + c * p.latentDim;
            v = bf16_to_float(xB[c]);
            for (uint d = 0; d < p.latentDim; d++) v += bf16_to_float(w[d]) * src[d];
        }
        X[r * p.hidden + c] = v;
    }

    // y += (pred + (pred - null) * cfg) * dt, and keep both predictions for parity.
    struct EulerParams { uint latentDim; uint predRow; uint nullRow; uint ldp; float cfg; float dt; };

    kernel void flow_euler(device float* y [[buffer(0)]],
                           device const float* pred [[buffer(1)]],
                           device float* capture [[buffer(2)]],
                           constant EulerParams& p [[buffer(3)]],
                           uint d [[thread_position_in_grid]]) {
        if (d >= p.latentDim) return;
        const float c = pred[p.predRow * p.ldp + d];
        const float u = pred[p.nullRow * p.ldp + d];
        capture[d] = c;
        capture[p.latentDim + d] = u;
        y[d] += (c + (c - u) * p.cfg) * p.dt;
    }

    """#

    // VAE decoder front and back: linear x4 upsample of fc1 outputs with one latent of lookahead,
    // the ISTFT frame synthesis (direct DFT), and the overlap-add emission.
    static let vae = #"""
    // Frames for latent n from h[n-1], h[n], h[n+1] (torch Upsample linear, align_corners=False,
    // scale 4). prev/next flags say whether those neighbours exist; absent ones clamp to h[n].
    struct UpsampleParams { uint hidden; uint hasPrev; uint hasNext; };

    kernel void upsample4(device const float* hPrev [[buffer(0)]],
                          device const float* hCur [[buffer(1)]],
                          device const float* hNext [[buffer(2)]],
                          device float* frames [[buffer(3)]],
                          constant UpsampleParams& p [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
        const uint c = gid.x, r = gid.y;
        if (c >= p.hidden) return;
        const float cur = hCur[c];
        float v;
        if (r == 0) v = p.hasPrev ? 0.625f * cur + 0.375f * hPrev[c] : cur;
        else if (r == 1) v = p.hasPrev ? 0.875f * cur + 0.125f * hPrev[c] : cur;
        else if (r == 2) v = p.hasNext ? 0.875f * cur + 0.125f * hNext[c] : cur;
        else v = p.hasNext ? 0.625f * cur + 0.375f * hNext[c] : cur;
        frames[r * p.hidden + c] = v;
    }

    // spec rows hold [log-magnitude (nfft/2+1), phase (nfft/2+1)]. Output frame f (ring slot
    // (frame0 + f) % ring) = irfft(mag * e^{i phase}) * window, irfft with 1/N scaling.
    struct IstftParams { uint nfft; uint ldSpec; uint frame0; uint ring; };
    constant int ISTFT_TG = 256;

    kernel void istft_frames(device const float* spec [[buffer(0)]],
                             device const float* cosTable [[buffer(1)]],
                             device const float* window [[buffer(2)]],
                             device float* frames [[buffer(3)]],
                             constant IstftParams& p [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]],
                             uint2 tpos [[thread_position_in_threadgroup]],
                             uint2 tgsize [[threads_per_threadgroup]]) {
        threadgroup float re[1766];
        threadgroup float im[1766];
        threadgroup float ctab[3528];
        const uint N = p.nfft, bins = N / 2 + 1;
        const uint f = gid.y;
        device const float* row = spec + f * p.ldSpec;
        for (uint k = tpos.x; k < bins; k += tgsize.x) {
            const float mag = min(exp(row[k]), 100.0f);
            const float ph = row[bins + k];
            re[k] = mag * cos(ph);
            im[k] = mag * sin(ph);
        }
        for (uint k = tpos.x; k < N; k += tgsize.x) ctab[k] = cosTable[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint t = gid.x;
        if (t >= N) return;
        float acc = re[0] + ((t & 1) ? -re[N / 2] : re[N / 2]);
        uint idx = 0;
        const uint quarter = N / 4;
        float sum = 0.0f;
        for (uint k = 1; k < N / 2; k++) {
            idx += t;
            if (idx >= N) idx -= N;
            uint sidx = idx + 3 * quarter;
            if (sidx >= N) sidx -= N;
            sum += re[k] * ctab[idx] - im[k] * ctab[sidx];
        }
        acc += 2.0f * sum;
        const uint slot = (p.frame0 + f) % p.ring;
        frames[slot * N + t] = acc / float(N) * window[t];
    }

    // out[i] for sample s = s0 + i: sum over frames f with 0 <= s + pad - f*hop < nfft and
    // f <= lastFrame of frame[f][s + pad - f*hop], divided by the matching window^2 envelope.
    struct EmitParams { uint nfft; uint hop; uint pad; uint s0; uint count; uint lastFrame; uint ring; };

    kernel void istft_emit(device const float* frames [[buffer(0)]],
                           device const float* window [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant EmitParams& p [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= p.count) return;
        const uint s = p.s0 + i + p.pad;
        const uint fHi = min(s / p.hop, p.lastFrame);
        const uint fLo = (s >= p.nfft) ? (s - p.nfft) / p.hop + 1 : 0;
        float acc = 0.0f, env = 0.0f;
        for (uint f = fLo; f <= fHi; f++) {
            const uint t = s - f * p.hop;
            if (t >= p.nfft) continue;
            const float w = window[t];
            acc += frames[(f % p.ring) * p.nfft + t];
            env += w * w;
        }
        out[i] = acc / env;
    }

    """#
}
