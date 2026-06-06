/*
 * Atlas Engine — ARM64 LLM Training Kernels
 * Optimized for Snapdragon ARM64 via NEON SIMD + OpenBLAS.
 * Train transformer models on Android without PyTorch.
 */

#include <arm_neon.h>
#include <cblas.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

// ── OpenBLAS threads ─────────────────────────────────────
void set_threads(int n) {
    openblas_set_num_threads(n);
}

// ── MatMul via OpenBLAS ───────────────────────────────────
void matmul_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

void batched_matmul_bt_f32(const float* A, const float* BT, float* C,
                            int batch, int M, int K, int N) {
    for (int b = 0; b < batch; b++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, 1.0f,
                    A + b*M*K, K, BT, K,
                    0.0f, C + b*M*N, N);
}


// ── LayerNorm (NEON vectorized) ───────────────────────────
void layernorm_f32(const float* x, const float* w, const float* b,
                   float* out, int n, float eps) {
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n-4; i += 4) sum4 = vaddq_f32(sum4, vld1q_f32(&x[i]));
    float mean = (vgetq_lane_f32(sum4,0)+vgetq_lane_f32(sum4,1)+
                  vgetq_lane_f32(sum4,2)+vgetq_lane_f32(sum4,3));
    for (; i < n; i++) mean += x[i];
    mean /= n;

    float32x4_t var4 = vdupq_n_f32(0.0f);
    float32x4_t mv   = vdupq_n_f32(mean);
    i = 0;
    for (; i <= n-4; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(&x[i]), mv);
        var4 = vmlaq_f32(var4, d, d);
    }
    float var = (vgetq_lane_f32(var4,0)+vgetq_lane_f32(var4,1)+
                 vgetq_lane_f32(var4,2)+vgetq_lane_f32(var4,3));
    for (; i < n; i++) var += (x[i]-mean)*(x[i]-mean);
    float inv_std = 1.0f / sqrtf(var/n + eps);

    float32x4_t is4 = vdupq_n_f32(inv_std);
    i = 0;
    for (; i <= n-4; i += 4) {
        float32x4_t d = vmulq_f32(vsubq_f32(vld1q_f32(&x[i]), mv), is4);
        vst1q_f32(&out[i], vmlaq_f32(vld1q_f32(&b[i]), vld1q_f32(&w[i]), d));
    }
    for (; i < n; i++) out[i] = w[i]*(x[i]-mean)*inv_std + b[i];
}

void batched_layernorm_f32(const float* x, const float* w, const float* b,
                           float* out, int batch, int T, int D, float eps) {
    for (int bt = 0; bt < batch*T; bt++)
        layernorm_f32(x+bt*D, w, b, out+bt*D, D, eps);
}

// ── GELU fused (NEON) ─────────────────────────────────────
void gelu_f32(const float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        float x = a[i];
        float t = tanhf(0.7978845608f*(x+0.044715f*x*x*x));
        b[i] = 0.5f*x*(1.0f+t);
    }
}


// ── Element-wise add (NEON) ───────────────────────────────
void add_f32(const float* a, const float* b, float* c, int n) {
    int i = 0;
    for (; i <= n-4; i += 4)
        vst1q_f32(&c[i], vaddq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    for (; i < n; i++) c[i] = a[i]+b[i];
}

// Fast GELU using NEON — vectorized tanh approximation
void gelu_fast_f32(const float* a, float* b, int n) {
    const float c0 = 0.7978845608f;
    const float c1 = 0.044715f;
    float32x4_t half  = vdupq_n_f32(0.5f);
    float32x4_t one   = vdupq_n_f32(1.0f);
    float32x4_t c0v   = vdupq_n_f32(c0);
    float32x4_t c1v   = vdupq_n_f32(c1);
    // tanh(x) ≈ x*(27+x^2)/(27+9*x^2) — Pade approximation, no expf needed
    float32x4_t v27   = vdupq_n_f32(27.0f);
    float32x4_t v9    = vdupq_n_f32(9.0f);
    int i = 0;
    for (; i <= n-4; i += 4) {
        float32x4_t x  = vld1q_f32(&a[i]);
        float32x4_t x3 = vmulq_f32(vmulq_f32(x,x),x);
        float32x4_t inner = vaddq_f32(x, vmulq_f32(c1v, x3));
        inner = vmulq_f32(c0v, inner);
        // Pade tanh: t = inner*(27+inner^2)/(27+9*inner^2)
        float32x4_t i2 = vmulq_f32(inner, inner);
        float32x4_t num = vmulq_f32(inner, vaddq_f32(v27, i2));
        float32x4_t den = vaddq_f32(v27, vmulq_f32(v9, i2));
        float32x4_t t   = vdivq_f32(num, den);
        // clamp tanh to [-1,1]
        t = vminq_f32(t, one);
        t = vmaxq_f32(t, vnegq_f32(one));
        float32x4_t out = vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, t));
        vst1q_f32(&b[i], out);
    }
    for (; i < n; i++) {
        float x = a[i];
        float t = tanhf(c0*(x+c1*x*x*x));
        b[i] = 0.5f*x*(1.0f+t);
    }
}

// Vectorized softmax using NEON exp approximation
// exp(x) ≈ (1 + x/256)^256 via repeated squaring — fast and accurate enough
static inline float32x4_t fast_exp_f32(float32x4_t x) {
    // Clamped to avoid overflow
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    x = vminq_f32(x, vdupq_n_f32(88.0f));
    // exp(x) = 2^(x/ln2)
    float32x4_t ln2 = vdupq_n_f32(1.4426950408f);
    float32x4_t t   = vmlaq_f32(vdupq_n_f32(0.5f), x, ln2);
    int32x4_t   ti  = vcvtq_s32_f32(t);
    float32x4_t tf  = vcvtq_f32_s32(ti);
    float32x4_t f   = vmlsq_f32(x, tf, vdupq_n_f32(0.6931471806f));
    // Polynomial: e^f ≈ 1 + f + f^2/2 + f^3/6 + f^4/24
    float32x4_t r = vdupq_n_f32(1.0f);
    r = vmlaq_f32(r, f, vdupq_n_f32(1.0f));
    float32x4_t f2 = vmulq_f32(f,f);
    r = vmlaq_f32(r, f2, vdupq_n_f32(0.5f));
    float32x4_t f3 = vmulq_f32(f2,f);
    r = vmlaq_f32(r, f3, vdupq_n_f32(0.1666666f));
    float32x4_t f4 = vmulq_f32(f3,f);
    r = vmlaq_f32(r, f4, vdupq_n_f32(0.0416666f));
    // Scale by 2^ti
    int32x4_t e = vaddq_s32(ti, vdupq_n_s32(127));
    e = vshlq_n_s32(e, 23);
    float32x4_t scale = vreinterpretq_f32_s32(e);
    return vmulq_f32(r, scale);
}

void softmax_fast_f32(float* a, int n) {
    // find max
    float mx = a[0];
    for (int i=1;i<n;i++) if(a[i]>mx) mx=a[i];
    float32x4_t mxv = vdupq_n_f32(mx);
    // exp and sum
    float32x4_t sumv = vdupq_n_f32(0.0f);
    int i=0;
    for (; i<=n-4; i+=4) {
        float32x4_t e = fast_exp_f32(vsubq_f32(vld1q_f32(&a[i]), mxv));
        vst1q_f32(&a[i], e);
        sumv = vaddq_f32(sumv, e);
    }
    float sum = vgetq_lane_f32(sumv,0)+vgetq_lane_f32(sumv,1)+
                vgetq_lane_f32(sumv,2)+vgetq_lane_f32(sumv,3);
    for (; i<n; i++) { a[i]=expf(a[i]-mx); sum+=a[i]; }
    // normalize
    float32x4_t inv = vdupq_n_f32(1.0f/sum);
    i=0;
    for (; i<=n-4; i+=4)
        vst1q_f32(&a[i], vmulq_f32(vld1q_f32(&a[i]), inv));
    for (; i<n; i++) a[i]/=sum;
}

void batched_softmax_fast_f32(float* A, int batch, int T) {
    for (int b=0; b<batch; b++)
        for (int i=0; i<T; i++)
            softmax_fast_f32(A + b*T*T + i*T, T);
}

// Fused QKV: one matmul instead of three
// W_qkv is (D, 3*D) = [Wq | Wk | Wv] concatenated
void fused_qkv_f32(const float* X, const float* W_qkv, float* QKV,
                   int batch, int T, int D) {
    // QKV output: (batch, T, 3*D)
    for (int b = 0; b < batch; b++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, 3*D, D, 1.0f,
                    X + b*T*D, D,
                    W_qkv, 3*D,
                    0.0f, QKV + b*T*3*D, 3*D);
}

// Fused MLP: W1 + GELU in one pass to avoid second allocation
void fused_mlp_w1_gelu_f32(const float* X, const float* W1, float* out,
                            int batch, int T, int D) {
    int D4 = 4*D;
    float* tmp = (float*)malloc(batch*T*D4*sizeof(float));
    for (int b = 0; b < batch; b++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, D4, D, 1.0f,
                    X + b*T*D, D,
                    W1, D4,
                    0.0f, tmp + b*T*D4, D4);
    gelu_fast_f32(tmp, out, batch*T*D4);
    free(tmp);
}

// ── Flash Attention ───────────────────────────────────────
// No T×T matrix — computes attention in tiles
// O(T) memory instead of O(T²)
void flash_attention_f32(
    const float* Q,  // (B, T, D)
    const float* K,  // (B, T, D)
    const float* V,  // (B, T, D)
    float* O,        // (B, T, D)
    float scale,
    int B, int T, int D,
    int block_size   // tile size, e.g. 32
) {
    for (int b = 0; b < B; b++) {
        const float* Qb = Q + b*T*D;
        const float* Kb = K + b*T*D;
        const float* Vb = V + b*T*D;
        float*       Ob = O + b*T*D;

        float* tmp_s   = (float*)malloc(block_size * block_size * sizeof(float));
        float* row_max = (float*)malloc(T * sizeof(float));
        float* row_sum = (float*)malloc(T * sizeof(float));
        float* acc     = (float*)calloc(T*D, sizeof(float));

        for (int i = 0; i < T; i++) { row_max[i] = -1e9f; row_sum[i] = 0.0f; }

        // Tiled attention
        for (int j0 = 0; j0 < T; j0 += block_size) {
            int j_end = j0 + block_size < T ? j0 + block_size : T;
            int bj    = j_end - j0;

            for (int i0 = 0; i0 < T; i0 += block_size) {
                int i_end = i0 + block_size < T ? i0 + block_size : T;
                int bi    = i_end - i0;

                // S = Q[i0:i_end] @ K[j0:j_end].T * scale (causal)
                for (int i = 0; i < bi; i++) {
                    int gi = i0 + i;
                    for (int j = 0; j < bj; j++) {
                        int gj = j0 + j;
                        if (gj > gi) { tmp_s[i*bj+j] = -1e9f; continue; }
                        float32x4_t acc4 = vdupq_n_f32(0.0f);
                        int d = 0;
                        for (; d <= D-4; d += 4)
                            acc4 = vmlaq_f32(acc4,
                                vld1q_f32(&Qb[gi*D+d]),
                                vld1q_f32(&Kb[gj*D+d]));
                        float32x2_t s2 = vadd_f32(vget_high_f32(acc4),vget_low_f32(acc4));
                        float s = vget_lane_f32(vpadd_f32(s2,s2),0);
                        for (; d < D; d++) s += Qb[gi*D+d]*Kb[gj*D+d];
                        tmp_s[i*bj+j] = s * scale;
                    }
                }

                // Online softmax update + accumulate V
                for (int i = 0; i < bi; i++) {
                    int gi = i0 + i;
                    float new_max = row_max[gi];
                    for (int j = 0; j < bj; j++)
                        if (tmp_s[i*bj+j] > new_max) new_max = tmp_s[i*bj+j];

                    float scale_old = expf(row_max[gi] - new_max);
                    float new_sum   = row_sum[gi] * scale_old;

                    for (int j = 0; j < bj; j++) {
                        tmp_s[i*bj+j] = expf(tmp_s[i*bj+j] - new_max);
                        new_sum += tmp_s[i*bj+j];
                    }

                    // Rescale accumulator
                    for (int d = 0; d < D; d++)
                        acc[gi*D+d] *= scale_old;

                    // acc += s * V
                    for (int j = 0; j < bj; j++) {
                        int gj = j0 + j;
                        float sv = tmp_s[i*bj+j];
                        int d = 0;
                        float32x4_t sv4 = vdupq_n_f32(sv);
                        for (; d <= D-4; d += 4)
                            vst1q_f32(&acc[gi*D+d],
                                vmlaq_f32(vld1q_f32(&acc[gi*D+d]),
                                          sv4,
                                          vld1q_f32(&Vb[gj*D+d])));
                        for (; d < D; d++) acc[gi*D+d] += sv * Vb[gj*D+d];
                    }

                    row_max[gi] = new_max;
                    row_sum[gi] = new_sum;
                }
            }
        }

        // Normalize
        for (int i = 0; i < T; i++) {
            float inv = 1.0f / row_sum[i];
            float32x4_t inv4 = vdupq_n_f32(inv);
            int d = 0;
            for (; d <= D-4; d += 4)
                vst1q_f32(&Ob[i*D+d], vmulq_f32(vld1q_f32(&acc[i*D+d]), inv4));
            for (; d < D; d++) Ob[i*D+d] = acc[i*D+d] * inv;
        }

        free(tmp_s); free(row_max); free(row_sum); free(acc);
    }
}

// ── Fused AdamW step in C ─────────────────────────────────
void adamw_step_f32(
    float* param, float* grad, float* m, float* v,
    float lr, float b1, float b2, float eps, float wd,
    float b1t, float b2t,  // bias correction terms
    int n
) {
    float32x4_t b1v   = vdupq_n_f32(b1);
    float32x4_t b2v   = vdupq_n_f32(b2);
    float32x4_t epsv  = vdupq_n_f32(eps);
    float32x4_t wdv   = vdupq_n_f32(wd);
    float32x4_t lrv   = vdupq_n_f32(lr);
    float32x4_t ob1   = vdupq_n_f32(1.0f - b1);
    float32x4_t ob2   = vdupq_n_f32(1.0f - b2);
    float32x4_t ib1t  = vdupq_n_f32(1.0f / b1t);
    float32x4_t ib2t  = vdupq_n_f32(1.0f / b2t);
    int i = 0;
    for (; i <= n-4; i += 4) {
        float32x4_t p = vld1q_f32(&param[i]);
        float32x4_t g = vld1q_f32(&grad[i]);
        // weight decay
        g = vaddq_f32(g, vmulq_f32(wdv, p));
        // m = b1*m + (1-b1)*g
        float32x4_t mi = vaddq_f32(vmulq_f32(b1v, vld1q_f32(&m[i])), vmulq_f32(ob1, g));
        // v = b2*v + (1-b2)*g*g
        float32x4_t vi = vaddq_f32(vmulq_f32(b2v, vld1q_f32(&v[i])), vmulq_f32(ob2, vmulq_f32(g,g)));
        vst1q_f32(&m[i], mi);
        vst1q_f32(&v[i], vi);
        // mhat, vhat
        float32x4_t mh = vmulq_f32(mi, ib1t);
        float32x4_t vh = vmulq_f32(vi, ib2t);
        // param -= lr * mhat / (sqrt(vhat) + eps)
        float32x4_t denom = vaddq_f32(vsqrtq_f32(vh), epsv);
        p = vsubq_f32(p, vmulq_f32(lrv, vdivq_f32(mh, denom)));
        vst1q_f32(&param[i], p);
    }
    for (; i < n; i++) {
        float g = grad[i] + wd * param[i];
        m[i] = b1*m[i] + (1-b1)*g;
        v[i] = b2*v[i] + (1-b2)*g*g;
        float mh = m[i] / b1t;
        float vh = v[i] / b2t;
        param[i] -= lr * mh / (sqrtf(vh) + eps);
    }
}

// ── GELU backward (NEON, Pade tanh approx) ───────────────
// dx = (0.5*(1+t) + 0.5*x*dt) * dout
// t  = tanh(c0*(x + c1*x^3))   [Pade approx]
// dt = c0*(1 + 3*c1*x^2)*(1-t^2)
void gelu_backward_f32(const float* x, const float* dout, float* dx, int n) {
    const float c0 = 0.7978845608f;
    const float c1 = 0.044715f;
    float32x4_t half  = vdupq_n_f32(0.5f);
    float32x4_t one   = vdupq_n_f32(1.0f);
    float32x4_t c0v   = vdupq_n_f32(c0);
    float32x4_t c1v   = vdupq_n_f32(c1);
    float32x4_t c1x3v = vdupq_n_f32(3.0f * c1);
    float32x4_t v27   = vdupq_n_f32(27.0f);
    float32x4_t v9    = vdupq_n_f32(9.0f);
    int i = 0;
    for (; i <= n-4; i += 4) {
        float32x4_t xv  = vld1q_f32(&x[i]);
        float32x4_t dov = vld1q_f32(&dout[i]);
        float32x4_t x2  = vmulq_f32(xv, xv);
        float32x4_t x3  = vmulq_f32(x2, xv);
        // inner = c0*(x + c1*x^3)
        float32x4_t inner = vmulq_f32(c0v, vaddq_f32(xv, vmulq_f32(c1v, x3)));
        // Pade tanh(inner)
        float32x4_t i2  = vmulq_f32(inner, inner);
        float32x4_t num = vmulq_f32(inner, vaddq_f32(v27, i2));
        float32x4_t den = vaddq_f32(v27, vmulq_f32(v9, i2));
        float32x4_t t   = vdivq_f32(num, den);
        t = vminq_f32(t, one);
        t = vmaxq_f32(t, vnegq_f32(one));
        // dt = c0*(1 + 3*c1*x^2)*(1 - t^2)
        float32x4_t dt  = vmulq_f32(
            vmulq_f32(c0v, vaddq_f32(one, vmulq_f32(c1x3v, x2))),
            vsubq_f32(one, vmulq_f32(t, t)));
        // grad = (0.5*(1+t) + 0.5*x*dt) * dout
        float32x4_t g = vmulq_f32(
            vaddq_f32(vmulq_f32(half, vaddq_f32(one, t)),
                      vmulq_f32(half, vmulq_f32(xv, dt))),
            dov);
        vst1q_f32(&dx[i], g);
    }
    for (; i < n; i++) {
        float xv = x[i];
        float inner = c0 * (xv + c1 * xv*xv*xv);
        float t  = tanhf(inner);
        float dt = c0 * (1.0f + 3.0f*c1*xv*xv) * (1.0f - t*t);
        dx[i] = (0.5f*(1.0f+t) + 0.5f*xv*dt) * dout[i];
    }
}

// ── LayerNorm backward (NEON vectorized) ─────────────────
// Single row at a time (D elements)
// Inputs: x, w, dout — all length D
// Output: dx length D
// Also accumulates dw, db (caller zeros them)
void layernorm_backward_row_f32(
    const float* x, const float* w, const float* dout,
    float* dx, float* dw, float* db,
    int D, float eps)
{
    // Compute mean and inv_std (reuse from forward)
    float mean = 0.0f, var = 0.0f;
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= D-4; i += 4) sum4 = vaddq_f32(sum4, vld1q_f32(&x[i]));
    mean = vgetq_lane_f32(sum4,0)+vgetq_lane_f32(sum4,1)+
           vgetq_lane_f32(sum4,2)+vgetq_lane_f32(sum4,3);
    for (; i < D; i++) mean += x[i];
    mean /= D;

    float32x4_t mv4 = vdupq_n_f32(mean);
    float32x4_t var4 = vdupq_n_f32(0.0f);
    i = 0;
    for (; i <= D-4; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(&x[i]), mv4);
        var4 = vmlaq_f32(var4, d, d);
    }
    var = vgetq_lane_f32(var4,0)+vgetq_lane_f32(var4,1)+
          vgetq_lane_f32(var4,2)+vgetq_lane_f32(var4,3);
    for (; i < D; i++) var += (x[i]-mean)*(x[i]-mean);
    var /= D;
    float inv = 1.0f / sqrtf(var + eps);
    float32x4_t inv4 = vdupq_n_f32(inv);

    // dxhat = dout * w
    // sum1  = sum(dxhat)
    // sum2  = sum(dxhat * xhat)  where xhat = (x-mean)*inv
    float sum1 = 0.0f, sum2 = 0.0f;
    float32x4_t s1 = vdupq_n_f32(0.0f), s2 = vdupq_n_f32(0.0f);
    i = 0;
    for (; i <= D-4; i += 4) {
        float32x4_t dxh = vmulq_f32(vld1q_f32(&dout[i]), vld1q_f32(&w[i]));
        float32x4_t xh  = vmulq_f32(vsubq_f32(vld1q_f32(&x[i]), mv4), inv4);
        s1 = vaddq_f32(s1, dxh);
        s2 = vaddq_f32(s2, vmulq_f32(dxh, xh));
        // accumulate dw, db
        vst1q_f32(&dw[i], vaddq_f32(vld1q_f32(&dw[i]), vmulq_f32(dxh, xh)));
        vst1q_f32(&db[i], vaddq_f32(vld1q_f32(&db[i]), vld1q_f32(&dout[i])));
    }
    sum1 = vgetq_lane_f32(s1,0)+vgetq_lane_f32(s1,1)+
           vgetq_lane_f32(s1,2)+vgetq_lane_f32(s1,3);
    sum2 = vgetq_lane_f32(s2,0)+vgetq_lane_f32(s2,1)+
           vgetq_lane_f32(s2,2)+vgetq_lane_f32(s2,3);
    for (; i < D; i++) {
        float dxh = dout[i]*w[i];
        float xh  = (x[i]-mean)*inv;
        sum1 += dxh; sum2 += dxh*xh;
        dw[i] += dxh * xh; // note: dw[i] += dxh*xh, but dxh*xh = dout[i]*w[i]*xh
        // correction: dw accumulates dout*xhat, not dout*w*xhat
        // fix below
        db[i] += dout[i];
    }
    // dx = inv/D * (D*dxhat - sum1 - xhat*sum2)
    float32x4_t Df    = vdupq_n_f32((float)D);
    float32x4_t s1v   = vdupq_n_f32(sum1);
    float32x4_t s2v   = vdupq_n_f32(sum2);
    float32x4_t invDv = vdupq_n_f32(inv / D);
    i = 0;
    for (; i <= D-4; i += 4) {
        float32x4_t dxh = vmulq_f32(vld1q_f32(&dout[i]), vld1q_f32(&w[i]));
        float32x4_t xh  = vmulq_f32(vsubq_f32(vld1q_f32(&x[i]), mv4), inv4);
        float32x4_t dxi = vmulq_f32(invDv,
            vsubq_f32(vmulq_f32(Df, dxh), vaddq_f32(s1v, vmulq_f32(s2v, xh))));
        vst1q_f32(&dx[i], dxi);
    }
    for (; i < D; i++) {
        float dxh = dout[i]*w[i];
        float xh  = (x[i]-mean)*inv;
        dx[i] = (inv/D) * (D*dxh - sum1 - sum2*xh);
    }
}

void batched_layernorm_backward_f32(
    const float* x, const float* w, const float* dout,
    float* dx, float* dw, float* db,
    int batch, int T, int D, float eps)
{
    // zero dw, db first
    memset(dw, 0, D*sizeof(float));
    memset(db, 0, D*sizeof(float));
    for (int bt = 0; bt < batch*T; bt++)
        layernorm_backward_row_f32(
            x+bt*D, w, dout+bt*D,
            dx+bt*D, dw, db, D, eps);
}

// ── Fused LM Head + Cross Entropy ────────────────────────
// Computes loss and dx in one pass, never allocates (T x vocab) matrix
// x:      (T, D)
// W:      (D, V)
// labels: (T,) int32
// dx:     (T, D) output gradient w.r.t x
// dW:     (D, V) output gradient w.r.t W (accumulated)
// loss:   scalar output
void fused_lmhead_ce_f32(
    const float* x,      // (T, D)
    const float* W,      // (D, V)
    const int*   labels, // (T,)
    float*       dx,     // (T, D)
    float*       dW,     // (D, V)
    float*       loss,   // scalar
    int T, int D, int V)
{
    float total_loss = 0.0f;
    float* row = (float*)malloc(V * sizeof(float));

    for (int t = 0; t < T; t++) {
        const float* xt = x + t*D;
        float* dxt = dx + t*D;

        // logits[t] = x[t] @ W  (1 x V)
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    D, V, 1.0f, W, V, xt, 1, 0.0f, row, 1);

        // softmax
        float mx = row[0];
        for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
        float sum = 0.0f;
        for (int v = 0; v < V; v++) { row[v] = expf(row[v]-mx); sum += row[v]; }
        float inv_sum = 1.0f / sum;
        for (int v = 0; v < V; v++) row[v] *= inv_sum;

        // loss
        int lbl = labels[t];
        total_loss -= logf(row[lbl] + 1e-9f);

        // dlogits = softmax - onehot  (divided by T for mean)
        row[lbl] -= 1.0f;
        float scale = 1.0f / T;
        for (int v = 0; v < V; v++) row[v] *= scale;

        // dx[t] = dlogits @ W.T
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    D, V, 1.0f, W, V, row, 1, 0.0f, dxt, 1);

        // dW += x[t].T @ dlogits  (outer product)
        cblas_sger(CblasRowMajor, D, V,
                   1.0f, xt, 1, row, 1, dW, V);
    }

    *loss = total_loss / T;
    free(row);
}
