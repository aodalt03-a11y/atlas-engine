# Atlas Engine

> Train a 162M parameter LLM on an Android phone. No PyTorch. No cloud. Just C and NEON.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-ARM64-green.svg)]()
[![No PyTorch](https://img.shields.io/badge/PyTorch-not%20required-red.svg)]()

Built from scratch on a Samsung S25 running Termux. Custom C/NEON SIMD kernels,
Flash Attention, fused AdamW — everything hand-written for ARM64.

---

## Why

Every LLM trainer assumes you have a GPU and PyTorch. Atlas Engine doesn't.

It runs on your phone, your Raspberry Pi, your AWS Graviton instance — anything
ARM64. No dependencies except OpenBLAS and a C compiler.

---

## Performance

Tested on Snapdragon 8 Elite (Samsung S25) — **162M parameters**:

| Phase          | Time    |
|----------------|---------|
| Forward pass   | ~400ms  |
| Backward pass  | ~620ms  |
| Optimizer step | ~270ms  |
| **Full step**  | **~1.4s** |

NEON kernel speedups vs scalar:

| Kernel  | Speedup |
|---------|---------|
| GELU    | **42x** |
| Softmax | **12x** |

---

## What's Inside

**Architecture:**
- GPT-style transformer with RoPE positional embeddings
- Flash Attention — O(T) memory, no T×T matrix materialized
- fp16 weight storage, fp32 compute
- Fused AdamW optimizer in C
- Memory-mapped token dataset

**C/NEON Kernels:**

| Kernel | Description |
|--------|-------------|
| `batched_matmul_bt_f32` | GEMM via OpenBLAS |
| `flash_attention_f32` | Tiled flash attention |
| `fused_qkv_f32` | Single matmul for Q, K, V |
| `gelu_fast_f32` | Padé approximation, no expf |
| `gelu_backward_f32` | NEON vectorized |
| `batched_layernorm_f32` | NEON vectorized |
| `batched_layernorm_backward_f32` | NEON vectorized |
| `adamw_step_f32` | Fused AdamW |
| `softmax_fast_f32` | NEON exp approximation |

---

## Supported Platforms

Any ARMv8.2+ hardware:

| Platform | Status |
|----------|--------|
| Android (Termux) — Snapdragon, Pixel, OnePlus | ✅ Tested |
| Apple Silicon — M1/M2/M3/M4 | ✅ Supported |
| Linux ARM64 — AWS Graviton, Raspberry Pi, Ampere | ✅ Supported |

---

## Setup

**Android (Termux):**
```bash
pkg install clang openblas python
pip install numpy tiktoken --break-system-packages
```

**Linux ARM64:**
```bash
apt install clang libopenblas-dev python3 python3-pip
pip install numpy tiktoken
```

**macOS Apple Silicon:**
```bash
brew install openblas python
pip install numpy tiktoken
```

**Build and train (all platforms):**
```bash
git clone https://github.com/aodalt03-a11y/atlas-engine
cd atlas-engine
clang -O3 -march=armv8.2-a+fp16+simd -ffast-math -funroll-loops \
  -shared -fPIC -o engine/libengine.so engine/matmul.c \
  -lopenblas -lm -lpthread
OPENBLAS_NUM_THREADS=8 python3 -c "from engine.trainer import train; train()"
```

---

## Research: ATLAS Analytical Weights

During development, a significant finding emerged about how transformer weights
relate to corpus statistics.

The dominant singular direction of a trained `lm_head` matrix is predictable
directly from token document frequency — **no gradient descent required**:

```
doc_freq^1.30  →  r = 0.976  (97.4% variance explained)
```

Top 500 tokens reconstructable at **0.986 cosine similarity** from corpus
statistics alone. This suggests a large portion of what transformers learn
during training is analytically derivable beforehand.

Full findings in [`RESEARCH.md`](RESEARCH.md) *(coming soon)*

---

## Roadmap

- [x] ARM64 NEON kernels
- [x] Flash Attention
- [x] fp16 storage
- [x] Fused AdamW
- [ ] Distributed training across devices
- [ ] Analytical weight initialization (ATLAS research)
- [ ] 1B parameter support
- [ ] GGUF export

---

## License

MIT — use it for anything.

---

*Built on a phone. Trained on a phone. No excuses.*

