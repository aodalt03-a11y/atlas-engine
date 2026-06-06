# Atlas Engine

ARM64-optimized LLM training engine. Runs on Android via Termux. No PyTorch required.

## What it is

A from-scratch transformer training engine built for ARM64 hardware. Uses custom C/NEON SIMD kernels and OpenBLAS for fast matrix operations on Snapdragon processors.

Tested on Samsung S25 (Snapdragon 8 Elite) running Termux.

## Performance (162M params, Snapdragon 8 Elite)

| Phase | Time |
|-------|------|
| Forward pass | ~400ms |
| Backward pass | ~620ms |
| Optimizer step | ~270ms |
| Full step | ~1.4s |
| GELU (NEON) | 42x faster than scalar |
| Softmax (NEON) | 12x faster than scalar |

## Architecture

- GPT with RoPE positional embeddings
- Flash Attention (O(T) memory, no T×T matrix)
- fp16 weight storage, fp32 compute
- Fused AdamW optimizer in C
- Memory-mapped token dataset

## C Kernels

| Kernel | Description |
|--------|-------------|
| `batched_matmul_bt_f32` | GEMM via OpenBLAS |
| `flash_attention_f32` | Tiled flash attention |
| `fused_qkv_f32` | Single matmul for Q, K, V |
| `gelu_fast_f32` | Pade approximation, no expf |
| `gelu_backward_f32` | NEON vectorized |
| `batched_layernorm_f32` | NEON vectorized |
| `batched_layernorm_backward_f32` | NEON vectorized |
| `adamw_step_f32` | Fused AdamW |
| `softmax_fast_f32` | NEON exp approximation |

## Setup (Termux)

```bash
pkg install clang openblas python
pip install numpy tiktoken --break-system-packages

git clone https://github.com/aodalt03-a11y/atlas-engine
cd atlas-engine

# Compile kernels
clang -O3 -march=armv8.2-a+fp16+simd -ffast-math -funroll-loops \
  -shared -fPIC \
  -I/data/data/com.termux/files/usr/include/openblas \
  -o engine/libengine.so engine/matmul.c \
  -L/data/data/com.termux/files/usr/lib -lopenblas -lm -lpthread

# Train
OPENBLAS_NUM_THREADS=8 python3 -c "from engine.trainer import train; train()"
License
MIT
