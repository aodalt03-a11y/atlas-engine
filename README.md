# Atlas Engine

ARM64-optimized LLM training engine. Runs on Android, Apple Silicon, and Linux ARM64. No PyTorch required.

## What it is

A from-scratch transformer training engine built for ARM64 hardware. Uses custom C/NEON SIMD kernels and OpenBLAS for fast matrix operations on Snapdragon, Apple Silicon, and ARM64 Linux processors.

Tested on Samsung S25 (Snapdragon 8 Elite) running Termux.

## Supported Platforms

Any ARM64 hardware:
- Android (Termux) - Samsung, Pixel, OnePlus
- Apple Silicon - M1/M2/M3/M4 Mac
- Linux ARM64 - AWS Graviton, Raspberry Pi, Ampere
- Any ARMv8.2+ device

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
- Flash Attention (O(T) memory, no T x T matrix)
- fp16 weight storage, fp32 compute
- Fused AdamW optimizer in C
- Memory-mapped token dataset

## C Kernels

| Kernel | Description |
|--------|-------------|
| batched_matmul_bt_f32 | GEMM via OpenBLAS |
| flash_attention_f32 | Tiled flash attention |
| fused_qkv_f32 | Single matmul for Q, K, V |
| gelu_fast_f32 | Pade approximation, no expf |
| gelu_backward_f32 | NEON vectorized |
| batched_layernorm_f32 | NEON vectorized |
| batched_layernorm_backward_f32 | NEON vectorized |
| adamw_step_f32 | Fused AdamW |
| softmax_fast_f32 | NEON exp approximation |

## Setup

Android (Termux):

    pkg install clang openblas python
    pip install numpy tiktoken --break-system-packages

Linux ARM64:

    apt install clang libopenblas-dev python3 python3-pip
    pip install numpy tiktoken

macOS Apple Silicon:

    brew install openblas python
    pip install numpy tiktoken

Build and train (all platforms):

    git clone https://github.com/aodalt03-a11y/atlas-engine
    cd atlas-engine
    clang -O3 -march=armv8.2-a+fp16+simd -ffast-math -funroll-loops -shared -fPIC -o engine/libengine.so engine/matmul.c -lopenblas -lm -lpthread
    OPENBLAS_NUM_THREADS=8 python3 -c "from engine.trainer import train; train()"

## License

MIT
