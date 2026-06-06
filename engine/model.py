"""
Atlas Engine — GPT Model
Transformer architecture with RoPE, Flash Attention, and fp16 weight storage.
"""
import numpy as np
import os, ctypes
from .tensor import Tensor, layernorm, cross_entropy, _lib, _ptr, FP, to_f16, to_f32

# ── RoPE ─────────────────────────────────────────────────
def build_rope(seq_len, head_dim, base=10000):
    i    = np.arange(0, head_dim, 2, dtype=np.float32)
    inv  = 1.0 / (base ** (i / head_dim))
    pos  = np.arange(seq_len, dtype=np.float32)
    freq = np.outer(pos, inv)           # (T, head_dim/2)
    cos  = np.cos(freq).astype(np.float32)
    sin  = np.sin(freq).astype(np.float32)
    return cos, sin

def apply_rope(x, cos, sin):
    # x: (B, T, n_head, head_dim)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos2 = cos[np.newaxis, :, np.newaxis, :]
    sin2 = sin[np.newaxis, :, np.newaxis, :]
    rx1 = x1*cos2 - x2*sin2
    rx2 = x1*sin2 + x2*cos2
    out = np.empty_like(x)
    out[..., ::2]  = rx1
    out[..., 1::2] = rx2
    return out

# ── Module base ───────────────────────────────────────────
class Parameter(Tensor):
    def __init__(self, data, fp32=False):
        self.requires_grad = True
        self.grad          = None
        self._backward     = lambda: None
        self._prev         = []
        self._fp32_only    = fp32
        self._dirty        = False
        if fp32:
            self._fp16   = None
            self._cache  = np.ascontiguousarray(data, dtype=np.float32)
        else:
            self._fp16   = to_f16(data)
            self._cache  = np.ascontiguousarray(to_f32(self._fp16))
        self._cacheT = np.ascontiguousarray(self._cache.T)

    @property
    def data(self): return self._cache
    @data.setter
    def data(self, v):
        if self._fp32_only:
            self._cache = np.ascontiguousarray(v, dtype=np.float32)
        else:
            self._fp16  = to_f16(v)
            self._cache = np.ascontiguousarray(to_f32(self._fp16))
        self._dirty = True

    @property
    def dataT(self):
        if self._dirty:
            self._cacheT = np.ascontiguousarray(self._cache.T)
            self._dirty  = False
        return self._cacheT

class Module:
    def parameters(self):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter): params.append(v)
            elif isinstance(v, Module):  params.extend(v.parameters())
            elif isinstance(v, list):
                for i in v:
                    if isinstance(i, Module): params.extend(i.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters(): p.grad = None

class Embedding(Module):
    def __init__(self, n, d):
        self.weight = Parameter(np.random.randn(n,d).astype(np.float32)*0.02, fp32=True)
    def __call__(self, idx):
        return Tensor(self.weight.data[idx])

class Linear(Module):
    def __init__(self, i, o):
        self.weight = Parameter(np.random.randn(i,o).astype(np.float32)*0.02)
    def __call__(self, x):
        return x.matmul(self.weight)

class LayerNorm(Module):
    def __init__(self, d):
        self.w = Parameter(np.ones(d,  dtype=np.float32))
        self.b = Parameter(np.zeros(d, dtype=np.float32))
    def __call__(self, x): return layernorm(x, self.w, self.b)
    def parameters(self): return [self.w, self.b]

# ── Flash Attention + RoPE ────────────────────────────────
class CausalSelfAttention(Module):
    def __init__(self, d_model, n_head):
        self.d_model  = d_model
        self.n_head   = n_head
        self.head_dim = d_model // n_head
        self.W_qkv    = Parameter(np.random.randn(d_model, 3*d_model).astype(np.float32)*0.02)
        self.W_o      = Parameter(np.random.randn(d_model, d_model).astype(np.float32)*0.02)

    def __call__(self, x, cos, sin):
        B,T,D = x.data.shape
        xd    = np.ascontiguousarray(x.data, dtype=np.float32)

        # Fused QKV
        qkv = np.empty((B,T,3*D), dtype=np.float32)
        _lib.fused_qkv_f32(_ptr(xd),
            _ptr(np.ascontiguousarray(self.W_qkv.data,dtype=np.float32)),
            _ptr(qkv), B, T, D)

        # Split and apply RoPE per head
        Q = qkv[:,:,:D].reshape(B,T,self.n_head,self.head_dim)
        K = qkv[:,:,D:2*D].reshape(B,T,self.n_head,self.head_dim)
        V = np.ascontiguousarray(qkv[:,:,2*D:])

        Q = apply_rope(Q, cos[:T], sin[:T]).reshape(B,T,D)
        K = apply_rope(K, cos[:T], sin[:T]).reshape(B,T,D)

        Q = np.ascontiguousarray(Q, dtype=np.float32)
        K = np.ascontiguousarray(K, dtype=np.float32)
        V = np.ascontiguousarray(V, dtype=np.float32)

        # Flash attention — no T×T matrix
        ctx = np.empty((B,T,D), dtype=np.float32)
        _lib.flash_attention_f32(
            _ptr(Q), _ptr(K), _ptr(V), _ptr(ctx),
            1.0/np.sqrt(self.head_dim), B, T, D, 32)

        ctx_t = Tensor(ctx, requires_grad=x.requires_grad)
        ctx_t._prev = [x]
        def _back():
            if x.requires_grad:
                x._init_grad()
                x.grad += ctx_t.grad @ self.W_qkv.data[:,:D].T
        ctx_t._backward = _back
        return ctx_t.matmul(self.W_o)

    def parameters(self): return [self.W_qkv, self.W_o]

    def infer_one(self, x, cos, sin, cache):
        # x: (1, 1, D) — single new token
        # cache: dict with "K": (1, t, D), "V": (1, t, D) or None
        D = self.d_model
        xd = np.ascontiguousarray(x, dtype=np.float32)  # (1,1,D)
        qkv = np.empty((1,1,3*D), dtype=np.float32)
        _lib.fused_qkv_f32(_ptr(xd),
            _ptr(np.ascontiguousarray(self.W_qkv.data, dtype=np.float32)),
            _ptr(qkv), 1, 1, D)
        Q = qkv[:,:,:D].reshape(1,1,self.n_head,self.head_dim)
        K = qkv[:,:,D:2*D].reshape(1,1,self.n_head,self.head_dim)
        V = np.ascontiguousarray(qkv[:,:,2*D:])  # (1,1,D)
        # apply RoPE at position t
        t = cache["K"].shape[1] if cache["K"] is not None else 0
        Q = apply_rope(Q, cos[t:t+1], sin[t:t+1]).reshape(1,1,D)
        K = apply_rope(K, cos[t:t+1], sin[t:t+1]).reshape(1,1,D)
        # append to cache
        if cache["K"] is None:
            cache["K"] = K
            cache["V"] = V
        else:
            cache["K"] = np.concatenate([cache["K"], K], axis=1)
            cache["V"] = np.concatenate([cache["V"], V], axis=1)
        # standard attention: Q(1,1,D) x K(1,T,D).T -> scores(1,1,T)
        Kc = np.ascontiguousarray(cache["K"].reshape(1,-1,D), dtype=np.float32)
        Vc = np.ascontiguousarray(cache["V"].reshape(1,-1,D), dtype=np.float32)
        Qc = np.ascontiguousarray(Q.reshape(1,1,D), dtype=np.float32)
        T2 = Kc.shape[1]
        scale = 1.0 / np.sqrt(self.head_dim)
        # scores: (1, T2)
        scores = (Qc.reshape(1,D) @ Kc.reshape(T2,D).T) * scale  # (1,T2)
        scores = scores - scores.max()
        w = np.exp(scores)
        w = w / w.sum()
        # context: (1, D)
        ctx = (w.reshape(1,T2) @ Vc.reshape(T2,D)).reshape(1,1,D)
        # output projection
        out = ctx.reshape(1,D) @ self.W_o.data
        return out.reshape(1,1,D)

class MLP(Module):
    def __init__(self, d):
        self.fc1 = Linear(d, 4*d)
        self.fc2 = Linear(4*d, d)
    def __call__(self, x):
        return self.fc2(self.fc1(x).gelu())

class Block(Module):
    def __init__(self, d, n_head):
        self.ln1  = LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.ln2  = LayerNorm(d)
        self.mlp  = MLP(d)

    def __call__(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

    def parameters(self):
        return (self.ln1.parameters() + self.attn.parameters() +
                self.ln2.parameters() + self.mlp.parameters())

    def infer_one(self, x, cos, sin, cache):
        # x: (1,1,D) numpy array
        xd = x
        # ln1
        from .tensor import Tensor
        xt = Tensor(xd.reshape(1,1,-1))
        ln1_out = self.ln1(xt).data
        # attn
        attn_out = self.attn.infer_one(ln1_out, cos, sin, cache)
        xd = xd + attn_out
        # ln2
        xt2 = Tensor(xd.reshape(1,1,-1))
        ln2_out = self.ln2(xt2).data
        # mlp
        h = ln2_out.reshape(1,1,-1) @ self.mlp.fc1.weight.data
        # gelu fast via C
        from .tensor import _lib, _ptr, FP
        import ctypes
        hf = np.ascontiguousarray(h.ravel(), dtype=np.float32)
        gf = np.empty_like(hf)
        _lib.gelu_fast_f32(hf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           gf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(hf))
        mlp_out = gf.reshape(1,1,-1) @ self.mlp.fc2.weight.data
        xd = xd + mlp_out
        return xd

class GPT(Module):
    def __init__(self, vocab_size=8000, d_model=768, n_head=12,
                 n_layer=12, block_size=512):
        self.d_model    = d_model
        self.n_head     = n_head
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.wte        = Embedding(vocab_size, d_model)
        self.blocks     = [Block(d_model, n_head) for _ in range(n_layer)]
        self.ln_f       = LayerNorm(d_model)
        self.lm_head    = Parameter(
            np.random.randn(d_model, vocab_size).astype(np.float32)*0.02, fp32=True)
        # weight tying
        self.wte.weight = Parameter(self.lm_head.data.T.copy())
        # RoPE cache
        head_dim = d_model // n_head
        self.cos, self.sin = build_rope(block_size, head_dim)
        n = sum((p._fp16.size*2 if p._fp16 is not None else p._cache.size*4) for p in self.parameters())
        print(f"GPT params: {sum((p._fp16.size if p._fp16 is not None else p._cache.size) for p in self.parameters())/1e6:.1f}M  "
              f"| fp16 RAM: {n/1e9:.2f}GB")

    def __call__(self, idx, targets=None):
        B,T   = idx.shape
        tok   = self.wte(idx)
        x     = Tensor(tok.data, requires_grad=True)
        cos   = self.cos[:T]
        sin   = self.sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        logits_d = x.data.reshape(B*T, self.d_model) @ self.lm_head.data
        logits   = Tensor(logits_d, requires_grad=True)
        logits._prev = [x]
        def _back():
            if x.requires_grad:
                x._init_grad()
                # sparse dX: only cols that appeared in targets
                _tgt = targets.ravel() if targets is not None else None
                if _tgt is not None:
                    unique = np.unique(_tgt)
                    # dX = dlogits[:, unique] @ W[unique].T  — sparse
                    x.grad += (logits.grad[:, unique] @ self.lm_head.data[:, unique].T).reshape(B,T,self.d_model)
                else:
                    x.grad += (logits.grad @ self.lm_head.dataT).reshape(B,T,self.d_model)
            if self.lm_head.requires_grad:
                if self.lm_head.grad is None:
                    self.lm_head.grad = np.zeros_like(self.lm_head.data)
                # sparse dW: only update columns that appeared
                _tgt = targets.ravel() if targets is not None else None
                if _tgt is not None:
                    unique = np.unique(_tgt)
                    xr = x.data.reshape(B*T, self.d_model)
                    self.lm_head.grad[:, unique] += xr.T @ logits.grad[:, unique]
                else:
                    self.lm_head.grad += x.data.reshape(B*T,self.d_model).T @ logits.grad
        logits._backward = _back
        loss = None
        if targets is not None:
            loss = cross_entropy(logits, targets.ravel())
        return logits, loss

    def generate(self, idx, max_new=200, temperature=0.8, top_k=50):
        B, T = idx.shape
        # init KV cache per layer
        caches = [{"K": None, "V": None} for _ in self.blocks]
        # prefill: run full forward on prompt to populate cache
        tok = self.wte.weight.data[idx[0]]  # (T, D)
        x = tok.copy()
        for i, block in enumerate(self.blocks):
            from .tensor import Tensor
            xt = Tensor(x.reshape(1, T, -1))
            ln1_out = block.ln1(xt).data
            # run full attention to fill cache
            D = self.d_model
            xd = np.ascontiguousarray(x.reshape(1,T,D), dtype=np.float32)
            from .tensor import _lib, _ptr
            qkv = np.empty((1,T,3*D), dtype=np.float32)
            _lib.fused_qkv_f32(_ptr(xd),
                _ptr(np.ascontiguousarray(block.attn.W_qkv.data, dtype=np.float32)),
                _ptr(qkv), 1, T, D)
            K = qkv[:,:,D:2*D].reshape(1,T,block.attn.n_head,block.attn.head_dim)
            V = np.ascontiguousarray(qkv[:,:,2*D:])
            K = apply_rope(K, self.cos[:T], self.sin[:T]).reshape(1,T,D)
            caches[i]["K"] = np.ascontiguousarray(K, dtype=np.float32)
            caches[i]["V"] = np.ascontiguousarray(V, dtype=np.float32)
            # full forward for x
            xt2 = Tensor(x.reshape(1,T,D))
            x = block(xt2, self.cos[:T], self.sin[:T]).data.reshape(T,D)
        from .tensor import Tensor
        xf = Tensor(x.reshape(1,T,D))
        x = self.ln_f(xf).data.reshape(T,D)
        # take last token logits
        last = x[-1:]  # (1,D)
        logits_d = last @ self.lm_head.data  # (1,vocab)

        for step in range(max_new):
            l = logits_d / temperature
            if top_k:
                kth = np.sort(l, axis=-1)[:, -top_k:-top_k+1]
                l   = np.where(l < kth, -1e9, l)
            e = np.exp(l - l.max(axis=-1, keepdims=True))
            p = e / e.sum(axis=-1, keepdims=True)
            nxt = np.random.choice(p.shape[-1], p=p[0])
            idx = np.concatenate([idx, np.array([[nxt]])], axis=1)
            # single token forward with KV cache
            tok1 = self.wte.weight.data[nxt:nxt+1]  # (1,D)
            xd = tok1.reshape(1,1,D)
            for i, block in enumerate(self.blocks):
                xd = block.infer_one(xd, self.cos, self.sin, caches[i])
            from .tensor import Tensor
            xf = Tensor(xd.reshape(1,1,D))
            xd = self.ln_f(xf).data.reshape(1,D)
            logits_d = xd @ self.lm_head.data
        return idx

    def parameters(self):
        p = [self.wte.weight, self.lm_head]
        for b in self.blocks: p.extend(b.parameters())
        p.extend(self.ln_f.parameters())
        return p
