"""
Atlas Engine — Autograd Tensor
Lightweight autograd with C/NEON backend for transformer training on ARM64.
"""
import numpy as np
import ctypes, os

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libengine.so'))
FP   = ctypes.POINTER(ctypes.c_float)
_ptr = lambda x: x.ctypes.data_as(FP)

_lib.set_threads.argtypes              = [ctypes.c_int]
_lib.set_threads.restype               = None
_lib.batched_matmul_bt_f32.argtypes    = [FP,FP,FP,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
_lib.batched_matmul_bt_f32.restype     = None
_lib.flash_attention_f32.argtypes      = [FP,FP,FP,FP,ctypes.c_float,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
_lib.flash_attention_f32.restype       = None
_lib.fused_qkv_f32.argtypes            = [FP,FP,FP,ctypes.c_int,ctypes.c_int,ctypes.c_int]
_lib.fused_qkv_f32.restype             = None
_lib.batched_layernorm_f32.argtypes    = [FP,FP,FP,FP,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float]
_lib.batched_layernorm_f32.restype     = None
_lib.gelu_fast_f32.argtypes            = [FP,FP,ctypes.c_int]
_lib.gelu_fast_f32.restype             = None
_lib.gelu_backward_f32.argtypes        = [FP,FP,FP,ctypes.c_int]
_lib.gelu_backward_f32.restype         = None
_lib.batched_layernorm_backward_f32.argtypes = [FP,FP,FP,FP,FP,FP,
                                                ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float]
_lib.batched_layernorm_backward_f32.restype  = None
_lib.fused_lmhead_ce_f32.argtypes = [FP, FP,
                                      ctypes.POINTER(ctypes.c_int),
                                      FP, FP, FP,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.fused_lmhead_ce_f32.restype  = None
_lib.adamw_step_f32.argtypes           = [FP,FP,FP,FP,
                                          ctypes.c_float,ctypes.c_float,ctypes.c_float,
                                          ctypes.c_float,ctypes.c_float,
                                          ctypes.c_float,ctypes.c_float,
                                          ctypes.c_int]
_lib.adamw_step_f32.restype            = None
_lib.set_threads(8)

# ── fp16 helpers ─────────────────────────────────────────
def to_f16(x): return x.astype(np.float16)
def to_f32(x): return x.astype(np.float32)

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data          = np.ascontiguousarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad          = None
        self._backward     = lambda: None
        self._prev         = []

    @property
    def shape(self): return self.data.shape

    def _init_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev = [self, other]
        def _back():
            if self.requires_grad:
                self._init_grad(); self.grad += out.grad
            if other.requires_grad:
                other._init_grad()
                g = out.grad
                while g.ndim > other.data.ndim: g = g.sum(axis=0)
                for i,(s,o) in enumerate(zip(other.data.shape, g.shape)):
                    if s==1: g = g.sum(axis=i, keepdims=True)
                other.grad += g
        out._backward = _back
        return out

    def matmul(self, other):
        A  = np.ascontiguousarray(self.data, dtype=np.float32)
        A  = np.ascontiguousarray(self.data, dtype=np.float32)
        WT = other.dataT if hasattr(other, 'dataT') else np.ascontiguousarray(other.data.T, dtype=np.float32)
        if A.ndim == 3:
            B,T,K = A.shape; N = other.data.shape[1]
            C = np.empty((B,T,N), dtype=np.float32)
            _lib.batched_matmul_bt_f32(_ptr(A),_ptr(WT),_ptr(C),B,T,K,N)
        else:
            M,K = A.shape; N = other.data.shape[1]
            C = np.empty((M,N), dtype=np.float32)
            _lib.batched_matmul_bt_f32(_ptr(A),_ptr(WT),_ptr(C),1,M,K,N)
        # capture arrays at forward time to avoid repeated property calls in backward
        _A   = A
        _W   = np.ascontiguousarray(other.data, dtype=np.float32)
        _WT  = other.dataT if hasattr(other, 'dataT') else np.ascontiguousarray(_W.T, dtype=np.float32)
        _self_data = self.data
        out = Tensor(C, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = [self, other]
        def _back():
            if self.requires_grad:
                self._init_grad()
                dC = np.ascontiguousarray(out.grad, dtype=np.float32)
                if dC.ndim == 3:
                    B,T,N = dC.shape; K = _W.shape[0]
                    dA = np.empty((B,T,K), dtype=np.float32)
                    _lib.batched_matmul_bt_f32(_ptr(dC), _ptr(_WT), _ptr(dA),B,T,N,K)
                else:
                    dA = out.grad @ _W
                self.grad += dA
            if other.requires_grad:
                other._init_grad()
                A2  = _self_data; dC2 = out.grad
                other.grad += (A2.reshape(-1,A2.shape[-1]).T @ dC2.reshape(-1,dC2.shape[-1])
                               if A2.ndim==3 else A2.T @ dC2)
        out._backward = _back
        return out

    def gelu(self):
        flat    = np.ascontiguousarray(self.data.ravel(), dtype=np.float32)
        out_flat= np.empty_like(flat)
        _lib.gelu_fast_f32(_ptr(flat),_ptr(out_flat),len(flat))
        out = Tensor(out_flat.reshape(self.data.shape), requires_grad=self.requires_grad)
        out._prev = [self]
        def _back():
            if self.requires_grad:
                self._init_grad()
                xf   = np.ascontiguousarray(self.data.ravel(), dtype=np.float32)
                dout = np.ascontiguousarray(out.grad.ravel(),  dtype=np.float32)
                dxf  = np.empty_like(xf)
                _lib.gelu_backward_f32(_ptr(xf), _ptr(dout), _ptr(dxf), len(xf))
                self.grad += dxf.reshape(self.data.shape)
        out._backward = _back
        return out

    def mean(self):
        out = Tensor(np.array(self.data.mean(),dtype=np.float32),
                     requires_grad=self.requires_grad)
        out._prev = [self]
        def _back():
            if self.requires_grad:
                self._init_grad()
                self.grad += np.full_like(self.data, out.grad/self.data.size)
        out._backward = _back
        return out


    def compile_backward(self):
        """Pre-build topo order once, reuse every step."""
        topo, visited = [], set()
        def build(t):
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._prev: build(p)
                topo.append(t)
        build(self)
        self._compiled_topo = list(reversed(topo))
        return self

    def backward(self):
        if hasattr(self, "_compiled_topo"):
            self.grad = np.ones_like(self.data)
            for t in self._compiled_topo:
                if t._prev: t._backward()
            return
        topo, visited = [], set()
        def build(t):
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._prev: build(p)
                topo.append(t)
        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo): t._backward()
    def __repr__(self):
        return f"Tensor(shape={self.shape})"


def layernorm(x, w, b, eps=1e-5):
    xd  = np.ascontiguousarray(x.data, dtype=np.float32)
    wd  = np.ascontiguousarray(w.data, dtype=np.float32)
    bd  = np.ascontiguousarray(b.data, dtype=np.float32)
    out = np.empty_like(xd)
    B,T,D = xd.shape
    _lib.batched_layernorm_f32(_ptr(xd),_ptr(wd),_ptr(bd),_ptr(out),B,T,D,eps)
    result = Tensor(out, requires_grad=x.requires_grad)
    result._prev = [x, w, b]
    def _back():
        B2,T2,D2 = xd.shape
        if x.requires_grad:
            x._init_grad()
        else:
            pass
        dout = np.ascontiguousarray(result.grad, dtype=np.float32)
        dx   = np.zeros_like(xd)
        dw2  = np.zeros_like(wd)
        db2  = np.zeros_like(bd)
        _lib.batched_layernorm_backward_f32(
            _ptr(xd), _ptr(wd), _ptr(dout),
            _ptr(dx), _ptr(dw2), _ptr(db2),
            B2, T2, D2, eps)
        if x.requires_grad:
            x.grad += dx
        if w.requires_grad:
            w._init_grad()
            w.grad += dw2
        if b.requires_grad:
            b._init_grad()
            b.grad += db2
    result._backward = _back
    return result

def cross_entropy(logits, targets):
    B  = logits.data.shape[0]
    e  = np.exp(logits.data - logits.data.max(axis=-1,keepdims=True))
    p  = e / e.sum(axis=-1,keepdims=True)
    lv = -np.log(p[np.arange(B),targets]+1e-9).mean()
    out= Tensor(np.array(lv,dtype=np.float32), requires_grad=logits.requires_grad)
    out._prev = [logits]
    def _back():
        if logits.requires_grad:
            logits._init_grad()
            g = p.copy(); g[np.arange(B),targets] -= 1
            logits.grad += g/B*out.grad
    out._backward = _back
    return out
