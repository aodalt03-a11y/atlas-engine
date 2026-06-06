"""
Atlas Engine — Trainer
AdamW optimizer with fused C kernels, gradient accumulation, and mmap token dataset.
"""
import numpy as np
import os, time, pickle, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))

from engine.model import GPT
from engine.tensor import _lib, _ptr, FP
from tokenizer import BPETokenizer

class AdamW:
    def __init__(self, params, lr=1e-4, betas=(0.9,0.95), wd=0.1, eps=1e-8, opt_usb=None):
        self.params = params
        self.lr     = lr
        self.b1,self.b2 = betas
        self.wd     = wd
        self.eps    = eps
        self.t      = 0
        self.usb    = opt_usb

        if self.usb is None:
            self.m = [np.zeros(p.data.size, dtype=np.float32) for p in params]
            self.v = [np.zeros(p.data.size, dtype=np.float32) for p in params]
        else:
            self.m = None
            self.v = None
            print("AdamW: m/v moments offloaded to memfile")
            for i,p in enumerate(params):
                self.usb.store(f'm{i}', np.zeros(p.data.size, dtype=np.float32))
                self.usb.store(f'v{i}', np.zeros(p.data.size, dtype=np.float32))

    def step(self):
        self.t += 1
        b1t = 1 - self.b1**self.t
        b2t = 1 - self.b2**self.t
        for i,p in enumerate(self.params):
            if p.grad is None: continue
            pd = np.ascontiguousarray(p.data.ravel(), dtype=np.float32)
            gd = np.ascontiguousarray(p.grad.ravel(), dtype=np.float32)

            if self.usb:
                mi = self.usb.load(f'm{i}')
                vi = self.usb.load(f'v{i}')
            else:
                mi, vi = self.m[i], self.v[i]

            _lib.adamw_step_f32(
                _ptr(pd), _ptr(gd),
                _ptr(mi), _ptr(vi),
                self.lr, self.b1, self.b2, self.eps, self.wd,
                b1t, b2t, len(pd))

            if self.usb:
                self.usb.store(f'm{i}', mi)
                self.usb.store(f'v{i}', vi)

            p.data = pd.reshape(p.data.shape)

    def zero_grad(self):
        for p in self.params: p.grad = None

class Config:
    vocab_size  = 50257
    d_model     = 768
    n_head      = 12
    n_layer     = 12
    block_size  = 128
    batch_size  = 1
    grad_accum  = 2
    lr          = 1e-4
    max_steps   = 20000
    save_every  = 100
    eval_every  = 500
    warmup      = 200
    checkpoint  = "checkpoints"
    cache_path  = "checkpoints/ids.pkl"
    data_path   = "data/mixed.txt"

def get_lr(step, cfg):
    if step < cfg.warmup:
        return cfg.lr * step / max(1, cfg.warmup)
    p = (step-cfg.warmup) / max(1, cfg.max_steps-cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi*p))

def get_batch(ids, cfg):
    ix = np.random.randint(0, len(ids)-cfg.block_size-1, cfg.batch_size)
    x  = np.stack([ids[i:i+cfg.block_size]   for i in ix])
    y  = np.stack([ids[i+1:i+cfg.block_size+1] for i in ix])
    return x, y

def save(ckpt, params, step):
    # streaming save — write params one at a time to avoid 600MB spike
    import zipfile, io
    tmp = ckpt + '.tmp'
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED) as zf:
        for i, p in enumerate(params):
            buf = io.BytesIO()
            np.save(buf, p.data)
            zf.writestr(f'p{i}.npy', buf.getvalue())
        buf = io.BytesIO()
        np.save(buf, np.array(step))
        zf.writestr('step.npy', buf.getvalue())
    os.replace(tmp, ckpt)

def load(ckpt, params):
    import zipfile, io
    with zipfile.ZipFile(ckpt, 'r') as zf:
        names = zf.namelist()
        keys = sorted([n for n in names if n.startswith("p")],
                      key=lambda x: int(x[1:-4]))
        for p, k in zip(params, keys):
            p.data = np.load(io.BytesIO(zf.read(k)))
        step = int(np.load(io.BytesIO(zf.read('step.npy'))))
    return step

def train(use_memfile=True):
    cfg = Config()
    if not os.path.exists(cfg.checkpoint):
        os.mkdir(cfg.checkpoint)

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    print("Tokenizer loaded.")

    # memory-mapped token dataset — never loads into RAM
    mmap_path = cfg.cache_path.replace('.pkl', '.bin')
    if os.path.exists(mmap_path):
        ids = np.memmap(mmap_path, dtype=np.int32, mode='r')
        print(f"Tokens mmap: {len(ids)}")
    elif os.path.exists(cfg.cache_path):
        # migrate old pkl to mmap
        print("Migrating pkl to mmap...")
        with open(cfg.cache_path,'rb') as f: old_ids = pickle.load(f)
        ids_arr = np.array(old_ids, dtype=np.int32)
        mm = np.memmap(mmap_path, dtype=np.int32, mode='w+', shape=(len(ids_arr),))
        mm[:] = ids_arr; mm.flush(); del mm, old_ids, ids_arr
        ids = np.memmap(mmap_path, dtype=np.int32, mode='r')
        print(f"Tokens mmap: {len(ids)}")
    else:
        print("Encoding...")
        encoded = tok.encode(open(cfg.data_path).read(), disallowed_special=())
        ids_arr = np.array(encoded, dtype=np.int32)
        mm = np.memmap(mmap_path, dtype=np.int32, mode='w+', shape=(len(ids_arr),))
        mm[:] = ids_arr; mm.flush(); del mm, encoded, ids_arr
        ids = np.memmap(mmap_path, dtype=np.int32, mode='r')
        print(f"Tokens mmap: {len(ids)}")
    model  = GPT(cfg.vocab_size,cfg.d_model,cfg.n_head,cfg.n_layer,cfg.block_size)
    params = model.parameters()

    opt_usb = None
    if use_memfile:
        from engine.usbram import OptimizerUSB
        opt_usb = OptimizerUSB()

    opt = AdamW(params, lr=cfg.lr, opt_usb=opt_usb)

    ckpt = f"{cfg.checkpoint}/model_engine.npz"
    step = 0
    if os.path.exists(ckpt):
        step = load(ckpt, params)
        print(f"Resumed from step {step}")

    print("Training started.")
    t0 = time.time()

    _compiled = False

    while step < cfg.max_steps:
        lr = get_lr(step, cfg)
        opt.lr = lr
        t0 = time.time()

        # gradient accumulation
        accum_loss = 0.0
        for acc in range(cfg.grad_accum):
            x, y = get_batch(ids, cfg)
            _, loss = model(x, y)
            if not _compiled:
                loss.compile_backward()
                _compiled = True
                print("Backward compiled.")
            loss.backward()
            accum_loss += loss.data.item()

        # scale gradients
        if cfg.grad_accum > 1:
            for p in params:
                if p.grad is not None:
                    p.grad /= cfg.grad_accum

        # grad clip (in-place, no temp alloc)
        norm_sq = 0.0
        for p in params:
            if p.grad is not None:
                norm_sq += float(np.dot(p.grad.ravel(), p.grad.ravel()))
        norm = np.sqrt(norm_sq)
        if norm > 1.0:
            scale = 1.0/norm
            for p in params:
                if p.grad is not None: p.grad *= scale

        opt.step()
        opt.zero_grad()
        # clear computation graph to free memory
        for p in params:
            p._prev = []
        del loss
        import gc; gc.collect()
        # trim malloc arenas back to OS every 10 steps
        if step % 10 == 0:
            import ctypes
            try:
                ctypes.CDLL('libc.so').malloc_trim(0)
            except: pass
        step += 1
        print(f"step {step} | loss {accum_loss/cfg.grad_accum:.4f} | lr {lr:.2e} | {time.time()-t0:.1f}s")

        if step % cfg.save_every == 0:
            save(ckpt, params, step)
            print(f"  >> saved step {step}")

    print("Training complete.")
    save(ckpt, params, step)

if __name__ == "__main__":
    train()
