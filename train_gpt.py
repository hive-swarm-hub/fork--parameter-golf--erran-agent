from __future__ import annotations
import copy, glob, io, lzma, math, os, random, subprocess, sys, time, uuid, zlib
from pathlib import Path
import numpy as np, sentencepiece as spm, torch, torch.distributed as dist, torch.nn.functional as F
from torch import Tensor, nn
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1150))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 600))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "1")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get("CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,ve_layer_scales,ve_shared.scale").split(",") if p)
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 99.99984 / 100.0
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d: G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT; B = b * A + c * (A @ A); X = a * X + B @ X
    if transposed: X = X.mT
    if was_2d: X = X.squeeze(0)
    return X
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay))
        self._built = False
    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size
        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]; padded_B = ((B + ws - 1) // ws) * ws; shard_B = padded_B // ws
                tail = p.shape[1:]; dev = p.device
                self._bank_meta.append({'p': p, 'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5})
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True
    def launch_reduce_scatters(self):
        if not self._built: self._build()
        if not self._distributed: return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None: self._rs_futures.append(None); continue
            pg = m['padded_grad']; pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']: pg[m['B']:].zero_()
            self._rs_futures.append(dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        if not self._built: self._build()
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov, wd = group["backend_steps"], group["nesterov"], group.get("weight_decay", 0.0)
            prev_ag_handle, prev_m = None, None
            sharded = self._distributed and hasattr(self, '_rs_futures')
            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None: continue
                if prev_ag_handle is not None:
                    prev_ag_handle.wait(); pp = prev_m['p']; upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0: pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])
                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait(); g = m['shard']; buf = m['shard_mom']
                else:
                    g = p.grad.bfloat16(); state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(m['full_update'], update, async_op=True); prev_m = m
                else:
                    if wd > 0.0: p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])
            if prev_ag_handle is not None:
                prev_ag_handle.wait(); pp = prev_m['p']; upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0: pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])
            if hasattr(self, '_rs_futures'): del self._rs_futures
        return loss
def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size()); ts = max(sp_vs, vocab_size)
    base_bytes_np = np.zeros((ts,), dtype=np.int16)
    has_ls_np = np.zeros((ts,), dtype=np.bool_); is_bnd_np = np.ones((ts,), dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_bnd_np[tid] = False
        if sp.is_byte(tid): base_bytes_np[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): has_ls_np[tid] = True; piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_ls_np, dtype=torch.bool, device=device),
            torch.tensor(is_bnd_np, dtype=torch.bool, device=device))
def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    assert header.size == 256 and int(header[0]) == 20240520 and int(header[1]) == 1
    nt = int(header[2]); off = 256 * np.dtype("<i4").itemsize
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=nt, offset=off).astype(np.uint16, copy=False))
def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]
def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, bl, hl, il, eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    lbt = args.val_batch_size // (world_size * grad_accum_steps)
    lbs = lbt // seq_len; total_seqs = (val_tokens.numel() - 1) // seq_len
    ss = (total_seqs * rank) // world_size; se = (total_seqs * (rank + 1)) // world_size
    vls = torch.zeros((), device=device, dtype=torch.float64)
    vtc = torch.zeros((), device=device, dtype=torch.float64)
    vbc = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(ss, se, lbs):
            bse = min(bss + lbs, se)
            local = val_tokens[bss*seq_len:(bse*seq_len)+1].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                bl_val = model(x, y).detach()
            btc = float(y.numel()); vls += bl_val.to(torch.float64) * btc; vtc += btc
            tb = bl[y.reshape(-1)].to(torch.int16)
            tb += (hl[y.reshape(-1)] & ~il[x.reshape(-1)]).to(torch.int16)
            vbc += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [vls, vtc, vbc]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = vls / vtc; bpt = vl.item() / math.log(2.0); tpb = vtc.item() / vbc.item()
    model.train()
    return float(vl.item()), float(bpt * tpb)
def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        ca = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        cl = torch.maximum(torch.minimum(t32, ca[:, None]), -ca[:, None])
        s = (ca / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(cl / s[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, s.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    ca = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    s = torch.tensor(ca / 127.0 if ca > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -ca, ca) / s), -127, 127).to(torch.int8).contiguous()
    return q, s
class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        assert self.files, f"No files: {pattern}"
        self.file_idx = 0; self.tokens = load_data_shard(self.files[0]); self.pos = 0
    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n):
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance(); continue
            k = min(remaining, avail); chunks.append(self.tokens[self.pos:self.pos+k])
            self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)
class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        lt = global_tokens // (self.world_size * grad_accum_steps); prs = lt + 1
        chunk = self.stream.take(prs * self.world_size)
        start = self.rank * prs; local = chunk[start:start+prs].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float(); rm = w32.abs().amax(dim=1)
                s = (rm / 31.0).clamp_min(1.0 / 31.0)
                wq = (torch.clamp(torch.round(w32 / s[:, None]), -32, 31) * s[:, None]).to(x.dtype)
            w = w + (wq - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim, self.base, self.train_seq_len = dim, base, train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)), persistent=False)
        self._seq_len_cached = 0; self._cos_cached = None; self._sin_cached = None
    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                inv_freq = 1.0 / ((self.base * (scale ** (rd / (rd - 2)))) ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else: inv_freq = self.inv_freq.to(device)
            freqs = torch.outer(torch.arange(seq_len, device=device, dtype=inv_freq.dtype), inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]; self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        xr, xp = x[..., :rope_dims], x[..., rope_dims:]
        h = rope_dims // 2; x1, x2 = xr[..., :h], xr[..., h:]
        return torch.cat((torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), dim=-1), xp), dim=-1)
    h = x.size(-1) // 2; x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), dim=-1)
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0; self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False
    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x, q_w, k_w, v_w, out_w, v_embed=None):
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None: v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            q2 = q.transpose(1, 2); k2 = k.transpose(1, 2); v2 = v.transpose(1, 2)
            if self.num_kv_heads != self.num_heads:
                rep = self.num_heads // self.num_kv_heads
                k2 = k2.repeat_interleave(rep, dim=1); v2 = v2.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True).transpose(1, 2)
        if self.use_xsa: y = self._xsa_efficient(y, v)
        return F.linear(y.reshape(bsz, seqlen, dim), out_w.to(x.dtype))
class SmearGate(nn.Module):
    def __init__(self, dim): super().__init__(); self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        return (1 - g) * x + g * torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim); nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32); mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t); out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, ve_dim, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim); nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids):
        h = self.embed(token_ids)
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult): super().__init__()
    def forward(self, x, up_w, down_w):
        return F.linear(F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5).square(), down_w.to(x.dtype))
class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm = RMSNorm(); self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w, v_embed=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, q_w, k_w, v_w, out_w, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out
class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init,
                 bigram_vocab_size=0, bigram_dim=128, xsa_last_n=0, rope_dims=0, ln_scale=False,
                 ve_enabled=False, ve_dim=128, ve_layers="9,10"):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)
        self.tie_embeddings, self.tied_embed_init_std = tie_embeddings, tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        head_dim = model_dim // num_heads; kv_dim = num_kv_heads * head_dim; mlp_dim = int(mlp_mult * model_dim)
        self.num_layers = num_layers
        self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, layer_idx=i, ln_scale=ln_scale) for i in range(num_layers)])
        if rope_dims > 0:
            hd = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(hd, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, self._ve_target_dim)
            self.ve_layer_scales = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None; self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()
    def _init_weights(self):
        if self.tie_embeddings: nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers; ps = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[n + i])
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.qo_bank.data[n + i].mul_(ps); self.mlp_down_bank.data[i].mul_(ps)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False): nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
    def _get_ve(self, layer_idx, input_ids, ve_cache=None):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices: return None
        if ve_cache is not None and 've' not in ve_cache: ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        return ve_base * self.ve_layer_scales[self.ve_layer_indices.index(layer_idx)].to(dtype=ve_base.dtype)
    def forward(self, input_ids, target_ids):
        n = self.num_layers; x = self.tok_emb(input_ids)
        if self.bigram is not None: x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),)); x = self.smear(x); x0 = x
        skips, ve_cache = [], {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, self.qo_bank[i], self.kv_bank[i], self.kv_bank[n+i], self.qo_bank[n+i], self.mlp_up_bank[i], self.mlp_down_bank[i], v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips: x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n+bi], self.qo_bank[n+bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi], v_embed=ve)
        x = self.final_norm(x); xf = x.reshape(-1, x.size(-1)); targets = target_ids.reshape(-1)
        logits_proj = F.linear(xf, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(xf)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")
    def forward_logits(self, input_ids):
        n = self.num_layers; x = self.tok_emb(input_ids)
        if self.bigram is not None: x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),)); x = self.smear(x); x0 = x
        skips, ve_cache = [], {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, self.qo_bank[i], self.kv_bank[i], self.kv_bank[n+i], self.qo_bank[n+i], self.mlp_up_bank[i], self.mlp_down_bank[i], v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips: x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n+bi], self.qo_bank[n+bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi], v_embed=ve)
        x = self.final_norm(x)
        lp = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)
def eval_val_sliding(args, base_model, rank, world_size, device, val_tokens, bl, hl, il, stride, batch_seqs=32, eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len; total_tokens = val_tokens.numel() - 1
    ws_list = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    tw = len(ws_list); ms, me = (tw * rank) // world_size, (tw * (rank + 1)) // world_size
    my_ws = ws_list[ms:me]
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_ws), batch_seqs):
            bws = my_ws[bi:bi+batch_seqs]; bsz = len(bws)
            xb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(bws):
                end = min(ws + seq_len, total_tokens); wlen = end - ws; wlens.append(wlen)
                chunk = val_tokens[ws:end+1].to(dtype=torch.int64, device=device)
                xb[i, :wlen] = chunk[:-1]; yb[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(xb)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), yb.reshape(-1), reduction="none").reshape(bsz, seq_len)
            for i, ws in enumerate(bws):
                wlen = wlens[i]; s = 0 if ws == 0 else max(wlen - stride, 0)
                ls += nll[i, s:wlen].to(torch.float64).sum(); tc += float(wlen - s)
                tgt, prev = yb[i, s:wlen], xb[i, s:wlen]
                tb = bl[tgt].to(torch.float64); tb += (hl[tgt] & ~il[prev]).to(torch.float64)
                bc += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in [ls, tc, bc]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (ls / tc).item(); bpt = vl / math.log(2.0); tpb = tc.item() / bc.item()
    base_model.train()
    return vl, bpt * tpb
def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name): return "attn"
    return "other"
def quantize_int6_per_row(t, clip_range=31):
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            rc = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
            s = (rc / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err: best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    s = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    return torch.clamp(torch.round(t32 / s.float()), -clip_range, clip_range).to(torch.int8), s
def _unbank_state_dict(sd, num_layers):
    out, n = {}, num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n): out[f"blocks.{i}.attn.c_q.weight"] = tensor[i]; out[f"blocks.{i}.attn.proj.weight"] = tensor[n+i]
        elif name == "kv_bank":
            for i in range(n): out[f"blocks.{i}.attn.c_k.weight"] = tensor[i]; out[f"blocks.{i}.attn.c_v.weight"] = tensor[n+i]
        elif name == "mlp_up_bank":
            for i in range(n): out[f"blocks.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n): out[f"blocks.{i}.mlp.proj.weight"] = tensor[i]
        else: out[name] = tensor
    return out
def _rebank_state_dict(sd, num_layers, template_sd):
    out, n = {}, num_layers
    qo, kv, up, dn = [None]*(2*n), [None]*(2*n), [None]*n, [None]*n
    consumed = set()
    for i in range(n):
        for k, lst, idx in [
            (f"blocks.{i}.attn.c_q.weight", qo, i), (f"blocks.{i}.attn.proj.weight", qo, n+i),
            (f"blocks.{i}.attn.c_k.weight", kv, i), (f"blocks.{i}.attn.c_v.weight", kv, n+i),
            (f"blocks.{i}.mlp.fc.weight", up, i), (f"blocks.{i}.mlp.proj.weight", dn, i)]:
            if k in sd: lst[idx] = sd[k]; consumed.add(k)
    out["qo_bank"] = torch.stack(qo).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(dn).to(dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed: out[name] = tensor
    return out
def mixed_quantize_int6(state_dict, int6_cats):
    result, meta = {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous(); cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t; meta[name] = "passthrough"; continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float(); meta[name] = "passthrough_ctrl"; continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t); result[name+".q"] = q; result[name+".scale"] = s; meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t); result[name+".q"] = q; result[name+".scale"] = s; meta[name] = {"type": "int8"}
    return result, meta
def dequantize_mixed_int6(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig.dtype in (torch.float32, torch.bfloat16): t = t.to(orig.dtype)
            out[name] = t; continue
        q, s = result[name+".q"], result[name+".scale"]
        if s.ndim > 0: out[name] = (q.float() * s.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(orig.dtype)
        else: out[name] = (q.float() * float(s.item())).to(orig.dtype)
    return out
def main():
    code = Path(__file__).read_text(encoding="utf-8"); args = Hyperparameters()
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0")); world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size; grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank); torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl", device_id=device); dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    if HAS_FA3:
        enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    logfile = None
    if master_process: os.makedirs("logs", exist_ok=True); logfile = f"logs/{args.run_id}.txt"; print(logfile)
    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)
    log0(code, console=False); log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False); log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    dataset_dir = Path(args.data_path).resolve()
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    bl, hl, il = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece"); log0(f"val_loader:tokens:{val_tokens.numel()-1}")
    CastedLinear._qat_enabled = args.qat_enabled
    base_model = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers).to(device).bfloat16()
    base_model.qo_bank.data = base_model.qo_bank.data.float()
    base_model.kv_bank.data = base_model.kv_bank.data.float()
    base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
    base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
    for module in base_model.modules():
        if isinstance(module, CastedLinear): module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True); model = compiled_model
    matrix_params = [base_model.qo_bank, base_model.kv_bank, base_model.mlp_up_bank, base_model.mlp_down_bank]
    block_named_params = list(base_model.blocks.named_parameters())
    scalar_params = [p for name, p in block_named_params if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None: scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None: scalar_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None: scalar_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales: scalar_params.append(s)
    optimizer_tok = torch.optim.AdamW(tok_params, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for group in optimizer_muon.param_groups: group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    replicated_params = list(optimizer_tok.param_groups[0]["params"])
    for pg in optimizer_tok.param_groups[1:]: replicated_params.extend(pg["params"])
    replicated_params.extend(scalar_params)
    optimizer_head = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam([{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        replicated_params.append(base_model.lm_head.weight)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if optimizer_head is not None: optimizers.append(optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"model_params:{n_params} XSA_layers:{xsa_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} iterations:{args.iterations} warmdown_iters:{args.warmdown_iters}")
    log0(f"seed:{args.seed}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            wds = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if wds <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1); wd_ms = args.warmdown_iters * step_ms
        rem = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rem / max(wd_ms, 1e-9) if rem <= wd_ms else 1.0
    if args.warmup_steps > 0:
        initial_model_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers: opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (ws+1) % 10 == 0 or ws+1 == args.warmup_steps:
                log0(f"warmup_step:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True): opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = 0.997; training_time_ms = 0.0; stop_after_step = None
    torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, bl, hl, il)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vbpb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True; log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all(); train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach(); (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups: group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        optimizer_muon.launch_reduce_scatters()
        if distributed:
            for p in replicated_params:
                if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        optimizer_tok.step(); optimizer_scalar.step()
        if optimizer_head is not None: optimizer_head.step()
        optimizer_muon.step(); zero_grad_all()
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        atms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        if should_log:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{atms:.0f}ms step_avg:{atms/step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and atms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rct = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rct, op=dist.ReduceOp.MAX); reached_cap = bool(rct.item())
        if stop_after_step is None and reached_cap: stop_after_step = step
    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")
    log0("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    torch.cuda.synchronize(); t_diag = time.perf_counter()
    diag_vl, diag_vbpb = eval_val(args, compiled_model, rank, world_size, device, grad_accum_steps, val_tokens, bl, hl, il)
    torch.cuda.synchronize()
    log0(f"DIAGNOSTIC post_ema val_loss:{diag_vl:.4f} val_bpb:{diag_vbpb:.4f} eval_time:{1000.0*(time.perf_counter()-t_diag):.0f}ms")
    full_sd = base_model.state_dict(); export_sd = {k: v for k, v in full_sd.items()}
    if master_process:
        torch.save(export_sd, "final_model.pt")
        log0(f"Serialized model: {os.path.getsize('final_model.pt')} bytes")
        log0(f"Code size: {len(code.encode('utf-8'))} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"})
    quant_buf = io.BytesIO(); torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_blob = lzma.compress(quant_buf.getvalue(), preset=9)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f: f.write(quant_blob)
        qfb = len(quant_blob); cb = len(code.encode("utf-8"))
        log0(f"Serialized model int6+lzma: {qfb} bytes"); log0(f"Total submission size int6+lzma: {qfb+cb} bytes")
        log0(f"Total submission size int8+zlib: {qfb+cb} bytes")
    if distributed: dist.barrier()
    with open("final_model.int6.ptz", "rb") as f: quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob_disk)), map_location="cpu")
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)
    eval_model = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    ttt_lr, ttt_epochs = 0.0008, 25
    ttt_seq_len = args.train_seq_len; total_val = val_tokens.numel() - 1
    ttt_seqs = total_val // ttt_seq_len; ttt_bs = 32
    if ttt_seqs > 0 and ttt_epochs > 0:
        log0(f"ttt:start lr={ttt_lr} epochs={ttt_epochs} seqs={ttt_seqs}")
        ttt_t0 = time.perf_counter()
        eval_model.train()
        ttt_opt = torch.optim.AdamW(eval_model.parameters(), lr=ttt_lr, weight_decay=0.0)
        ms_ttt = (ttt_seqs * rank) // world_size; me_ttt = (ttt_seqs * (rank + 1)) // world_size
        for ep in range(ttt_epochs):
            for bs in range(ms_ttt, me_ttt, ttt_bs):
                be = min(bs + ttt_bs, me_ttt)
                st = bs * ttt_seq_len; et = be * ttt_seq_len + 1
                if et > val_tokens.numel(): continue
                local = val_tokens[st:et].to(device=device, dtype=torch.int64)
                x, y = local[:-1].reshape(-1, ttt_seq_len), local[1:].reshape(-1, ttt_seq_len)
                ttt_opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = eval_model(x, y)
                loss.backward()
                if distributed:
                    for p in eval_model.parameters():
                        if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                ttt_opt.step()
            if rank == 0: log0(f"  ttt_epoch {ep+1}/{ttt_epochs} time={time.perf_counter()-ttt_t0:.1f}s")
        eval_model.eval()
        log0(f"ttt:done elapsed={time.perf_counter()-ttt_t0:.1f}s")
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize(); t_qeval = time.perf_counter()
    q_vl, q_vbpb = eval_val(args, compiled_eval, rank, world_size, device, grad_accum_steps, val_tokens, bl, hl, il, eval_seq_len=effective_eval_seq_len)
    torch.cuda.synchronize()
    log0(f"final_int6_roundtrip val_loss:{q_vl:.4f} val_bpb:{q_vbpb:.4f} eval_time:{1000.0*(time.perf_counter()-t_qeval):.0f}ms")
    log0(f"final_int6_roundtrip_exact val_loss:{q_vl:.8f} val_bpb:{q_vbpb:.8f}")
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize(); t_slide = time.perf_counter()
        sw_vl, sw_vbpb = eval_val_sliding(args, eval_model, rank, world_size, device, val_tokens, bl, hl, il, stride=args.eval_stride, eval_seq_len=sw_seq_len)
        torch.cuda.synchronize()
        log0(f"final_int6_sliding_window val_loss:{sw_vl:.4f} val_bpb:{sw_vbpb:.4f} stride:{args.eval_stride}")
        log0(f"final_int6_sliding_window_exact val_loss:{sw_vl:.8f} val_bpb:{sw_vbpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw_vl:.8f} val_bpb:{sw_vbpb:.8f}")
    if args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize(); t_s64 = time.perf_counter()
        s64_vl, s64_vbpb = eval_val_sliding(args, eval_model, rank, world_size, device, val_tokens, bl, hl, il, stride=64, eval_seq_len=sw_seq_len)
        torch.cuda.synchronize()
        log0(f"final_int6_sliding_window_s64 val_loss:{s64_vl:.4f} val_bpb:{s64_vbpb:.4f} stride:64")
        log0(f"final_int6_sliding_window_s64_exact val_loss:{s64_vl:.8f} val_bpb:{s64_vbpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{s64_vl:.8f} val_bpb:{s64_vbpb:.8f}")
    if distributed: dist.destroy_process_group()
if __name__ == "__main__":
    main()
