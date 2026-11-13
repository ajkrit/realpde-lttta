"""RealPDE Track 2 submission (v14, LB final_score 80.854482).

Distilled 4-block factorized space-time transformer (d=384, 12 heads, MLP x2)
forecasting the next 20 frames from 20 context frames, plus three online
corrections in ttt_step: shrunk Welford bias, TKE amplitude map, and
per-lead-time interval multipliers K chosen online for the SPS score.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

T, NH, NW, D, HEADS, N_UNCOND = 20, 16, 32, 384, 12, 2
MEAN_UV = [0.15496256947517395, -0.0005177936982363462]
STD_UV = [0.09681040793657303, 0.015963643789291382]
SIGMA_GLOBAL = 0.0563870259
K_GRID = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
          1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0, 6.0, 10.0]
K_WARMUP, K_EVERY, K_SUBSAMPLE = 50, 5, 4
SHRINK_N0 = 10


def get_ttt_model(submission_dir: str, device: str = "cuda"):
    return V14(Path(submission_dir) / "model.pth", device)


def _self_attn(a, x):
    B, N, _ = x.shape
    q, k, v = a["qkv"](x).reshape(B, N, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4).unbind(0)
    return a["out"](F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(B, N, D))


class V14(nn.Module):
    def __init__(self, ckpt_path, device):
        super().__init__()
        lin, md = nn.Linear, lambda **kw: nn.ModuleDict(kw)
        self.patch_embed = md(proj=lin(8, D))
        self.t_pos_embed = nn.Parameter(torch.zeros(1, T, 1, 1, D))
        self.blocks = nn.ModuleList()
        for i in range(4):
            blk = md(s_attn=md(qkv=lin(D, 3 * D, bias=False), out=lin(D, D, bias=False)),
                     t_attn=md(qkv=lin(D, 3 * D, bias=False), out=lin(D, D, bias=False)),
                     mlp=nn.Sequential(lin(D, 2 * D), nn.GELU(), lin(2 * D, D)))
            if i < N_UNCOND:
                blk.update(md(norm1=nn.LayerNorm(D, eps=1e-6), norm2=nn.LayerNorm(D, eps=1e-6),
                              norm3=nn.LayerNorm(D, eps=1e-6)))
            else:
                blk["adaLN"] = md(proj=nn.Sequential(nn.SiLU(), lin(D, 6 * D)))
            self.blocks.append(blk)
        self.regime_norm = nn.LayerNorm(D, eps=1e-6)
        self.regime_mlp = nn.Sequential(lin(D, D), nn.GELU(), lin(D, D))
        self.out_time_embed = nn.Parameter(torch.zeros(T, D))
        self.out_xattn = md(q=lin(D, D, bias=False), kv=lin(D, 2 * D, bias=False), out=lin(D, D, bias=False))
        self.out_norm = nn.LayerNorm(D, eps=1e-6)
        self.head_proj = lin(D, 26 * 4)   # per pixel: mean(2), log-sigma(2), 22 unused quantile channels
        self.tke_head = lin(2 * D, 8)
        self.register_buffer("sp_pos_embed", torch.zeros(1, 1, NH, NW, D))   # 2D sin-cos, from ckpt

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
        self.load_state_dict({k: v for k, v in sd.items() if not k.startswith("aux_")})
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(self.device).to(torch.bfloat16).eval()
        self.q_tok = (self.out_time_embed[:, None, :]
                      + self.sp_pos_embed.reshape(NH * NW, D)[None]).reshape(T * NH * NW, D)[None]

        # Online K state persists across trajectories (never reset).
        f32 = dict(dtype=torch.float32, device=self.device)
        self.k_grid = torch.tensor(K_GRID, **f32)
        self.mean_uv = torch.tensor(MEAN_UV, **f32)
        self.std_uv = torch.tensor(STD_UV, **f32)
        self.k_scores = torch.zeros(2, T, len(K_GRID), **f32)
        self.K = torch.tensor([[1.5] * T, [2.0] * T], **f32)
        self.k_n = 0
        self.step_n = 0
        self.reset_ttt_state()

    def forward(self, ctx):
        B = ctx.shape[0]
        x = ctx.reshape(B, T, NH, 2, NW, 2, 2).permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = self.patch_embed["proj"](x.reshape(B, T, NH, NW, 8)) + self.sp_pos_embed + self.t_pos_embed
        for i, b in enumerate(self.blocks):
            if i < N_UNCOND:
                f1, f2, f3, g1, g2 = b["norm1"], b["norm2"], b["norm3"], 1.0, 1.0
            else:
                if i == N_UNCOND:
                    w = self.regime_mlp(self.regime_norm(x.reshape(B, -1, D)).mean(dim=1))
                s1, c1, g1, s2, c2, g2 = (v[:, None, None, None, :]
                                          for v in b["adaLN"]["proj"](w).chunk(6, dim=-1))
                f1 = f2 = lambda y: F.layer_norm(y, (D,), eps=1e-6) * (1.0 + c1) + s1
                f3 = lambda y: F.layer_norm(y, (D,), eps=1e-6) * (1.0 + c2) + s2
            x = x + g1 * _self_attn(b["s_attn"], f1(x).reshape(B * T, NH * NW, D)).reshape(B, T, NH, NW, D)
            xt = _self_attn(b["t_attn"], f2(x).permute(0, 2, 3, 1, 4).reshape(B * NH * NW, T, D))
            x = x + g1 * xt.reshape(B, NH, NW, T, D).permute(0, 3, 1, 2, 4)
            x = x + g2 * b["mlp"](f3(x))

        xa, Nq = self.out_xattn, T * NH * NW
        q = xa["q"](self.out_norm(self.q_tok.expand(B, -1, -1))).reshape(B, Nq, HEADS, D // HEADS).permute(0, 2, 1, 3)
        k, v = xa["kv"](x.reshape(B, -1, D)).reshape(B, -1, 2, HEADS, D // HEADS).permute(2, 0, 3, 1, 4).unbind(0)
        out = xa["out"](F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(B, Nq, D))
        out = out.reshape(B, T, NH, NW, D)

        raw = self.head_proj(out).reshape(B, T, NH, NW, 2, 2, 26).permute(0, 1, 2, 4, 3, 5, 6)
        raw = raw.contiguous().reshape(B, T, 2 * NH, 2 * NW, 26)
        var, mean = torch.var_mean(out, dim=1)
        a = self.tke_head(torch.cat([mean, (var + 1e-6).sqrt()], dim=-1))
        a = a.reshape(B, NH, NW, 2, 2, 2).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, 2 * NH, 2 * NW, 2)
        return raw[..., :2], F.softplus(raw[..., 2:4]).clamp(min=1e-4), F.softplus(a)

    def reset_ttt_state(self):
        self.prev_m = self.prev_mb = self.prev_sig = self.bias = None
        self.n_wf = 0

    @torch.no_grad()
    def ttt_step(self, input_norm, prev_target_norm=None):
        ctx = torch.as_tensor(input_norm, dtype=torch.float32).to(self.device)[..., :2]
        m, sigma, a = (t.float() for t in self(ctx.to(torch.bfloat16)))
        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = torch.nan_to_num(sigma, nan=1e-4, posinf=1e-4, neginf=1e-4)
        pt = None if prev_target_norm is None else \
            torch.as_tensor(prev_target_norm, dtype=torch.float32).to(self.device)

        # 1. Score every K on the previous window, per lead time and channel.
        self.step_n += 1
        if pt is not None and self.prev_mb is not None and (
                self.k_n < K_WARMUP or self.step_n % K_EVERY == 0):
            for c in (0, 1):
                flat = lambda t: t.movedim(1, 0).reshape(T, -1)[:, ::K_SUBSAMPLE]
                mu = flat(self.prev_mb[..., c] * self.std_uv[c] + self.mean_uv[c])
                s = flat(self.prev_sig[..., c] * self.std_uv[c]).clamp(min=1e-8)
                y = flat(pt[..., c] * self.std_uv[c] + self.mean_uv[c])
                scored = y != 0.0
                n_sc = scored.sum(dim=1)
                inside = ((y - mu).abs() / s)[:, None, :] <= self.k_grid[None, :, None]
                width = self.k_grid[None, :, None] * s[:, None, :] * (2.0 / SIGMA_GLOBAL)
                contrib = ((inside & scored[:, None, :]) * torch.exp(-width)).sum(dim=2)
                self.k_scores[c] += torch.where((n_sc > 0)[:, None], contrib / n_sc[:, None].clamp(min=1),
                                                torch.zeros_like(contrib))
            self.k_n += 1
            self.K = self.k_grid[self.k_scores.argmax(dim=-1)]

        # 2. Welford running-mean bias, shrunk toward zero while few windows are seen.
        if pt is not None and self.prev_m is not None:
            r = pt[..., :2] - self.prev_m
            self.n_wf += 1
            self.bias = r if self.bias is None else self.bias + (r - self.bias) / self.n_wf
        m_b = m
        if self.bias is not None:
            m_b = m + self.n_wf / (self.n_wf + SHRINK_N0) * self.bias.mean(dim=1, keepdim=True)

        # 3. TKE amplitude correction around the temporal mean.
        m_bar = m_b.mean(dim=1, keepdim=True)
        m_b = m_bar + a[:, None] * (m_b - m_bar)

        self.prev_m = m
        keep = self.k_n < K_WARMUP or (self.step_n + 1) % K_EVERY == 0
        self.prev_mb, self.prev_sig = (m_b, sigma) if keep else (None, None)

        # 4. Bounds m_b +- K * sigma, K per lead time and channel.
        half = [self.K[c].reshape(1, T, 1, 1) * sigma[..., c] for c in (0, 1)]
        lo = torch.stack([m_b[..., 0] - half[0], m_b[..., 1] - half[1]], dim=-1)
        hi = torch.stack([m_b[..., 0] + half[0], m_b[..., 1] + half[1]], dim=-1)
        pad = torch.zeros(*m_b.shape[:-1], 1, dtype=m_b.dtype, device=self.device)
        out = torch.stack([torch.cat([t, pad], dim=-1) for t in (m_b, lo, hi)], dim=0).cpu()
        return out[0], {"adapt_loss": None, "lower": out[1], "upper": out[2]}
