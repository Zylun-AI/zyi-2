"""
ZYI 2: Diffusion Transformer (DiT) texto→imagem.

Cada bloco:
    x = x + gate1 * SelfAttn(modulate(LN(x), shift1, scale1))   # adaLN-Zero + RoPE 2D
    x = x + CrossAttn(LN(x), texto_clip)                        # condicionamento de texto
    x = x + gate2 * MLP(modulate(LN(x), shift2, scale2))        # adaLN-Zero

O vetor de condição c = emb(timestep) + proj(texto pooled do CLIP) gera shift/scale/gate
de cada bloco. Os gates e as projeções de saída começam em zero, então cada bloco começa
como identidade (é o "Zero" do adaLN-Zero).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class DiTConfig:
    in_channels: int = 4          # canais do latente do VAE do SD
    patch_size: int = 2           # latente 32x32 -> grid 16x16 = 256 tokens
    dim: int = 1024
    depth: int = 16
    heads: int = 16
    mlp_ratio: float = 4.0
    text_dim: int = 768           # CLIP ViT-L/14
    rope_base_grid: int = 16      # grid de tokens da fase 1; grids maiores têm as posições interpoladas para ele
    rope_theta: float = 10000.0
    grad_checkpointing: bool = False


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def timestep_embedding(t, dim, max_period=10000):
    """Embedding senoidal de t em [0, 1] (escalado para [0, 1000] como no DiT)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def rope_2d(h, w, head_dim, base_grid, theta, device):
    """RoPE 2D axial: metade da cabeça gira com a linha, metade com a coluna.

    As posições são interpoladas para o grid base: na fase 2 (grid 32x32) as posições vão de 0 a
    15.5 em passos de 0.5, cobrindo o mesmo intervalo que o modelo viu na fase 1 (grid 16x16).
    Retorna (cos, sin), cada um com shape (h*w, head_dim/2).
    """
    quarter = head_dim // 4
    freqs = 1.0 / theta ** (torch.arange(quarter, device=device, dtype=torch.float32) / quarter)
    ys = torch.arange(h, device=device, dtype=torch.float32) * (base_grid / h)
    xs = torch.arange(w, device=device, dtype=torch.float32) * (base_grid / w)
    ang_y = (ys[:, None] * freqs[None]).repeat_interleave(w, dim=0)   # (h*w, quarter)
    ang_x = (xs[:, None] * freqs[None]).repeat(h, 1)                  # (h*w, quarter)
    ang = torch.cat([ang_y, ang_x], dim=-1)                           # (h*w, head_dim/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, N, heads, head_dim). Gira os pares (i, i + head_dim/2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Self-attention (context=None) ou cross-attention, com QK-norm e SDPA (Flash Attention na GPU)."""

    def __init__(self, dim, heads, context_dim=None):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(context_dim or dim, 2 * dim)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context=None, rope=None):
        B, N, D = x.shape
        context = x if context is None else context
        q = self.q(x).view(B, N, self.heads, self.head_dim)
        k, v = self.kv(context).view(B, context.shape[1], 2, self.heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q.float()), self.k_norm(k.float())   # QK-norm e RoPE em fp32
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        q, k = q.to(v.dtype), k.to(v.dtype)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d, cfg.heads)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.cross = Attention(d, cfg.heads, context_dim=cfg.text_dim)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        hidden = int(d * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))

    def forward(self, x, c, text, rope):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c)[:, None].chunk(6, dim=-1)
        x = x + gate1 * self.attn(modulate(self.norm1(x), shift1, scale1), rope=rope)
        x = x + self.cross(self.norm2(x), context=text)
        x = x + gate2 * self.mlp(modulate(self.norm3(x), shift2, scale2))
        return x


class DiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.dim, cfg.patch_size
        assert d % cfg.heads == 0 and (d // cfg.heads) % 4 == 0, "head_dim precisa ser múltiplo de 4 (RoPE 2D)"
        self.patch_embed = nn.Conv2d(cfg.in_channels, d, kernel_size=p, stride=p)
        self.t_embed = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.pooled_embed = nn.Sequential(nn.Linear(cfg.text_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.depth)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.final_proj = nn.Linear(d, p * p * cfg.in_channels)
        self.init_weights()

    def init_weights(self):
        # inicialização do DiT: xavier nas lineares, adaLN e saídas zeradas (blocos começam como identidade)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        w = self.patch_embed.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        for emb in (self.t_embed, self.pooled_embed):
            nn.init.normal_(emb[0].weight, std=0.02)
            nn.init.normal_(emb[2].weight, std=0.02)
        zero = [self.final_ada[-1], self.final_proj]
        for block in self.blocks:
            zero += [block.ada[-1], block.cross.proj]
        for m in zero:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, t, text, pooled):
        """x: latente ruidoso (B, C, H, W); t: (B,) em [0, 1]; text: (B, L, text_dim); pooled: (B, text_dim).
        Retorna a velocidade prevista, com o mesmo shape de x."""
        B, C, H, W = x.shape
        p = self.cfg.patch_size
        h, w = H // p, W // p
        x = self.patch_embed(x).flatten(2).transpose(1, 2).float()   # (B, h*w, d); residual em fp32
        c = self.t_embed(timestep_embedding(t, 256)) + self.pooled_embed(pooled)
        rope = rope_2d(h, w, self.cfg.dim // self.cfg.heads, self.cfg.rope_base_grid, self.cfg.rope_theta, x.device)
        for block in self.blocks:
            if self.cfg.grad_checkpointing and self.training:
                x = checkpoint(block, x, c, text, rope, use_reentrant=False)
            else:
                x = block(x, c, text, rope)
        shift, scale = self.final_ada(c)[:, None].chunk(2, dim=-1)
        x = self.final_proj(modulate(self.final_norm(x), shift, scale))   # (B, h*w, p*p*C)
        return x.view(B, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
