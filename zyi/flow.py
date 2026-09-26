"""
Rectified flow / flow matching (como no SD3).

Convenção: t=0 é o dado limpo, t=1 é ruído puro.
    x_t = (1 - t) * x0 + t * eps
    v   = eps - x0                  # o modelo prevê essa velocidade
Treino: t ~ logit-normal. Amostragem: integra dx/dt = v de t=1 até t=0 com Euler.
"""
import torch
import torch.nn.functional as F


def shift_time(t, shift):
    """Desloca t para o lado do ruído (SD3, eq. 23). Resoluções maiores precisam de mais ruído: shift ~ sqrt(pixels / pixels_base)."""
    if shift == 1.0:
        return t
    return shift * t / (1 + (shift - 1) * t)


def sample_timesteps(n, device, mean=0.0, std=1.0, shift=1.0):
    """Logit-normal: t = sigmoid(N(mean, std)), concentra o treino nos t intermediários."""
    t = torch.sigmoid(torch.randn(n, device=device) * std + mean)
    return shift_time(t, shift)


def flow_loss(model, x0, text, pooled, mean=0.0, std=1.0, shift=1.0):
    """MSE entre a velocidade prevista e a real."""
    t = sample_timesteps(x0.shape[0], x0.device, mean, std, shift)
    eps = torch.randn_like(x0)
    tt = t.view(-1, 1, 1, 1)
    xt = (1 - tt) * x0 + tt * eps
    v_pred = model(xt, t, text, pooled)
    return F.mse_loss(v_pred.float(), (eps - x0).float())


@torch.no_grad()
def euler_sample(model, noise, text, pooled, null_text, null_pooled, steps=30, cfg=5.0, shift=1.0):
    """Sampler Euler com classifier-free guidance.

    noise: (B, C, H, W); text/pooled: condição (B, ...); null_text/null_pooled: prompt vazio
    (ou negativo), com batch 1 ou B. cfg=1 desliga o guidance (metade do custo).
    """
    B = noise.shape[0]
    ts = shift_time(torch.linspace(1.0, 0.0, steps + 1, device=noise.device), shift)
    null_text = null_text.expand(B, *null_text.shape[1:]).to(text.dtype)
    null_pooled = null_pooled.expand(B, *null_pooled.shape[1:]).to(pooled.dtype)
    if cfg != 1.0:
        text, pooled = torch.cat([text, null_text]), torch.cat([pooled, null_pooled])
    x = noise.float()
    for i in range(steps):
        t = ts[i].expand(B)
        if cfg != 1.0:
            v_cond, v_uncond = model(torch.cat([x, x]), torch.cat([t, t]), text, pooled).float().chunk(2)
            v = v_uncond + cfg * (v_cond - v_uncond)
        else:
            v = model(x, t, text, pooled).float()
        x = x + (ts[i + 1] - ts[i]) * v
    return x
