"""
Treino do ZYI 2 (DiT + rectified flow) num arquivo só, no estilo nanoGPT.

    python train.py --config configs/zyi2_256.yaml     # fase 1: 256px, ~8h
    python train.py --config configs/zyi2_512.yaml     # fase 2: 512px, ~4h, parte dos pesos da fase 1

Qualquer chave pode ser sobrescrita na linha de comando (use pontos para o modelo):
    python train.py --config configs/zyi2_256.yaml --batch_size=64 --model.depth=8 --logger=none

Rodar o mesmo comando de novo retoma automaticamente de <out_dir>/ckpt.pt. O treino para sozinho
quando o tempo total (somado entre resumes) atinge time_budget_hours; o LR segue um cosseno
sobre esse orçamento, então chega ao mínimo exatamente no fim.
"""
import copy
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from zyi.data import (InfiniteSampler, ShardDataset, image_grid, latents_to_rgb_approx, load_vae, to_pil,
                      vae_decode)
from zyi.flow import euler_sample, flow_loss
from zyi.model import DiT, DiTConfig

# ----------------------------------------------------------------------------- configuração padrão
DEFAULTS = dict(
    run_name="zyi2",
    out_dir="checkpoints/zyi2",
    data_dir="data/cc3m_256",
    max_samples=None,            # usa só as primeiras N amostras do dataset (None = todas)
    resume=True,                 # retoma de <out_dir>/ckpt.pt se existir
    init_from=None,              # checkpoint de onde vêm os pesos iniciais (fase 2 <- fase 1)
    init_weights="ema",          # "ema" ou "model"
    model={},                    # campos de DiTConfig (zyi/model.py)
    batch_size=256,              # amostras por passo do otimizador
    micro_batch_size=128,        # amostras por forward; acumulação = batch_size / micro_batch_size
    lr=2e-4,
    min_lr=2e-5,
    warmup_steps=1000,
    weight_decay=0.0,
    betas=[0.9, 0.95],
    grad_clip=1.0,
    ema_decay=0.9999,
    cfg_dropout=0.1,             # fração de prompts vazios (para classifier-free guidance)
    t_mean=0.0,                  # t ~ sigmoid(N(t_mean, t_std))  (logit-normal)
    t_std=1.0,
    time_shift=1.0,              # desloca t para mais ruído (~2.0 em 512px); o sampler usa o mesmo valor
    time_budget_hours=8.0,       # para sozinho ao atingir este tempo de treino
    max_steps=1_000_000,         # limite de passos (normalmente o orçamento de tempo para antes)
    checkpoint_every_minutes=30,
    sample_every_minutes=60,     # 0 desliga as amostras no log
    num_preview=8,
    preview_steps=30,
    preview_cfg=5.0,
    preview_vae=True,            # decodifica as prévias com o VAE (False: aproximação linear, sem baixar nada)
    log_every=50,
    logger="tensorboard",        # "tensorboard", "wandb" ou "none"
    wandb_project="zyi2",
    dtype="bfloat16",            # "bfloat16" ou "float32"
    compile=True,
    num_workers=8,
    seed=0,
    device="auto",
)


def parse_value(text):
    value = yaml.safe_load(text)
    if isinstance(value, str):  # o YAML lê "3e-4" como string
        try:
            value = float(value)
        except ValueError:
            pass
    return value


def load_config(argv):
    """DEFAULTS <- arquivo YAML (--config) <- overrides --chave=valor da linha de comando."""
    cfg = copy.deepcopy(DEFAULTS)
    overrides, args = [], iter(argv)
    for arg in args:
        if not arg.startswith("--"):
            sys.exit(f"argumento inválido: {arg}")
        key, value = arg[2:].split("=", 1) if "=" in arg else (arg[2:], next(args))
        if key == "config":
            with open(value) as f:
                overrides = list((yaml.safe_load(f) or {}).items()) + overrides
        else:
            overrides.append((key, parse_value(value)))
    for key, value in overrides:
        *parents, last = key.split(".")
        if parents[:1] != ["model"] and key not in cfg:
            sys.exit(f"chave desconhecida: {key}")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        if isinstance(value, dict):
            node[last].update({k: parse_value(v) if isinstance(v, str) else v for k, v in value.items()})
        else:
            node[last] = parse_value(value) if isinstance(value, str) else value
    return cfg


class Logger:
    """Escalares e imagens em tensorboard ou wandb."""

    def __init__(self, cfg, out_dir):
        self.kind, self.tb, self.wandb = cfg["logger"], None, None
        if self.kind == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(out_dir / "tb")
        elif self.kind == "wandb":
            import wandb
            wandb.init(project=cfg["wandb_project"], name=cfg["run_name"], id=cfg["run_name"], resume="allow", config=cfg)
            self.wandb = wandb

    def scalars(self, values, step):
        if self.tb:
            for k, v in values.items():
                self.tb.add_scalar(k, v, step)
        if self.wandb:
            self.wandb.log(values, step=step)

    def image(self, name, pil, step):
        if self.tb:
            self.tb.add_image(name, np.asarray(pil), step, dataformats="HWC")
        if self.wandb:
            self.wandb.log({name: self.wandb.Image(pil)}, step=step)


def get_lr(step, progress, cfg):
    """Warmup linear (em passos) e depois cosseno sobre o progresso (fração do orçamento já usada)."""
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * min(progress, 1.0)))


def main():
    cfg = load_config(sys.argv[1:])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"] if cfg["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = torch.autocast(device.type, dtype=torch.bfloat16) if cfg["dtype"] == "bfloat16" else nullcontext()

    # ------------------------------------------------------------------ dados
    data = ShardDataset(cfg["data_dir"], cfg["max_samples"])
    meta = data.meta
    print(f"dados: {len(data):,} amostras de {cfg['data_dir']} ({meta['resolution']}px, latente {meta['latent_size']}x{meta['latent_size']})")

    # ------------------------------------------------------------------ modelo (novo, resume ou init_from)
    ckpt_path = out_dir / "ckpt.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False) if cfg["resume"] and ckpt_path.exists() else None
    init = torch.load(cfg["init_from"], map_location="cpu", weights_only=False) if cfg["init_from"] and state is None else None
    arch = (state or init or {}).get("model_config", {})   # resume e fase 2 herdam a arquitetura do checkpoint
    mcfg = DiTConfig(**{**arch, **({} if arch else cfg["model"]), "text_dim": meta["text_dim"],
                        "in_channels": meta["latent_channels"],
                        "grad_checkpointing": cfg["model"].get("grad_checkpointing", False)})
    model = DiT(mcfg).to(device)
    if state:
        model.load_state_dict(state["model"])
        print(f"retomando de {ckpt_path}: passo {state['step']}, {state['train_time'] / 3600:.2f}h já treinadas")
    elif init:
        weights = init[cfg["init_weights"]] if cfg["init_weights"] in init else init["ema"]
        model.load_state_dict({k: v.float() for k, v in weights.items()})
        print(f"pesos iniciais de {cfg['init_from']} ({cfg['init_weights']})")
    ema = copy.deepcopy(model).requires_grad_(False).eval()   # EMA em fp32
    if state:
        ema.load_state_dict(state["ema"])
    print(f"DiT: {model.num_params() / 1e6:.1f}M parâmetros | {asdict(mcfg)}")

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=cfg["lr"], betas=tuple(cfg["betas"]), fused=device.type == "cuda")
    if state:
        optimizer.load_state_dict(state["optimizer"])
    train_model = torch.compile(model) if cfg["compile"] else model
    ema_params, model_params = list(ema.parameters()), list(model.parameters())

    step = state["step"] if state else 0
    samples_seen = state["samples_seen"] if state else 0
    time_before = state["train_time"] if state else 0.0
    del state, init

    micro = cfg["micro_batch_size"]
    accum = cfg["batch_size"] // micro
    assert accum * micro == cfg["batch_size"], "batch_size precisa ser múltiplo de micro_batch_size"
    loader = DataLoader(data, batch_size=micro, sampler=InfiniteSampler(len(data), cfg["seed"], samples_seen),
                        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda", drop_last=True,
                        persistent_workers=cfg["num_workers"] > 0, prefetch_factor=4 if cfg["num_workers"] > 0 else None)
    batches = iter(loader)
    null_text, null_pooled = (x.to(device).float() for x in data.null_embedding())
    logger = Logger(cfg, out_dir)
    vae = None

    # ------------------------------------------------------------------ checkpoint e amostras
    session_start = time.time()

    def elapsed():
        return time_before + time.time() - session_start

    def save_checkpoint(final=False):
        ckpt = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    samples_seen=samples_seen, train_time=elapsed(), config=cfg, model_config=asdict(mcfg), meta=meta)
        torch.save(ckpt, out_dir / "ckpt.tmp")
        os.replace(out_dir / "ckpt.tmp", ckpt_path)
        if final:  # só os pesos EMA em bf16: é o arquivo para sample.py / app.py
            light = dict(ema={k: v.to(torch.bfloat16) for k, v in ema.state_dict().items()}, step=step,
                         config=cfg, model_config=asdict(mcfg), meta=meta)
            torch.save(light, out_dir / "ema.pt")
        print(f"checkpoint salvo (passo {step}){' + ema.pt' if final else ''}")

    def preview():
        nonlocal vae
        text, pooled = (x.to(device).float() for x in data.eval_embedding(cfg["num_preview"]))
        noise = torch.randn(len(text), meta["latent_channels"], meta["latent_size"], meta["latent_size"],
                            generator=torch.Generator().manual_seed(0)).to(device)
        with amp:
            latents = euler_sample(ema, noise, text, pooled, null_text, null_pooled, cfg["preview_steps"],
                                   cfg["preview_cfg"], cfg["time_shift"])
        if cfg["preview_vae"] and vae is None:
            try:
                vae = load_vae(device)
            except Exception as e:  # sem internet: cai na aproximação linear
                print(f"VAE indisponível ({type(e).__name__}); usando prévia aproximada")
                cfg["preview_vae"] = False
        images = vae_decode(vae, latents) if cfg["preview_vae"] else latents_to_rgb_approx(latents)
        grid = image_grid([to_pil(x) for x in images], cols=min(4, len(images)))
        (out_dir / "samples").mkdir(exist_ok=True)
        grid.save(out_dir / "samples" / f"step_{step:07d}.png")
        logger.image("samples", grid, step)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))   # ex.: preempção da nuvem

    # ------------------------------------------------------------------ laço de treino
    print(f"treinando: batch {cfg['batch_size']} (micro {micro} x {accum}), orçamento {cfg['time_budget_hours']}h")
    last_ckpt = last_sample = last_log = time.time()
    loss_sum, log_steps = torch.zeros((), device=device), 0
    try:
        while True:
            progress = max(step / cfg["max_steps"], elapsed() / (cfg["time_budget_hours"] * 3600))
            if progress >= 1.0 or stop["flag"]:
                break
            lr = get_lr(step, progress, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            for _ in range(accum):
                latents, text, pooled = (x.to(device, non_blocking=True).float() for x in next(batches))
                drop = torch.rand(latents.shape[0], device=device) < cfg["cfg_dropout"]   # prompt vazio p/ CFG
                text = torch.where(drop[:, None, None], null_text, text)
                pooled = torch.where(drop[:, None], null_pooled, pooled)
                with amp:
                    loss = flow_loss(train_model, latents, text, pooled, cfg["t_mean"], cfg["t_std"], cfg["time_shift"])
                (loss / accum).backward()
                loss_sum += loss.detach() / accum
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                torch._foreach_lerp_(ema_params, model_params, 1 - min(cfg["ema_decay"], (1 + step) / (10 + step)))
            step += 1
            samples_seen += cfg["batch_size"]
            log_steps += 1

            now = time.time()
            if step % cfg["log_every"] == 0:
                loss_avg = loss_sum.item() / log_steps
                speed = log_steps * cfg["batch_size"] / (now - last_log)
                hours = elapsed() / 3600
                print(f"passo {step} | loss {loss_avg:.4f} | lr {lr:.2e} | grad {grad_norm.item():.2f} | "
                      f"{speed:.0f} amostras/s | {hours:.2f}/{cfg['time_budget_hours']}h | época {samples_seen / len(data):.2f}")
                logger.scalars({"loss": loss_avg, "lr": lr, "grad_norm": grad_norm.item(), "samples_per_sec": speed,
                                "hours": hours, "epoch": samples_seen / len(data)}, step)
                loss_sum.zero_()
                log_steps, last_log = 0, now
            if now - last_ckpt > cfg["checkpoint_every_minutes"] * 60:
                save_checkpoint()
                last_ckpt = now
            if cfg["sample_every_minutes"] and now - last_sample > cfg["sample_every_minutes"] * 60:
                preview()
                last_sample = now
    except KeyboardInterrupt:
        print("interrompido; salvando checkpoint")
        save_checkpoint()
        return
    save_checkpoint(final=not stop["flag"])
    if cfg["sample_every_minutes"] and not stop["flag"]:
        preview()
    print(f"fim: {step} passos, {samples_seen:,} amostras, {elapsed() / 3600:.2f}h")


if __name__ == "__main__":
    main()
