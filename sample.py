"""
Gera imagens com o ZYI 2 e salva um grid PNG.

    python sample.py --ckpt checkpoints/zyi2_512/ema.pt --prompt "a red car parked on a city street" --n 4
    python sample.py --ckpt checkpoints/zyi2_512/ema.pt --prompts_file eval/prompts.txt --out samples/eval.png

Aceita tanto o ckpt.pt completo quanto o ema.pt (só pesos EMA) gerados pelo train.py.
"""
import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from zyi.data import TextEncoder, image_grid, load_vae, to_pil, vae_decode
from zyi.flow import euler_sample
from zyi.model import DiT, DiTConfig


class ZYIPipeline:
    """Checkpoint + CLIP + VAE -> imagens. Usado por sample.py, app.py e eval/evaluate.py."""

    def __init__(self, ckpt, device=None, weights="ema", random_encoders=False):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg, self.meta = state["config"], state["meta"]
        self.model = DiT(DiTConfig(**{**state["model_config"], "grad_checkpointing": False}))
        self.model.load_state_dict({k: v.float() for k, v in state[weights].items()})
        self.model.to(self.device).eval()
        self.step = state.get("step")
        del state
        random_encoders = random_encoders or self.meta.get("random_encoders", False)
        self.text_encoder = TextEncoder(self.device, self.meta["text_len"], random_init=random_encoders)
        self.vae = load_vae(self.device, random_init=random_encoders)
        self.null_text, self.null_pooled = self.text_encoder([""])
        self.amp = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()

    @torch.no_grad()
    def __call__(self, prompts, negative_prompt="", steps=30, cfg=5.0, seed=0, resolution=None, shift=None, batch_size=8):
        """Lista de prompts -> lista de PIL. A mesma seed gera o mesmo ruído (reprodutível)."""
        res = resolution or self.meta["resolution"]
        shift = self.cfg.get("time_shift", 1.0) if shift is None else shift
        noise = torch.randn(len(prompts), self.meta["latent_channels"], res // 8, res // 8,
                            generator=torch.Generator().manual_seed(seed))
        null = self.text_encoder([negative_prompt]) if negative_prompt else (self.null_text, self.null_pooled)
        images = []
        for i in range(0, len(prompts), batch_size):
            text, pooled = self.text_encoder(prompts[i : i + batch_size])
            with self.amp:
                latents = euler_sample(self.model, noise[i : i + batch_size].to(self.device), text.float(), pooled.float(),
                                       null[0].float(), null[1].float(), steps, cfg, shift)
            images += [to_pil(x) for x in vae_decode(self.vae, latents)]
        return images


def main():
    parser = argparse.ArgumentParser(description="ZYI 2: prompt -> grid PNG")
    parser.add_argument("--ckpt", default="checkpoints/zyi2_512/ema.pt")
    parser.add_argument("--prompt", action="append", help="pode repetir: --prompt 'a' --prompt 'b'")
    parser.add_argument("--prompts_file", help="um prompt por linha")
    parser.add_argument("--n", type=int, default=4, help="imagens por prompt")
    parser.add_argument("--steps", type=int, default=30, help="passos do Euler (20-50)")
    parser.add_argument("--cfg", type=float, default=5.0, help="classifier-free guidance (1 = desligado)")
    parser.add_argument("--negative", default="", help="prompt negativo (substitui o prompt vazio no CFG)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=None, help="padrão: a do treino")
    parser.add_argument("--shift", type=float, default=None, help="time shift (padrão: o do treino)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--out", default="samples/grid.png")
    parser.add_argument("--weights", default="ema", choices=["ema", "model"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--random_encoders", action="store_true", help="CLIP/VAE aleatórios (só smoke test offline)")
    args = parser.parse_args()

    prompts = list(args.prompt or [])
    if args.prompts_file:
        prompts += [p.strip() for p in Path(args.prompts_file).read_text().splitlines() if p.strip()]
    if not prompts:
        parser.error("passe --prompt ou --prompts_file")

    pipe = ZYIPipeline(args.ckpt, args.device, args.weights, args.random_encoders)
    all_prompts = [p for p in prompts for _ in range(args.n)]
    images = pipe(all_prompts, args.negative, args.steps, args.cfg, args.seed, args.resolution, args.shift, args.batch_size)
    cols = args.cols or (args.n if len(prompts) > 1 and args.n > 1 else math.ceil(math.sqrt(len(images))))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image_grid(images, cols).save(args.out)
    print(f"{len(images)} imagens -> {args.out}")


if __name__ == "__main__":
    main()
