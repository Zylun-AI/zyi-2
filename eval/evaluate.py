"""
Avaliação fixa para comparar versões do ZYI (mesmos prompts, seeds e subconjunto em toda versão).

  CLIP score  50 prompts de eval/prompts.txt x --seeds seeds fixas; 100 * cos(imagem, texto), com CLIP ViT-L/14
  FID         legendas das primeiras --fid_samples amostras válidas (em ordem) dos .tar de validação do CC3M,
              imagens geradas vs. reais (Inception, via torchmetrics)

    python eval/evaluate.py --ckpt checkpoints/zyi2_512/ema.pt
    python prepare_data.py download --split val --out data/cc3m_val_wds
    python eval/evaluate.py --ckpt checkpoints/zyi2_512/ema.pt --fid_wds data/cc3m_val_wds --fid_samples 5000

Salva <pasta do ckpt>/eval_<nome>.json e um grid com os 50 prompts (seed 0).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sample import ZYIPipeline  # noqa: E402
from zyi.data import center_crop_resize, image_grid, iter_tar, load_image, tokenize  # noqa: E402

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]


def to_tensor(images):
    """Lista de PIL -> (B, 3, H, W) float em [0, 1]."""
    return torch.stack([torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1) for im in images]).float() / 255


def features(out):
    return out if torch.is_tensor(out) else out.pooler_output   # transformers 4.x devolve tensor; 5.x, um output


class ClipScorer:
    def __init__(self, device, name, random_init=False):
        from transformers import CLIPConfig, CLIPModel, CLIPTokenizer
        if random_init:  # só para smoke test offline
            torch.manual_seed(0)
            self.model, self.tokenizer = CLIPModel(CLIPConfig()), None
        else:
            self.model, self.tokenizer = CLIPModel.from_pretrained(name), CLIPTokenizer.from_pretrained(name)
        self.model = self.model.to(device).eval()
        self.device, self.size = device, self.model.config.vision_config.image_size

    @torch.no_grad()
    def __call__(self, images, prompts):
        pixels = (to_tensor([center_crop_resize(im, self.size) for im in images]) - CLIP_MEAN) / CLIP_STD
        ids = tokenize(self.tokenizer, prompts, 77)
        img = features(self.model.get_image_features(pixel_values=pixels.to(self.device)))
        txt = features(self.model.get_text_features(input_ids=ids.to(self.device)))
        return (100 * F.cosine_similarity(img, txt).clamp(min=0)).tolist()


class RandomFeatures(torch.nn.Module):
    """Extrator aleatório no lugar do Inception: só para smoke test offline (o FID resultante não significa nada)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 8, 8), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())

    def forward(self, x):
        return self.net(x.float())


def compute_fid(pipe, args, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    metric = FrechetInceptionDistance(feature=RandomFeatures() if args.random_encoders else 2048, normalize=True).to(device)
    res = pipe.meta["resolution"]
    reals, captions, n = [], [], 0

    def flush():
        nonlocal n
        fakes = pipe(captions, steps=args.steps, cfg=args.cfg, seed=args.seed + n, batch_size=args.batch_size)
        metric.update(to_tensor(reals).to(device), real=True)
        metric.update(to_tensor(fakes).to(device), real=False)
        n += len(reals)
        reals.clear()
        captions.clear()
        print(f"FID: {n}/{args.fid_samples}")

    for tar in sorted(Path(args.fid_wds).glob("*.tar")):
        for data, caption in iter_tar(tar):
            img = load_image(data)
            if img is None or not caption:
                continue
            reals.append(center_crop_resize(img, res))
            captions.append(caption)
            if len(reals) == args.batch_size or n + len(reals) == args.fid_samples:
                flush()
            if n == args.fid_samples:
                break
        if n == args.fid_samples:
            break
    if reals:
        flush()
    return float(metric.compute()), n


def main():
    parser = argparse.ArgumentParser(description="CLIP score + FID fixos para comparar versões do ZYI")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompts", default=str(ROOT / "eval" / "prompts.txt"))
    parser.add_argument("--seeds", type=int, default=4, help="imagens por prompt no CLIP score")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--fid_wds", default=None, help="pasta com .tar de validação (sem isso, pula o FID)")
    parser.add_argument("--fid_samples", type=int, default=5000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--random_encoders", action="store_true", help="CLIP/VAE/Inception aleatórios (só smoke test offline)")
    args = parser.parse_args()

    pipe = ZYIPipeline(args.ckpt, args.device, random_encoders=args.random_encoders)
    device = pipe.device
    prompts = [p.strip() for p in Path(args.prompts).read_text().splitlines() if p.strip()]
    t0 = time.time()

    # CLIP score: cada seed gera um lote com todos os prompts
    scorer = ClipScorer(device, args.clip_model, args.random_encoders)
    scores, first = [], None
    for s in range(args.seeds):
        images = pipe(prompts, steps=args.steps, cfg=args.cfg, seed=args.seed + 1000 * s, batch_size=args.batch_size)
        first = first or images
        scores += scorer(images, prompts)
    del scorer
    result = dict(ckpt=str(args.ckpt), step=pipe.step, resolution=pipe.meta["resolution"], steps=args.steps, cfg=args.cfg,
                  clip_model=args.clip_model, clip_score=float(np.mean(scores)), clip_images=len(scores))
    if args.fid_wds:
        result["fid"], result["fid_samples"] = compute_fid(pipe, args, device)
    result["minutes"] = round((time.time() - t0) / 60, 2)

    ckpt = Path(args.ckpt)
    out = Path(args.out) if args.out else ckpt.parent / f"eval_{ckpt.stem}.json"
    out.write_text(json.dumps(result, indent=2))
    image_grid(first, cols=10).save(out.with_suffix(".png"))
    print(json.dumps(result, indent=2))
    print(f"-> {out} e {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
