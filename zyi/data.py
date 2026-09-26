"""
Dados do ZYI 2.

1) Encoders congelados (VAE do SD e CLIP ViT-L/14): usados só em prepare_data.py (codificar),
   sample.py/app.py (texto e decodificação) e na pré-visualização de amostras do log.
   O laço de treino nunca roda VAE nem CLIP.
2) ShardDataset: lê os shards pré-computados (.npy com memmap).

Layout de um diretório de dados (gerado por prepare_data.py):
    meta.json                         resolução, tamanho do texto, modelos usados
    null_text.npy, null_pooled.npy    embedding do prompt vazio (para CFG)
    eval_text.npy, eval_pooled.npy    embeddings de eval/prompts.txt (amostras no log)
    <shard>.latents.npy   (N, 4, H/8, W/8) float16, já multiplicado por 0.18215
    <shard>.text.npy      (N, L, 768)      float16, last_hidden_state do CLIP
    <shard>.pooled.npy    (N, 768)         float16, pooler_output do CLIP
    <shard>.txt           N legendas, uma por linha
"""
import io
import json
import math
import os
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

VAE_NAME = "stabilityai/sd-vae-ft-ema"
CLIP_NAME = "openai/clip-vit-large-patch14"
VAE_SCALE = 0.18215
IMG_EXTS = ("jpg", "jpeg", "png", "webp")


# ----------------------------------------------------------------------------- encoders congelados

def load_vae(device, random_init=False):
    """VAE do SD (4 canais, fator 8). random_init=True cria um VAE pequeno com pesos aleatórios
    e a mesma interface: só para smoke tests sem internet."""
    from diffusers import AutoencoderKL
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if random_init:
        torch.manual_seed(0)   # mesmos pesos aleatórios em todos os scripts
        vae = AutoencoderKL(down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                            block_out_channels=(32, 32, 64, 64), layers_per_block=1, latent_channels=4)
    else:
        vae = AutoencoderKL.from_pretrained(VAE_NAME, torch_dtype=dtype)
    if vae.dtype != dtype:
        vae = vae.to(dtype=dtype)
    return vae.to(device).eval().requires_grad_(False)


@torch.no_grad()
def vae_encode(vae, images):
    """images: (B, 3, H, W) em [-1, 1] -> latentes escalados (B, 4, H/8, W/8)."""
    return vae.encode(images.to(vae.device, vae.dtype)).latent_dist.sample() * VAE_SCALE


@torch.no_grad()
def vae_decode(vae, latents):
    """Latentes escalados -> imagens (B, 3, H, W) em [0, 1]."""
    x = vae.decode(latents.to(vae.device, vae.dtype) / VAE_SCALE).sample
    return (x.float() * 0.5 + 0.5).clamp(0, 1)


def latents_to_rgb_approx(latents):
    """Prévia barata sem VAE: projeção linear 4->3 canais (aproximação conhecida para o latente do SD 1.x)."""
    factors = torch.tensor([[0.3512, 0.2297, 0.3227], [0.3250, 0.4974, 0.2350],
                            [-0.2829, 0.1762, 0.2721], [-0.2120, -0.2616, -0.7177]], device=latents.device)
    rgb = torch.einsum("bchw,cr->brhw", latents.float(), factors)
    rgb = torch.nn.functional.interpolate(rgb, scale_factor=8, mode="nearest")
    return (rgb * 0.5 + 0.5).clamp(0, 1)


def tokenize(tokenizer, prompts, max_length):
    """Tokeniza no formato do CLIP (BOS ... EOS, completado com EOS). tokenizer=None usa um
    tokenizador de bytes: só para smoke tests sem internet."""
    if tokenizer is not None:
        return tokenizer(prompts, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt").input_ids
    ids = torch.full((len(prompts), max_length), 49407, dtype=torch.long)
    for i, p in enumerate(prompts):
        body = list(p.encode("utf-8"))[: max_length - 2]
        ids[i, : len(body) + 1] = torch.tensor([49406] + body)
    return ids


class TextEncoder:
    """CLIP ViT-L/14 congelado. __call__(prompts) -> (tokens (B, L, 768), pooled (B, 768))."""

    def __init__(self, device, max_length=32, random_init=False):
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
        self.device, self.max_length = device, max_length
        if random_init:
            torch.manual_seed(0)
            self.tokenizer = None
            model = CLIPTextModel(CLIPTextConfig(hidden_size=768, intermediate_size=1024, num_hidden_layers=2,
                                                 num_attention_heads=12, eos_token_id=49407))
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_NAME)
            model = CLIPTextModel.from_pretrained(CLIP_NAME)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = model.to(device, dtype).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prompts):
        ids = tokenize(self.tokenizer, list(prompts), self.max_length).to(self.device)
        out = self.model(input_ids=ids)
        return out.last_hidden_state, out.pooler_output


# ----------------------------------------------------------------------------- imagens

def iter_tar(path):
    """Itera (bytes da imagem, legenda) de um .tar no formato webdataset (key.jpg + key.txt + ...)."""
    sample, key = {}, None
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            folder, name = os.path.split(m.name)
            stem, _, ext = name.partition(".")
            if os.path.join(folder, stem) != key:
                if sample:
                    yield from _pair(sample)
                sample, key = {}, os.path.join(folder, stem)
            sample[ext.lower()] = tf.extractfile(m).read()
    if sample:
        yield from _pair(sample)


def _pair(sample):
    img = next((sample[e] for e in IMG_EXTS if e in sample), None)
    if img is not None and "txt" in sample:
        yield img, sample["txt"].decode("utf-8", "ignore").strip()


def center_crop_resize(img, size):
    """PIL -> quadrado central redimensionado para size x size."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC)


def load_image(data):
    """bytes -> PIL RGB, ou None se a imagem estiver corrompida."""
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def to_pil(x):
    """Tensor (3, H, W) em [0, 1] -> PIL."""
    return Image.fromarray((x.clamp(0, 1).permute(1, 2, 0).cpu().float().numpy() * 255).round().astype(np.uint8))


def image_grid(images, cols=None):
    """Lista de PIL (mesmo tamanho) -> um grid PIL."""
    cols = cols or math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(images):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    return grid


# ----------------------------------------------------------------------------- dataset de shards

class ShardDataset(torch.utils.data.Dataset):
    """Amostras (latente, texto, pooled) em float16 a partir dos shards .npy (memmap, acesso aleatório)."""

    def __init__(self, data_dir, max_samples=None):
        self.dir = Path(data_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.shards = sorted(p.name[: -len(".latents.npy")] for p in self.dir.glob("*.latents.npy"))
        assert self.shards, f"nenhum shard em {self.dir}; rode prepare_data.py antes"
        sizes = [np.load(self.dir / f"{s}.latents.npy", mmap_mode="r").shape[0] for s in self.shards]
        self.offsets = np.cumsum([0] + sizes)
        self.n = int(self.offsets[-1]) if not max_samples else min(int(self.offsets[-1]), int(max_samples))
        self.arrays = {}   # memmaps abertos sob demanda (em cada worker)
        try:  # cada memmap segura um descritor de arquivo; libera o limite para muitos shards
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        except (ImportError, ValueError, OSError):
            pass

    def __len__(self):
        return self.n

    def _shard(self, i):
        if i not in self.arrays:
            self.arrays[i] = [np.load(self.dir / f"{self.shards[i]}.{k}.npy", mmap_mode="r") for k in ("latents", "text", "pooled")]
        return self.arrays[i]

    def __getitem__(self, idx):
        i = int(np.searchsorted(self.offsets, idx, side="right")) - 1
        j = idx - int(self.offsets[i])
        return tuple(torch.from_numpy(np.array(a[j])) for a in self._shard(i))

    def _load(self, name):
        return torch.from_numpy(np.load(self.dir / name))

    def null_embedding(self):
        """Embedding do prompt vazio: ((1, L, D), (1, D))."""
        return self._load("null_text.npy")[None], self._load("null_pooled.npy")[None]

    def eval_embedding(self, n):
        """Embeddings dos n primeiros prompts de eval (para as amostras do log)."""
        return self._load("eval_text.npy")[:n], self._load("eval_pooled.npy")[:n]


class InfiniteSampler(torch.utils.data.Sampler):
    """Permutação nova a cada época, para sempre. `start` pula as amostras já vistas (resume exato)."""

    def __init__(self, n, seed=0, start=0):
        self.n, self.seed, self.start = n, seed, start

    def __iter__(self):
        epoch, offset = divmod(self.start, self.n)
        while True:
            perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + epoch))
            yield from perm[offset:].tolist()
            epoch, offset = epoch + 1, 0
