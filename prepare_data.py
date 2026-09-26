"""
Prepara o CC3M para o ZYI 2: baixa, filtra e pré-computa latentes do VAE + embeddings do CLIP.

Etapas (cada uma é retomável: rodar de novo continua de onde parou):
  download   baixa o TSV oficial do CC3M e as imagens com img2dataset (.tar webdataset, lado menor <= 512px)
  encode     filtra, recorta/redimensiona, roda VAE + CLIP e grava shards .npy (uma vez por resolução)
  synthetic  gera shards sintéticos (sem internet) para smoke tests

Exemplos:
  python prepare_data.py download --out data/cc3m_wds
  python prepare_data.py encode --wds data/cc3m_wds --out data/cc3m_256 --resolution 256
  python prepare_data.py encode --wds data/cc3m_wds --out data/cc3m_512 --resolution 512
  python prepare_data.py download --split val --out data/cc3m_val_wds          # para o FID
  python prepare_data.py synthetic --out data/synthetic --resolution 64
"""
import argparse
import json
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from zyi.data import (CLIP_NAME, VAE_NAME, VAE_SCALE, TextEncoder, center_crop_resize, iter_tar, load_image,
                      load_vae, vae_encode)

CC3M_TSV = {
    "train": "https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv",
    "val": "https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv",
}
EVAL_PROMPTS = Path(__file__).parent / "eval" / "prompts.txt"


# ----------------------------------------------------------------------------- download

def cmd_download(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tsv = Path(args.tsv) if args.tsv else out / f"cc3m_{args.split}.tsv"
    if not tsv.exists():
        print(f"baixando {CC3M_TSV[args.split]} -> {tsv}")
        urllib.request.urlretrieve(CC3M_TSV[args.split], tsv)
    # o TSV oficial não tem cabeçalho (colunas: legenda, url); o img2dataset precisa de um
    url_list = out / f"cc3m_{args.split}_urls.tsv"
    with open(tsv, encoding="utf-8") as src, open(url_list, "w", encoding="utf-8") as dst:
        dst.write("caption\turl\n")
        for i, line in enumerate(src):
            if i == 0 and line.startswith("caption\t"):
                continue
            if args.max_urls and i >= args.max_urls:
                break
            dst.write(line)

    from img2dataset import download
    download(
        url_list=str(url_list), input_format="tsv", url_col="url", caption_col="caption",
        output_format="webdataset", output_folder=str(out),
        image_size=args.image_size, resize_mode="keep_ratio", resize_only_if_bigger=True,  # lado menor <= 512
        min_image_size=args.min_image_size, max_aspect_ratio=args.max_aspect,
        processes_count=args.processes, thread_count=args.threads, number_sample_per_shard=10000,
        encode_quality=95, timeout=10, retries=1, incremental_mode="incremental",
    )


# ----------------------------------------------------------------------------- encode

class TarImages(IterableDataset):
    """Decodifica e filtra imagens dos .tar em paralelo (cada worker pega alguns tars inteiros).
    Emite (nome_do_tar, imagem uint8 (3, r, r), legenda) e, ao fim de cada tar, (nome_do_tar, None, None)."""

    def __init__(self, tars, args):
        self.tars, self.a = tars, args

    def keep(self, img, caption):
        words = len(caption.split())
        if not (self.a.min_words <= words <= self.a.max_words):
            return False
        w, h = img.size
        return min(w, h) >= self.a.min_size and max(w, h) / min(w, h) <= self.a.max_aspect

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        for tar in self.tars[wid::nw]:
            for data, caption in iter_tar(tar):
                img = load_image(data)
                if img is not None and self.keep(img, caption):
                    img = center_crop_resize(img, self.a.resolution)
                    yield tar.stem, torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1), caption.replace("\n", " ")
            yield tar.stem, None, None


def save_arrays(out, name, arrays, captions):
    """Grava um shard; o .latents.npy vai por último e é o que marca o shard como completo."""
    (out / f"{name}.txt").write_text("\n".join(captions) + "\n", encoding="utf-8")
    for key in ("text", "pooled", "latents"):
        tmp = out / f"{name}.{key}.tmp.npy"
        np.save(tmp, arrays[key])
        os.replace(tmp, out / f"{name}.{key}.npy")


def write_fixed(out, text_encoder, args, extra=None):
    """meta.json + embeddings do prompt vazio (CFG) e dos prompts de eval (amostras no log)."""
    meta = dict(resolution=args.resolution, latent_size=args.resolution // 8, latent_channels=4,
                text_len=args.text_len, text_dim=768, vae=VAE_NAME, text_encoder=CLIP_NAME, vae_scale=VAE_SCALE,
                random_encoders=bool(getattr(args, "random_encoders", False)), **(extra or {}))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    prompts = [p.strip() for p in EVAL_PROMPTS.read_text().splitlines() if p.strip()]
    for name, texts in (("null", [""]), ("eval", prompts)):
        tokens, pooled = text_encoder(texts)
        tokens, pooled = tokens.half().cpu().numpy(), pooled.half().cpu().numpy()
        np.save(out / f"{name}_text.npy", tokens[0] if name == "null" else tokens)
        np.save(out / f"{name}_pooled.npy", pooled[0] if name == "null" else pooled)


def cmd_encode(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.min_size = args.min_size or args.resolution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(device, random_init=args.random_encoders)
    text_encoder = TextEncoder(device, args.text_len, random_init=args.random_encoders)
    write_fixed(out, text_encoder, args)

    tars = sorted(Path(args.wds).glob("*.tar"))
    done = {p.name[: -len(".latents.npy")] for p in out.glob("*.latents.npy")}
    count = sum(np.load(out / f"{s}.latents.npy", mmap_mode="r").shape[0] for s in done)
    todo = [t for t in tars if t.stem not in done]
    print(f"{len(tars)} tars, {len(done)} já codificados ({count} amostras), {len(todo)} por fazer")
    loader = DataLoader(TarImages(todo, args), batch_size=None, num_workers=args.workers)

    buffers = defaultdict(lambda: defaultdict(list))   # tar -> {"latents": [...], "text": [...], ...}
    pending = []                                       # (tar, imagem, legenda) esperando o próximo lote

    def encode_pending():
        if not pending:
            return
        images = torch.stack([p[1] for p in pending]).to(device).float() / 127.5 - 1
        captions = [p[2] for p in pending]
        latents = vae_encode(vae, images).half().cpu().numpy()
        tokens, pooled = (x.half().cpu().numpy() for x in text_encoder(captions))
        for i, (name, _, caption) in enumerate(pending):
            buf = buffers[name]
            buf["latents"].append(latents[i])
            buf["text"].append(tokens[i])
            buf["pooled"].append(pooled[i])
            buf["captions"].append(caption)
        pending.clear()

    def flush(name):
        buf = buffers.pop(name, None)
        if buf:
            save_arrays(out, name, {k: np.stack(buf[k]) for k in ("latents", "text", "pooled")}, buf["captions"])

    t0, max_seconds = time.time(), (args.max_hours or float("inf")) * 3600
    for name, image, caption in loader:
        if image is None:        # fim de um tar: codifica o que falta e grava o shard
            encode_pending()
            flush(name)
            continue
        pending.append((name, image, caption))
        count += 1
        if len(pending) == args.batch_size:
            encode_pending()
        if count % 10000 == 0:
            print(f"{count} amostras | {count / (time.time() - t0):.0f}/s")
        if (args.max_samples and count >= args.max_samples) or time.time() - t0 > max_seconds:
            print("limite de amostras/tempo atingido; gravando shards parciais")
            break
    encode_pending()
    for name in list(buffers):
        flush(name)
    print(f"pronto: {count} amostras em {out}")


# ----------------------------------------------------------------------------- synthetic

def cmd_synthetic(args):
    """Dados falsos com estrutura aprendível: K 'conceitos', cada um com um latente-protótipo e um
    embedding de texto. Se o treino funciona, a loss cai bem abaixo de ~1.0 (valor de um modelo que não aprendeu nada)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    s, K, L = args.resolution // 8, args.concepts, args.text_len
    protos = torch.nn.functional.interpolate(torch.randn(K, 4, 4, 4, generator=g), size=(s, s), mode="bilinear")
    texts, pooled = torch.randn(K, L, 768, generator=g), torch.randn(K, 768, generator=g)
    labels = torch.randint(K, (args.num_samples,), generator=g)
    latents = protos[labels] + 0.1 * torch.randn(args.num_samples, 4, s, s, generator=g)
    for i in range(0, args.num_samples, args.shard_size):
        idx = labels[i : i + args.shard_size]
        arrays = dict(latents=latents[i : i + args.shard_size], text=texts[idx], pooled=pooled[idx])
        save_arrays(out, f"synthetic-{i // args.shard_size:05d}", {k: v.half().numpy() for k, v in arrays.items()},
                    [f"concept {k}" for k in idx.tolist()])
    meta = dict(resolution=args.resolution, latent_size=s, latent_channels=4, text_len=L, text_dim=768,
                vae=None, text_encoder=None, vae_scale=VAE_SCALE, random_encoders=True, synthetic=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    np.save(out / "null_text.npy", np.zeros((L, 768), np.float16))
    np.save(out / "null_pooled.npy", np.zeros(768, np.float16))
    np.save(out / "eval_text.npy", texts.half().numpy())
    np.save(out / "eval_pooled.npy", pooled.half().numpy())
    print(f"{args.num_samples} amostras sintéticas em {out}")


# ----------------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="baixa o CC3M com img2dataset")
    d.add_argument("--out", default="data/cc3m_wds")
    d.add_argument("--split", default="train", choices=["train", "val"])
    d.add_argument("--tsv", default=None, help="TSV do CC3M já baixado (senão baixa o oficial)")
    d.add_argument("--max_urls", type=int, default=None, help="usa só as primeiras N URLs")
    d.add_argument("--image_size", type=int, default=512, help="lado menor máximo guardado")
    d.add_argument("--min_image_size", type=int, default=256)
    d.add_argument("--max_aspect", type=float, default=2.0)
    d.add_argument("--processes", type=int, default=16)
    d.add_argument("--threads", type=int, default=64)

    e = sub.add_parser("encode", help="pré-computa latentes do VAE + embeddings do CLIP")
    e.add_argument("--wds", default="data/cc3m_wds", help="pasta com os .tar")
    e.add_argument("--out", required=True)
    e.add_argument("--resolution", type=int, default=256)
    e.add_argument("--min_size", type=int, default=None, help="lado menor mínimo da imagem (padrão: a resolução)")
    e.add_argument("--max_aspect", type=float, default=2.0)
    e.add_argument("--min_words", type=int, default=3)
    e.add_argument("--max_words", type=int, default=40)
    e.add_argument("--text_len", type=int, default=32, help="tokens do CLIP guardados (legendas do CC3M são curtas)")
    e.add_argument("--max_samples", type=int, default=None, help="padrão: todas")
    e.add_argument("--max_hours", type=float, default=None, help="para ao atingir este tempo (padrão: sem limite)")
    e.add_argument("--batch_size", type=int, default=128)
    e.add_argument("--workers", type=int, default=16)
    e.add_argument("--random_encoders", action="store_true", help="VAE/CLIP com pesos aleatórios (só smoke test offline)")

    s = sub.add_parser("synthetic", help="dados sintéticos para smoke tests")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--resolution", type=int, default=64)
    s.add_argument("--num_samples", type=int, default=2048)
    s.add_argument("--concepts", type=int, default=8)
    s.add_argument("--text_len", type=int, default=8)
    s.add_argument("--shard_size", type=int, default=512)

    args = parser.parse_args()
    {"download": cmd_download, "encode": cmd_encode, "synthetic": cmd_synthetic}[args.cmd](args)


if __name__ == "__main__":
    main()
