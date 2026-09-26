# ZYI 2

**Gerador de imagens texto→imagem 512px treinado do zero em 12 horas de uma única A100 80GB.**

<p align="center">
  <i>[ espaço reservado: grid de amostras do ZYI 2 (assets/samples.png) ]</i>
</p>

O ZYI 2 é uma receita reproduzível no estilo do [nanoGPT](https://github.com/karpathy/nanoGPT): pouco código,
fácil de ler e de rodar. Um Diffusion Transformer (DiT) de ~366M parâmetros é treinado com
rectified flow sobre latentes do VAE do Stable Diffusion, condicionado por texto do CLIP ViT-L/14. VAE e CLIP
são pré-treinados e congelados; só o modelo de difusão é treinado.

## Quickstart

```bash
git clone https://github.com/Zylun-AI/zyi-2.git && cd zyi-2
pip install -r requirements.txt

# 1) dados (fora do orçamento de 12h): baixa o CC3M, filtra e pré-computa latentes + embeddings
python prepare_data.py download --out data/cc3m_wds        # URLs do Hugging Face + img2dataset
# ou, sem links mortos: .tar com as imagens já baixadas (pixparse/cc3m-wds; --max_shards N limita)
# python prepare_data.py download --source wds --out data/cc3m_wds
python prepare_data.py encode --wds data/cc3m_wds --out data/cc3m_256 --resolution 256
python prepare_data.py encode --wds data/cc3m_wds --out data/cc3m_512 --resolution 512

# 2) treino: fase 1 em 256px (~8h) e fase 2 em 512px (~4h, parte dos pesos da fase 1)
python train.py --config configs/zyi2_256.yaml
python train.py --config configs/zyi2_512.yaml

# 3) amostras
python sample.py --ckpt checkpoints/zyi2_512/ema.pt --prompt "a dog running on the beach at sunset" --n 4
python sample.py --ckpt checkpoints/zyi2_512/ema.pt --prompts_file eval/prompts.txt --n 1 --out samples/eval.png

# 4) demo e avaliação
python app.py --ckpt checkpoints/zyi2_512/ema.pt
python eval/evaluate.py --ckpt checkpoints/zyi2_512/ema.pt
```

Se o treino cair (ou for interrompido com Ctrl+C / SIGTERM), rode **o mesmo comando** de novo: ele retoma de
`<out_dir>/ckpt.pt` com o otimizador, o EMA, a posição exata nos dados e o tempo já gasto.

Qualquer chave do YAML pode ser sobrescrita na linha de comando: `--batch_size=128`, `--model.depth=12`,
`--logger=wandb`, `--time_budget_hours=6`...

## Evolução do ZYI

| Versão | Horas GPU | Resolução | Params | CLIP score | Amostra |
|---|---|---|---|---|---|
| ZYI 1.0 | ~7–8 h (A100 80GB) | 256×256 | TBD | TBD | TBD |
| ZYI 1.1 tiny | 1 h (A100 80GB) | 256×256 | TBD | TBD | TBD |
| ZYI 1.2 | TBD | TBD | TBD | TBD | TBD |
| **ZYI 2** | **12 h (A100 80GB)** | **512×512** (256→512) | **366M** | TBD | TBD |

O ZYI 1.2 está em treino com 150k imagens do CC3M. O CLIP score é medido com `eval/evaluate.py`: mesmos 50 prompts
e mesmas seeds em todas as versões.

## Orçamento de 12h

| | Fase 1 | Fase 2 |
|---|---|---|
| Config | `configs/zyi2_256.yaml` | `configs/zyi2_512.yaml` |
| Resolução | 256px (latente 32×32 → 256 tokens) | 512px (latente 64×64 → 1024 tokens) |
| Tempo | 8 h | 4 h |
| Batch | 256 (2 × 128) | 128 (4 × 32) |
| LR | 2e-4 → 2e-5 (cosseno) | 1e-4 → 1e-5 (cosseno) |
| Custo por amostra | ~0,38 TFLOP | ~1,65 TFLOP |
| Throughput estimado | ~250–330 amostras/s | ~55–75 amostras/s |
| Amostras vistas | ~7–9,5M (~3 épocas) | ~0,8–1,1M |
| Passos | ~28–37k | ~6–8,5k |

- **Fase 1** aprende o grosso (semântica, composição, cores) em baixa resolução, onde cada amostra custa ~4× menos.
- **Fase 2** retoma os pesos EMA da fase 1 e aprende detalhes em 512px. As posições do RoPE 2D são interpoladas
  para o grid da fase 1 (0, 0,5, 1, …, 15,5), e o *time shift* sobe para 2,0 (mais ruído em resolução maior, como no SD3).
- O treino para sozinho quando o tempo acaba. O LR segue um cosseno **sobre o orçamento de tempo** (não sobre um
  número fixo de passos), então chega ao mínimo exatamente no fim, seja qual for a velocidade da máquina.

O throughput é uma estimativa (FLOPs ≈ 6 × 235M params por token + atenção; A100 com 312 TFLOPS bf16 de pico e
30–40% de utilização), ainda não medido numa A100. O valor real aparece no log (`amostras/s`). Os micro-batches
foram escolhidos para ~45 GiB de ativações. Com `torch.compile` costuma sobrar memória, e dá para subir
`micro_batch_size` (256 na fase 1, 64 na fase 2).

**Fora das 12h:** a preparação dos dados. O download do CC3M leva de 2 a 6 h, dependendo da banda (espere que só 50–75%
das 3,3M URLs ainda funcionem). Codificar em 256px e em 512px leva ~1 h cada numa A100. Reserve ~400 GB de disco:
imagens (~150 GB), `cc3m_256` (~57 KB/amostra) e `cc3m_512` (~82 KB/amostra, só imagens com lado ≥ 512).

## Como funciona

| Peça | Escolha |
|---|---|
| Modelo (`zyi/model.py`) | DiT, patch 2, dim 1024, 16 blocos, 16 cabeças, MLP 4×, ~366M params |
| Condição de tempo | adaLN-Zero: `c = emb(t) + proj(CLIP pooled)` gera shift/scale/gate de cada bloco |
| Condição de texto | cross-attention em todos os blocos sobre os 32 tokens do CLIP ViT-L/14 (768-d) |
| Posição | RoPE 2D axial com interpolação de posições entre resoluções |
| Atenção | `F.scaled_dot_product_attention` (Flash Attention na GPU) + QK-norm |
| Objetivo (`zyi/flow.py`) | rectified flow: `x_t = (1-t)·x0 + t·ε`, prevê `v = ε - x0`, t ~ logit-normal(0, 1) |
| CFG | 10% de prompts vazios no treino; sampler Euler (20–50 passos) com guidance configurável e prompt negativo |
| VAE | `stabilityai/sd-vae-ft-ema`: 4 canais, fator 8, escala 0,18215 |
| Dados (`zyi/data.py`) | shards `.npy` pré-computados, lidos por memmap. O treino nunca roda VAE nem CLIP |
| Performance | bf16, `torch.compile`, AdamW fused, EMA, gradient checkpointing opcional |
| Robustez | checkpoint a cada 30 min, resume automático, orçamento de horas, SIGTERM salva e sai |
| Logs | tensorboard (`<out_dir>/tb`) ou wandb (`--logger=wandb`); loss, LR, grad norm, amostras/s e um grid de amostras a cada hora |

O VAE só é carregado no treino para decodificar as 8 amostras de pré-visualização do log. Com `preview_vae: false`,
ele usa uma aproximação linear do latente e não baixa nada.

## Dados

`prepare_data.py` tem três etapas, todas retomáveis:

- `download`: por padrão (`--source urls`) pega legendas + URLs do espelho do CC3M no Hugging Face
  ([google-research-datasets/conceptual_captions](https://huggingface.co/datasets/google-research-datasets/conceptual_captions);
  o TSV no Google Storage não é mais público) e baixa as imagens com [img2dataset](https://github.com/rom1504/img2dataset)
  (webdataset `.tar`, lado menor ≤ 512px). `--max_urls N` limita o download. Com `--source wds` baixa os `.tar` já
  prontos de [pixparse/cc3m-wds](https://huggingface.co/datasets/pixparse/cc3m-wds): mais rápido e sem links mortos,
  mas com imagens no tamanho original (mais disco); `--max_shards N` limita.
- `encode`: filtra (legenda com 3–40 palavras, lado menor ≥ resolução, proporção ≤ 2:1), faz o recorte central e
  grava por `.tar` de entrada um shard com `latents` (float16), `text` (32×768), `pooled` (768) e legendas. Também
  grava `meta.json`, o embedding do prompt vazio e os dos prompts de eval. `--max_samples N` e `--max_hours H`
  limitam o tamanho. O padrão é usar tudo.
- `synthetic`: dados falsos com estrutura aprendível, para testar o pipeline sem internet.

Para treinar com menos dados, use `--max_samples` no `encode` ou `max_samples` no YAML de treino.

## Avaliação

```bash
python prepare_data.py download --split val --out data/cc3m_val_wds
python eval/evaluate.py --ckpt checkpoints/zyi2_512/ema.pt --fid_wds data/cc3m_val_wds --fid_samples 5000
```

- **CLIP score**: 50 prompts fixos (`eval/prompts.txt`) × 4 seeds fixas, `100 · cos(imagem, texto)` com CLIP ViT-L/14.
- **FID**: legendas das primeiras 5000 amostras válidas do CC3M de validação (em ordem), imagens geradas vs. reais.
- Resultado em `eval_<ckpt>.json` e um grid dos 50 prompts ao lado do checkpoint.

Compare versões sempre com os mesmos `--steps`, `--cfg`, `--seeds` e `--fid_samples`.

## Demo (Gradio / Hugging Face Spaces)

`python app.py --ckpt checkpoints/zyi2_512/ema.pt` abre a demo local. Para publicar num
[Space](https://huggingface.co/docs/hub/spaces-sdks-gradio) (SDK Gradio, `app_file: app.py`), suba o `ema.pt` num
repositório de modelo do Hub e defina a variável `ZYI_HF_REPO=<usuario>/<modelo>` (e `ZYI_HF_FILE`, se o nome não for
`ema.pt`). Em Spaces com ZeroGPU, a geração roda sob `@spaces.GPU` automaticamente.

## Smoke test (sem GPU, sem internet)

Um teste rápido do pipeline inteiro com um modelo minúsculo em dados sintéticos (roda em ~1 min na CPU):

```bash
python prepare_data.py synthetic --out data/synthetic --resolution 64
python train.py --config configs/zyi2_256.yaml --data_dir=data/synthetic --out_dir=checkpoints/smoke \
  --model.dim=128 --model.depth=2 --model.heads=4 --model.rope_base_grid=4 \
  --batch_size=32 --micro_batch_size=16 --lr=1e-3 --warmup_steps=20 --max_steps=300 \
  --compile=false --num_workers=2 --preview_vae=false --logger=none
python sample.py --ckpt checkpoints/smoke/ema.pt --prompt "concept 1" --n 4 --steps 20 --out samples/smoke.png
```

A loss deve cair de ~1,07 para ~0,43. `prepare_data.py encode`, `sample.py`, `app.py` e `eval/evaluate.py` aceitam
`--random_encoders` (VAE/CLIP com pesos aleatórios) para testar sem baixar nada.

## Estrutura

```
zyi/model.py       DiT (adaLN-Zero, cross-attention, RoPE 2D)
zyi/flow.py        loss de flow matching + sampler Euler com CFG
zyi/data.py        VAE/CLIP congelados, leitura dos shards, utilitários de imagem
prepare_data.py    download do CC3M, filtro, latentes + embeddings
train.py           treino (um arquivo), com resume e orçamento de tempo
sample.py          prompt → grid PNG
app.py             demo Gradio
eval/              prompts fixos + CLIP score e FID
configs/           fase 1 (256px) e fase 2 (512px)
```

## Créditos e licenças

| Componente | Link | Licença |
|---|---|---|
| VAE `sd-vae-ft-ema` (Stability AI) | [huggingface.co/stabilityai/sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema) | MIT |
| CLIP ViT-L/14 (OpenAI) | [huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) · [github.com/openai/CLIP](https://github.com/openai/CLIP) | MIT |
| Conceptual Captions 3M (Google) | [ai.google.com/research/ConceptualCaptions](https://ai.google.com/research/ConceptualCaptions/) · [github.com/google-research-datasets/conceptual-captions](https://github.com/google-research-datasets/conceptual-captions) | Uso livre para qualquer finalidade, com agradecimento ao Google LLC como fonte; as imagens pertencem aos seus autores |
| img2dataset | [github.com/rom1504/img2dataset](https://github.com/rom1504/img2dataset) | MIT |
| FID (Inception via torchmetrics / torch-fidelity) | [github.com/Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics) | Apache-2.0 |

Referências: DiT ([Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)), rectified flow / SD3
([Esser et al., 2024](https://arxiv.org/abs/2403.03206)), [nanoGPT](https://github.com/karpathy/nanoGPT).

O código do ZYI 2 é MIT (veja `LICENSE`). Os pesos treinados herdam as condições dos dados e modelos acima.
