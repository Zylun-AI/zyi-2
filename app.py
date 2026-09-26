"""
Demo Gradio do ZYI 2 (compatível com Hugging Face Spaces).

Local:
    python app.py --ckpt checkpoints/zyi2_512/ema.pt

Hugging Face Spaces (SDK gradio, app_file: app.py): suba este repositório e defina uma variável:
    ZYI_CKPT=<caminho do checkpoint dentro do Space>          ou
    ZYI_HF_REPO=<usuario/modelo-no-hub>  (+ ZYI_HF_FILE, padrão "ema.pt")
Em Spaces com ZeroGPU, a geração roda dentro de @spaces.GPU automaticamente.
"""
import argparse
import os

import gradio as gr

from sample import ZYIPipeline

try:  # ZeroGPU no Hugging Face Spaces
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn

EXAMPLES = [
    "a watercolor painting of a lighthouse at sunset",
    "a dog running on the beach at sunset",
    "a bowl of fresh fruit on a wooden table",
    "the city skyline at night with lights reflecting on the river",
]


def resolve_ckpt(path):
    if path:
        return path
    if os.environ.get("ZYI_CKPT"):
        return os.environ["ZYI_CKPT"]
    if os.environ.get("ZYI_HF_REPO"):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(os.environ["ZYI_HF_REPO"], os.environ.get("ZYI_HF_FILE", "ema.pt"))
    return "checkpoints/zyi2_512/ema.pt"


def build_demo(pipe):
    @gpu
    def generate(prompt, negative, steps, cfg, seed, n):
        return pipe([prompt] * int(n), negative_prompt=negative, steps=int(steps), cfg=float(cfg), seed=int(seed))

    res = pipe.meta["resolution"]
    with gr.Blocks(title="ZYI 2") as demo:
        gr.Markdown(f"# ZYI 2\nGerador texto→imagem {res}px (DiT + rectified flow), treinado do zero em 12h de uma A100. "
                    "Foi treinado com legendas do CC3M, então prompts em inglês funcionam melhor.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value=EXAMPLES[0], lines=2)
                negative = gr.Textbox(label="Prompt negativo (opcional)")
                with gr.Row():
                    steps = gr.Slider(10, 50, value=30, step=1, label="Passos (Euler)")
                    cfg = gr.Slider(1.0, 12.0, value=5.0, step=0.5, label="Guidance (CFG)")
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed")
                    n = gr.Slider(1, 4, value=4, step=1, label="Imagens")
                button = gr.Button("Gerar", variant="primary")
                gr.Examples(EXAMPLES, inputs=prompt)
            gallery = gr.Gallery(label="Resultado", columns=2, height="auto")
        inputs = [prompt, negative, steps, cfg, seed, n]
        button.click(generate, inputs, gallery)
        prompt.submit(generate, inputs, gallery)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo Gradio do ZYI 2")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true", help="link público temporário do Gradio")
    parser.add_argument("--random_encoders", action="store_true", help="CLIP/VAE aleatórios (só smoke test offline)")
    args = parser.parse_args()
    pipe = ZYIPipeline(resolve_ckpt(args.ckpt), random_encoders=args.random_encoders)
    build_demo(pipe).queue().launch(server_port=args.port, share=args.share)
