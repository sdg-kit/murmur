"""
serve.py — your own inference engine on a rented GPU (Modal), with LoRA personas.

    pip install modal
    python -m modal setup                 # one-time login (opens browser)
    modal deploy serve.py                 # prints a URL like https://you--murmur-vllm-serve.modal.run
    # then in murmur.py:  VLLM_URL = "<that url>/v1"

The server sleeps (and stops billing) after 10 idle minutes and wakes on the
next request. First cold start downloads the model (~1 min), later ones ~30s.

LoRA adapters: every folder in the "murmur-loras" volume becomes a selectable
model named after the folder. train_lora.py puts them there. Restart the
server (modal deploy again) after training new ones.
"""
import modal

BASE_MODEL = "Qwen/Qwen3-1.7B"   # keep in sync with murmur.py VLLM_BASE and train_lora.py
GPU        = "L4"                 # L4 ≈ cheap. "A100" or "H100" for tens of thousands of agents.
N_REPLICAS = 1                    # max GPU servers Modal may spin up under load (each = 1 GPU bill)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
app = modal.App("murmur-vllm")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)      # model weights, downloaded once
loras    = modal.Volume.from_name("murmur-loras", create_if_missing=True)   # one folder per persona adapter


@app.function(
    image=image, gpu=GPU, max_containers=N_REPLICAS,
    volumes={"/root/.cache/huggingface": hf_cache, "/loras": loras},
    scaledown_window=10 * 60, timeout=24 * 60 * 60,
)
@modal.concurrent(max_inputs=1024)          # requests in flight per GPU; vLLM batches them
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve():
    import os, subprocess
    adapters = [f"{d}={os.path.join('/loras', d)}" for d in sorted(os.listdir("/loras"))
                if os.path.isdir(os.path.join("/loras", d))]
    cmd = [
        "vllm", "serve", BASE_MODEL, "--port", "8000",
        "--seed", "0",                    # server-level seed; per-request seeds come from murmur.py
        "--max-model-len", "2048",        # our prompts are short; small window = more agents per batch
        "--enable-prefix-caching",        # RULES prefix is shared by every agent
        "--gpu-memory-utilization", "0.9",
        "--max-num-seqs", "512",          # concurrent sequences per batch; raise on bigger GPUs
    ]
    if adapters:
        cmd += ["--enable-lora", "--max-loras", str(min(len(adapters), 16)),
                "--max-lora-rank", "16", "--lora-modules", *adapters]
    print("starting:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)
