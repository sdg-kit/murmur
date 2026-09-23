"""
train_lora.py — fine-tune one persona adapter on a rented GPU and drop it
into the volume that serve.py reads from.

    modal run train_lora.py --persona Scout_gloomy
    modal run train_lora.py --persona Liar --epochs 3

Reads lora_data/<persona>.jsonl (made by make_lora_data.py), writes the adapter
to the "murmur-loras" volume as /loras/<persona>. Then `modal deploy serve.py`
to reload the server with the new adapter. A few hundred examples on a 1.7B
model takes ~2–5 minutes on an L4.
"""
import modal

BASE_MODEL = "Qwen/Qwen3-1.7B"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "peft", "trl", "datasets", "accelerate")
)
app = modal.App("murmur-train")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
loras    = modal.Volume.from_name("murmur-loras", create_if_missing=True)


@app.function(image=image, gpu="L4", timeout=60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache, "/loras": loras})
def train(persona: str, data: str, epochs: int = 2, rank: int = 16):
    import json
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in data.splitlines() if l.strip()]
    ds = Dataset.from_list(rows)                       # each row: {"messages": [...]}
    print(f"{persona}: {len(ds)} examples")

    trainer = SFTTrainer(
        model=BASE_MODEL,
        train_dataset=ds,
        peft_config=LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir="/tmp/out", num_train_epochs=epochs,
            per_device_train_batch_size=8, gradient_accumulation_steps=2,
            learning_rate=2e-4, bf16=True, logging_steps=10,
            assistant_only_loss=True,        # learn the action, not the observation
            max_length=2048, report_to="none",
        ),
    )
    trainer.train()
    trainer.model.save_pretrained(f"/loras/{persona}")
    loras.commit()
    print(f"saved /loras/{persona}")


@app.local_entrypoint()
def main(persona: str, epochs: int = 2, rank: int = 16):
    data = open(f"lora_data/{persona}.jsonl").read()   # read locally, ship to the GPU function
    train.remote(persona, data, epochs, rank)
