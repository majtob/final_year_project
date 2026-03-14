"""
Fine-tune Qwen 2.5 3B Instruct with QLoRA for Saudi credit verdicts.

Uses 4-bit quantization via bitsandbytes + peft (without unsloth)
for Windows compatibility. LoRA config mirrors ZiGong (Lei et al., 2025).

HF cache redirected to D: drive due to limited C: drive space.
"""

import json
import os
import torch
from pathlib import Path

os.environ["HF_HOME"] = "D:\\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:\\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "finetune_dataset.jsonl"
ADAPTER_DIR = PROJECT_ROOT / "models" / "lora_adapter"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_STEPS = 10
LOGGING_STEPS = 5


def load_dataset_jsonl():
    """Load instruction-format JSONL and convert to HF Dataset."""
    examples = []
    with open(DATASET_FILE) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


def main():
    print("=" * 60)
    print("QWEN 2.5 7B INSTRUCT -- QLoRA FINE-TUNING")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No CUDA GPU detected. Training will be very slow on CPU.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\nLoading {MODEL_NAME} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder="D:\\offload_tmp",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Model and tokenizer loaded")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied: {trainable:,} trainable / {total:,} total ({trainable/total:.2%})")

    dataset = load_dataset_jsonl()

    def formatting_func(example):
        messages = [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    training_args = SFTConfig(
        output_dir="D:\\training_output",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        fp16=True,
        bf16=False,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_func,
    )

    print(f"\nStarting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} (x{GRADIENT_ACCUMULATION} accumulation)")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

    train_result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Total steps: {train_result.global_step}")
    print(f"  Training loss: {train_result.training_loss:.4f}")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"\nLoRA adapter saved to {ADAPTER_DIR}")

    adapter_size = sum(
        f.stat().st_size for f in ADAPTER_DIR.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    print(f"Adapter size: {adapter_size:.1f} MB")

    config = {
        "base_model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "max_seq_length": MAX_SEQ_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "training_loss": train_result.training_loss,
        "total_steps": train_result.global_step,
        "training_examples": len(dataset),
    }
    config_file = ADAPTER_DIR / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {config_file}")


if __name__ == "__main__":
    main()
