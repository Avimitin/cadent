"""Qwen3 LoRA SFT with chart-only loss and identical inference prompting."""

import json
import math
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, set_seed,
)

from .data import SCHEMA, SPAN, SYSTEM, compact, validate_events


def prompt_ids(tokenizer, context):
    if context.get("schema") != SCHEMA:
        raise ValueError(f"Expected feature schema {SCHEMA}")
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": compact(context)}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )


def encode(tokenizer, example, max_length):
    context = example["conditioning"]
    validate_events(example["events"], SPAN, context["initial_holds"], context["final_window"])
    prefix = prompt_ids(tokenizer, context)
    answer = tokenizer(compact(example["events"]), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prefix) + len(answer) > max_length:
        raise ValueError(
            f"Example from {example.get('source', '?')} needs {len(prefix) + len(answer)} tokens "
            f"but max_length={max_length}; increase it or revise the window schema. No silent truncation."
        )
    return {"input_ids": prefix + answer, "attention_mask": [1] * (len(prefix) + len(answer)),
            "labels": [-100] * len(prefix) + answer}


def read_examples(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def train(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a new output directory; refusing to overwrite {output}")
    seed = int(config.get("seed", 42))
    set_seed(seed)
    train_rows = read_examples(config["train_data"])
    validation_rows = read_examples(config["validation_data"])
    if {r["song_id"] for r in train_rows} & {r["song_id"] for r in validation_rows}:
        raise ValueError("Audio identity leakage between training and validation")
    device = config.get("device", "cuda")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use smoke-test here, or explicitly select device: cpu for real training.")
    quantized = config.get("load_in_4bit", False)
    if quantized and device != "cuda":
        raise ValueError("4-bit QLoRA requires the CUDA environment for this draft")
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16 if device == "cuda" else torch.float32
    model_id = config["model_name_or_path"]
    revision = config.get("revision")
    if not Path(model_id).exists() and not revision:
        raise ValueError("Pin a revision when loading a model from the Hub")
    source = {"revision": revision, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **source)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError("A chat tokenizer with EOS is required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_length = int(config.get("max_length", 4096))
    # Fail on malformed/oversized data before allocating the base-model weights.
    train_data = Dataset.from_list([encode(tokenizer, row, max_length) for row in train_rows])
    validation_data = Dataset.from_list([encode(tokenizer, row, max_length) for row in validation_rows])
    load_options = {"dtype": dtype, **source}
    if quantized:
        load_options.update(
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            ),
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_options)
    if max_length > model.config.max_position_embeddings:
        raise ValueError("max_length exceeds the base model's context limit")
    checkpointing = bool(config.get("gradient_checkpointing", True))
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **config["lora"]))
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=str(output), use_cpu=device == "cpu", seed=seed,
        num_train_epochs=float(config.get("epochs", 3)), max_steps=int(config.get("max_steps", -1)),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(config.get("batch_size", 1)),
        per_device_eval_batch_size=int(config.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 16)),
        gradient_checkpointing=checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, fp16=device == "cuda" and not bf16, optim="adamw_torch",
        lr_scheduler_type="cosine", warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        logging_steps=int(config.get("logging_steps", 10)), logging_first_step=True,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=["tensorboard"], logging_dir=str(output / "logs"),
        dataloader_pin_memory=device == "cuda", prediction_loss_only=True,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_data, eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    result = trainer.train()
    metrics = trainer.evaluate()
    if not math.isfinite(result.training_loss) or not math.isfinite(metrics["eval_loss"]):
        raise ValueError("Non-finite training/validation loss")
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    manifest = {"schema": SCHEMA, "config": config, "metrics": metrics,
                "base_model": model_id, "revision": revision,
                "train_songs": sorted({r["song_id"] for r in train_rows}),
                "validation_songs": sorted({r["song_id"] for r in validation_rows})}
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Adapter, tokenizer and run manifest saved to {output}")
    return model, tokenizer, metrics
