"""Offline integration check; its random adapter is not a usable chart model."""

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from peft import PeftModel
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from .data import SYSTEM, Tempo, compact, prepare, write_sm
from .training import train


def make_songs(root):
    events = [[0, "1000"], [48, "0100"], [96, "0010"], [144, "0001"],
              [336, "2000"], [384, "0100"], [432, "3000"], [480, "0040"],
              [576, "0030"], [624, "M000"], [672, "0011"]]
    for index in range(2):
        directory = Path(root) / f"song-{index}"
        directory.mkdir(parents=True)
        sr = 22050
        audio = np.zeros(int(8.5 * sr), dtype=np.float32)
        tempo = Tempo([[0, 120 + index * 10]], -0.25)
        for beat in range(16):
            start = round(float(tempo.time_at(beat)) * sr)
            t = np.arange(1000) / sr
            audio[start:start + len(t)] += (0.5 * np.sin(2 * np.pi * (150 + 120 * index) * t) * np.exp(-t * 60)).astype(np.float32)
        sf.write(directory / "music.wav", audio, sr)
        write_sm(directory / "chart.sm", events, tempo, "music.wav", 8, f"Synthetic {index}")


def smoke_test(output):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    torch.set_num_threads(2)
    torch.manual_seed(42)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Use a new smoke-test output directory: {output}")
    make_songs(output / "songs")
    train_rows, validation_rows = prepare(output / "songs", output / "data", "synthetic", 0.5)
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [SYSTEM] + [compact(r) for r in train_rows + validation_rows],
        trainers.BpeTrainer(vocab_size=512, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    base = output / "base"
    tokenizer.save_pretrained(base)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=4096, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    ))
    model.save_pretrained(base)
    config = yaml.safe_load(Path("configs/train.yaml").read_text())
    config.update({
        "model_name_or_path": str(base), "revision": None, "device": "cpu",
        "train_data": str(output / "data/train.jsonl"), "validation_data": str(output / "data/validation.jsonl"),
        "output_dir": str(output / "adapter"), "epochs": 1, "max_steps": 2,
        "gradient_accumulation_steps": 1, "logging_steps": 1, "learning_rate": 0.001,
    })
    config_path = output / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))
    trained, _, metrics = train(config_path)
    if not any(torch.count_nonzero(p).item() for name, p in trained.named_parameters() if "lora_B" in name):
        raise AssertionError("No LoRA B weights changed from their zero initialization")
    reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), output / "adapter")
    trained.eval()
    reloaded.eval()
    probe = torch.tensor([[tokenizer.bos_token_id, 10, 20, 30]])
    with torch.inference_mode():
        torch.testing.assert_close(trained(probe).logits, reloaded(probe).logits)
    assert (output / "adapter/tokenizer.json").exists()
    assert list((output / "adapter/logs").glob("events.out.tfevents.*"))
    print(f"PASS: audio -> song split -> chart-only LoRA training -> adapter reload; eval loss {metrics['eval_loss']:.4f}")
    print("Synthetic fixtures and a tiny random model only. No pretrained weights or external datasets downloaded.")
