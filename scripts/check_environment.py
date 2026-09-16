"""Check the local ML stack without downloading any model or dataset."""

import argparse
import importlib.metadata
import io
import sys
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from accelerate import Accelerator
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from trl import SFTConfig, SFTTrainer

    print(f"Python: {sys.version.split()[0]}")
    for package in (
        "torch", "transformers", "peft", "trl", "accelerate", "datasets",
        "pydantic", "PyYAML", "tensorboard",
    ):
        print(f"{package}: {importlib.metadata.version(package)}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda or 'CPU build'}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA verification failed: check the host NVIDIA driver and GPU access. "
            "A Nix development shell does not install a kernel driver."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        for index in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(index)
            print(f"GPU {index}: {gpu.name}, {gpu.total_memory / 1024**3:.1f} GiB")

    # Exercise a real LoRA backward/optimizer pass on a tiny random Qwen3.
    # Use the tutorial's adapter settings, with a much smaller random base model.
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = get_peft_model(
        Qwen3ForCausalLM(config),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        ),
    ).to(device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    tokens = torch.randint(3, config.vocab_size, (2, 16), device=device)
    loss = model(input_ids=tokens, labels=tokens).loss
    if not torch.isfinite(loss):
        raise RuntimeError("Tiny Qwen3 produced a non-finite loss")
    loss.backward()
    optimizer.step()
    if not any(not torch.equal(old, new) for old, new in zip(before, trainable)):
        raise RuntimeError("The LoRA optimizer step did not update any adapter weights")
    print(f"Tiny Qwen3 LoRA training step: passed on {device} (loss={loss.item():.4f})")

    with TemporaryDirectory(prefix="itg-ai-tensorboard-") as log_dir:
        with SummaryWriter(log_dir) as writer:
            writer.add_scalar("train/loss", loss.item(), 1)
        events = EventAccumulator(log_dir).Reload().Scalars("train/loss")
        if len(events) != 1 or not np.isclose(events[0].value, loss.item()):
            raise RuntimeError("TensorBoard scalar logging failed")
    print("TensorBoard scalar logging: passed")

    if args.require_cuda:
        import bitsandbytes as bnb

        # Test the actual NF4 CUDA kernel used by QLoRA, not only its import.
        layer = bnb.nn.Linear4bit(64, 64, quant_type="nf4", compute_dtype=torch.float16)
        layer = layer.to(device)
        inputs = torch.randn(2, 64, device=device, dtype=torch.float16, requires_grad=True)
        layer(inputs).float().square().mean().backward()
        if inputs.grad is None or not torch.isfinite(inputs.grad).all():
            raise RuntimeError("NF4 backward pass failed")
        torch.cuda.synchronize()
        print(f"bitsandbytes {bnb.__version__} NF4 CUDA forward/backward: passed")

    # Exercise the dataset and native audio libraries too.
    dataset = Dataset.from_dict({"text": ["1000", "0100"]})
    if len(dataset) != 2:
        raise RuntimeError("Dataset construction failed")
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, np.zeros(2048, dtype=np.float32), 22050, format="WAV")
    audio_buffer.seek(0)
    audio, sample_rate = sf.read(audio_buffer, dtype="float32")
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=256, hop_length=128, n_mels=32
    )
    if mel.shape[0] != 32 or not np.isfinite(mel).all():
        raise RuntimeError("Audio feature extraction failed")
    print("Datasets, audio round trip, and Mel features: passed")
    print(f"Training interfaces: {Accelerator.__name__}, {SFTConfig.__name__}, {SFTTrainer.__name__}")
    print("Environment check passed. No pretrained weights were downloaded.")


if __name__ == "__main__":
    main()
