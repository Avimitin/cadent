# Cadent

Experimental music-to-ITG chart generation with locally fine-tuned Qwen models.

The development environment targets x86_64 Linux/NixOS.

Nix pins Python 3.12, uv, a compiler, FFmpeg, and native libraries through `flake.lock`. Python training packages and
their CPU/CUDA variants are pinned by `uv.lock`. Python packages are installed into `.venv`; this is a Nix development
shell with a locked uv environment, not a fully Nix-packaged Python closure.

**Start on this development machine:**

```bash
nix develop
bash scripts/setup.sh cpu
```

The setup installs the locked packages and runs an offline check: imports the training stack, performs a LoRA optimizer
step on a tiny randomly initialized Qwen3 using the tutorial's adapter settings, writes and reads a TensorBoard metric,
and checks audio feature extraction. It downloads no pretrained model. CPU mode is for development and validation; full
Qwen fine-tuning should be planned on a suitable training machine.

**On an NVIDIA training machine:**

```bash
nix develop
bash scripts/setup.sh cuda
```

This selects PyTorch's CUDA 12.8 wheels and bitsandbytes for QLoRA. It also requires a successful GPU check and an NF4
forward/backward pass. CUDA userspace libraries arrive with the Python wheels; a compatible NVIDIA driver and visible
GPU must already be configured on the host. The flake exposes NixOS's `/run/opengl-driver/lib` without changing system
configuration. It does not install a kernel driver or the full CUDA development toolkit.

The CPU and CUDA extras are mutually exclusive. Run the setup command again to switch the same `.venv`. Always select an
extra when syncing manually, and use `--no-sync` when running commands so uv preserves the selected backend:

```bash
uv run --no-sync python scripts/check_environment.py
uv run --no-sync accelerate env
uv run --no-sync python train_itg.py --help
```

Use `uv run --no-sync python scripts/check_environment.py --require-cuda` to require GPU/QLoRA validation. Enter the Nix
shell each time so shared-library paths are available. Existing direnv users can run `direnv allow` to enable the
included `.envrc`.

The stack includes PyTorch 2.9.1, Transformers 4.57.6, PEFT 0.18.1, TRL 0.24.0, Accelerate 1.12.0, Datasets 4.4.2,
librosa, and SoundFile. The tutorial's requirements are covered, including Pydantic, PyYAML for its configuration files,
and TensorBoard for its training logger. The CUDA extra adds bitsandbytes 0.49.2 for future QLoRA experiments. The
tutorial itself uses ordinary LoRA with Hugging Face Trainer; it does not require bitsandbytes, TRL, ms-swift, Unsloth,
DeepSpeed, or FlashAttention.

**The supplied tutorial uses Qwen3-1.7B (1.7 billion parameters).** Its
[configuration](https://github.com/junqiangchen/qwen3_Lora_sft_medicalQA/blob/2906f6a68fb2e4f5048f08433853f1fbc6f49b60/config/sft_config.yaml)
identifies the checkpoint; its README suggests Qwen3-8B as a larger alternative. The official
[Qwen/Qwen3-1.7B checkpoint](https://huggingface.co/Qwen/Qwen3-1.7B) is a text model, so music-to-chart work will need
an audio feature pipeline. The source review and adaptation notes are in
[docs/tutorial-reference.md](docs/tutorial-reference.md).

When ready for a separate model download, use the verified repository and pinned revision:

```bash
export MODEL_ID='Qwen/Qwen3-1.7B'
export MODEL_REVISION='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
uv run --no-sync hf download "$MODEL_ID" --revision "$MODEL_REVISION"
```

Hugging Face uses its normal user cache; set `HF_HOME` to a larger disk if needed. Model weights and data are separate
from the dependency lock files. Local `data/`, `models/`, `checkpoints/`, and `outputs/` directories are ignored by Git.

For dependency maintenance, update `pyproject.toml`, run `uv lock` inside the shell, then rerun setup and its check. Use
`nix flake update` to deliberately refresh native dependencies. A Nix revision change can require recreating `.venv` if
its Python interpreter has been garbage-collected.

The broader model/data/integration assessment is in [docs/feasibility.md](docs/feasibility.md). Backend selection
follows
[uv's PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies).

**Training draft:** [train_itg.py](train_itg.py) prepares paired music/simfiles, trains a Qwen3 LoRA adapter, and
generates draft `.sm` files. Read [docs/training.md](docs/training.md) for the representation, supported charts,
commands, and limits. Start with an offline CPU integration check using synthetic songs and a tiny random model:

```bash
uv run --no-sync python train_itg.py smoke-test
```

It writes fixtures and a test adapter to `outputs/smoke-test/`; choose a new `--output` directory to repeat it.
