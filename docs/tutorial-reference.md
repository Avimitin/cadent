Tutorial reference: Qwen3 LoRA environment

Reviewed September 16, 2026. Source:
[junqiangchen/qwen3_Lora_sft_medicalQA](https://github.com/junqiangchen/qwen3_Lora_sft_medicalQA/tree/2906f6a68fb2e4f5048f08433853f1fbc6f49b60),
commit `2906f6a68fb2e4f5048f08433853f1fbc6f49b60`.

The
[training configuration](https://github.com/junqiangchen/qwen3_Lora_sft_medicalQA/blob/2906f6a68fb2e4f5048f08433853f1fbc6f49b60/config/sft_config.yaml)
selects **Qwen3-1.7B**, resolving the earlier “Qwen3.7 7B” naming ambiguity. The
[official model card](https://huggingface.co/Qwen/Qwen3-1.7B) describes a 1.7B text-generation model and requires
Transformers 4.51.0 or later. This project's Transformers 4.57.6 baseline supports its native model class.

The recipe uses PEFT LoRA and Transformers Trainer. Its
[adapter settings](https://github.com/junqiangchen/qwen3_Lora_sft_medicalQA/blob/2906f6a68fb2e4f5048f08433853f1fbc6f49b60/config/lora_config.yaml)
are rank 8, alpha 16, dropout 0.05, and the `q_proj`, `v_proj`, `k_proj`, and `o_proj` attention projections. There is
no quantized base-model loading. Our CPU environment check uses these settings on a tiny random Qwen3.

The
[requirements file](https://github.com/junqiangchen/qwen3_Lora_sft_medicalQA/blob/2906f6a68fb2e4f5048f08433853f1fbc6f49b60/requirements.txt)
lists PEFT, Datasets, Transformers, and Pydantic without pins. Source imports and Trainer configuration additionally
require PyTorch, Accelerate, PyYAML, and TensorBoard. The local `pyproject.toml` and `uv.lock` provide a pinned,
compatible baseline; they do not reproduce an unpublished upstream lock file.

Validation passed on this CPU workstation: the regular environment check covered LoRA weight updates, TensorBoard
logging, and audio features. A separate integration check invoked the original `LORASFTtrainModel` class from the pinned
source with a tiny random local Qwen3, a local tokenizer, and two synthetic examples. One optimizer step produced a
finite loss, changed adapter weights, saved `adapter_model.safetensors`, and wrote readable TensorBoard events. This
verifies the training path, not pretrained-model quality or the supplied inference class. CUDA remains untested. The
upstream `Trainer(tokenizer=...)` argument emits a deprecation warning; use `processing_class` when adapting it,
particularly before upgrading Transformers to version 5.

Before adapting the training code:

- Replace its Windows paths with local model, dataset, and output paths.
- Forward intended options explicitly to `TrainingArguments`. The
  [trainer](https://github.com/junqiangchen/qwen3_Lora_sft_medicalQA/blob/2906f6a68fb2e4f5048f08433853f1fbc6f49b60/model/__init__.py)
  currently passes only output directory, batch size, epoch count, learning rate, logging steps, save strategy, and
  TensorBoard reporting. Settings such as gradient accumulation, checkpointing, precision, scheduling, and evaluation
  remain at library defaults. Supplying an evaluation dataset alone does not enable evaluation.
- Choose and wire a sequence length. The
  [dataset loader](https://github.com/junqiangchen/qwen3_Lora_sft_medicalQA/blob/2906f6a68fb2e4f5048f08433853f1fbc6f49b60/model/dataset_loader.py)
  defaults to 1,024 tokens; the configured 2,048-token limit is unused. It concatenates input and output text and trains
  on both. For ITG, define consistent training/inference formatting and an explicit target loss mask.
- Save the tokenizer alongside the adapter and retain the base model ID/revision. The final explicit save call writes
  the PEFT model, while the supplied inference class expects a tokenizer in its model directory.

For ITG, retain the LoRA approach but replace the medical dataset with aligned rhythm/feature inputs and note-event
targets. A text model cannot consume a music file directly. First prepare beat-aligned audio features and a compact
event schema; then evaluate phrase-level generation, valid holds, and pad playability on held-out songs. The
[training draft](training.md) now implements paired data preparation, LoRA training, and standalone `.sm` generation.
ArrowVortex integration remains a later step.

Setup downloads dependencies only. The optional model download command in the README pins `Qwen/Qwen3-1.7B` to revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Model weights, training data, and real fine-tuning remain separate steps.
