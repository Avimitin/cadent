"""Generate sequential windows and export only structurally valid predictions."""

import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import SCHEMA, SPAN, conditioning, read_audio, validate_events, window_count, write_sm
from .training import prompt_ids


def generate(adapter, music, tempo, output, meter, style, device="cpu", max_new_tokens=2048):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    if meter < 1 or max_new_tokens < 1:
        raise ValueError("Meter and max_new_tokens must be positive")
    run = json.loads((Path(adapter) / "run.json").read_text())
    if run["schema"] != SCHEMA:
        raise ValueError("Adapter feature schema does not match this code")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        run["base_model"], revision=run["revision"], dtype=dtype, trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, adapter).to(device).eval()
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=False)
    audio = read_audio(str(Path(music).resolve()))
    count, events, active = window_count(audio, tempo), [], "0000"
    for window in range(count):
        context = conditioning(audio, tempo, window * SPAN, events, active, meter, style, window == count - 1)
        ids = prompt_ids(tokenizer, context)
        if len(ids) + max_new_tokens > model.config.max_position_embeddings:
            raise ValueError("Generation would exceed the model context limit")
        tokens = torch.tensor([ids], device=device)
        with torch.inference_mode():
            result = model.generate(
                input_ids=tokens, attention_mask=torch.ones_like(tokens),
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(result[0, len(ids):], skip_special_tokens=True).strip()
        try:
            predicted = json.loads(text)
            active = validate_events(predicted, SPAN, active, final=window == count - 1)
            for tick, _ in predicted:
                seconds = tempo.time_at((window * SPAN + tick) / 48)
                if seconds < -0.05 or seconds >= audio.duration:
                    raise ValueError("Predicted note lies outside the music")
        except (ValueError, TypeError) as error:
            raise ValueError(f"Window {window} prediction rejected: {error}. Raw output: {text[:500]}") from error
        events.extend([[window * SPAN + tick, row] for tick, row in predicted])
        print(f"Generated window {window + 1}/{count}")
    output.mkdir(parents=True)
    audio_name = "music" + Path(music).suffix.lower()
    shutil.copyfile(music, output / audio_name)
    write_sm(output / "chart.sm", events, tempo, audio_name, meter)
    (output / "events.json").write_text(json.dumps(events) + "\n")
    print(f"Draft saved to {output}/chart.sm. Review timing and footwork in ArrowVortex.")
