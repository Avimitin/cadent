"""Shared training/inference representation. No learned audio encoder is implied."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import simfile
from simfile.sm import SMChart, SMSimfile
from simfile.timing import TimingData

SCHEMA = "itg-audio-events-v1"
TICKS = 48
WINDOW_BEATS = 8
SPAN = TICKS * WINDOW_BEATS
FEATURES_PER_BEAT = 12
SAMPLE_RATE = 22050
HOP = 220
SYSTEM = (
    "Map music features to a four-panel dance-single ITG chart. Return ONLY a JSON array of "
    "[tick,row] events, sorted by unique tick within this window. There are 48 ticks per beat; "
    "omit empty rows. Row columns are left,down,up,right. Symbols: 0 empty,1 tap,2 hold start,"
    "3 hold/roll end,4 roll start,M mine. Preserve initial holds; no other note may occupy a "
    "held lane until its end. Close all holds in the final window. previous contains prior "
    "events at negative relative ticks. audio contains 12 bins per beat; each four-digit "
    "hex string encodes onset strength,loudness,bass energy,treble energy (0-F). Follow the "
    "requested difficulty and style, using musical accents, rests and coherent panel patterns."
)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass
class Tempo:
    bpms: list
    offset: float

    def __post_init__(self):
        self.beats = np.array([p[0] for p in self.bpms], dtype=float)
        self.values = np.array([p[1] for p in self.bpms], dtype=float)
        if (not len(self.beats) or not np.isfinite(self.beats).all()
                or not np.isfinite(self.values).all() or not math.isfinite(self.offset)
                or self.beats[0] != 0 or np.any(np.diff(self.beats) <= 0)
                or np.any(self.values <= 0)):
            raise ValueError("Require finite, positive BPMs, unique sorted beats starting at beat 0")
        self.seconds = np.r_[0, np.cumsum(np.diff(self.beats) * 60 / self.values[:-1])] - self.offset

    @classmethod
    def from_chart(cls, song, chart):
        timing = TimingData(song, chart)
        if timing.stops or timing.delays or timing.warps:
            raise ValueError("Stops, delays and warps are not supported in schema v1")
        # Reject these even when inherited/overridden ambiguously; report, never flatten.
        if any(obj.get("FAKES", "").strip() for obj in (song, chart)):
            raise ValueError("Fake regions are not supported in schema v1")
        for obj in (song, chart):
            signatures = obj.get("TIMESIGNATURES", "").strip()
            if signatures and any(p.strip().split("=")[1:] != ["4", "4"] for p in signatures.split(",")):
                raise ValueError("Only 4/4 time signatures are supported in schema v1")
        return cls([[float(b), float(v)] for b, v in timing.bpms], float(timing.offset))

    def time_at(self, beat):
        beat = np.asarray(beat)
        index = np.clip(np.searchsorted(self.beats, beat, side="right") - 1, 0, len(self.beats) - 1)
        return self.seconds[index] + (beat - self.beats[index]) * 60 / self.values[index]

    def beat_at(self, seconds):
        index = np.clip(np.searchsorted(self.seconds, seconds, side="right") - 1, 0, len(self.beats) - 1)
        return float(self.beats[index] + (seconds - self.seconds[index]) * self.values[index] / 60)

    def window_bpms(self, start):
        index = max(0, np.searchsorted(self.beats, start, side="right") - 1)
        return [[0, float(self.values[index])]] + [
            [float(b - start), float(v)] for b, v in self.bpms if start < b < start + WINDOW_BEATS
        ]


def validate_events(events, span, active="0000", final=False):
    if not isinstance(events, list) or len(active) != 4 or set(active) - set("024"):
        raise ValueError("Invalid event list or initial hold state")
    state, previous = list(active), -1
    for event in events:
        if not isinstance(event, list) or len(event) != 2:
            raise ValueError("Each event must be [integer tick, four-character row]")
        tick, row = event
        if type(tick) is not int or not previous < tick < span:
            raise ValueError("Event ticks must be sorted, unique and inside the window")
        if not isinstance(row, str) or len(row) != 4 or set(row) - set("01234M") or row == "0000":
            raise ValueError(f"Unsupported note row: {row!r}")
        for lane, symbol in enumerate(row):
            if symbol == "3":
                if state[lane] == "0":
                    raise ValueError("Hold tail without a preceding head")
                state[lane] = "0"
            elif symbol != "0":
                if state[lane] != "0":
                    raise ValueError("Note overlaps an active hold")
                if symbol in "24":
                    state[lane] = symbol
        previous = tick
    if final and any(s != "0" for s in state):
        raise ValueError("Unclosed hold/roll at chart end")
    return "".join(state)


def parse_notes(notes):
    events = []
    for measure, text in enumerate(re.sub(r"//[^\n]*", "", notes).split(",")):
        rows = text.split()
        if not rows:
            raise ValueError("Empty measure")
        for index, row in enumerate(rows):
            if len(row) != 4 or set(row) - set("01234M"):
                raise ValueError(f"Unsupported four-panel note row: {row!r}")
            if row != "0000":
                tick, remainder = divmod(index * 4 * TICKS, len(rows))
                if remainder:
                    raise ValueError("Note is off the 1/48-beat grid; refusing to round it")
                events.append([measure * 4 * TICKS + tick, row])
    validate_events(events, max(1, (measure + 1) * 4 * TICKS), final=True)
    return events


@dataclass
class Audio:
    digest: str
    duration: float
    times: np.ndarray
    channels: np.ndarray

    def bins(self, tempo, start):
        beats = start + np.arange(WINDOW_BEATS * FEATURES_PER_BEAT) / FEATURES_PER_BEAT
        # Center bins on the beat grid, matching the centered STFT timestamps.
        edges = tempo.time_at(np.r_[beats - 0.5 / FEATURES_PER_BEAT, beats[-1] + 0.5 / FEATURES_PER_BEAT])
        result = []
        for left, right in zip(edges[:-1], edges[1:]):
            lo, hi = np.searchsorted(self.times, [left, right])
            values = self.channels[:, lo:hi].max(axis=1) if hi > lo else np.zeros(4)
            result.append("".join(f"{int(v):X}" for v in values))
        return result


@lru_cache(maxsize=2)
def read_audio(path):
    # Hash decoded mono PCM so exact audio copies share a split, independent of path/tags.
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < 1024 or not np.isfinite(y).all():
        raise ValueError("Audio must contain at least 1024 finite samples")
    digest = hashlib.sha256(y.astype("<f4").tobytes()).hexdigest()
    magnitude = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP, center=True))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=1024)
    channels = np.stack([
        np.maximum(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]), 0).mean(axis=0),
        np.sqrt(np.mean(magnitude ** 2, axis=0)),
        magnitude[(frequencies >= 30) & (frequencies < 250)].mean(axis=0),
        magnitude[frequencies >= 2000].mean(axis=0),
    ])
    scale = np.maximum(np.percentile(channels, 95, axis=1, keepdims=True), 1e-8)
    quantized = np.rint(np.clip(channels / scale, 0, 1) * 15).astype(np.uint8)
    return Audio(digest, len(y) / sr, np.arange(magnitude.shape[1]) * HOP / sr, quantized)


def conditioning(audio, tempo, start_tick, history, active, meter, style, final):
    start = start_tick / TICKS
    return {
        "schema": SCHEMA, "start_beat": start, "span_ticks": SPAN,
        "meter": meter, "style": style, "bpms": tempo.window_bpms(start),
        "initial_holds": active, "final_window": final,
        "previous": [[t - start_tick, row] for t, row in history if start_tick - 4 * TICKS <= t < start_tick],
        "audio": audio.bins(tempo, start),
    }


def window_count(audio, tempo):
    return max(1, math.ceil(tempo.beat_at(audio.duration) / WINDOW_BEATS))


def split_examples(examples, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be strictly between 0 and 1")
    songs = sorted({row["song_id"] for row in examples}, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    if len(songs) < 2:
        raise ValueError("Need at least two distinct decoded songs for separate train/validation splits")
    count = min(len(songs) - 1, max(1, round(len(songs) * validation_fraction)))
    validation = set(songs[:count])
    return ([r for r in examples if r["song_id"] not in validation],
            [r for r in examples if r["song_id"] in validation])


def prepare(root, output, style="mixed", validation_fraction=0.1, seed=42):
    root, output = Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".sm", ".ssc"))
    if not files:
        raise ValueError(f"No .sm/.ssc files found in {root}")
    examples, skipped, seen = [], [], set()
    for path in files:
        try:
            song = simfile.open(str(path))
        except Exception as error:
            skipped.append({"file": str(path), "reason": str(error)})
            continue
        for index, chart in enumerate(song.charts):
            try:
                if chart.stepstype != "dance-single":
                    raise ValueError("Only dance-single charts are supported")
                tempo = Tempo.from_chart(song, chart)
                music = chart.get("MUSIC") or song.music
                if not music:
                    raise ValueError("Missing MUSIC tag")
                audio = read_audio(str((path.parent / music.replace("\\", "/")).resolve()))
                events = parse_notes(chart.notes or "")
                if not events:
                    raise ValueError("Chart contains no events")
                event_times = tempo.time_at(np.array([t for t, _ in events]) / TICKS)
                if event_times.min() < -0.05 or event_times.max() >= audio.duration:
                    raise ValueError("Notes lie outside the music; inspect offset/audio matching")
                meter = int(chart.meter)
                if meter < 1:
                    raise ValueError("Meter must be positive")
                identity = compact([audio.digest, tempo.bpms, tempo.offset, events, meter, style])
                fingerprint = hashlib.sha256(identity.encode()).hexdigest()
                if fingerprint in seen:
                    raise ValueError("Duplicate chart/audio pair")
                count, active, chart_examples = window_count(audio, tempo), "0000", []
                for window in range(count):
                    start = window * SPAN
                    target = [[t - start, row] for t, row in events if start <= t < start + SPAN]
                    context = conditioning(audio, tempo, start, events, active, meter, style, window == count - 1)
                    active = validate_events(target, SPAN, active, final=window == count - 1)
                    chart_examples.append({"song_id": audio.digest, "source": str(path.relative_to(root)),
                                           "chart_index": index, "conditioning": context, "events": target})
                examples.extend(chart_examples)
                seen.add(fingerprint)
            except Exception as error:
                skipped.append({"file": str(path), "chart": index, "reason": f"{type(error).__name__}: {error}"})
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "files": len(files), "charts": len(seen), "examples": len(examples),
              "style": style, "seed": seed, "validation_fraction": validation_fraction, "skipped": skipped}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    train, validation = split_examples(examples, validation_fraction, seed)
    for name, rows in (("train", train), ("validation", validation)):
        (output / f"{name}.jsonl").write_text("".join(compact(row) + "\n" for row in rows), encoding="utf-8")
    print(f"Prepared {len(train)} train / {len(validation)} validation windows; {len(skipped)} skips. See {output}/report.json")
    return train, validation


def write_sm(path, events, tempo, music, meter, title="AI draft"):
    validate_events(events, max([t for t, _ in events], default=0) + 1, final=True)
    rows = ["0000"] * (max(1, max([t for t, _ in events], default=0) // (4 * TICKS) + 1) * 4 * TICKS)
    for tick, row in events:
        rows[tick] = row
    song, chart = SMSimfile.blank(), SMChart.blank()
    song.title, song.music, song.offset = title, music, str(tempo.offset)
    song.bpms = ",".join(f"{b:g}={v:g}" for b, v in tempo.bpms)
    chart.stepstype, chart.difficulty, chart.meter = "dance-single", "Edit", str(meter)
    chart.description = "Qwen LoRA draft"
    chart.notes = "\n,\n".join("\n".join(rows[i:i + 4 * TICKS]) for i in range(0, len(rows), 4 * TICKS))
    song.charts.append(chart)
    with Path(path).open("x", encoding="utf-8") as output:
        song.serialize(output)
