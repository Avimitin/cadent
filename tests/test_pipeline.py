import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import simfile
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast

from itg_ai.data import (
    SCHEMA, SPAN, Audio, Tempo, parse_notes, prepare,
    split_examples, validate_events, write_sm,
)
from itg_ai.smoke import make_songs
from itg_ai.training import encode


class PipelineTests(unittest.TestCase):
    def test_offset_and_bpm_changes(self):
        tempo = Tempo([[0, 120], [4, 60]], 0.25)
        np.testing.assert_allclose(tempo.time_at([0, 2, 4, 5]), [-0.25, 0.75, 1.75, 2.75])
        for beat in [0, 2, 4, 5, 20]:
            self.assertAlmostEqual(tempo.beat_at(tempo.time_at(beat)), beat)

    def test_ssc_chart_timing(self):
        song = simfile.loads(
            "#VERSION:0.83;#BPMS:0=120;#OFFSET:1;#STOPS:4=1;"
            "#NOTEDATA:;#STEPSTYPE:dance-single;#BPMS:0=180;#OFFSET:-0.5;"
            "#NOTES:1000\n0000\n0000\n0000;"
        )
        # Chart timing replaces song timing, including the global stop.
        tempo = Tempo.from_chart(song, song.charts[0])
        self.assertAlmostEqual(float(tempo.time_at(3)), 1.5)

    def test_unsupported_timing(self):
        song = simfile.loads("#BPMS:0=120;#STOPS:4=1;#NOTES:dance-single::Hard:8:0,0,0,0,0:1000;")
        with self.assertRaisesRegex(ValueError, "Stops"):
            Tempo.from_chart(song, song.charts[0])

    def test_hold_crosses_window_and_roundtrip(self):
        events = [[336, "2000"], [384, "0100"], [432, "3000"], [480, "00M1"]]
        state = validate_events(events[:1], SPAN)
        self.assertEqual(state, "2000")
        self.assertEqual(validate_events([[0, "0100"], [48, "3000"], [96, "00M1"]], SPAN, state, True), "0000")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sm"
            write_sm(path, events, Tempo([[0, 150]], -0.5), "music.ogg", 9)
            self.assertEqual(parse_notes(simfile.open(str(path)).charts[0].notes), events)

    def test_reject_invalid_chart_predictions(self):
        invalid = [ [[-1, "1000"]], [[384, "1000"]], [[0, "3000"]], [[0, "2000"]],
                    [[0, "1000"], [0, "0100"]], [[True, "1000"]], [[0, "F000"]],
                    [[0, "2000"], [48, "1000"], [96, "3000"]] ]
        for events in invalid:
            with self.subTest(events=events), self.assertRaises(ValueError):
                validate_events(events, SPAN, final=True)

    def test_triplets_preserved_and_offgrid_rejected(self):
        rows = ["0000"] * 12
        rows[1] = "1000"
        self.assertEqual(parse_notes("\n".join(rows)), [[16, "1000"]])
        with self.assertRaisesRegex(ValueError, "refusing to round"):
            parse_notes("0000\n1000\n0000\n0000\n0000")

    def test_audio_features_align_and_respond_to_music(self):
        times = np.arange(0, 5, 0.01)
        channels = np.zeros((4, len(times)), dtype=np.uint8)
        channels[:, 125] = 15
        audio = Audio("test", 5, times, channels)
        # Beat 2 lands at 1.25 seconds with this negative StepMania offset.
        bins = audio.bins(Tempo([[0, 120]], -0.25), 0)
        self.assertEqual(bins[24], "FFFF")
        self.assertEqual(bins[0], "0000")
        self.assertNotEqual(bins, Audio("silent", 5, times, channels * 0).bins(Tempo([[0, 120]], -0.25), 0))

    def test_prepare_splits_all_charts_of_same_audio_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_songs(root / "songs")
            # A copied file must be deduplicated, not become an independent validation song.
            original = root / "songs/song-0/chart.sm"
            original.with_name("copy.sm").write_text(original.read_text())
            train, validation = prepare(root / "songs", root / "dataset", "synthetic", 0.5)
            self.assertTrue(train and validation)
            self.assertFalse({r["song_id"] for r in train} & {r["song_id"] for r in validation})
            self.assertTrue(any(r["conditioning"]["initial_holds"] == "2000" for r in train + validation))
            report = json.loads((root / "dataset/report.json").read_text())
            self.assertEqual(report["charts"], 2)
            self.assertTrue(any("Duplicate" in skip["reason"] for skip in report["skipped"]))
        with self.assertRaisesRegex(ValueError, "two distinct"):
            split_examples([{"song_id": "same"}, {"song_id": "same"}], 0.5, 42)

    def test_chart_only_labels_and_padding(self):
        backend = Tokenizer(models.WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2}, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
        tokenizer.chat_template = "{% for m in messages %}{{m['content']}} {% endfor %}assistant: "
        context = {"schema": SCHEMA, "initial_holds": "0000", "final_window": True}
        row = {"conditioning": context, "events": [[0, "1000"]]}
        encoded = encode(tokenizer, row, 4096)
        first_target = next(i for i, label in enumerate(encoded["labels"]) if label != -100)
        self.assertGreater(first_target, 0)
        self.assertEqual(encoded["labels"][-1], tokenizer.eos_token_id)
        self.assertEqual(encoded["labels"][first_target:], encoded["input_ids"][first_target:])
        shorter = encode(tokenizer, {**row, "events": []}, 4096)
        batch = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)([encoded, shorter])
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        with self.assertRaisesRegex(ValueError, "No silent truncation"):
            encode(tokenizer, row, 4)


if __name__ == "__main__":
    unittest.main()
