"""Prepare music/chart pairs, train Qwen LoRA, and generate a draft simfile."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert a simfile pack and its music into song-split examples")
    prepare.add_argument("--songs", required=True)
    prepare.add_argument("--output", default="data/processed")
    prepare.add_argument("--style", default="mixed", help="Corpus style label, e.g. stamina or technical")
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train", help="Fine-tune the actual model; normally run on a GPU machine")
    train.add_argument("--config", default="configs/train.yaml")
    smoke = commands.add_parser("smoke-test", help="Offline CPU test using synthetic music and a tiny random model")
    smoke.add_argument("--output", default="outputs/smoke-test")
    generate = commands.add_parser("generate", help="Generate from music with a human-verified BPM/offset map")
    generate.add_argument("--adapter", required=True)
    generate.add_argument("--music", required=True)
    generate.add_argument("--bpms", required=True, help="StepMania beat=BPM pairs, e.g. '0=150,64=160'")
    generate.add_argument("--offset", type=float, required=True, help="StepMania OFFSET in seconds")
    generate.add_argument("--meter", type=int, required=True)
    generate.add_argument("--style", default="mixed")
    generate.add_argument("--output", required=True, help="New song directory for chart.sm and a copy of the music")
    generate.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    generate.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.command == "prepare":
        from itg_ai.data import prepare
        prepare(args.songs, args.output, args.style, args.validation_fraction, args.seed)
    elif args.command == "train":
        from itg_ai.training import train
        train(args.config)
    elif args.command == "smoke-test":
        from itg_ai.smoke import smoke_test
        smoke_test(args.output)
    elif args.command == "generate":
        from itg_ai.data import Tempo
        from itg_ai.generation import generate
        tempo = Tempo([[float(x) for x in pair.split("=")] for pair in args.bpms.split(",")], args.offset)
        generate(args.adapter, args.music, tempo, args.output, args.meter, args.style, args.device, args.max_new_tokens)


if __name__ == "__main__":
    main()
