"""Command-line interface for Resonance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_midi
from .arrangement.baseline import arrange
from .arrangement.transpose import auto_transpose_midi
from .pipeline import run
from .transcription.basic_pitch import transcribe
from .transcription.yourmt3 import DEFAULT_MODEL_KEY, YourMT3Transcriber


def build_parser():
    parser = argparse.ArgumentParser(
        prog="resonance",
        description="Local AI music transcription and arrangement.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("transcribe", help="Transcribe audio to MIDI.")
    p.add_argument("audio", type=Path)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument(
        "--backend",
        choices=["basic-pitch", "yourmt3"],
        default="basic-pitch",
        help="Transcription backend.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_KEY,
        help="YourMT3 model key (default: yptf_moe_nops).",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="YourMT3 inference device.",
    )
    p.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable YourMT3 adaptive transcription.",
    )

    p = subparsers.add_parser("analyze", help="Analyze musical structure in MIDI.")
    p.add_argument("midi", type=Path)
    p.add_argument("--output", "-o", type=Path)

    p = subparsers.add_parser("arrange", help="Arrange MIDI for a target instrument.")
    p.add_argument("midi", type=Path)
    p.add_argument("--instrument", "-i", required=True)
    p.add_argument("--parts", "-p", type=int, default=1)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument(
        "--auto-transpose",
        action="store_true",
        help="Choose the chromatic shift with the lowest pitch/range cost.",
    )

    p = subparsers.add_parser("run", help="Transcribe audio and arrange it.")
    p.add_argument("audio", type=Path)
    p.add_argument("--instrument", "-i", required=True)
    p.add_argument("--parts", "-p", type=int, default=1)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument(
        "--backend",
        choices=["basic-pitch", "yourmt3"],
        default="basic-pitch",
        help="Transcription backend.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_KEY, help="YourMT3 model key.")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="YourMT3 inference device.",
    )
    p.add_argument("--adaptive", action="store_true", help="Enable YourMT3 adaptive transcription.")
    p.add_argument("--auto-transpose", action="store_true")

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "transcribe":
        if args.backend == "yourmt3":
            transcription = YourMT3Transcriber(
                model=args.model,
                device=args.device,
                adaptive=args.adaptive,
            ).transcribe(args.audio)
            from .midi import transcription_to_midi

            transcription_to_midi(transcription, args.output)
            result = args.output
        else:
            result = transcribe(args.audio, args.output)

    elif args.command == "analyze":
        text = json.dumps(analyze_midi(args.midi), indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
            print(f"Created: {args.output}")
            return
        print(text)
        return

    elif args.command == "arrange":
        result = arrange(args.midi, args.output, args.instrument, args.parts)
        if args.auto_transpose:
            result_info = auto_transpose_midi(result, result, args.instrument)
            print(
                f"Auto-transposed: {result_info.key} "
                f"({result_info.semitones:+d} semitones), "
                f"{result_info.black_key_notes}/{result_info.total_notes} black-key notes"
            )

    elif args.command == "run":
        result = run(
            args.audio,
            args.output,
            args.instrument,
            args.parts,
            backend=args.backend,
            yourmt3_model=args.model,
            yourmt3_device=args.device,
        )
        if args.auto_transpose:
            result_info = auto_transpose_midi(result, result, args.instrument)
            print(
                f"Auto-transposed: {result_info.key} "
                f"({result_info.semitones:+d} semitones), "
                f"{result_info.black_key_notes}/{result_info.total_notes} black-key notes"
            )
    else:
        raise RuntimeError("Unknown command")

    print(f"Created: {result}")


if __name__ == "__main__":
    main()
