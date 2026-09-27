"""Command-line interface for Resonance."""

from __future__ import annotations

import argparse
from pathlib import Path

from .arrangement.baseline import arrange
from .pipeline import run
from .transcription.basic_pitch import transcribe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resonance",
        description="Local AI music transcription and target-instrument arrangement.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio to MIDI.")
    transcribe_parser.add_argument("audio", type=Path)
    transcribe_parser.add_argument("--output", "-o", type=Path, required=True)

    arrange_parser = subparsers.add_parser(
        "arrange",
        help="Arrange existing MIDI for a target instrument.",
    )
    arrange_parser.add_argument("midi", type=Path)
    arrange_parser.add_argument("--instrument", "-i", required=True)
    arrange_parser.add_argument("--parts", "-p", type=int, default=1)
    arrange_parser.add_argument("--output", "-o", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Transcribe audio and arrange it for target instruments.",
    )
    run_parser.add_argument("audio", type=Path)
    run_parser.add_argument("--instrument", "-i", required=True)
    run_parser.add_argument("--parts", "-p", type=int, default=1)
    run_parser.add_argument("--output", "-o", type=Path, required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "transcribe":
        result = transcribe(args.audio, args.output)
    elif args.command == "arrange":
        result = arrange(args.midi, args.output, args.instrument, args.parts)
    elif args.command == "run":
        result = run(args.audio, args.output, args.instrument, args.parts)
    else:
        parser.error("Unknown command")
        return

    print(f"Created: {result}")
