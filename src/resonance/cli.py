"""Command-line interface for Resonance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .analysis import analyze_midi
from .arrangement.baseline import arrange
from .pipeline import run
from .transcription.basic_pitch import transcribe

def build_parser():
    parser=argparse.ArgumentParser(prog="resonance",description="Local AI music transcription and arrangement.")
    subparsers=parser.add_subparsers(dest="command",required=True)
    p=subparsers.add_parser("transcribe",help="Transcribe audio to MIDI.")
    p.add_argument("audio",type=Path); p.add_argument("--output","-o",type=Path,required=True)
    p=subparsers.add_parser("analyze",help="Analyze musical structure in MIDI.")
    p.add_argument("midi",type=Path); p.add_argument("--output","-o",type=Path)
    p=subparsers.add_parser("arrange",help="Arrange MIDI for a target instrument.")
    p.add_argument("midi",type=Path); p.add_argument("--instrument","-i",required=True); p.add_argument("--parts","-p",type=int,default=1); p.add_argument("--output","-o",type=Path,required=True)
    p=subparsers.add_parser("run",help="Transcribe audio and arrange it.")
    p.add_argument("audio",type=Path); p.add_argument("--instrument","-i",required=True); p.add_argument("--parts","-p",type=int,default=1); p.add_argument("--output","-o",type=Path,required=True)
    return parser

def main():
    args=build_parser().parse_args()
    if args.command=="transcribe":
        result=transcribe(args.audio,args.output)
    elif args.command=="analyze":
        text=json.dumps(analyze_midi(args.midi),indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(text+"\n",encoding="utf-8")
            print(f"Created: {args.output}")
            return
        print(text); return
    elif args.command=="arrange":
        result=arrange(args.midi,args.output,args.instrument,args.parts)
    elif args.command=="run":
        result=run(args.audio,args.output,args.instrument,args.parts)
    else:
        raise RuntimeError("Unknown command")
    print(f"Created: {result}")

if __name__=="__main__":
    main()
