"""Standalone YourMT3+ runtime bridge used by Resonance.

This module keeps the official YourMT3 implementation outside Resonance.
Set YOURMT3_ROOT to a checkout of mimbres/YourMT3. The official model loader,
tokenizer and decoder are then used unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import platform
import sys
from collections import Counter
from pathlib import Path

CHECKPOINT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"


def progress(stage: str, percent: int) -> None:
    print(
        f'{{"stage":"{stage}","percent":{max(0, min(100, percent))}}}',
        file=sys.stderr,
        flush=True,
    )


def _runtime_root() -> Path:
    value = os.environ.get("YOURMT3_ROOT")
    if not value:
        raise RuntimeError(
            "YOURMT3_ROOT is not set. Clone the official YourMT3 repository "
            "and point YOURMT3_ROOT at that checkout."
        )
    root = Path(value).expanduser().resolve()
    if not (root / "amt" / "src").is_dir():
        raise RuntimeError(f"YOURMT3_ROOT does not contain amt/src: {root}")
    return root


def _add_runtime_to_path(root: Path) -> None:
    for path in (root / "amt" / "src", root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _model_args(checkpoint: str, precision: str) -> list[str]:
    return [
        checkpoint,
        "-p", "2024",
        "-tk", "mc13_full_plus_256",
        "-dec", "multi-t5",
        "-nl", "26",
        "-enc", "perceiver-tf",
        "-sqr", "1",
        "-ff", "moe",
        "-wf", "4",
        "-nmoe", "8",
        "-kmoe", "2",
        "-act", "silu",
        "-epe", "rope",
        "-rp", "1",
        "-ac", "spec",
        "-hop", "300",
        "-atc", "1",
        "-pr", precision,
    ]


def transcribe(audio_path: Path, output_path: Path, checkpoint: str = CHECKPOINT) -> None:
    import librosa
    import numpy as np
    import torch

    root = _runtime_root()
    _add_runtime_to_path(root)

    from model_helper import load_model_checkpoint
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes
    from utils.utils import write_model_output_as_midi

    precision = os.environ.get("YOURMT3_PRECISION", "32")
    if os.environ.get("YOURMT3_DEVICE"):
        device = os.environ["YOURMT3_DEVICE"]
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    progress("loading_model", 2)
    old_cwd = os.getcwd()
    os.chdir(root)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = load_model_checkpoint(
                args=_model_args(checkpoint, precision),
                device=device,
            )
    finally:
        os.chdir(old_cwd)

    if device == "cpu" and platform.machine() in ("arm64", "aarch64"):
        torch.backends.quantized.engine = "qnnpack"

    if device == "cpu":
        model = torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

    progress("preparing_audio", 10)
    sample_rate = model.audio_cfg["sample_rate"]
    input_frames = model.audio_cfg["input_frames"]
    audio_np, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
    audio_segments = slice_padded_array(audio_tensor, input_frames, input_frames)
    audio_segments = torch.from_numpy(
        np.asarray(audio_segments, dtype=np.float32)
    ).unsqueeze(1)

    progress("transcribing", 15)
    batch_size = int(os.environ.get("YOURMT3_BATCH_SIZE", "8"))
    n_items = audio_segments.shape[0]
    if n_items == 0:
        raise RuntimeError("The audio file produced no model input segments.")

    predictions = []
    with torch.inference_mode():
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            batch = audio_segments[start:end].to(model.device)
            with contextlib.redirect_stdout(sys.stderr):
                predicted = model.inference(batch).detach().cpu().numpy()
            predictions.append(predicted)
            progress("transcribing", 15 + int(55 * end / n_items))

    progress("extracting_notes", 70)
    num_channels = model.task_manager.num_decoding_channels
    segment_starts = [
        input_frames * index / sample_rate
        for index in range(n_items)
    ]

    predicted_parts = []
    errors = Counter()
    for channel in range(num_channels):
        channel_batches = [batch[:, channel, :] for batch in predictions]
        zipped, _events, channel_errors = model.task_manager.detokenize_list_batches(
            channel_batches,
            segment_starts,
            return_events=True,
        )
        notes, note_errors = merge_zipped_note_events_and_ties_to_notes(zipped)
        predicted_parts.append(notes)
        errors.update(channel_errors)
        errors.update(note_errors)

    predicted_notes = mix_notes(predicted_parts)

    progress("writing_midi", 85)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    track_name = output_path.stem
    with contextlib.redirect_stdout(sys.stderr):
        write_model_output_as_midi(
            predicted_notes,
            str(output_path.parent),
            track_name,
            model.midi_output_inverse_vocab,
        )

    generated = output_path.parent / "model_output" / f"{track_name}.mid"
    if not generated.is_file():
        raise RuntimeError("YourMT3 produced no MIDI output.")
    generated.replace(output_path)
    progress("done", 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YourMT3+ for Resonance.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    audio = args.audio.resolve()
    if not audio.is_file():
        parser.error(f"Audio file does not exist: {audio}")

    try:
        transcribe(audio, args.output.resolve(), args.checkpoint)
    except Exception:
        import traceback

        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
