"""High-level Resonance pipeline."""

from __future__ import annotations

from pathlib import Path

from .arrangement.baseline import arrange
from .transcription.basic_pitch import transcribe
from .transcription.yourmt3 import YourMT3Transcriber


def run(
    audio_path: Path,
    output_path: Path,
    instrument: str,
    parts: int,
    backend: str = "basic-pitch",
    yourmt3_model: str = "yourmt3",
    yourmt3_device: str = "auto",
) -> Path:
    """Transcribe audio and arrange the resulting MIDI."""
    temporary_midi = output_path.parent / f".{output_path.stem}.source.mid"
    temporary_midi.parent.mkdir(parents=True, exist_ok=True)

    if backend == "yourmt3":
        transcription = YourMT3Transcriber(
            model=yourmt3_model,
            device=yourmt3_device,
        ).transcribe(audio_path)
        from .midi import transcription_to_midi

        transcription_to_midi(transcription, temporary_midi)
    elif backend == "basic-pitch":
        transcribe(audio_path, temporary_midi)
    else:
        raise ValueError(f"Unknown transcription backend: {backend}")

    try:
        return arrange(
            source_path=temporary_midi,
            output_path=output_path,
            instrument_name=instrument,
            parts=parts,
        )
    finally:
        temporary_midi.unlink(missing_ok=True)
