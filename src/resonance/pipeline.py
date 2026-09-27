"""High-level Resonance pipeline."""

from __future__ import annotations

from pathlib import Path

from .arrangement.baseline import arrange
from .transcription.basic_pitch import transcribe


def run(audio_path: Path, output_path: Path, instrument: str, parts: int) -> Path:
    """Transcribe audio and arrange the resulting MIDI for target instruments."""
    temporary_midi = output_path.parent / f".{output_path.stem}.source.mid"
    temporary_midi.parent.mkdir(parents=True, exist_ok=True)

    transcribe(audio_path, temporary_midi)
    try:
        return arrange(
            source_path=temporary_midi,
            output_path=output_path,
            instrument_name=instrument,
            parts=parts,
        )
    finally:
        temporary_midi.unlink(missing_ok=True)
