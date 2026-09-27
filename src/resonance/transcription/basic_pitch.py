"""Basic Pitch transcription backend.

Basic Pitch is the initial transcription backend for Resonance. The interface is
kept deliberately small so a future Resonance-trained transcription model can
replace it without changing the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


def validate_audio_path(path: Path) -> Path:
    """Validate a supported audio input path."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_AUDIO:
        supported = ", ".join(sorted(SUPPORTED_AUDIO))
        raise ValueError(f"Unsupported audio format '{path.suffix}'. Supported: {supported}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def transcribe(audio_path: Path, output_path: Path) -> Path:
    """Transcribe audio to MIDI using the installed Basic Pitch model."""
    validate_audio_path(audio_path)

    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise RuntimeError("Basic Pitch is not installed.") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, midi_data, _ = predict(audio_path)
    midi_data.write(str(output_path))
    return output_path
