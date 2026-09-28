"""Shared representation for multi-instrument transcription backends.

A transcription backend should return independent musical parts instead of
forcing the result into a fixed vocals/bass/other stem layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class TranscribedNote:
    """One musical note emitted by an AMT model."""

    pitch: int
    start: float
    end: float
    velocity: int = 100

    def __post_init__(self) -> None:
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"MIDI pitch must be 0..127, got {self.pitch}")
        if self.start < 0:
            raise ValueError("Note start cannot be negative")
        if self.end <= self.start:
            raise ValueError("Note end must be after note start")
        if not 0 <= self.velocity <= 127:
            raise ValueError("MIDI velocity must be 0..127")


@dataclass(slots=True)
class SourcePart:
    """An independently detected instrument/part from a recording."""

    name: str
    instrument: str
    notes: list[TranscribedNote] = field(default_factory=list)
    confidence: float | None = None
    program: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Source part name cannot be empty")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Part confidence must be between 0 and 1")


@dataclass(slots=True)
class MultiInstrumentTranscription:
    """Model output before arrangement."""

    parts: list[SourcePart]
    duration: float
    tempo: float | None = None
    model: str = "unknown"

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ValueError("Duration cannot be negative")
        if self.tempo is not None and self.tempo <= 0:
            raise ValueError("Tempo must be positive")


class MultiInstrumentTranscriber(Protocol):
    """Interface implemented by browser/desktop multi-instrument backends."""

    def transcribe(self, audio_path: Path) -> MultiInstrumentTranscription:
        """Transcribe one complete recording into independent parts."""
        ...


def merge_short_parts(
    transcription: MultiInstrumentTranscription,
    minimum_notes: int = 3,
) -> MultiInstrumentTranscription:
    """Merge tiny model fragments into the nearest compatible instrument part.

    This is deliberately a post-processing operation. It does not invent an
    instrument or split a model output into artificial voices.
    """
    if minimum_notes < 1:
        raise ValueError("minimum_notes must be positive")

    substantial = [
        part for part in transcription.parts if len(part.notes) >= minimum_notes
    ]
    fragments = [
        part for part in transcription.parts if len(part.notes) < minimum_notes
    ]

    if not fragments or not substantial:
        return transcription

    for fragment in fragments:
        same_instrument = [
            part for part in substantial
            if part.instrument.lower() == fragment.instrument.lower()
        ]
        candidates = same_instrument or substantial

        fragment_center = _pitch_center(fragment.notes)
        target = min(
            candidates,
            key=lambda part: abs(_pitch_center(part.notes) - fragment_center),
        )
        target.notes.extend(fragment.notes)
        target.notes.sort(key=lambda note: (note.start, note.pitch))

    return MultiInstrumentTranscription(
        parts=substantial,
        duration=transcription.duration,
        tempo=transcription.tempo,
        model=transcription.model,
    )


def _pitch_center(notes: Sequence[TranscribedNote]) -> float:
    if not notes:
        return 0.0
    return sum(note.pitch for note in notes) / len(notes)
