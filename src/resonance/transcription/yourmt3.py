"""YourMT3+ multi-instrument transcription backend.

YourMT3+ is used as a true multi-track AMT backend: the model receives the
complete recording and returns independent MIDI tracks. Resonance preserves
those tracks as SourcePart objects instead of collapsing them into a fixed
vocals/bass/other representation.

The maintained MT3-Infer package supplies the PyTorch model and checkpoint
management. Resonance only owns the adapter and the normalized musical data
model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .multi_instrument import (
    MultiInstrumentTranscription,
    SourcePart,
    TranscribedNote,
    merge_short_parts,
)

DEFAULT_MODEL_KEY = "yptf_moe_nops"


class YourMT3Error(RuntimeError):
    """Raised when the YourMT3 runtime is unavailable or fails."""


class YourMT3Transcriber:
    """Transcribe one recording into independent model-produced instrument parts."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL_KEY,
        device: str = "auto",
        adaptive: bool = False,
        merge_fragments: bool = False,
        minimum_notes: int = 3,
    ) -> None:
        self.model = model
        self.device = device
        self.adaptive = adaptive
        self.merge_fragments = merge_fragments
        self.minimum_notes = minimum_notes

    def transcribe(self, audio_path: Path) -> MultiInstrumentTranscription:
        """Run YourMT3 and normalize every output MIDI track into a SourcePart."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        try:
            import numpy as np
            import soundfile as sf
        except ImportError as exc:
            raise YourMT3Error(
                "YourMT3 audio dependencies are missing. Install the optional "
                "dependency set with: python -m pip install -e ".[yourmt3]""
            ) from exc

        try:
            from mt3_infer import transcribe as mt3_transcribe
        except ImportError as exc:
            raise YourMT3Error(
                "MT3-Infer is not installed. Install the optional dependency set "
                "with: python -m pip install -e ".[yourmt3]""
            ) from exc

        try:
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1, dtype=np.float32)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size == 0:
                raise ValueError("audio file contains no samples")

            kwargs: dict[str, Any] = {
                "model": self.model,
                "sr": int(sample_rate),
            }
            if self.device != "auto":
                kwargs["device"] = self.device

            if self.adaptive:
                kwargs["adaptive"] = True

            midi = mt3_transcribe(audio, **kwargs)
        except Exception as exc:
            raise YourMT3Error(
                f"YourMT3 transcription failed with model {self.model!r}: {exc}"
            ) from exc

        transcription = self._midi_to_transcription(midi)
        if self.merge_fragments:
            transcription = merge_short_parts(
                transcription,
                minimum_notes=self.minimum_notes,
            )
        return transcription

    @staticmethod
    def _midi_to_transcription(midi) -> MultiInstrumentTranscription:
        """Convert an MT3-Infer mido.MidiFile into independent Resonance parts."""
        import mido

        parts: list[SourcePart] = []

        tempo = _midi_tempo(midi)
        for track_index, track in enumerate(midi.tracks):
            tick = 0
            active: dict[int, list[tuple[int, int]]] = {}
            notes: list[TranscribedNote] = []
            program: int | None = None
            name = f"Part {track_index + 1}"

            for message in track:
                tick += message.time

                if message.type == "track_name" and message.name.strip():
                    name = message.name.strip()
                elif message.type == "program_change":
                    program = message.program
                elif message.type == "note_on" and message.velocity > 0:
                    active.setdefault(message.note, []).append(
                        (tick, message.velocity)
                    )
                elif message.type == "note_off" or (
                    message.type == "note_on" and message.velocity == 0
                ):
                    starts = active.get(message.note)
                    if starts:
                        start_tick, velocity = starts.pop(0)
                        start = _tick_to_seconds(start_tick, midi, tempo)
                        end = _tick_to_seconds(tick, midi, tempo)
                        notes.append(
                            TranscribedNote(
                                pitch=message.note,
                                start=start,
                                end=max(start + 0.001, end),
                                velocity=velocity,
                            )
                        )

            if notes:
                notes.sort(key=lambda note: (note.start, note.pitch, note.end))
                parts.append(
                    SourcePart(
                        name=name,
                        instrument=_instrument_name(program, name),
                        notes=notes,
                        program=program,
                    )
                )

        duration = max(
            (note.end for part in parts for note in part.notes),
            default=0.0,
        )
        return MultiInstrumentTranscription(
            parts=parts,
            duration=duration,
            tempo=tempo,
            model="YourMT3+",
        )


def _midi_tempo(midi) -> float:
    for track in midi.tracks:
        for message in track:
            if message.type == "set_tempo":
                return 60_000_000 / message.tempo
    return 120.0


def _tick_to_seconds(tick: int, midi, tempo_bpm: float) -> float:
    import mido

    return mido.tick2second(
        tick,
        midi.ticks_per_beat,
        mido.bpm2tempo(tempo_bpm),
    )


def _instrument_name(program: int | None, track_name: str) -> str:
    """Prefer a model-provided track name, otherwise map General MIDI."""
    lowered = track_name.lower()
    known = (
        ("piano", "Piano"),
        ("guitar", "Guitar"),
        ("bass", "Bass"),
        ("violin", "Violin"),
        ("viola", "Viola"),
        ("cello", "Cello"),
        ("string", "Strings"),
        ("flute", "Flute"),
        ("clarinet", "Clarinet"),
        ("sax", "Saxophone"),
        ("trumpet", "Trumpet"),
        ("trombone", "Trombone"),
        ("organ", "Organ"),
        ("synth", "Synth"),
        ("drum", "Drums"),
    )
    for token, instrument in known:
        if token in lowered:
            return instrument

    if program is None:
        return "Unknown"

    ranges = (
        (0, 7, "Piano"),
        (24, 31, "Guitar"),
        (32, 39, "Bass"),
        (40, 47, "Strings"),
        (48, 55, "Ensemble"),
        (56, 63, "Brass"),
        (64, 71, "Reed"),
        (72, 79, "Pipe"),
        (80, 87, "Synth Lead"),
        (88, 95, "Synth Pad"),
    )
    for low, high, name in ranges:
        if low <= program <= high:
            return name
    return f"MIDI Program {program + 1}"
