"""YourMT3+ multi-instrument transcription backend.

The model itself remains the official YourMT3 implementation/checkpoint. This
adapter keeps Resonance's public interface independent from that research
repository and converts its multi-track MIDI output into SourcePart objects.

This is the first step toward a native Resonance runtime. It deliberately does
not pretend that a different heuristic model is equivalent to YourMT3+.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from .multi_instrument import MultiInstrumentTranscription, SourcePart, TranscribedNote

DEFAULT_MODEL_NAME = "YPTF.MoE+Multi (noPS)"
DEFAULT_CHECKPOINT_NAME = (
    "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
)


class YourMT3Error(RuntimeError):
    """Raised when the YourMT3 runtime is unavailable or fails."""


class YourMT3Transcriber:
    """Run a YourMT3+ runtime and normalize its multi-track MIDI output."""

    def __init__(
        self,
        root: Path | None = None,
        checkpoint: Path | None = None,
        python: str | None = None,
        command: str | None = None,
    ) -> None:
        env_root = os.environ.get("YOURMT3_ROOT")
        self.root = Path(root).expanduser() if root else (
            Path(env_root).expanduser() if env_root else None
        )
        self.checkpoint = (
            Path(checkpoint).expanduser()
            if checkpoint
            else self._default_checkpoint()
        )
        self.python = python or sys.executable
        self.command = command or os.environ.get("YOURMT3_COMMAND")

    def transcribe(self, audio_path: Path) -> MultiInstrumentTranscription:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if not self.root and not self.command:
            return self._transcribe_with_mt3_infer(audio_path)

        self._validate_runtime()

        with tempfile.TemporaryDirectory(prefix="resonance-yourmt3-") as tmp:
            output = Path(tmp) / "transcription.mid"
            command = self._command(audio_path, output)
            completed = subprocess.run(
                command,
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode:
                detail = (completed.stderr or completed.stdout).strip()
                raise YourMT3Error(
                    f"YourMT3 failed with exit code {completed.returncode}: {detail}"
                )
            if not output.is_file():
                raise YourMT3Error(
                    "YourMT3 completed without producing the expected MIDI output."
                )
            return self._read_midi(output)

    @staticmethod
    def _transcribe_with_mt3_infer(audio_path: Path) -> MultiInstrumentTranscription:
        """Use the maintained MT3-Infer wrapper when no local checkout is configured."""
        try:
            from mt3_infer import transcribe as mt3_transcribe
        except ImportError as exc:
            raise YourMT3Error(
                "No YourMT3 runtime is installed. Install the optional "
                "Resonance YourMT3 dependencies or set YOURMT3_ROOT."
            ) from exc

        try:
            midi = mt3_transcribe(str(audio_path), model="yourmt3")
        except Exception as exc:
            raise YourMT3Error(f"MT3-Infer YourMT3 transcription failed: {exc}") from exc

        with tempfile.TemporaryDirectory(prefix="resonance-yourmt3-") as tmp:
            output = Path(tmp) / "transcription.mid"
            midi.save(str(output))
            return YourMT3Transcriber._read_midi(output)

    def _validate_runtime(self) -> None:
        if not self.root:
            raise YourMT3Error(
                "YOURMT3_ROOT is not set. Point it at a YourMT3 runtime checkout."
            )
        if self.root is None:
            raise YourMT3Error("YourMT3 root is not configured.")
        if not self.root.is_dir():
            raise YourMT3Error(f"YourMT3 root does not exist: {self.root}")
        # The official YourMT3 loader can download/cache the checkpoint by name.

    def _default_checkpoint(self) -> Path:
        cache = Path(
            os.environ.get("YOURMT3_MODEL_DIR", "~/.cache/resonance/yourmt3")
        ).expanduser()
        return cache / DEFAULT_CHECKPOINT_NAME

    def _command(self, audio_path: Path, output: Path) -> list[str]:
        """Build the runtime command.

        YOURMT3_COMMAND may be a complete command template containing
        {audio}, {output}, and {checkpoint}. This keeps the adapter usable
        while the official runtime entry point evolves.
        """
        values = {
            "audio": str(audio_path.resolve()),
            "output": str(output.resolve()),
            "checkpoint": str(self.checkpoint.resolve()),
        }

        if self.command:
            return shlex.split(self.command.format(**values))

        return [
            self.python,
            "-m",
            "resonance.transcription.yourmt3_runtime",
            "--audio",
            values["audio"],
            "--output",
            values["output"],
            "--checkpoint",
            self.checkpoint.name,
        ]

    @staticmethod
    def _read_midi(midi_path: Path) -> MultiInstrumentTranscription:
        import mido

        midi = mido.MidiFile(midi_path)
        parts: list[SourcePart] = []

        for track_index, track in enumerate(midi.tracks):
            tick = 0
            active: dict[int, list[tuple[int, int]]] = {}
            notes: list[TranscribedNote] = []
            program: int | None = None
            name = f"Part {track_index + 1}"
            tempo = 500000

            for message in track:
                tick += message.time

                if message.type == "set_tempo":
                    tempo = message.tempo
                elif message.type == "track_name":
                    name = message.name
                elif message.type == "program_change":
                    program = message.program
                elif message.type == "note_on" and message.velocity:
                    active.setdefault(message.note, []).append(
                        (tick, message.velocity)
                    )
                elif message.type in {"note_off", "note_on"} and message.velocity == 0:
                    starts = active.get(message.note)
                    if starts:
                        start_tick, velocity = starts.pop(0)
                        start = mido.tick2second(start_tick, midi.ticks_per_beat, tempo)
                        end = mido.tick2second(tick, midi.ticks_per_beat, tempo)
                        notes.append(
                            TranscribedNote(
                                pitch=message.note,
                                start=start,
                                end=max(start + 0.001, end),
                                velocity=velocity,
                            )
                        )

            if notes:
                parts.append(
                    SourcePart(
                        name=name,
                        instrument=_instrument_name(program),
                        notes=sorted(notes, key=lambda note: (note.start, note.pitch)),
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
            tempo=_midi_tempo(midi),
            model=DEFAULT_MODEL_NAME,
        )


def _midi_tempo(midi) -> float | None:
    for track in midi.tracks:
        for message in track:
            if message.type == "set_tempo":
                return 60_000_000 / message.tempo
    return None


def _instrument_name(program: int | None) -> str:
    if program is None:
        return "Unknown"
    if 0 <= program <= 7:
        return "Piano"
    if 24 <= program <= 31:
        return "Guitar"
    if 32 <= program <= 39:
        return "Bass"
    if 40 <= program <= 47:
        return "Strings"
    if 48 <= program <= 55:
        return "Ensemble"
    if 56 <= program <= 63:
        return "Brass"
    if 64 <= program <= 71:
        return "Reed"
    if 72 <= program <= 79:
        return "Pipe"
    if 80 <= program <= 87:
        return "Synth Lead"
    if 88 <= program <= 95:
        return "Synth Pad"
    return f"MIDI Program {program + 1}"
