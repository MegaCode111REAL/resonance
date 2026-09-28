from pathlib import Path

import mido

from resonance.transcription.multi_instrument import (
    MultiInstrumentTranscription,
    SourcePart,
    TranscribedNote,
    merge_short_parts,
)
from resonance.transcription.yourmt3 import YourMT3Transcriber


def test_source_parts_are_independent():
    piano = SourcePart(
        name="Piano",
        instrument="piano",
        notes=[TranscribedNote(60, 0.0, 0.5)],
    )
    guitar = SourcePart(
        name="Guitar",
        instrument="guitar",
        notes=[TranscribedNote(64, 0.0, 0.25)],
    )

    transcription = MultiInstrumentTranscription(
        parts=[piano, guitar],
        duration=1.0,
        tempo=120,
        model="test",
    )

    assert [part.instrument for part in transcription.parts] == ["piano", "guitar"]


def test_short_fragment_merges_into_same_instrument():
    main = SourcePart(
        name="Guitar",
        instrument="guitar",
        notes=[
            TranscribedNote(52, 0.0, 0.5),
            TranscribedNote(55, 0.5, 1.0),
            TranscribedNote(59, 1.0, 1.5),
        ],
    )
    fragment = SourcePart(
        name="Guitar 2",
        instrument="guitar",
        notes=[TranscribedNote(60, 1.5, 1.75)],
    )

    result = merge_short_parts(
        MultiInstrumentTranscription(
            parts=[main, fragment],
            duration=2.0,
            model="test",
        )
    )

    assert len(result.parts) == 1
    assert len(result.parts[0].notes) == 4


def test_invalid_note_is_rejected():
    try:
        TranscribedNote(128, 0, 1)
    except ValueError as exc:
        assert "0..127" in str(exc)
    else:
        raise AssertionError("Expected invalid MIDI pitch to be rejected")


def test_yourmt3_midi_output_becomes_independent_parts(tmp_path: Path):
    midi = mido.MidiFile(ticks_per_beat=480)

    piano = mido.MidiTrack()
    piano.append(mido.MetaMessage("track_name", name="Piano"))
    piano.append(mido.Message("program_change", program=0, time=0))
    piano.append(mido.Message("note_on", note=60, velocity=100, time=0))
    piano.append(mido.Message("note_off", note=60, velocity=0, time=480))

    bass = mido.MidiTrack()
    bass.append(mido.MetaMessage("track_name", name="Bass"))
    bass.append(mido.Message("program_change", program=33, time=0))
    bass.append(mido.Message("note_on", note=36, velocity=90, time=0))
    bass.append(mido.Message("note_off", note=36, velocity=0, time=480))

    midi.tracks.extend([piano, bass])
    path = tmp_path / "transcription.mid"
    midi.save(path)

    result = YourMT3Transcriber._read_midi(path)

    assert len(result.parts) == 2
    assert result.parts[0].name == "Piano"
    assert result.parts[1].name == "Bass"
    assert result.parts[0].notes[0].pitch == 60
    assert result.parts[1].notes[0].pitch == 36
    assert result.parts[0].notes[0].end > result.parts[0].notes[0].start


def test_yourmt3_adapter_defaults_to_public_model_name():
    transcriber = YourMT3Transcriber()
    assert transcriber.model == "yourmt3"
    assert transcriber.device == "auto"
    assert transcriber.adaptive is False


def test_yourmt3_midi_conversion_preserves_separate_same_instrument_tracks(tmp_path: Path):
    midi = mido.MidiFile(ticks_per_beat=480)

    first = mido.MidiTrack()
    first.append(mido.MetaMessage("track_name", name="Guitar Lead"))
    first.append(mido.Message("program_change", program=24, time=0))
    first.append(mido.Message("note_on", note=64, velocity=100, time=0))
    first.append(mido.Message("note_off", note=64, velocity=0, time=240))

    second = mido.MidiTrack()
    second.append(mido.MetaMessage("track_name", name="Guitar Harmony"))
    second.append(mido.Message("program_change", program=24, time=0))
    second.append(mido.Message("note_on", note=52, velocity=90, time=0))
    second.append(mido.Message("note_off", note=52, velocity=0, time=240))

    midi.tracks.extend([first, second])
    path = tmp_path / "same-instrument.mid"
    midi.save(path)

    result = YourMT3Transcriber._midi_to_transcription(mido.MidiFile(path))

    assert len(result.parts) == 2
    assert [part.name for part in result.parts] == ["Guitar Lead", "Guitar Harmony"]
    assert [part.notes[0].pitch for part in result.parts] == [64, 52]
