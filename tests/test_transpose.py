from pathlib import Path

import mido

from resonance.arrangement.transpose import (
    BLACK_KEY_PITCH_CLASSES,
    auto_transpose_midi,
    choose_best_transposition,
    transpose_midi,
)


def _make_midi(path: Path, notes: list[int]) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=100, time=0))
        track.append(mido.Message("note_off", note=note, velocity=0, time=120))
    midi.tracks.append(track)
    midi.save(path)


def test_choose_best_transposition_minimizes_black_key_notes():
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    for note in (61, 65, 68):
        track.append(mido.Message("note_on", note=note, velocity=100, time=0))
        track.append(mido.Message("note_off", note=note, velocity=0, time=120))
    midi.tracks.append(track)

    result = choose_best_transposition(midi)

    assert result.black_key_notes == 0
    assert result.semitones in {-1, -4, -6, -8, -10, 2, 3, 5, 7, 9, 10, 11}


def test_auto_transpose_rewrites_notes(tmp_path: Path):
    source = tmp_path / "source.mid"
    output = tmp_path / "output.mid"
    _make_midi(source, [61, 65, 68])

    result = auto_transpose_midi(source, output)

    assert output.exists()
    assert result.black_key_notes == 0

    midi = mido.MidiFile(output)
    notes = [
        message.note
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    ]
    assert all(note % 12 not in BLACK_KEY_PITCH_CLASSES for note in notes)


def test_transpose_rejects_midi_overflow(tmp_path: Path):
    source = tmp_path / "source.mid"
    output = tmp_path / "output.mid"
    _make_midi(source, [127])

    try:
        transpose_midi(source, output, 1)
    except ValueError as exc:
        assert "MIDI note 128" in str(exc)
    else:
        raise AssertionError("Expected MIDI overflow to be rejected")
