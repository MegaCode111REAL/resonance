from pathlib import Path

import mido

from resonance.midi import load_midi, save_midi


def test_midi_round_trip(tmp_path: Path):
    source = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Test"))
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    source.tracks.append(track)

    path = tmp_path / "test.mid"
    save_midi(source, path)

    loaded = load_midi(path)
    assert len(loaded.tracks) == 1
    assert loaded.tracks[0][1].note == 60


def test_multi_instrument_transcription_exports_separate_tracks(tmp_path: Path):
    from resonance.midi import transcription_to_midi
    from resonance.transcription.multi_instrument import (
        MultiInstrumentTranscription,
        SourcePart,
        TranscribedNote,
    )

    transcription = MultiInstrumentTranscription(
        parts=[
            SourcePart(
                name="Piano",
                instrument="piano",
                program=0,
                notes=[TranscribedNote(60, 0.0, 0.5)],
            ),
            SourcePart(
                name="Bass",
                instrument="bass",
                program=32,
                notes=[TranscribedNote(36, 0.25, 0.75)],
            ),
        ],
        duration=1.0,
        tempo=120,
        model="test",
    )

    path = tmp_path / "multi.mid"
    transcription_to_midi(transcription, path)

    loaded = load_midi(path)
    assert len(loaded.tracks) == 2
    assert loaded.tracks[0][0].type == "set_tempo"
    assert any(message.type == "track_name" and message.name == "Piano" for message in loaded.tracks[0])
    assert any(message.type == "track_name" and message.name == "Bass" for message in loaded.tracks[1])
