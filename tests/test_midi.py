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
