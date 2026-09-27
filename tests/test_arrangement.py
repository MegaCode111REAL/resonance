from pathlib import Path

import mido

from resonance.arrangement.baseline import arrange


def test_arranger_can_generate_an_instrument_absent_from_source(tmp_path: Path):
    source = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Original Piano"))
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    source.tracks.append(track)

    source_path = tmp_path / "source.mid"
    output_path = tmp_path / "marimba.mid"
    source.save(source_path)

    arrange(source_path, output_path, "marimba", 4)

    result = mido.MidiFile(output_path)
    assert len(result.tracks) == 5
    assert any(
        message.type == "program_change" and message.program == 12
        for track in result.tracks
        for message in track
    )
