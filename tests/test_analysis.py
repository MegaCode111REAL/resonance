from pathlib import Path
import mido
from resonance.analysis import analyze_midi

def test_analysis_extracts_core_features(tmp_path: Path):
    path=tmp_path/"song.mid"
    midi=mido.MidiFile(ticks_per_beat=480)
    track=mido.MidiTrack()
    track.append(mido.MetaMessage("track_name",name="Piano"))
    for pitch in (60,64,67): track.append(mido.Message("note_on",note=pitch,velocity=100,time=0))
    for pitch in (60,64,67): track.append(mido.Message("note_off",note=pitch,velocity=0,time=480))
    midi.tracks.append(track); midi.save(path)
    result=analyze_midi(path)
    assert result["note_count"]==3
    assert result["key"]["key"]
    assert result["melody"][0]["pitch"]==67
    assert result["chords"][0]["chord"].startswith("C")
    assert result["sections"]
