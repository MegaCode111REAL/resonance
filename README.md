# Resonance
[![CI](https://github.com/MegaCode111REAL/resonance/actions/workflows/ci.yml/badge.svg)](https://github.com/MegaCode111REAL/resonance/actions/workflows/ci.yml) [![pages-build-deployment](https://github.com/MegaCode111REAL/resonance/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/MegaCode111REAL/resonance/actions/workflows/pages/pages-build-deployment)

Resonance is an open-source AI system for understanding recorded music and creating new MIDI arrangements for user-selected instruments.

The core idea is:

~~~text
Audio
  ↓
Music transcription
  ↓
MIDI
  ↓
Musical analysis
  ↓
Target instrumentation
  ↓
Arrangement model
  ↓
MIDI
~~~

The source instruments do **not** constrain the output instruments. A pop song containing vocals, synth, guitar, bass and drums can be transcribed and arranged for four marimbas, even if there is no marimba in the original recording.

## Current status

Resonance is being built in stages.

The first implementation provides:

- Audio-file validation.
- Audio → MIDI transcription using Spotify Basic Pitch as the initial transcription backend.
- An audio-first browser interface: MP3/WAV/FLAC/OGG/M4A/AIFF → local transcription → arrangement → Auto Transpose → MIDI download.
- Browser-local transcription using Spotify's TypeScript Basic Pitch package; the recording is decoded and processed in the browser rather than uploaded by Resonance.
- MIDI normalization utilities.
- A target-instrument model independent of the source instrumentation.
- A deterministic baseline arranger that redistributes musical material across requested target parts.
- MIDI export.
- A command-line interface.
- A structure designed so the baseline components can later be replaced by Resonance-trained models.
- A future browser deployment path using browser-compatible model formats.

The baseline arranger is deliberately **not** the final AI arranger. It exists so that we have a complete, testable pipeline and a reference point for evaluating future learned models.

## Architecture

~~~text
                    AUDIO
                      │
                      ▼
             ┌──────────────────┐
             │   Transcription  │
             │  model/backend   │
             └────────┬─────────┘
                      │
                      ▼
                    MIDI
                      │
             ┌────────┴─────────┐
             │                  │
             ▼                  ▼
       MIDI analysis       Source metadata
             │
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ Arrangement      │
             │ planner/model    │
             └────────┬─────────┘
                      │
                      ▼
                ARRANGED MIDI
                      │
             ┌────────┴─────────┐
             ▼                  ▼
          .mid               MusicXML
                              (planned)
~~~

### Important design rule

MIDI is the central musical representation. Resonance does **not** use an LLM to generate the notes of an arrangement.

An optional language model may eventually translate natural-language requests into structured arrangement settings, but the musical arrangement itself is produced by the arranger.

## Development

Requirements:

- Python 3.11+
- FFmpeg is recommended for broad audio-format support.
- A machine with enough memory for the transcription model.

Install the project in editable mode:

~~~bash
python -m pip install -e ".[dev]"
~~~

Transcribe an audio file:

~~~bash
resonance transcribe song.wav --output output/song.mid
~~~

Create a four-part marimba arrangement:

~~~bash
resonance arrange output/song.mid   --instrument marimba   --parts 4   --output output/marimba-arrangement.mid
~~~

Run the complete baseline pipeline:

~~~bash
resonance run song.wav   --instrument marimba   --parts 4   --output output/marimba-arrangement.mid
~~~

Run tests:

~~~bash
python -m pytest
~~~

## Planned model development

### Transcription

The first backend uses Basic Pitch. It is a lightweight automatic music transcription model that can produce MIDI and is explicitly designed to work across instruments. It is a starting point, not the final Resonance transcription model.

Future Resonance models will work toward:

- instrument identification
- source-aware multi-instrument transcription
- better note separation
- drum-event transcription
- chord recognition
- musical section detection
- vocal melody transcription
- lyrics transcription



### Multi-instrument transcription branch

The `feature/yourmt3-transcription` branch adds a real multi-track transcription backend based on YourMT3+. Unlike the earlier fixed stem pipeline, the model can emit independent instrument tracks rather than forcing everything into vocals/bass/other. The upstream YourMT3+ project documents a multi-instrument, multi-track model, and the maintained MT3-Infer wrapper exposes it as the `yourmt3` backend. citeturn0search4turn1search0

Install the optional backend:

~~~bash
python -m pip install -e ".[yourmt3]"
~~~

Then:

~~~bash
resonance transcribe song.wav --backend yourmt3 --output output/song.mid
~~~

The first YourMT3 run downloads/caches its model checkpoint through the MT3-Infer backend. The checkpoint is roughly 536 MB according to the current MT3-Infer model table. citeturn1search0

For development against the official YourMT3 checkout instead, set:

~~~bash
export YOURMT3_ROOT=/path/to/YourMT3
~~~

The branch also contains a small Resonance runtime bridge for the official model implementation. It keeps the model implementation and weights outside the Resonance repository rather than copying the upstream research code into the project.

The browser version is intentionally **not** switched to this backend yet. YourMT3 is an autoregressive PyTorch model, and the current MT3-Infer project explicitly does not provide an ONNX export. citeturn1search0 The next browser step is therefore a dedicated browser-runtime conversion rather than pretending the Python backend can run on GitHub Pages.

### Arrangement

The final arranger will learn from MIDI arrangements and will be conditioned on:

- source musical material
- target instruments
- target part count
- instrument ranges
- ensemble role
- section
- musical importance
- difficulty/playability constraints

It will be possible to request instruments that do not exist in the original recording.

### Browser

The web interface is now audio-first rather than MIDI-first. MIDI remains the internal musical representation and an export format; users normally begin by dropping an audio recording.

The current browser flow is:

~~~text
MP3 / WAV / FLAC / OGG / M4A / AIFF
              ↓
     Browser-local transcription
              ↓
        Musical notes
              ↓
       Target instruments
              ↓
          Arrangement
              ↓
       Auto Transpose
              ↓
            MIDI

The long-term goal is a completely local web application:

~~~text
Browser
  ├── audio processing
  ├── transcription model
  ├── MIDI analysis
  ├── arrangement model
  └── MIDI / notation export
~~~

No server should be required for normal operation.

## Repository layout

~~~text
resonance/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── resonance/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── instruments.py
│       ├── midi.py
│       ├── pipeline.py
│       ├── arrangement/
│       │   ├── __init__.py
│       │   └── baseline.py
│       └── transcription/
│           ├── __init__.py
│           └── basic_pitch.py
├── tests/
│   ├── test_arrangement.py
│   ├── test_instruments.py
│   └── test_midi.py
├── .gitignore
├── pyproject.toml
└── README.md
~~~

## License

The Resonance source code is currently distributed under the MIT License. Model weights and third-party models retain their own licenses; see their respective repositories and model cards before redistribution.
