"""Tiny browser-facing decoder specification.

The neural model is intentionally kept in ONNX Runtime Web. This module is
used by the build tooling to emit the stable model metadata consumed by the
browser. Musical token decoding remains deterministic and versioned alongside
the exact YourMT3 configuration.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_metadata(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model": "YPTF.MoE+Multi (noPS)",
                "task": "mc13_full_plus_256",
                "channels": 13,
                "sample_rate": 16000,
                "segment_samples": 32767,
                "max_note_tokens_per_channel": 256,
                "vocab_size": 596,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
