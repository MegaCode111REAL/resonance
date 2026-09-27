"""Target-instrument definitions used by the arrangement system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Instrument:
    """Musical constraints and capabilities for an arrangement instrument."""

    name: str
    program: int
    low_midi: int
    high_midi: int
    preferred_low_midi: int
    preferred_high_midi: int
    max_simultaneous_notes: int
    roles: tuple[str, ...]


INSTRUMENTS: Final[dict[str, Instrument]] = {
    "piano": Instrument("piano", 0, 21, 108, 36, 96, 10, ("melody", "harmony", "bass", "rhythm")),
    "marimba": Instrument("marimba", 12, 48, 96, 52, 88, 4, ("melody", "harmony", "bass", "rhythm")),
    "vibraphone": Instrument("vibraphone", 11, 53, 89, 55, 84, 4, ("melody", "harmony", "texture")),
    "xylophone": Instrument("xylophone", 13, 65, 108, 67, 100, 4, ("melody", "rhythm")),
    "violin": Instrument("violin", 40, 55, 103, 60, 96, 2, ("melody", "harmony", "countermelody")),
    "cello": Instrument("cello", 42, 36, 96, 48, 84, 2, ("bass", "harmony", "melody")),
    "flute": Instrument("flute", 73, 60, 96, 67, 91, 1, ("melody", "countermelody")),
    "acoustic_guitar": Instrument("acoustic_guitar", 25, 40, 88, 45, 81, 6, ("melody", "harmony", "rhythm")),
    "electric_guitar": Instrument("electric_guitar", 27, 40, 96, 45, 88, 6, ("melody", "harmony", "rhythm", "countermelody")),
    "bass": Instrument("electric_bass", 33, 28, 67, 31, 55, 2, ("bass", "rhythm")),
}


def get_instrument(name: str) -> Instrument:
    """Return an instrument by case-insensitive name."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "marimbas": "marimba",
        "vibes": "vibraphone",
        "xylophones": "xylophone",
        "guitars": "acoustic_guitar",
        "acoustic": "acoustic_guitar",
        "electric": "electric_guitar",
        "vi": "violin",
    }
    key = aliases.get(key, key)

    try:
        return INSTRUMENTS[key]
    except KeyError as exc:
        available = ", ".join(sorted(INSTRUMENTS))
        raise ValueError(f"Unknown instrument '{name}'. Available: {available}") from exc
