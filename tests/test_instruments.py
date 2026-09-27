from resonance.instruments import get_instrument


def test_marimba_is_independent_target_instrument():
    instrument = get_instrument("marimba")
    assert instrument.name == "marimba"
    assert instrument.low_midi < instrument.high_midi


def test_instrument_aliases():
    assert get_instrument("Marimbas").name == "marimba"
    assert get_instrument("vibes").name == "vibraphone"
