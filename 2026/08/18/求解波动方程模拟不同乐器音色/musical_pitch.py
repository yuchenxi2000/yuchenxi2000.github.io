"""Musical note parsing and frequency conversion for the simulators."""

from __future__ import annotations

import math
import re


_NATURAL_PITCH_CLASS = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_NATURAL_FIFTH_COORDINATE = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": -1,
    "G": 1,
    "A": 3,
    "B": 5,
}

_QUARTER_COMMA_MEANTONE_NAMES = {
    "quarter-comma-meantone",
    "1/4-comma-meantone",
}

# A pure fifth narrowed by one quarter of the syntonic comma 81/80.
# This is also exactly the fourth root of five, so four such fifths minus
# two octaves form a pure 5:4 major third.
_QUARTER_COMMA_MEANTONE_FIFTH = 5.0 ** 0.25

_JUST_RATIOS = (
    1.0,
    16.0 / 15.0,
    9.0 / 8.0,
    6.0 / 5.0,
    5.0 / 4.0,
    4.0 / 3.0,
    45.0 / 32.0,
    3.0 / 2.0,
    8.0 / 5.0,
    5.0 / 3.0,
    9.0 / 5.0,
    15.0 / 8.0,
)

_PYTHAGOREAN_RATIOS = (
    1.0,
    256.0 / 243.0,
    9.0 / 8.0,
    32.0 / 27.0,
    81.0 / 64.0,
    4.0 / 3.0,
    729.0 / 512.0,
    3.0 / 2.0,
    128.0 / 81.0,
    27.0 / 16.0,
    16.0 / 9.0,
    243.0 / 128.0,
)


def _normalize_accidentals(value: str) -> str:
    return value.strip().replace("♯", "#").replace("♭", "b")


def parse_pitch_class(value: str) -> int:
    normalized = _normalize_accidentals(value)
    match = re.fullmatch(r"([A-Ga-g])([#b]?)", normalized)
    if match is None:
        raise ValueError(
            f"Invalid tonic {value!r}; use a pitch class such as C, F# or Bb"
        )
    letter, accidental = match.groups()
    pitch_class = _NATURAL_PITCH_CLASS[letter.upper()]
    if accidental == "#":
        pitch_class += 1
    elif accidental == "b":
        pitch_class -= 1
    return pitch_class % 12


def parse_note(value: str) -> tuple[int, str]:
    normalized = _normalize_accidentals(value)
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", normalized)
    if match is None:
        raise ValueError(
            f"Invalid note {value!r}; use scientific pitch notation such as "
            "A3, C#4 or Bb3"
        )
    letter, accidental, octave_text = match.groups()
    pitch_class = _NATURAL_PITCH_CLASS[letter.upper()]
    if accidental == "#":
        pitch_class += 1
    elif accidental == "b":
        pitch_class -= 1
    octave = int(octave_text)
    midi_note = 12 * (octave + 1) + pitch_class
    canonical = f"{letter.upper()}{accidental}{octave}"
    return midi_note, canonical


def _temperament_ratios(temperament: str) -> tuple[float, ...]:
    if temperament == "12tet":
        return tuple(2.0 ** (degree / 12.0) for degree in range(12))
    if temperament == "just":
        return _JUST_RATIOS
    if temperament == "pythagorean":
        return _PYTHAGOREAN_RATIOS
    raise ValueError(f"Unknown temperament: {temperament}")


def _relative_coordinate(
    midi_note: int, tonic_pitch_class: int, ratios: tuple[float, ...]
) -> float:
    pitch_class = midi_note % 12
    scale_degree = (pitch_class - tonic_pitch_class) % 12
    tonic_instance = midi_note - scale_degree
    octave_index = (tonic_instance - tonic_pitch_class) // 12
    return math.ldexp(ratios[scale_degree], octave_index)


def _quarter_comma_meantone_coordinate(note: str) -> float:
    """Return an octave-aware coordinate on the meantone fifths chain.

    Unlike a twelve-pitch-class lookup table, this preserves the written
    accidental: for example, C-sharp and D-flat have distinct coordinates.
    """
    normalized = _normalize_accidentals(note)
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", normalized)
    if match is None:
        raise ValueError(
            f"Invalid note {note!r}; use scientific pitch notation such as "
            "A3, C#4 or Bb3"
        )

    letter_text, accidental, octave_text = match.groups()
    letter = letter_text.upper()
    accidental_step = 1 if accidental == "#" else -1 if accidental == "b" else 0
    fifth_coordinate = (
        _NATURAL_FIFTH_COORDINATE[letter] + 7 * accidental_step
    )

    octave = int(octave_text)
    chromatic_coordinate = (
        octave
        + (_NATURAL_PITCH_CLASS[letter] + accidental_step) / 12.0
    )
    raw_log2_coordinate = fifth_coordinate * math.log2(
        _QUARTER_COMMA_MEANTONE_FIFTH
    )
    octave_shift = round(chromatic_coordinate - raw_log2_coordinate)
    return 2.0 ** (raw_log2_coordinate + octave_shift)


def note_frequency(
    note: str,
    a4_frequency: float = 440.0,
    temperament: str = "12tet",
    tonic: str = "C",
) -> tuple[float, str]:
    """Return note frequency and canonical note name.

    Table-based non-equal temperaments use twelve chromatic ratios relative
    to ``tonic``. Quarter-comma meantone follows the written note spelling,
    so enharmonic names such as C-sharp and D-flat remain distinct. Every
    temperament is globally scaled so that A4 equals ``a4_frequency``.
    """
    if a4_frequency <= 0.0:
        raise ValueError("A4 frequency must be positive")
    midi_note, canonical = parse_note(note)
    tonic_pitch_class = parse_pitch_class(tonic)
    if temperament in _QUARTER_COMMA_MEANTONE_NAMES:
        note_coordinate = _quarter_comma_meantone_coordinate(canonical)
        a4_coordinate = _quarter_comma_meantone_coordinate("A4")
        return a4_frequency * note_coordinate / a4_coordinate, canonical
    ratios = _temperament_ratios(temperament)
    note_coordinate = _relative_coordinate(
        midi_note, tonic_pitch_class, ratios
    )
    a4_coordinate = _relative_coordinate(69, tonic_pitch_class, ratios)
    return a4_frequency * note_coordinate / a4_coordinate, canonical


def resolve_fundamental(
    fundamental: float | None,
    note: str,
    a4_frequency: float,
    temperament: str,
    tonic: str,
) -> tuple[float, str | None]:
    if fundamental is not None:
        if fundamental <= 0.0:
            raise ValueError("Fundamental frequency must be positive")
        return fundamental, None
    return note_frequency(note, a4_frequency, temperament, tonic)


def pitch_filename_label(note: str | None, fundamental: float) -> str:
    if note is None:
        return f"{fundamental:.2f}Hz"
    return (
        note.replace("#", "sharp")
        .replace("b", "flat")
        .replace("-", "minus")
    )
