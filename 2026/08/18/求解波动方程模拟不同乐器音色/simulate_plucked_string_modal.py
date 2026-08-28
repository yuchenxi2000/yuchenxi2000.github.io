#!/usr/bin/env python3
"""Synthesize a plucked damped ideal string by analytic modal superposition.

The string is released from a triangular displacement profile with zero
initial velocity.  Bending stiffness is omitted and all modes share one
viscous decay rate.  The audio signal is the dynamic transverse force at the
right bridge.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from array import array
from pathlib import Path

from musical_pitch import pitch_filename_label, resolve_fundamental


def triangular_modal_amplitude(
    mode_number: int, pluck_ratio: float, release_displacement: float
) -> float:
    """Fourier-sine coefficient of the triangular release profile."""
    n_pi = mode_number * math.pi
    return (
        2.0
        * release_displacement
        * math.sin(n_pi * pluck_ratio)
        / (n_pi * n_pi * pluck_ratio * (1.0 - pluck_ratio))
    )


def synthesize_bridge_force(
    sample_rate: int,
    duration: float,
    fundamental: float,
    string_length: float,
    tension: float,
    pluck_ratio: float,
    release_displacement: float,
    requested_modes: int,
    decay_rate: float,
) -> tuple[array, int, float]:
    """Return T*u_x(L,t), the retained mode count, and highest frequency."""
    sample_count = round(sample_rate * duration)
    samples = array("d", [0.0]) * sample_count

    retained_modes = min(
        requested_modes,
        int(0.47 * sample_rate / fundamental),
    )
    if retained_modes < 1:
        raise ValueError("No string modes fall below the audio Nyquist limit")

    amplitude_step = math.exp(-decay_rate / sample_rate)
    highest_frequency = 0.0

    for mode_number in range(1, retained_modes + 1):
        undamped_omega = 2.0 * math.pi * mode_number * fundamental
        if decay_rate >= undamped_omega:
            raise ValueError(f"Mode {mode_number} is not underdamped")
        damped_omega = math.sqrt(undamped_omega**2 - decay_rate**2)
        highest_frequency = damped_omega / (2.0 * math.pi)

        initial_amplitude = triangular_modal_amplitude(
            mode_number, pluck_ratio, release_displacement
        )
        # q_n(t) = exp(-sigma*t) * [A*cos(Omega*t) + B*sin(Omega*t)]
        # with B=sigma*A/Omega so that q_n_dot(0)=0.
        cosine_coefficient = initial_amplitude
        sine_coefficient = decay_rate * initial_amplitude / damped_omega

        wave_number = mode_number * math.pi / string_length
        endpoint_sign = -1.0 if mode_number % 2 else 1.0
        force_gain = tension * wave_number * endpoint_sign

        phase_step = damped_omega / sample_rate
        step_cosine = math.cos(phase_step)
        step_sine = math.sin(phase_step)
        oscillator_cosine = 1.0
        oscillator_sine = 0.0
        envelope = 1.0

        force_cosine = force_gain * cosine_coefficient
        force_sine = force_gain * sine_coefficient

        for sample_index in range(sample_count):
            samples[sample_index] += envelope * (
                force_cosine * oscillator_cosine
                + force_sine * oscillator_sine
            )
            envelope *= amplitude_step
            next_cosine = (
                oscillator_cosine * step_cosine
                - oscillator_sine * step_sine
            )
            oscillator_sine = (
                oscillator_sine * step_cosine
                + oscillator_cosine * step_sine
            )
            oscillator_cosine = next_cosine

    return samples, retained_modes, highest_frequency


def remove_quasistatic_force(
    samples: array, sample_rate: int, cutoff_hz: float
) -> None:
    """One-pole high-pass filter initialized at the pre-release force."""
    if not samples:
        return
    coefficient = math.exp(-2.0 * math.pi * cutoff_hz / sample_rate)
    previous_input = samples[0]
    previous_output = 0.0
    samples[0] = 0.0

    for index in range(1, len(samples)):
        current_input = samples[index]
        current_output = coefficient * (
            previous_output + current_input - previous_input
        )
        samples[index] = current_output
        previous_input = current_input
        previous_output = current_output


def normalize_and_fade_out(
    samples: array,
    sample_rate: int,
    normalization_gain: float | None = None,
    peak_level: float = 0.92,
    fade_out_seconds: float = 0.03,
) -> float:
    peak = max(abs(value) for value in samples)
    if peak == 0.0 or not math.isfinite(peak):
        raise ValueError("Simulation produced an invalid or silent signal")

    gain = (
        peak_level / peak
        if normalization_gain is None
        else normalization_gain
    )
    for index in range(len(samples)):
        samples[index] *= gain

    fade_samples = min(round(fade_out_seconds * sample_rate), len(samples))
    for offset in range(fade_samples):
        factor = 0.5 - 0.5 * math.cos(
            math.pi * offset / max(1, fade_samples - 1)
        )
        samples[-1 - offset] *= factor
    return gain


def write_pcm16_wav(path: Path, samples: array, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        chunk_size = 8192
        for start in range(0, len(samples), chunk_size):
            pcm = bytearray()
            for value in samples[start : start + chunk_size]:
                integer = round(max(-1.0, min(1.0, value)) * 32767.0)
                pcm.extend(struct.pack("<h", integer))
            wav_file.writeframesraw(pcm)


def reconstructed_release_displacement(
    position_ratio: float,
    mode_count: int,
    pluck_ratio: float,
    release_displacement: float,
) -> float:
    return math.fsum(
        triangular_modal_amplitude(mode_number, pluck_ratio, release_displacement)
        * math.sin(mode_number * math.pi * position_ratio)
        for mode_number in range(1, mode_count + 1)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize a triangular-release pluck with modal damping."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="WAV path; generated from the selected pitch when omitted",
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--duration", type=float, default=4.0)
    pitch_group = parser.add_mutually_exclusive_group()
    pitch_group.add_argument(
        "--fundamental", type=float, default=None,
        help="direct fundamental frequency in Hz",
    )
    pitch_group.add_argument(
        "--note", default="A3",
        help="scientific pitch name, for example A3, C#4 or Bb3",
    )
    parser.add_argument(
        "--a4-frequency", "--reference-frequency",
        dest="a4_frequency", type=float, default=440.0,
        help="reference frequency of A4 in Hz (default: 440)",
    )
    parser.add_argument(
        "--temperament",
        choices=(
            "12tet",
            "just",
            "pythagorean",
            "quarter-comma-meantone",
            "1/4-comma-meantone",
        ),
        default="12tet",
        help="tuning system (default: twelve-tone equal temperament)",
    )
    parser.add_argument(
        "--tonic", default="C",
        help="tonic pitch class for table-based just or Pythagorean tuning",
    )
    parser.add_argument("--string-length", type=float, default=0.65)
    parser.add_argument("--linear-density", type=float, default=0.003)
    parser.add_argument("--pluck-position", type=float, default=0.18)
    parser.add_argument("--release-displacement", type=float, default=0.003)
    parser.add_argument("--modes", type=int, default=128)
    parser.add_argument("--t60", type=float, default=3.5)
    parser.add_argument("--high-pass-cutoff", type=float, default=20.0)
    parser.add_argument(
        "--normalization-gain",
        type=float,
        default=None,
        help=(
            "fixed output gain for comparable loudness between runs; "
            "when omitted, normalize the peak to 0.92"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        fundamental, canonical_note = resolve_fundamental(
            args.fundamental,
            args.note,
            args.a4_frequency,
            args.temperament,
            args.tonic,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    output_path = args.output
    if output_path is None:
        pitch_label = pitch_filename_label(canonical_note, fundamental)
        output_path = Path(
            f"audio_output/plucked_string_modal_{pitch_label}.wav"
        )

    positive_values = {
        "sample-rate": args.sample_rate,
        "duration": args.duration,
        "A4 frequency": args.a4_frequency,
        "string-length": args.string_length,
        "linear-density": args.linear_density,
        "release-displacement": args.release_displacement,
        "modes": args.modes,
        "t60": args.t60,
        "high-pass-cutoff": args.high_pass_cutoff,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise SystemExit(f"{name} must be positive")
    if not 0.0 < args.pluck_position < 1.0:
        raise SystemExit("pluck-position must be between 0 and 1")
    if args.high_pass_cutoff >= 0.5 * args.sample_rate:
        raise SystemExit("high-pass-cutoff must be below Nyquist")
    if args.normalization_gain is not None and (
        not math.isfinite(args.normalization_gain)
        or args.normalization_gain <= 0.0
    ):
        raise SystemExit("normalization-gain must be finite and positive")

    wave_speed = 2.0 * args.string_length * fundamental
    tension = args.linear_density * wave_speed**2
    decay_rate = 3.0 * math.log(10.0) / args.t60

    samples, retained_modes, highest_frequency = synthesize_bridge_force(
        sample_rate=args.sample_rate,
        duration=args.duration,
        fundamental=fundamental,
        string_length=args.string_length,
        tension=tension,
        pluck_ratio=args.pluck_position,
        release_displacement=args.release_displacement,
        requested_modes=args.modes,
        decay_rate=decay_rate,
    )

    reconstructed_peak = reconstructed_release_displacement(
        args.pluck_position,
        retained_modes,
        args.pluck_position,
        args.release_displacement,
    )
    reconstruction_error = (
        reconstructed_peak - args.release_displacement
    ) / args.release_displacement

    remove_quasistatic_force(samples, args.sample_rate, args.high_pass_cutoff)
    peak_before_gain = max(abs(value) for value in samples)
    normalization_gain = normalize_and_fade_out(
        samples,
        args.sample_rate,
        args.normalization_gain,
    )
    peak_after_gain = max(abs(value) for value in samples)
    clipped_sample_count = sum(abs(value) > 1.0 for value in samples)
    write_pcm16_wav(output_path, samples, args.sample_rate)

    summary = {
        "output_wav": str(output_path),
        "sample_rate_hz": args.sample_rate,
        "duration_s": args.duration,
        "fundamental_hz": fundamental,
        "note": canonical_note,
        "a4_frequency_hz": args.a4_frequency,
        "temperament": args.temperament,
        "tonic": args.tonic,
        "string_length_m": args.string_length,
        "linear_density_kg_per_m": args.linear_density,
        "wave_speed_m_per_s": wave_speed,
        "tension_n": tension,
        "pluck_position_fraction": args.pluck_position,
        "release_displacement_m": args.release_displacement,
        "t60_s": args.t60,
        "decay_rate_per_s": decay_rate,
        "retained_modes": retained_modes,
        "highest_mode_hz": highest_frequency,
        "reconstructed_displacement_m": reconstructed_peak,
        "relative_reconstruction_error": reconstruction_error,
        "high_pass_cutoff_hz": args.high_pass_cutoff,
        "normalization_mode": (
            "automatic" if args.normalization_gain is None else "fixed"
        ),
        "normalization_gain": normalization_gain,
        "peak_before_gain": peak_before_gain,
        "peak_after_gain": peak_after_gain,
        "clipped_sample_count": clipped_sample_count,
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(output_path)
    print(summary_path)
    print(
        f"modes={retained_modes}, highest={highest_frequency:.1f} Hz, "
        f"release reconstruction error={100.0 * reconstruction_error:.3f}%"
    )
    print(
        f"fundamental={fundamental:.3f} Hz, tension={tension:.2f} N, "
        f"T60={args.t60:.2f} s"
    )


if __name__ == "__main__":
    main()
