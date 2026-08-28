#!/usr/bin/env python3
"""Modal simulation of a damped stiff piano string struck by a soft hammer.

The contact stage integrates the coupled nonlinear modal ODEs with RK4.  Once
the hammer separates, every string mode is propagated with its exact damped
free-vibration solution.  The output signal is the transverse reaction force
at the right bridge.

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


def raised_cosine_with_plateau(
    offset: float, contact_width: float, platform_width: float
) -> float:
    """Normalized contact density g(offset), in inverse metres."""
    distance = abs(offset)
    half_width = 0.5 * contact_width
    half_platform = 0.5 * platform_width
    normalization = 2.0 / (contact_width + platform_width)

    if distance <= half_platform:
        return normalization
    if distance >= half_width:
        return 0.0

    shoulder_phase = math.pi * (
        (2.0 * distance - platform_width)
        / (contact_width - platform_width)
    )
    return normalization * 0.5 * (1.0 + math.cos(shoulder_phase))


def coupling_coefficient(
    wave_number: float,
    strike_position: float,
    contact_width: float,
    platform_width: float,
    quadrature_steps: int,
) -> float:
    """Compute G_n = integral g(x-x_h) sin(k_n*x) dx by Simpson's rule."""
    # The symmetric footprint removes the odd part around x_h, leaving
    # sin(k*x_h) times the cosine transform of g.
    half_width = 0.5 * contact_width
    step = half_width / quadrature_steps

    total = raised_cosine_with_plateau(0.0, contact_width, platform_width)
    total += raised_cosine_with_plateau(
        half_width, contact_width, platform_width
    ) * math.cos(wave_number * half_width)

    for index in range(1, quadrature_steps):
        offset = index * step
        integrand = raised_cosine_with_plateau(
            offset, contact_width, platform_width
        ) * math.cos(wave_number * offset)
        total += (4.0 if index % 2 else 2.0) * integrand

    cosine_transform = 2.0 * step * total / 3.0
    return math.sin(wave_number * strike_position) * cosine_transform


def state_metrics(
    state: list[float], coupling: list[float], hammer_stiffness: float,
    hammer_exponent: float
) -> tuple[float, float, float]:
    """Return raw compression, relative velocity, and contact force."""
    mode_count = len(coupling)
    weighted_displacement = math.fsum(
        coupling[index] * state[index] for index in range(mode_count)
    )
    weighted_velocity = math.fsum(
        coupling[index] * state[mode_count + index]
        for index in range(mode_count)
    )
    compression = state[-2] - weighted_displacement
    relative_velocity = state[-1] - weighted_velocity
    force = (
        hammer_stiffness * compression**hammer_exponent
        if compression > 0.0
        else 0.0
    )
    return compression, relative_velocity, force


def derivative(
    state: list[float],
    coupling: list[float],
    modal_frequency_squared: list[float],
    modal_decay_rate: list[float],
    forcing_gain: list[float],
    hammer_mass: float,
    hammer_stiffness: float,
    hammer_exponent: float,
) -> list[float]:
    mode_count = len(coupling)
    _, _, force = state_metrics(
        state, coupling, hammer_stiffness, hammer_exponent
    )
    result = [0.0] * (2 * mode_count + 2)

    for index in range(mode_count):
        displacement = state[index]
        velocity = state[mode_count + index]
        result[index] = velocity
        result[mode_count + index] = (
            -2.0 * modal_decay_rate[index] * velocity
            - modal_frequency_squared[index] * displacement
            + forcing_gain[index] * force
        )

    result[-2] = state[-1]
    result[-1] = -force / hammer_mass
    return result


def rk4_step(
    state: list[float],
    time_step: float,
    derivative_arguments: tuple,
) -> list[float]:
    k1 = derivative(state, *derivative_arguments)
    stage = [
        value + 0.5 * time_step * slope
        for value, slope in zip(state, k1)
    ]
    k2 = derivative(stage, *derivative_arguments)
    stage = [
        value + 0.5 * time_step * slope
        for value, slope in zip(state, k2)
    ]
    k3 = derivative(stage, *derivative_arguments)
    stage = [value + time_step * slope for value, slope in zip(state, k3)]
    k4 = derivative(stage, *derivative_arguments)

    return [
        value
        + time_step
        * (slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4)
        / 6.0
        for value, slope1, slope2, slope3, slope4 in zip(
            state, k1, k2, k3, k4
        )
    ]


def bridge_force(
    state: list[float],
    wave_numbers: list[float],
    tension: float,
    bending_rigidity: float,
    linear_density: float,
    sigma1: float,
) -> float:
    """Right-bridge reaction, up to the overall direction convention."""
    mode_count = len(wave_numbers)
    terms = []
    for index, wave_number in enumerate(wave_numbers):
        mode_number = index + 1
        endpoint_sign = -1.0 if mode_number % 2 else 1.0
        displacement = state[index]
        velocity = state[mode_count + index]
        elastic_gain = tension * wave_number + bending_rigidity * wave_number**3
        loss_gain = 2.0 * linear_density * sigma1 * wave_number
        terms.append(
            endpoint_sign
            * (elastic_gain * displacement + loss_gain * velocity)
        )
    return math.fsum(terms)


def integrate_contact(
    sample_rate: int,
    oversample: int,
    output_samples: array,
    coupling: list[float],
    wave_numbers: list[float],
    modal_frequency_squared: list[float],
    modal_decay_rate: list[float],
    forcing_gain: list[float],
    hammer_mass: float,
    hammer_velocity: float,
    hammer_stiffness: float,
    hammer_exponent: float,
    tension: float,
    bending_rigidity: float,
    linear_density: float,
    sigma1: float,
    max_contact_time: float,
) -> tuple[float, list[float], float, float, int]:
    """Integrate until separation and return its time, state and diagnostics."""
    mode_count = len(coupling)
    state = [0.0] * (2 * mode_count + 2)
    state[-1] = hammer_velocity

    internal_rate = sample_rate * oversample
    time_step = 1.0 / internal_rate
    maximum_steps = math.ceil(max_contact_time * internal_rate)
    derivative_arguments = (
        coupling,
        modal_frequency_squared,
        modal_decay_rate,
        forcing_gain,
        hammer_mass,
        hammer_stiffness,
        hammer_exponent,
    )

    peak_force = 0.0
    peak_compression = 0.0
    contact_started = False
    last_audio_index = 0
    output_samples[0] = bridge_force(
        state,
        wave_numbers,
        tension,
        bending_rigidity,
        linear_density,
        sigma1,
    )

    for step_index in range(1, maximum_steps + 1):
        state = rk4_step(state, time_step, derivative_arguments)
        compression, relative_velocity, force = state_metrics(
            state, coupling, hammer_stiffness, hammer_exponent
        )
        peak_force = max(peak_force, force)
        peak_compression = max(peak_compression, compression)

        if compression > 0.0:
            contact_started = True

        if step_index % oversample == 0:
            audio_index = step_index // oversample
            if audio_index < len(output_samples):
                output_samples[audio_index] = bridge_force(
                    state,
                    wave_numbers,
                    tension,
                    bending_rigidity,
                    linear_density,
                    sigma1,
                )
                last_audio_index = audio_index

        if contact_started and compression <= 0.0 and relative_velocity < 0.0:
            separation_time = step_index * time_step
            return (
                separation_time,
                state,
                peak_force,
                peak_compression,
                last_audio_index,
            )

    raise RuntimeError(
        "Hammer did not separate before max-contact-time; increase that limit "
        "or inspect the hammer parameters"
    )


def add_free_vibration(
    samples: array,
    sample_rate: int,
    start_index: int,
    separation_time: float,
    separation_state: list[float],
    wave_numbers: list[float],
    modal_frequency_squared: list[float],
    modal_decay_rate: list[float],
    tension: float,
    bending_rigidity: float,
    linear_density: float,
    sigma1: float,
) -> None:
    """Add the exact underdamped free response to the remaining samples."""
    mode_count = len(wave_numbers)
    initial_tau = start_index / sample_rate - separation_time

    for index, wave_number in enumerate(wave_numbers):
        decay_rate = modal_decay_rate[index]
        damped_frequency = math.sqrt(
            modal_frequency_squared[index] - decay_rate * decay_rate
        )
        initial_displacement = separation_state[index]
        initial_velocity = separation_state[mode_count + index]

        cosine_coefficient = initial_displacement
        sine_coefficient = (
            initial_velocity + decay_rate * initial_displacement
        ) / damped_frequency

        endpoint_sign = -1.0 if (index + 1) % 2 else 1.0
        elastic_gain = tension * wave_number + bending_rigidity * wave_number**3
        loss_gain = 2.0 * linear_density * sigma1 * wave_number

        velocity_cosine = (
            -decay_rate * cosine_coefficient
            + damped_frequency * sine_coefficient
        )
        velocity_sine = (
            -decay_rate * sine_coefficient
            - damped_frequency * cosine_coefficient
        )
        force_cosine = endpoint_sign * (
            elastic_gain * cosine_coefficient + loss_gain * velocity_cosine
        )
        force_sine = endpoint_sign * (
            elastic_gain * sine_coefficient + loss_gain * velocity_sine
        )

        phase = damped_frequency * initial_tau
        oscillator_cosine = math.cos(phase)
        oscillator_sine = math.sin(phase)
        phase_step = damped_frequency / sample_rate
        step_cosine = math.cos(phase_step)
        step_sine = math.sin(phase_step)
        amplitude = math.exp(-decay_rate * initial_tau)
        amplitude_step = math.exp(-decay_rate / sample_rate)

        for sample_index in range(start_index, len(samples)):
            samples[sample_index] += amplitude * (
                force_cosine * oscillator_cosine
                + force_sine * oscillator_sine
            )
            amplitude *= amplitude_step
            next_cosine = (
                oscillator_cosine * step_cosine
                - oscillator_sine * step_sine
            )
            oscillator_sine = (
                oscillator_sine * step_cosine
                + oscillator_cosine * step_sine
            )
            oscillator_cosine = next_cosine


def normalize_and_fade_out(
    samples: array,
    sample_rate: int,
    normalization_gain: float | None = None,
    peak_level: float = 0.92,
    fade_out_seconds: float = 0.03,
) -> float:
    peak = max(abs(value) for value in samples)
    if peak == 0.0 or not math.isfinite(peak):
        raise ValueError("Simulation produced an invalid or silent output")

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate a nonlinear hammer striking a damped stiff string."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="WAV path; generated from the selected pitch when omitted",
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--duration", type=float, default=4.0)
    pitch_group = parser.add_mutually_exclusive_group()
    pitch_group.add_argument(
        "--fundamental", type=float, default=None,
        help="direct fundamental frequency in Hz",
    )
    pitch_group.add_argument(
        "--note", default="C4",
        help="scientific pitch name, for example C4, F#3 or Bb4",
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
    parser.add_argument("--length", type=float, default=0.65)
    parser.add_argument("--string-radius", type=float, default=0.0005)
    parser.add_argument("--steel-density", type=float, default=7850.0)
    parser.add_argument("--young-modulus", type=float, default=2.0e11)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument(
        "--strike-position", type=float, default=0.125,
        help="fraction of string length measured from the left end",
    )
    parser.add_argument("--contact-width", type=float, default=0.012)
    parser.add_argument("--platform-width", type=float, default=0.002)
    parser.add_argument("--hammer-mass", type=float, default=0.008)
    parser.add_argument("--hammer-velocity", type=float, default=1.5)
    parser.add_argument("--hammer-stiffness", type=float, default=4.0e9)
    parser.add_argument("--hammer-exponent", type=float, default=3.0)
    parser.add_argument("--t60-fundamental", type=float, default=5.0)
    parser.add_argument("--t60-high", type=float, default=0.8)
    parser.add_argument("--damping-high-mode", type=int, default=40)
    parser.add_argument("--max-contact-time", type=float, default=0.03)
    parser.add_argument("--quadrature-steps", type=int, default=512)
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
            f"audio_output/struck_string_modal_{pitch_label}.wav"
        )

    positive_values = {
        "sample-rate": args.sample_rate,
        "oversample": args.oversample,
        "duration": args.duration,
        "A4 frequency": args.a4_frequency,
        "length": args.length,
        "string-radius": args.string_radius,
        "steel-density": args.steel_density,
        "young-modulus": args.young_modulus,
        "modes": args.modes,
        "contact-width": args.contact_width,
        "hammer-mass": args.hammer_mass,
        "hammer-velocity": args.hammer_velocity,
        "hammer-stiffness": args.hammer_stiffness,
        "hammer-exponent": args.hammer_exponent,
        "t60-fundamental": args.t60_fundamental,
        "t60-high": args.t60_high,
        "damping-high-mode": args.damping_high_mode,
        "max-contact-time": args.max_contact_time,
        "quadrature-steps": args.quadrature_steps,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise SystemExit(f"{name} must be positive")
    if not 0.0 < args.strike_position < 1.0:
        raise SystemExit("strike-position must be between 0 and 1")
    if not 0.0 <= args.platform_width < args.contact_width:
        raise SystemExit("platform-width must satisfy 0 <= platform < contact width")
    if args.quadrature_steps % 2:
        raise SystemExit("quadrature-steps must be even for Simpson integration")
    if args.normalization_gain is not None and (
        not math.isfinite(args.normalization_gain)
        or args.normalization_gain <= 0.0
    ):
        raise SystemExit("normalization-gain must be finite and positive")

    area = math.pi * args.string_radius**2
    second_moment = math.pi * args.string_radius**4 / 4.0
    linear_density = args.steel_density * area
    bending_rigidity = args.young_modulus * second_moment
    first_wave_number = math.pi / args.length
    target_omega = 2.0 * math.pi * fundamental
    tension = (
        linear_density * target_omega**2
        - bending_rigidity * first_wave_number**4
    ) / first_wave_number**2
    if tension <= 0.0:
        raise SystemExit("parameters imply non-positive string tension")

    strike_position_metres = args.strike_position * args.length
    if (
        strike_position_metres - 0.5 * args.contact_width <= 0.0
        or strike_position_metres + 0.5 * args.contact_width >= args.length
    ):
        raise SystemExit("the hammer footprint must lie entirely on the string")

    alpha_first = 3.0 * math.log(10.0) / args.t60_fundamental
    alpha_high = 3.0 * math.log(10.0) / args.t60_high
    high_wave_number = args.damping_high_mode * math.pi / args.length
    sigma1 = (alpha_high - alpha_first) / (
        high_wave_number**2 - first_wave_number**2
    )
    sigma0 = alpha_first - sigma1 * first_wave_number**2
    if sigma0 < 0.0 or sigma1 < 0.0:
        raise SystemExit("the requested T60 values imply negative damping")

    wave_numbers: list[float] = []
    modal_frequency_squared: list[float] = []
    modal_decay_rate: list[float] = []
    coupling: list[float] = []

    for mode_number in range(1, args.modes + 1):
        wave_number = mode_number * math.pi / args.length
        omega_squared = (
            tension / linear_density * wave_number**2
            + bending_rigidity / linear_density * wave_number**4
        )
        frequency = math.sqrt(omega_squared) / (2.0 * math.pi)
        if frequency >= 0.47 * args.sample_rate:
            break
        decay_rate = sigma0 + sigma1 * wave_number**2
        if decay_rate >= math.sqrt(omega_squared):
            raise SystemExit(f"mode {mode_number} is not underdamped")

        wave_numbers.append(wave_number)
        modal_frequency_squared.append(omega_squared)
        modal_decay_rate.append(decay_rate)
        coupling.append(
            coupling_coefficient(
                wave_number,
                strike_position_metres,
                args.contact_width,
                args.platform_width,
                args.quadrature_steps,
            )
        )

    if not wave_numbers:
        raise SystemExit("no modes fall below the audio Nyquist limit")

    maximum_phase_step = (
        math.sqrt(modal_frequency_squared[-1])
        / (args.sample_rate * args.oversample)
    )
    if maximum_phase_step > 1.0:
        raise SystemExit(
            "internal time step is too large for the highest mode; "
            "increase --oversample"
        )

    forcing_gain = [
        2.0 * coefficient / (linear_density * args.length)
        for coefficient in coupling
    ]
    samples = array("d", [0.0]) * round(args.duration * args.sample_rate)

    (
        separation_time,
        separation_state,
        peak_force,
        peak_compression,
        last_contact_sample,
    ) = integrate_contact(
        sample_rate=args.sample_rate,
        oversample=args.oversample,
        output_samples=samples,
        coupling=coupling,
        wave_numbers=wave_numbers,
        modal_frequency_squared=modal_frequency_squared,
        modal_decay_rate=modal_decay_rate,
        forcing_gain=forcing_gain,
        hammer_mass=args.hammer_mass,
        hammer_velocity=args.hammer_velocity,
        hammer_stiffness=args.hammer_stiffness,
        hammer_exponent=args.hammer_exponent,
        tension=tension,
        bending_rigidity=bending_rigidity,
        linear_density=linear_density,
        sigma1=sigma1,
        max_contact_time=args.max_contact_time,
    )

    free_start_index = last_contact_sample + 1
    add_free_vibration(
        samples=samples,
        sample_rate=args.sample_rate,
        start_index=free_start_index,
        separation_time=separation_time,
        separation_state=separation_state,
        wave_numbers=wave_numbers,
        modal_frequency_squared=modal_frequency_squared,
        modal_decay_rate=modal_decay_rate,
        tension=tension,
        bending_rigidity=bending_rigidity,
        linear_density=linear_density,
        sigma1=sigma1,
    )

    mode_count = len(wave_numbers)
    string_energy = linear_density * args.length / 4.0 * math.fsum(
        separation_state[mode_count + index] ** 2
        + modal_frequency_squared[index] * separation_state[index] ** 2
        for index in range(mode_count)
    )
    hammer_energy = 0.5 * args.hammer_mass * separation_state[-1] ** 2
    initial_energy = 0.5 * args.hammer_mass * args.hammer_velocity**2
    retained_energy_ratio = (string_energy + hammer_energy) / initial_energy

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
        "string_length_m": args.length,
        "string_tension_n": tension,
        "linear_density_kg_per_m": linear_density,
        "bending_rigidity_n_m2": bending_rigidity,
        "sigma0_per_s": sigma0,
        "sigma1_m2_per_s": sigma1,
        "retained_modes": mode_count,
        "highest_mode_hz": math.sqrt(modal_frequency_squared[-1])
        / (2.0 * math.pi),
        "contact_duration_ms": 1000.0 * separation_time,
        "peak_contact_force_n": peak_force,
        "peak_compression_mm": 1000.0 * peak_compression,
        "energy_ratio_at_separation": retained_energy_ratio,
        "normalization_mode": (
            "automatic" if args.normalization_gain is None else "fixed"
        ),
        "normalization_gain": normalization_gain,
        "peak_before_gain": peak_before_gain,
        "peak_after_gain": peak_after_gain,
        "clipped_sample_count": clipped_sample_count,
        "hammer_mass_kg": args.hammer_mass,
        "hammer_initial_velocity_m_per_s": args.hammer_velocity,
        "hammer_stiffness_n_per_m_power_p": args.hammer_stiffness,
        "hammer_exponent": args.hammer_exponent,
        "strike_position_fraction": args.strike_position,
        "contact_width_m": args.contact_width,
        "platform_width_m": args.platform_width,
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(output_path)
    print(summary_path)
    print(
        f"contact={summary['contact_duration_ms']:.3f} ms, "
        f"peak force={peak_force:.2f} N, "
        f"peak compression={summary['peak_compression_mm']:.3f} mm"
    )
    print(
        f"modes={mode_count}, highest={summary['highest_mode_hz']:.1f} Hz, "
        f"energy ratio at separation={retained_energy_ratio:.6f}"
    )


if __name__ == "__main__":
    main()
