#!/usr/bin/env python3
"""
cycle6_kuramoto_holographic_sonification.py - Cycle 6: 6π Phase Completion

Integrates:
1. Duopyramid Holographic Entropy (from Cycle 5)
2. Kuramoto Synchronization in Hilbert Space
3. Hexagonal Weaving Patterns (6-fold symmetry)
4. Octagonal Signal Normalization (8-fold symmetry)
5. Sonification of Entropy-Gravity Dynamics

The system couples:
- System A: Decagonal Duopyramid (D₁₀h symmetry, closed logic)
- System B: Non-binary Hilbert Field (open, superposition)
- Kuramoto Layer: Phase synchronization across oscillators
- Weaving Layer: Hexagonal pattern generation
- Normalization Layer: Octagonal signal processing
- Sonification Layer: Audio parameter mapping

Part of the Prismatic Self Project - 6π Cycle Completion
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import json
from enum import Enum
import wave
import struct


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1: FUNDAMENTAL CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

# Physical Constants
PHYSICS = {
    'G': 6.67430e-11,          # Gravitational constant [m³/kg·s²]
    'c': 2.99792458e8,         # Speed of light [m/s]
    'hbar': 1.054571817e-34,   # Reduced Planck constant [J·s]
    'k_B': 1.380649e-23,       # Boltzmann constant [J/K]
    'planck_length': 1.616255e-35,  # Planck length [m]
    'solar_mass': 1.989e30     # Solar mass [kg]
}

# L₄ Framework Constants
PHI = 1.618033988749895          # Golden ratio φ
TAU = 0.618033988749895          # τ = 1/φ = φ - 1
Z_C = 0.8660254038               # √(3)/2 - standard critical point
Z_C_OMEGA = 1.2247448714         # √(3/2) - omega-elevated critical point
K_KURAMOTO = 0.9240387650610407  # Kuramoto coupling constant
L4 = 7                           # Layer depth (φ⁴ + φ⁻⁴ = 7)

# Geometric Constants
HEXAGONAL_ANGLES = np.array([i * np.pi / 3 for i in range(6)])  # 60° intervals
OCTAGONAL_ANGLES = np.array([i * np.pi / 4 for i in range(8)])  # 45° intervals
DECAGONAL_ANGLES = np.array([i * np.pi / 5 for i in range(10)]) # 36° intervals

# Sonification Constants
SAMPLE_RATE = 44100  # Audio sample rate
BASE_FREQUENCY = 432  # A4 tuning (Hz)
OVERTONE_SERIES = [1, 2, 3, 4, 5, 6, 7, 8]  # Harmonic series


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2: KURAMOTO OSCILLATOR NETWORK
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class KuramotoOscillator:
    """
    Single Kuramoto oscillator with natural frequency and phase.

    The Kuramoto model describes synchronization:
    dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)

    where:
    - θᵢ: phase of oscillator i
    - ωᵢ: natural frequency of oscillator i
    - K: coupling strength
    - N: number of oscillators
    """
    index: int
    natural_frequency: float
    phase: float = 0.0
    amplitude: float = 1.0

    # Geometric embedding
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Coupling to duopyramid node
    duopyramid_node: Optional[int] = None

    def evolve(self, dt: float, coupling_term: float):
        """Evolve phase by one timestep."""
        self.phase += dt * (self.natural_frequency + coupling_term)
        self.phase = self.phase % (2 * np.pi)

    def get_complex_amplitude(self) -> complex:
        """Return complex representation: A·e^(iθ)"""
        return self.amplitude * np.exp(1j * self.phase)


@dataclass
class KuramotoNetwork:
    """
    Network of coupled Kuramoto oscillators with various coupling topologies.

    Supports:
    - All-to-all coupling (mean field)
    - Hexagonal coupling (6 neighbors)
    - Octagonal coupling (8 neighbors)
    - Decagonal coupling (10 neighbors, duopyramid)
    """
    n_oscillators: int
    coupling_strength: float = K_KURAMOTO
    natural_freq_spread: float = 0.1

    oscillators: List[KuramotoOscillator] = field(default_factory=list)
    coupling_matrix: np.ndarray = field(default=None)

    # Order parameter tracking
    order_parameter_history: List[complex] = field(default_factory=list)

    def __post_init__(self):
        """Initialize oscillators with distributed natural frequencies."""
        if not self.oscillators:
            self.oscillators = []
            for i in range(self.n_oscillators):
                # Natural frequencies distributed around 1.0
                omega = 1.0 + self.natural_freq_spread * (np.random.random() - 0.5)

                # Initial phases uniformly distributed
                phase = np.random.random() * 2 * np.pi

                self.oscillators.append(KuramotoOscillator(
                    index=i,
                    natural_frequency=omega,
                    phase=phase
                ))

        # Initialize coupling matrix (all-to-all by default)
        if self.coupling_matrix is None:
            self.coupling_matrix = np.ones((self.n_oscillators, self.n_oscillators))
            np.fill_diagonal(self.coupling_matrix, 0)

    def set_hexagonal_coupling(self):
        """
        Set hexagonal coupling topology (6-fold symmetry).
        Each oscillator couples to 6 neighbors arranged hexagonally.
        """
        self.coupling_matrix = np.zeros((self.n_oscillators, self.n_oscillators))

        for i in range(self.n_oscillators):
            # Connect to 6 neighbors (with wraparound)
            for k in range(6):
                j = (i + k * self.n_oscillators // 6) % self.n_oscillators
                if i != j:
                    self.coupling_matrix[i, j] = 1.0

    def set_octagonal_coupling(self):
        """
        Set octagonal coupling topology (8-fold symmetry).
        Each oscillator couples to 8 neighbors.
        """
        self.coupling_matrix = np.zeros((self.n_oscillators, self.n_oscillators))

        for i in range(self.n_oscillators):
            # Connect to 8 neighbors
            for k in range(8):
                j = (i + k * self.n_oscillators // 8) % self.n_oscillators
                if i != j:
                    self.coupling_matrix[i, j] = 1.0

    def set_decagonal_coupling(self):
        """
        Set decagonal coupling topology (10-fold symmetry).
        Matches duopyramid structure.
        """
        self.coupling_matrix = np.zeros((self.n_oscillators, self.n_oscillators))

        for i in range(self.n_oscillators):
            # Connect to 10 neighbors
            for k in range(10):
                j = (i + k * self.n_oscillators // 10) % self.n_oscillators
                if i != j:
                    self.coupling_matrix[i, j] = 1.0

    def compute_coupling_term(self, i: int) -> float:
        """
        Compute coupling term for oscillator i:
        (K/N) Σⱼ Aᵢⱼ sin(θⱼ - θᵢ)
        """
        theta_i = self.oscillators[i].phase
        coupling_sum = 0.0

        for j in range(self.n_oscillators):
            if self.coupling_matrix[i, j] > 0:
                theta_j = self.oscillators[j].phase
                coupling_sum += self.coupling_matrix[i, j] * np.sin(theta_j - theta_i)

        return (self.coupling_strength / self.n_oscillators) * coupling_sum

    def compute_order_parameter(self) -> complex:
        """
        Compute Kuramoto order parameter:
        r·e^(iψ) = (1/N) Σⱼ e^(iθⱼ)

        |r| measures synchronization:
        - r ≈ 0: incoherent (phases uniformly distributed)
        - r ≈ 1: synchronized (phases aligned)
        """
        z = sum(osc.get_complex_amplitude() for osc in self.oscillators)
        return z / self.n_oscillators

    def evolve(self, dt: float):
        """Evolve all oscillators by one timestep."""
        # Compute all coupling terms first
        coupling_terms = [self.compute_coupling_term(i) for i in range(self.n_oscillators)]

        # Then evolve all oscillators
        for i, osc in enumerate(self.oscillators):
            osc.evolve(dt, coupling_terms[i])

        # Track order parameter
        self.order_parameter_history.append(self.compute_order_parameter())

    def get_phase_vector(self) -> np.ndarray:
        """Return array of all phases."""
        return np.array([osc.phase for osc in self.oscillators])

    def get_synchronization_level(self) -> float:
        """Return magnitude of order parameter (synchronization level)."""
        return abs(self.compute_order_parameter())

    def to_dict(self) -> Dict:
        """Export network state."""
        r = self.compute_order_parameter()
        return {
            "n_oscillators": self.n_oscillators,
            "coupling_strength": self.coupling_strength,
            "order_parameter": {
                "magnitude": abs(r),
                "phase": np.angle(r)
            },
            "phases": [osc.phase for osc in self.oscillators],
            "natural_frequencies": [osc.natural_frequency for osc in self.oscillators]
        }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3: HEXAGONAL WEAVING PATTERNS
# ════════════════════════════════════════════════════════════════════════════════

class WeavingMode(Enum):
    """Hexagonal weaving modes."""
    RADIAL = "radial"          # Outward from center
    SPIRAL = "spiral"          # Logarithmic spiral
    LATTICE = "lattice"        # Regular hexagonal lattice
    INTERFERENCE = "interference"  # Wave interference pattern


@dataclass
class HexagonalWeave:
    """
    Hexagonal weaving pattern generator.

    Creates 6-fold symmetric patterns by superimposing waves
    at hexagonal angles (0°, 60°, 120°, 180°, 240°, 300°).
    """
    n_rings: int = 6
    base_wavelength: float = 1.0
    phase_offset: float = 0.0
    mode: WeavingMode = WeavingMode.LATTICE

    # Modulation parameters
    amplitude_decay: float = 0.9  # Amplitude decay per ring
    phase_rotation: float = PHI   # Phase rotation factor (golden ratio)

    def compute_hex_coordinates(self, ring: int, index: int) -> Tuple[float, float]:
        """
        Compute (x, y) position for hexagonal lattice point.

        Ring 0: center point
        Ring n: 6n points arranged hexagonally
        """
        if ring == 0:
            return (0.0, 0.0)

        # Angle within the ring
        angle = HEXAGONAL_ANGLES[index % 6] + (index // 6) * (np.pi / (3 * ring))
        radius = ring * self.base_wavelength

        return (radius * np.cos(angle), radius * np.sin(angle))

    def compute_pattern_value(self, x: float, y: float, time: float = 0.0) -> float:
        """
        Compute weaving pattern intensity at point (x, y).

        Superimposes 6 plane waves at hexagonal angles.
        """
        value = 0.0

        for i, angle in enumerate(HEXAGONAL_ANGLES):
            # Wave vector direction
            kx = np.cos(angle) / self.base_wavelength
            ky = np.sin(angle) / self.base_wavelength

            # Phase for this wave
            phase = self.phase_offset + i * self.phase_rotation + time

            # Plane wave contribution
            wave = np.cos(2 * np.pi * (kx * x + ky * y) + phase)

            value += wave

        # Normalize to [-1, 1]
        return value / 6.0

    def generate_lattice_points(self) -> List[Tuple[float, float, float]]:
        """
        Generate hexagonal lattice points with amplitude values.

        Returns list of (x, y, amplitude) tuples.
        """
        points = []

        # Center point
        points.append((0.0, 0.0, 1.0))

        for ring in range(1, self.n_rings + 1):
            # Number of points in this ring
            n_points = 6 * ring
            amplitude = self.amplitude_decay ** ring

            for i in range(n_points):
                x, y = self.compute_hex_coordinates(ring, i)
                points.append((x, y, amplitude))

        return points

    def compute_spiral_pattern(self, theta: float, time: float = 0.0) -> Tuple[float, float, float]:
        """
        Compute point on golden spiral with hexagonal modulation.

        r = a · φ^(θ/π) (logarithmic spiral with golden ratio growth)
        """
        # Golden spiral radius
        r = self.base_wavelength * (PHI ** (theta / np.pi))

        # Hexagonal modulation
        hex_mod = 0.0
        for angle in HEXAGONAL_ANGLES:
            hex_mod += np.cos(theta - angle + time)
        hex_mod = 1.0 + 0.1 * hex_mod / 6

        r *= hex_mod

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        amplitude = self.amplitude_decay ** (theta / (2 * np.pi))

        return (x, y, amplitude)

    def to_dict(self) -> Dict:
        """Export weave state."""
        return {
            "n_rings": self.n_rings,
            "base_wavelength": self.base_wavelength,
            "mode": self.mode.value,
            "n_lattice_points": 1 + sum(6 * r for r in range(1, self.n_rings + 1)),
            "phase_rotation": self.phase_rotation
        }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4: OCTAGONAL SIGNAL NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class OctagonalNormalizer:
    """
    8-fold symmetric signal normalization.

    Implements octagonal wavelet-like decomposition for signal processing.
    Each of 8 channels corresponds to an octagonal direction.
    """
    n_channels: int = 8

    # Normalization parameters
    target_rms: float = 1.0
    smoothing_factor: float = 0.1

    # Channel state
    channel_energies: np.ndarray = field(default=None)
    channel_phases: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.channel_energies is None:
            self.channel_energies = np.zeros(self.n_channels)
        if self.channel_phases is None:
            self.channel_phases = OCTAGONAL_ANGLES.copy()

    def decompose_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Decompose signal into 8 octagonal channels.

        Uses projection onto 8 basis functions at octagonal angles.
        """
        n_samples = len(signal)
        channels = np.zeros((self.n_channels, n_samples))

        t = np.linspace(0, 2 * np.pi, n_samples)

        for i, angle in enumerate(OCTAGONAL_ANGLES):
            # Basis function: cos(t + angle) and sin(t + angle)
            basis_cos = np.cos(t + angle)
            basis_sin = np.sin(t + angle)

            # Project signal onto basis
            coef_cos = np.sum(signal * basis_cos) / n_samples
            coef_sin = np.sum(signal * basis_sin) / n_samples

            # Reconstruct channel
            channels[i] = coef_cos * basis_cos + coef_sin * basis_sin

            # Update channel energy
            self.channel_energies[i] = np.sqrt(coef_cos**2 + coef_sin**2)
            self.channel_phases[i] = np.arctan2(coef_sin, coef_cos)

        return channels

    def normalize_channels(self, channels: np.ndarray) -> np.ndarray:
        """
        Apply octagonal normalization to equalize channel energies.
        """
        normalized = np.zeros_like(channels)

        total_energy = np.sum(self.channel_energies)
        if total_energy == 0:
            return channels

        target_per_channel = self.target_rms / np.sqrt(self.n_channels)

        for i in range(self.n_channels):
            if self.channel_energies[i] > 0:
                scale = target_per_channel / self.channel_energies[i]
                # Smooth scaling to avoid artifacts
                scale = 1.0 + self.smoothing_factor * (scale - 1.0)
                normalized[i] = channels[i] * scale
            else:
                normalized[i] = channels[i]

        return normalized

    def reconstruct_signal(self, channels: np.ndarray) -> np.ndarray:
        """Reconstruct signal from octagonal channels."""
        return np.sum(channels, axis=0) / self.n_channels

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Full octagonal normalization pipeline."""
        channels = self.decompose_signal(signal)
        normalized = self.normalize_channels(channels)
        return self.reconstruct_signal(normalized)

    def get_octagonal_spectrum(self) -> Dict:
        """Return octagonal energy spectrum."""
        return {
            f"channel_{i}": {
                "angle_deg": int(np.degrees(OCTAGONAL_ANGLES[i])),
                "energy": float(self.channel_energies[i]),
                "phase": float(self.channel_phases[i])
            }
            for i in range(self.n_channels)
        }

    def to_dict(self) -> Dict:
        """Export normalizer state."""
        return {
            "n_channels": self.n_channels,
            "target_rms": self.target_rms,
            "channel_energies": self.channel_energies.tolist(),
            "octagonal_spectrum": self.get_octagonal_spectrum()
        }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5: HOLOGRAPHIC ENTROPY SONIFICATION
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class SonificationMapping:
    """
    Maps physical quantities to audio parameters.

    Physics → Audio mappings:
    - Entropy → Base frequency (higher entropy = higher pitch)
    - Temperature → Timbre brightness (hotter = brighter)
    - Time dilation → Tempo (slower time = slower tempo)
    - Synchronization → Consonance (more sync = more harmonic)
    - Hexagonal pattern → Spatial panning (6 channels)
    - Octagonal norm → Dynamic range compression
    """

    # Frequency mapping
    min_frequency: float = 55.0      # A1
    max_frequency: float = 880.0     # A5

    # Tempo mapping
    min_bpm: float = 40.0
    max_bpm: float = 180.0

    # Amplitude mapping
    min_amplitude: float = 0.1
    max_amplitude: float = 1.0

    def entropy_to_frequency(self, entropy_normalized: float) -> float:
        """
        Map normalized entropy [0, 1] to frequency.

        Uses golden ratio scaling for harmonic relationships.
        """
        # Logarithmic mapping (musical pitch perception)
        log_min = np.log2(self.min_frequency)
        log_max = np.log2(self.max_frequency)
        log_freq = log_min + entropy_normalized * (log_max - log_min)
        return 2 ** log_freq

    def temperature_to_brightness(self, temp_normalized: float) -> List[float]:
        """
        Map normalized temperature [0, 1] to overtone amplitudes.

        Hotter = more high-frequency content.
        """
        overtones = []
        for n in OVERTONE_SERIES:
            # Higher harmonics emphasized for higher temperature
            amp = (1.0 / n) * (temp_normalized ** (1.0 / n))
            overtones.append(amp)
        return overtones

    def time_dilation_to_tempo(self, dilation: float) -> float:
        """
        Map time dilation factor [0, 1] to BPM.

        dilation = 0 (horizon) → slowest
        dilation = 1 (far) → fastest
        """
        return self.min_bpm + dilation * (self.max_bpm - self.min_bpm)

    def synchronization_to_consonance(self, sync_level: float) -> List[float]:
        """
        Map Kuramoto synchronization to harmonic ratios.

        sync ≈ 0: dissonant intervals (minor 2nd, tritone)
        sync ≈ 1: consonant intervals (octave, fifth, fourth)
        """
        consonant_ratios = [1.0, 2.0, 1.5, 4/3, 5/4, 6/5]  # Octave, 5th, 4th, M3, m3
        dissonant_ratios = [16/15, 45/32, 7/5, 9/8]        # m2, tritone, etc.

        ratios = []
        for i in range(len(consonant_ratios)):
            ratio = consonant_ratios[i] * sync_level + dissonant_ratios[i % len(dissonant_ratios)] * (1 - sync_level)
            ratios.append(ratio)

        return ratios

    def hexagonal_to_panning(self, hex_pattern: List[float]) -> List[Tuple[float, float]]:
        """
        Map 6 hexagonal values to stereo panning positions.

        Returns list of (left_gain, right_gain) tuples.
        """
        panning = []
        for i, val in enumerate(hex_pattern[:6]):
            angle = HEXAGONAL_ANGLES[i]
            # Map angle to stereo position
            pan = np.cos(angle)  # -1 (left) to +1 (right)
            left = np.sqrt((1 - pan) / 2)
            right = np.sqrt((1 + pan) / 2)
            panning.append((left * abs(val), right * abs(val)))
        return panning


@dataclass
class HolographicSonifier:
    """
    Sonification engine for holographic entropy dynamics.

    Generates audio representations of:
    - Duopyramid state (10 nodes → 10 frequencies)
    - Hilbert field (superposition → chord voicing)
    - Kuramoto sync (order parameter → timbre)
    - Entropy flow (gradient → filter sweep)
    """

    sample_rate: int = SAMPLE_RATE
    mapping: SonificationMapping = field(default_factory=SonificationMapping)
    normalizer: OctagonalNormalizer = field(default_factory=OctagonalNormalizer)

    # State
    current_frequencies: List[float] = field(default_factory=list)
    current_amplitudes: List[float] = field(default_factory=list)
    phase_accumulators: List[float] = field(default_factory=list)

    def __post_init__(self):
        # Initialize 10 oscillators for duopyramid nodes
        self.current_frequencies = [BASE_FREQUENCY * (PHI ** (i / 10)) for i in range(10)]
        self.current_amplitudes = [0.1] * 10
        self.phase_accumulators = [0.0] * 10

    def generate_tone(self, frequency: float, duration: float,
                      amplitude: float = 0.5, overtones: List[float] = None) -> np.ndarray:
        """
        Generate a tone with specified frequency and overtones.
        """
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)

        # Fundamental
        signal = amplitude * np.sin(2 * np.pi * frequency * t)

        # Add overtones
        if overtones:
            for i, ot_amp in enumerate(overtones):
                harmonic = i + 2  # 2nd harmonic, 3rd harmonic, etc.
                signal += ot_amp * amplitude * np.sin(2 * np.pi * frequency * harmonic * t)

        # Apply envelope (ADSR)
        attack = int(0.01 * n_samples)
        decay = int(0.1 * n_samples)
        release = int(0.1 * n_samples)

        envelope = np.ones(n_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack+decay] = np.linspace(1, 0.7, decay)
        envelope[-release:] = np.linspace(0.7, 0, release)

        return signal * envelope

    def sonify_duopyramid_state(self, node_activities: List[float],
                                 duration: float = 1.0) -> np.ndarray:
        """
        Sonify duopyramid node activities as 10-voice chord.

        Each node's activity determines amplitude of its frequency.
        """
        n_samples = int(duration * self.sample_rate)
        signal = np.zeros(n_samples)
        t = np.linspace(0, duration, n_samples)

        for i, activity in enumerate(node_activities):
            # Map node to frequency (golden ratio spacing)
            freq = BASE_FREQUENCY * (PHI ** (i / 5))
            amp = activity * 0.1  # Scale down for mixing

            # Generate partial with phase continuity
            phase_start = self.phase_accumulators[i]
            signal += amp * np.sin(2 * np.pi * freq * t + phase_start)

            # Update phase accumulator
            self.phase_accumulators[i] = (phase_start + 2 * np.pi * freq * duration) % (2 * np.pi)

        return signal

    def sonify_kuramoto_state(self, phases: np.ndarray, sync_level: float,
                               duration: float = 1.0) -> np.ndarray:
        """
        Sonify Kuramoto oscillator phases as frequency-modulated tones.

        Synchronization level affects harmonic content.
        """
        n_samples = int(duration * self.sample_rate)
        signal = np.zeros(n_samples)
        t = np.linspace(0, duration, n_samples)

        # Get consonance ratios based on sync
        ratios = self.mapping.synchronization_to_consonance(sync_level)

        for i, phase in enumerate(phases):
            # Base frequency modulated by phase
            base_freq = BASE_FREQUENCY * ratios[i % len(ratios)]

            # Phase modulation
            mod_depth = 0.1 * (1 - sync_level)  # Less modulation when synchronized
            freq = base_freq * (1 + mod_depth * np.sin(phase + 2 * np.pi * t))

            # Generate tone
            amp = 0.1 / len(phases)
            signal += amp * np.sin(2 * np.pi * base_freq * t + phase)

        return signal

    def sonify_entropy_gradient(self, entropy_values: List[float],
                                 duration: float = 1.0) -> np.ndarray:
        """
        Sonify entropy gradient as filter sweep.

        Higher entropy → higher cutoff frequency.
        """
        n_samples = int(duration * self.sample_rate)
        signal = np.zeros(n_samples)
        t = np.linspace(0, duration, n_samples)

        # Generate noise source
        noise = np.random.randn(n_samples) * 0.3

        # Create time-varying filter based on entropy
        for i, (start, end) in enumerate(zip(entropy_values[:-1], entropy_values[1:])):
            # Interpolate entropy over this segment
            segment_length = n_samples // (len(entropy_values) - 1)
            segment_start = i * segment_length
            segment_end = min((i + 1) * segment_length, n_samples)

            for j in range(segment_start, segment_end):
                t_local = (j - segment_start) / segment_length
                entropy = start + t_local * (end - start)

                # Map entropy to frequency
                freq = self.mapping.entropy_to_frequency(entropy)

                # Simple resonant filter approximation
                signal[j] = noise[j] * (0.5 + 0.5 * np.sin(2 * np.pi * freq * j / self.sample_rate))

        return signal

    def apply_hexagonal_spatialization(self, signal: np.ndarray,
                                        hex_pattern: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply hexagonal panning to create stereo output.
        """
        panning = self.mapping.hexagonal_to_panning(hex_pattern.tolist())

        left = np.zeros_like(signal)
        right = np.zeros_like(signal)

        # Mix with hexagonal weights
        for i, (l_gain, r_gain) in enumerate(panning):
            # Modulate signal by hexagonal pattern
            mod_signal = signal * (0.5 + 0.5 * hex_pattern[i % len(hex_pattern)])
            left += l_gain * mod_signal / 6
            right += r_gain * mod_signal / 6

        return left, right

    def generate_full_sonification(self,
                                    duopyramid_nodes: List[float],
                                    kuramoto_phases: np.ndarray,
                                    sync_level: float,
                                    entropy_gradient: List[float],
                                    hex_pattern: np.ndarray,
                                    duration: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete stereo sonification combining all elements.
        """
        # Generate individual layers
        duopyramid_signal = self.sonify_duopyramid_state(duopyramid_nodes, duration)
        kuramoto_signal = self.sonify_kuramoto_state(kuramoto_phases, sync_level, duration)
        entropy_signal = self.sonify_entropy_gradient(entropy_gradient, duration)

        # Mix layers
        mixed = 0.4 * duopyramid_signal + 0.3 * kuramoto_signal + 0.3 * entropy_signal

        # Apply octagonal normalization
        mixed = self.normalizer.process(mixed)

        # Apply hexagonal spatialization
        left, right = self.apply_hexagonal_spatialization(mixed, hex_pattern)

        # Final limiting
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > 0.95:
            left = left * 0.95 / max_val
            right = right * 0.95 / max_val

        return left, right

    def export_wav(self, left: np.ndarray, right: np.ndarray, filename: str):
        """Export stereo audio to WAV file."""
        # Normalize to 16-bit range
        left_int = (left * 32767).astype(np.int16)
        right_int = (right * 32767).astype(np.int16)

        # Interleave channels
        stereo = np.column_stack((left_int, right_int)).flatten()

        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(stereo.tobytes())


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6: INTEGRATED CYCLE 6 SYSTEM
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class DuopyramidHilbertCoupling:
    """
    Coupling between duopyramid (closed) and Hilbert field (open) systems.

    Implements bidirectional state transfer:
    - Duopyramid → Hilbert: Discrete collapse informs continuous phase
    - Hilbert → Duopyramid: Phase coherence guides node activation
    """
    n_nodes: int = 10
    n_basis: int = 10
    coupling_strength: float = 0.1

    # Duopyramid state (System A - closed)
    node_activities: np.ndarray = field(default=None)
    alpha_pole: float = 0.0  # Upper pole (projection)
    omega_pole: float = 0.0  # Lower pole (collapse)

    # Hilbert state (System B - open)
    phases: np.ndarray = field(default=None)
    weights: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.node_activities is None:
            self.node_activities = np.zeros(self.n_nodes)
        if self.phases is None:
            self.phases = np.random.uniform(0, 2 * np.pi, self.n_basis)
        if self.weights is None:
            self.weights = np.ones(self.n_basis) / self.n_basis

    def hilbert_to_duopyramid(self):
        """
        Transfer Hilbert phase hint to duopyramid collapse.
        """
        # Find dominant phase region
        dominant_idx = np.argmax(self.weights)
        hint = dominant_idx / self.n_basis

        # Collapse duopyramid toward node
        focus_idx = int(round(hint * self.n_nodes)) % self.n_nodes
        self.node_activities *= 0.9  # Decay
        self.node_activities[focus_idx] += 0.1 * self.coupling_strength

        # Update poles based on total activity
        total_activity = np.sum(self.node_activities)
        self.alpha_pole = np.max(self.node_activities) / (total_activity + 1e-10)
        self.omega_pole = 1 - self.alpha_pole

    def duopyramid_to_hilbert(self):
        """
        Transfer duopyramid node pattern to Hilbert phases.
        """
        # Compute weighted phase shift from node activities
        total_activity = np.sum(self.node_activities)
        if total_activity > 0:
            weighted_shift = np.sum(self.node_activities * np.arange(self.n_nodes)) / total_activity
        else:
            weighted_shift = 0

        # Modulate Hilbert phases
        for i in range(self.n_basis):
            self.phases[i] += self.coupling_strength * np.sin(i - weighted_shift)
            self.phases[i] %= (2 * np.pi)

        # Update weights based on phase coherence
        self.weights = np.abs(np.cos(self.phases))
        self.weights /= np.sum(self.weights) + 1e-10

    def evolve(self, time: float):
        """Evolve coupled system."""
        # Update Hilbert phases (perceptual drift)
        self.phases += 0.05 * np.sin(time / 3.0)
        self.weights = np.abs(np.cos(self.phases))
        self.weights /= np.sum(self.weights) + 1e-10

        # Bidirectional coupling
        self.hilbert_to_duopyramid()
        self.duopyramid_to_hilbert()

    def get_entropy_gradient(self) -> List[float]:
        """
        Compute entropy gradient from alpha pole to omega pole.
        """
        # Entropy varies from high (alpha) to low (omega)
        return [
            self.alpha_pole * (1 - i / 10) + self.omega_pole * (i / 10)
            for i in range(11)
        ]


@dataclass
class Cycle6System:
    """
    Complete Cycle 6 system integrating all components.

    Achieves 6π phase completion through:
    1. Duopyramid-Hilbert coupling (from Mirror System)
    2. Kuramoto synchronization layer
    3. Hexagonal weaving patterns
    4. Octagonal signal normalization
    5. Holographic entropy sonification
    """

    # Core systems
    coupling: DuopyramidHilbertCoupling = field(default_factory=DuopyramidHilbertCoupling)
    kuramoto: KuramotoNetwork = field(default=None)
    hexweave: HexagonalWeave = field(default_factory=HexagonalWeave)
    normalizer: OctagonalNormalizer = field(default_factory=OctagonalNormalizer)
    sonifier: HolographicSonifier = field(default_factory=HolographicSonifier)

    # State
    time: float = 0.0
    history: List[Dict] = field(default_factory=list)

    # L₄ coupling constants
    phi_coupling: float = PHI
    tau_damping: float = TAU
    z_critical: float = Z_C_OMEGA

    def __post_init__(self):
        if self.kuramoto is None:
            # Initialize Kuramoto network with decagonal coupling
            self.kuramoto = KuramotoNetwork(n_oscillators=40, coupling_strength=K_KURAMOTO)
            self.kuramoto.set_decagonal_coupling()

    def evolve(self, dt: float = 0.1):
        """Evolve the complete system by one timestep."""
        self.time += dt

        # 1. Evolve Kuramoto network
        self.kuramoto.evolve(dt)
        sync_level = self.kuramoto.get_synchronization_level()

        # 2. Evolve duopyramid-Hilbert coupling
        self.coupling.evolve(self.time)

        # 3. Modulate coupling based on Kuramoto sync
        self.coupling.coupling_strength = 0.1 * (0.5 + 0.5 * sync_level)

        # 4. Update hexagonal weave based on Hilbert phases
        self.hexweave.phase_offset = np.mean(self.coupling.phases)

        # 5. Record state
        self.history.append(self.get_state_snapshot())

    def get_state_snapshot(self) -> Dict:
        """Get current state snapshot."""
        return {
            "time": self.time,
            "kuramoto_sync": self.kuramoto.get_synchronization_level(),
            "kuramoto_order_phase": np.angle(self.kuramoto.compute_order_parameter()),
            "duopyramid_nodes": self.coupling.node_activities.tolist(),
            "hilbert_weights": self.coupling.weights.tolist(),
            "alpha_pole": self.coupling.alpha_pole,
            "omega_pole": self.coupling.omega_pole,
            "hexweave_phase": self.hexweave.phase_offset
        }

    def generate_sonification(self, duration: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate audio sonification of current state."""
        # Get hexagonal pattern values
        lattice_points = self.hexweave.generate_lattice_points()
        hex_pattern = np.array([p[2] for p in lattice_points[:6]])  # Use first 6 amplitudes

        return self.sonifier.generate_full_sonification(
            duopyramid_nodes=self.coupling.node_activities.tolist(),
            kuramoto_phases=self.kuramoto.get_phase_vector(),
            sync_level=self.kuramoto.get_synchronization_level(),
            entropy_gradient=self.coupling.get_entropy_gradient(),
            hex_pattern=hex_pattern,
            duration=duration
        )

    def compute_6pi_closure(self) -> Dict:
        """Compute 6π cycle closure metrics."""
        # Total accumulated phase
        total_phase = 6 * np.pi
        phase_coherence = np.cos(total_phase) ** 2

        # Kuramoto-Hilbert alignment
        kuramoto_phase = np.angle(self.kuramoto.compute_order_parameter())
        hilbert_mean_phase = np.mean(self.coupling.phases)
        phase_alignment = np.cos(kuramoto_phase - hilbert_mean_phase)

        # Hexagonal-octagonal resonance
        hex_oct_resonance = np.cos(6 * self.hexweave.phase_offset) * np.cos(8 * self.hexweave.phase_offset)

        return {
            "total_phase_pi": 6.0,
            "phase_coherence": phase_coherence,
            "kuramoto_sync": self.kuramoto.get_synchronization_level(),
            "phase_alignment": phase_alignment,
            "hex_oct_resonance": hex_oct_resonance,
            "duopyramid_entropy": np.sum(self.coupling.node_activities * np.log(self.coupling.node_activities + 1e-10)),
            "hilbert_coherence": np.max(self.coupling.weights)
        }

    def run_simulation(self, duration: float = 100.0, dt: float = 0.1) -> List[Dict]:
        """Run complete simulation."""
        n_steps = int(duration / dt)
        for _ in range(n_steps):
            self.evolve(dt)
        return self.history

    def export_state(self, filepath: str):
        """Export complete state to JSON."""
        state = {
            "cycle": 6,
            "phase_pi": 6.0,
            "framework": "kuramoto_holographic_sonification",
            "L4_constants": {
                "phi": PHI,
                "tau": TAU,
                "z_c": Z_C,
                "z_c_omega": Z_C_OMEGA,
                "K_kuramoto": K_KURAMOTO,
                "L4": L4
            },
            "kuramoto": self.kuramoto.to_dict(),
            "duopyramid_hilbert": {
                "n_nodes": self.coupling.n_nodes,
                "n_basis": self.coupling.n_basis,
                "node_activities": self.coupling.node_activities.tolist(),
                "hilbert_phases": self.coupling.phases.tolist(),
                "hilbert_weights": self.coupling.weights.tolist(),
                "alpha_pole": self.coupling.alpha_pole,
                "omega_pole": self.coupling.omega_pole
            },
            "hexagonal_weave": self.hexweave.to_dict(),
            "octagonal_normalizer": self.normalizer.to_dict(),
            "closure_metrics": self.compute_6pi_closure(),
            "history_length": len(self.history)
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        return state


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7: UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def create_standard_cycle6_system() -> Cycle6System:
    """Create a standard Cycle 6 system with default parameters."""
    return Cycle6System()


def print_reference_card():
    """Print the Cycle 6 reference card."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    CYCLE 6: KURAMOTO-HOLOGRAPHIC SONIFICATION                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  KURAMOTO MODEL:                                                              ║
║  dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)                                        ║
║  Order parameter: r·e^(iψ) = (1/N) Σⱼ e^(iθⱼ)                                ║
║  K_critical = 0.9240387650610407                                              ║
║                                                                               ║
║  HEXAGONAL WEAVING (6-fold symmetry):                                         ║
║  Pattern: Σᵢ cos(k⃗ᵢ · r⃗ + φᵢ)   where k⃗ᵢ at 60° intervals                   ║
║  Golden spiral modulation: r = a·φ^(θ/π)                                      ║
║                                                                               ║
║  OCTAGONAL NORMALIZATION (8-fold symmetry):                                   ║
║  8-channel decomposition at 45° intervals                                     ║
║  Energy equalization across octagonal directions                              ║
║                                                                               ║
║  SONIFICATION MAPPINGS:                                                       ║
║  Entropy → Frequency (log scale)                                              ║
║  Temperature → Brightness (overtone content)                                  ║
║  Time dilation → Tempo (BPM)                                                  ║
║  Synchronization → Consonance (harmonic ratios)                               ║
║  Hexagonal pattern → Spatial panning                                          ║
║                                                                               ║
║  COUPLING TOPOLOGY:                                                           ║
║  Duopyramid (10 nodes) ↔ Hilbert (10 basis) ↔ Kuramoto (40 oscillators)     ║
║                                                                               ║
║  L₄ CONSTANTS:                                                                ║
║  φ = 1.618033988749895   τ = 0.618033988749895                               ║
║  z_c = 0.8660254038      z_c_Ω = 1.2247448714                                ║
║  K = 0.9240387650610407  L₄ = 7                                              ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cycle 6: Kuramoto-Holographic Sonification System"
    )
    parser.add_argument("--duration", "-t", type=float, default=100.0,
                        help="Simulation duration (default: 100.0)")
    parser.add_argument("--output", "-o", type=str, default="cycle6_state.json",
                        help="Output JSON file (default: cycle6_state.json)")
    parser.add_argument("--audio", "-a", type=str, default=None,
                        help="Output WAV file for sonification")
    parser.add_argument("--reference", "-r", action="store_true",
                        help="Print reference card")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if args.reference:
        print_reference_card()

    # Create and run system
    system = create_standard_cycle6_system()

    if args.verbose:
        print("Cycle 6: Kuramoto-Holographic Sonification System")
        print(f"  Kuramoto oscillators: {system.kuramoto.n_oscillators}")
        print(f"  Duopyramid nodes: {system.coupling.n_nodes}")
        print(f"  Hilbert basis: {system.coupling.n_basis}")
        print(f"  Hexagonal rings: {system.hexweave.n_rings}")
        print(f"\nRunning simulation for {args.duration} time units...")

    system.run_simulation(args.duration)
    state = system.export_state(args.output)

    if args.verbose:
        print(f"\n6π Closure Metrics:")
        closure = state["closure_metrics"]
        for key, value in closure.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"\nState exported to: {args.output}")

    # Generate audio if requested
    if args.audio:
        if args.verbose:
            print(f"\nGenerating sonification...")

        left, right = system.generate_sonification(duration=5.0)
        system.sonifier.export_wav(left, right, args.audio)

        if args.verbose:
            print(f"Audio exported to: {args.audio}")
