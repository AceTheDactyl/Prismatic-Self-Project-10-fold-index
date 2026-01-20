#!/usr/bin/env python3
"""
Cycle 9: SPHERE SIGNAL PROPAGATION
===================================

The 8π closure generates a sphere in the RRRR-lattice.
This cycle propagates the signal through that sphere using
BFADGS operator logic, simulating black hole sonification
with holographic entropy.

Mathematical Foundation:
    S₉ = ∮_Σ BFADGS · dA  (Surface integral over 8π sphere)

    Where BFADGS operators are:
        B = Bekenstein (entropy bound)
        F = Flux (information flow)
        A = Area (holographic surface)
        D = Dimension (reduction operator)
        G = Gravity (entropic force)
        S = Sonification (audio mapping)

Phase Transitions:
    Each 2π rotation marks a phase boundary in the RRRR-lattice:
        R₁: 0   → 2π  (First rotation)
        R₂: 2π  → 4π  (Second rotation)
        R₃: 4π  → 6π  (Third rotation)
        R₄: 6π  → 8π  (Fourth rotation)

    RRRR = R₁ ⊗ R₂ ⊗ R₃ ⊗ R₄

Lucas Extension:
    L₉ = φ⁹ + φ⁻⁹ = 76

Author: VaultNode Genesis System
License: Prismatic Self Project
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import json

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
TAU = PHI - 1

# Physical constants (normalized units where c = G = ℏ = k_B = 1)
C_LIGHT = 1.0
G_NEWTON = 1.0
HBAR = 1.0
K_BOLTZMANN = 1.0

# Lucas numbers
LUCAS_SEQUENCE = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199]
L4 = 7   # Cycle 7
L8 = 47  # Cycle 8
L9 = 76  # Cycle 9

# Phase boundaries
PHASE_BOUNDARIES = [0, 2*np.pi, 4*np.pi, 6*np.pi, 8*np.pi]
RRRR_ROTATIONS = ['R₁', 'R₂', 'R₃', 'R₄']

# Sphere parameters
SPHERE_RADIUS = 8 * np.pi  # Radius equals total rotation

# Sonification base frequency (A4 = 432 Hz in natural tuning)
BASE_FREQUENCY = 432.0


# =============================================================================
# RRRR-LATTICE STRUCTURE
# =============================================================================

class RotationPhase(Enum):
    """The four rotation phases of the RRRR-lattice"""
    R1 = 1  # 0 → 2π
    R2 = 2  # 2π → 4π
    R3 = 3  # 4π → 6π
    R4 = 4  # 6π → 8π


@dataclass
class RRRRLattice:
    """
    The RRRR-lattice: 4D tensor product of rotations.

    R₁ ⊗ R₂ ⊗ R₃ ⊗ R₄

    Each Rᵢ represents a 2π rotation contributing to the 8π closure.
    """

    @staticmethod
    def get_phase(angle: float) -> RotationPhase:
        """Determine which rotation phase contains the angle"""
        angle = angle % (8 * np.pi)
        if angle < 2 * np.pi:
            return RotationPhase.R1
        elif angle < 4 * np.pi:
            return RotationPhase.R2
        elif angle < 6 * np.pi:
            return RotationPhase.R3
        else:
            return RotationPhase.R4

    @staticmethod
    def phase_transition_factor(angle: float) -> float:
        """
        Compute the phase transition factor.
        Peaks at phase boundaries (2π, 4π, 6π).
        """
        # Distance to nearest phase boundary
        boundaries = np.array([2*np.pi, 4*np.pi, 6*np.pi])
        angle_mod = angle % (8 * np.pi)
        distances = np.abs(boundaries - angle_mod)
        min_dist = np.min(distances)

        # Gaussian peak at boundaries
        sigma = 0.5
        return np.exp(-min_dist**2 / (2 * sigma**2))

    @staticmethod
    def lattice_coordinates(angle: float) -> Tuple[int, int, int, int]:
        """
        Map an angle to RRRR-lattice coordinates.
        Returns (r1, r2, r3, r4) where each is 0 or 1.
        """
        phase = RRRRLattice.get_phase(angle)
        coords = [0, 0, 0, 0]
        coords[phase.value - 1] = 1
        return tuple(coords)

    @staticmethod
    def tensor_product_state(angles: List[float]) -> np.ndarray:
        """
        Compute the tensor product state for multiple angles.
        Returns a 2⁴ = 16 dimensional state vector.
        """
        state = np.zeros(16)
        for angle in angles:
            coords = RRRRLattice.lattice_coordinates(angle)
            # Convert coords to index
            idx = coords[0] * 8 + coords[1] * 4 + coords[2] * 2 + coords[3]
            state[idx] += 1
        state /= np.linalg.norm(state) if np.linalg.norm(state) > 0 else 1
        return state


# =============================================================================
# BFADGS OPERATORS
# =============================================================================

@dataclass
class BekensteinOperator:
    """
    B: Bekenstein entropy bound operator.

    S ≤ 2πkER/ℏc = 2πER (in natural units)

    Maps area to maximum entropy.
    """
    name: str = "Bekenstein"
    symbol: str = "B"

    def apply(self, area: float, energy: float) -> float:
        """Compute Bekenstein-bounded entropy"""
        radius = np.sqrt(area / (4 * np.pi))
        max_entropy = 2 * np.pi * energy * radius
        return max_entropy

    def entropy_density(self, r: float, theta: float, phi: float) -> float:
        """Entropy density at spherical coordinates"""
        # Entropy increases toward the center (black hole core)
        return 1.0 / (r + 0.1)**2


@dataclass
class FluxOperator:
    """
    F: Information flux operator.

    Φ = ∫ J · dA

    Measures information flow through holographic surface.
    """
    name: str = "Flux"
    symbol: str = "F"

    def apply(self, current_density: np.ndarray, area_element: np.ndarray) -> float:
        """Compute flux through area element"""
        return np.dot(current_density, area_element)

    def radial_flux(self, r: float, time: float) -> float:
        """Outward radial information flux"""
        # Hawking radiation flux: increases with temperature (1/M ~ 1/r)
        return np.exp(-r / SPHERE_RADIUS) * np.sin(2 * np.pi * time)


@dataclass
class AreaOperator:
    """
    A: Holographic area operator.

    A = 4πr²

    The holographic screen area that bounds information.
    """
    name: str = "Area"
    symbol: str = "A"

    def apply(self, radius: float) -> float:
        """Compute sphere area"""
        return 4 * np.pi * radius**2

    def area_element(self, r: float, theta: float, dphi: float, dtheta: float) -> float:
        """Infinitesimal area element in spherical coordinates"""
        return r**2 * np.sin(theta) * dphi * dtheta


@dataclass
class DimensionOperator:
    """
    D: Dimensional reduction operator.

    Maps 3D bulk to 2D boundary (holographic principle).
    """
    name: str = "Dimension"
    symbol: str = "D"

    def apply(self, bulk_data: np.ndarray) -> np.ndarray:
        """Project 3D data to 2D holographic surface"""
        # Radial average projects to angular distribution
        if len(bulk_data.shape) == 3:
            return np.mean(bulk_data, axis=0)
        return bulk_data

    def holographic_projection(self, r: float, theta: float, phi: float) -> Tuple[float, float]:
        """Project 3D point to 2D holographic coordinates"""
        # Stereographic projection from north pole
        x = 2 * r * np.sin(theta) * np.cos(phi) / (1 + np.cos(theta))
        y = 2 * r * np.sin(theta) * np.sin(phi) / (1 + np.cos(theta))
        return (x, y)


@dataclass
class GravityOperator:
    """
    G: Entropic gravity operator.

    F = T∇S = GMm/r²

    Gravity emerges from entropy gradients.
    """
    name: str = "Gravity"
    symbol: str = "G"

    def apply(self, mass: float, radius: float) -> float:
        """Compute gravitational acceleration"""
        return G_NEWTON * mass / radius**2 if radius > 0 else 0

    def entropic_force(self, temperature: float, entropy_gradient: float) -> float:
        """F = T∇S"""
        return temperature * entropy_gradient

    def redshift_factor(self, r: float, schwarzschild_radius: float) -> float:
        """Gravitational redshift: √(1 - rs/r)"""
        if r <= schwarzschild_radius:
            return 0.0
        return np.sqrt(1 - schwarzschild_radius / r)


@dataclass
class SonificationOperator:
    """
    S: Sonification operator.

    Maps physical quantities to audio parameters:
        - Entropy → Amplitude
        - Temperature → Frequency
        - Area → Harmonic complexity
        - Redshift → Pitch bend
    """
    name: str = "Sonification"
    symbol: str = "S"
    base_frequency: float = BASE_FREQUENCY

    def entropy_to_amplitude(self, entropy: float, max_entropy: float = 100.0) -> float:
        """Map entropy to amplitude [0, 1]"""
        return min(1.0, entropy / max_entropy)

    def temperature_to_frequency(self, temperature: float) -> float:
        """Map Hawking temperature to frequency"""
        # Higher temperature = higher frequency
        # T ~ 1/M ~ 1/r for black hole
        return self.base_frequency * (1 + temperature)

    def area_to_harmonics(self, area: float) -> int:
        """Map area to number of harmonics"""
        # Larger area = more complex harmonic structure
        return max(1, int(np.log(area + 1) / np.log(PHI)))

    def redshift_to_pitch_bend(self, redshift_factor: float) -> float:
        """Map gravitational redshift to pitch bend ratio"""
        # Redshift lowers frequency
        return redshift_factor  # 0 at horizon, 1 at infinity

    def synthesize_sample(self, entropy: float, temperature: float,
                         area: float, redshift: float, time: float) -> float:
        """Generate audio sample from physical parameters"""
        amplitude = self.entropy_to_amplitude(entropy)
        base_freq = self.temperature_to_frequency(temperature)
        freq = base_freq * self.redshift_to_pitch_bend(redshift)
        n_harmonics = self.area_to_harmonics(area)

        # Additive synthesis with harmonics
        sample = 0.0
        for n in range(1, n_harmonics + 1):
            harmonic_amp = amplitude / n
            sample += harmonic_amp * np.sin(2 * np.pi * freq * n * time)

        return sample


@dataclass
class BFADGSOperatorSystem:
    """
    Complete BFADGS operator system.

    Combines all six operators into a unified framework.
    """
    B: BekensteinOperator = field(default_factory=BekensteinOperator)
    F: FluxOperator = field(default_factory=FluxOperator)
    A: AreaOperator = field(default_factory=AreaOperator)
    D: DimensionOperator = field(default_factory=DimensionOperator)
    G: GravityOperator = field(default_factory=GravityOperator)
    S: SonificationOperator = field(default_factory=SonificationOperator)

    def apply_chain(self, r: float, theta: float, phi: float,
                   time: float, mass: float = 1.0) -> Dict[str, float]:
        """Apply the full BFADGS operator chain"""

        # A: Area at this radius
        area = self.A.apply(r)

        # B: Bekenstein entropy bound
        energy = mass  # E = Mc² = M in natural units
        max_entropy = self.B.apply(area, energy)
        entropy = self.B.entropy_density(r, theta, phi) * area

        # F: Flux
        flux = self.F.radial_flux(r, time)

        # D: Dimensional projection
        holo_x, holo_y = self.D.holographic_projection(r, theta, phi)

        # G: Gravity
        schwarzschild_r = 2 * G_NEWTON * mass  # rs = 2GM
        gravity = self.G.apply(mass, r)
        redshift = self.G.redshift_factor(r, schwarzschild_r)

        # Temperature (Hawking)
        if r > schwarzschild_r:
            temperature = HBAR * C_LIGHT**3 / (8 * np.pi * G_NEWTON * mass * K_BOLTZMANN)
            temperature *= (schwarzschild_r / r)**2  # Scaled by position
        else:
            temperature = float('inf')

        # S: Sonification
        audio_sample = self.S.synthesize_sample(entropy, temperature, area, redshift, time)

        return {
            'r': r,
            'theta': theta,
            'phi': phi,
            'area': area,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'flux': flux,
            'holo_x': holo_x,
            'holo_y': holo_y,
            'gravity': gravity,
            'redshift': redshift,
            'temperature': temperature,
            'audio_sample': audio_sample
        }


# =============================================================================
# SPHERE SIGNAL
# =============================================================================

@dataclass
class SphereSignal:
    """
    The signal propagating through the 8π closure sphere.

    The sphere is divided into 4 phase regions corresponding
    to the RRRR-lattice rotations.
    """
    radius: float = SPHERE_RADIUS
    n_theta: int = 32
    n_phi: int = 64
    operators: BFADGSOperatorSystem = field(default_factory=BFADGSOperatorSystem)

    def __post_init__(self):
        # Create angular grids
        self.theta_grid = np.linspace(0, np.pi, self.n_theta)
        self.phi_grid = np.linspace(0, 2*np.pi, self.n_phi)

    def radial_layers(self) -> List[float]:
        """
        Define radial layers with phase transitions at 2π intervals.
        """
        n_layers = 32
        return np.linspace(0.1, self.radius, n_layers)

    def phase_at_radius(self, r: float) -> RotationPhase:
        """Determine phase based on radial distance"""
        # Map radius to rotation phase
        phase_idx = int((r / self.radius) * 4) % 4
        return list(RotationPhase)[phase_idx]

    def is_phase_boundary(self, r: float, tolerance: float = 0.5) -> bool:
        """Check if radius is near a phase boundary"""
        boundaries = np.array([2*np.pi, 4*np.pi, 6*np.pi])
        distances = np.abs(boundaries - r)
        return np.min(distances) < tolerance

    def compute_signal_at_point(self, r: float, theta: float, phi: float,
                                time: float, mass: float = 1.0) -> Dict[str, Any]:
        """Compute the complete signal at a point"""
        # Basic BFADGS computation
        result = self.operators.apply_chain(r, theta, phi, time, mass)

        # Add phase information
        phase = self.phase_at_radius(r)
        result['phase'] = phase.name
        result['phase_value'] = phase.value
        result['is_boundary'] = self.is_phase_boundary(r)

        # Add RRRR-lattice coordinates
        lattice_coords = RRRRLattice.lattice_coordinates(r)
        result['rrrr_coords'] = lattice_coords

        # Phase transition factor
        result['transition_factor'] = RRRRLattice.phase_transition_factor(r)

        return result

    def generate_radial_profile(self, theta: float, phi: float,
                                time: float) -> List[Dict[str, Any]]:
        """Generate signal profile along a radial line"""
        profile = []
        for r in self.radial_layers():
            point_data = self.compute_signal_at_point(r, theta, phi, time)
            profile.append(point_data)
        return profile

    def generate_surface_map(self, r: float, time: float) -> np.ndarray:
        """Generate 2D map of signal on spherical surface at radius r"""
        signal_map = np.zeros((self.n_theta, self.n_phi))

        for i, theta in enumerate(self.theta_grid):
            for j, phi in enumerate(self.phi_grid):
                result = self.compute_signal_at_point(r, theta, phi, time)
                signal_map[i, j] = result['audio_sample']

        return signal_map


# =============================================================================
# BLACK HOLE SONIFICATION
# =============================================================================

@dataclass
class BlackHoleSonifier:
    """
    Simulates black hole sonification using holographic entropy.

    The sound of a black hole is generated from:
    - Hawking radiation (high frequency hiss)
    - Gravitational waves (low frequency rumble)
    - Accretion disk (mid-range tones)
    - Photon sphere resonances (harmonic overtones)
    """
    mass: float = 1.0  # Solar masses (normalized)
    sample_rate: int = 44100
    operators: BFADGSOperatorSystem = field(default_factory=BFADGSOperatorSystem)

    @property
    def schwarzschild_radius(self) -> float:
        """Event horizon radius"""
        return 2 * G_NEWTON * self.mass

    @property
    def hawking_temperature(self) -> float:
        """Hawking temperature"""
        return HBAR * C_LIGHT**3 / (8 * np.pi * G_NEWTON * self.mass * K_BOLTZMANN)

    @property
    def photon_sphere_radius(self) -> float:
        """Radius of photon sphere"""
        return 1.5 * self.schwarzschild_radius

    @property
    def isco_radius(self) -> float:
        """Innermost stable circular orbit"""
        return 3 * self.schwarzschild_radius

    def hawking_radiation_sample(self, time: float) -> float:
        """Generate Hawking radiation noise"""
        # White noise modulated by temperature
        temp_factor = min(1.0, self.hawking_temperature)
        freq = BASE_FREQUENCY * (1 + 10 * temp_factor)
        return 0.1 * temp_factor * np.sin(2 * np.pi * freq * time + np.random.random() * np.pi)

    def gravitational_wave_sample(self, time: float, inspiral_phase: float = 0) -> float:
        """Generate gravitational wave tone"""
        # Low frequency chirp
        freq = 20 + 50 * inspiral_phase  # Increases during inspiral
        return 0.3 * np.sin(2 * np.pi * freq * time)

    def accretion_disk_sample(self, time: float, r: float) -> float:
        """Generate accretion disk tone at radius r"""
        if r <= self.isco_radius:
            return 0.0

        # Keplerian frequency
        orbital_freq = np.sqrt(G_NEWTON * self.mass / r**3) / (2 * np.pi)
        scaled_freq = BASE_FREQUENCY * orbital_freq * 1000  # Scale to audible

        return 0.2 * np.sin(2 * np.pi * scaled_freq * time)

    def photon_sphere_resonance(self, time: float) -> float:
        """Generate photon sphere resonance harmonics"""
        r = self.photon_sphere_radius
        base_freq = C_LIGHT / (2 * np.pi * r)  # Light orbital frequency
        scaled_freq = BASE_FREQUENCY * base_freq * 100

        # Multiple harmonics (quasi-normal modes)
        sample = 0.0
        for n in range(1, 5):
            damping = np.exp(-n * 0.5 * time)  # Modes decay
            sample += 0.1 * damping * np.sin(2 * np.pi * scaled_freq * n * time) / n

        return sample

    def composite_sample(self, time: float, r: float = None) -> float:
        """Generate composite black hole sound"""
        if r is None:
            r = 2 * self.isco_radius

        sample = 0.0
        sample += self.hawking_radiation_sample(time)
        sample += self.gravitational_wave_sample(time)
        sample += self.accretion_disk_sample(time, r)
        sample += self.photon_sphere_resonance(time)

        return np.clip(sample, -1.0, 1.0)

    def generate_audio_buffer(self, duration: float) -> np.ndarray:
        """Generate audio buffer for given duration"""
        n_samples = int(duration * self.sample_rate)
        buffer = np.zeros(n_samples)

        for i in range(n_samples):
            time = i / self.sample_rate
            buffer[i] = self.composite_sample(time)

        return buffer


# =============================================================================
# CYCLE 9 STATE
# =============================================================================

@dataclass
class Cycle9State:
    """
    Complete state for Cycle 9: Sphere Signal Propagation.

    Integrates:
    - RRRR-lattice structure
    - BFADGS operator system
    - Sphere signal propagation
    - Black hole sonification
    """
    sphere: SphereSignal = field(default_factory=SphereSignal)
    sonifier: BlackHoleSonifier = field(default_factory=BlackHoleSonifier)
    lattice: RRRRLattice = field(default_factory=RRRRLattice)
    time: float = 0.0

    def step(self, dt: float = 0.01):
        """Advance time"""
        self.time += dt

    def sample_sphere_signal(self, n_points: int = 100) -> List[Dict[str, Any]]:
        """Sample the sphere signal at random points"""
        samples = []
        for _ in range(n_points):
            r = np.random.uniform(0.1, self.sphere.radius)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)

            sample = self.sphere.compute_signal_at_point(r, theta, phi, self.time)
            samples.append(sample)

        return samples

    def compute_phase_statistics(self) -> Dict[str, Any]:
        """Compute statistics for each phase region"""
        stats = {}
        for phase in RotationPhase:
            # Sample points in this phase
            phase_samples = []
            for _ in range(50):
                # Map phase to radius range
                r_min = (phase.value - 1) * 2 * np.pi + 0.1
                r_max = phase.value * 2 * np.pi
                r = np.random.uniform(r_min, min(r_max, SPHERE_RADIUS))
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2*np.pi)

                sample = self.sphere.compute_signal_at_point(r, theta, phi, self.time)
                phase_samples.append(sample)

            # Compute statistics
            entropies = [s['entropy'] for s in phase_samples]
            audio = [s['audio_sample'] for s in phase_samples]

            stats[phase.name] = {
                'mean_entropy': np.mean(entropies),
                'mean_audio': np.mean(audio),
                'std_audio': np.std(audio),
                'n_samples': len(phase_samples)
            }

        return stats

    def unified_state(self) -> Dict[str, Any]:
        """Get complete unified state"""
        phase_stats = self.compute_phase_statistics()

        return {
            'time': self.time,
            'L9': L9,
            'sphere_radius': self.sphere.radius,
            'phase_boundaries': PHASE_BOUNDARIES,
            'phase_statistics': phase_stats,
            'black_hole': {
                'mass': self.sonifier.mass,
                'schwarzschild_radius': self.sonifier.schwarzschild_radius,
                'hawking_temperature': self.sonifier.hawking_temperature,
                'photon_sphere': self.sonifier.photon_sphere_radius,
                'isco': self.sonifier.isco_radius
            },
            'operators': ['B', 'F', 'A', 'D', 'G', 'S'],
            'total_rotation': '9π'
        }

    def to_json(self) -> str:
        """Export as JSON"""
        state = self.unified_state()

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, float)):
                if np.isinf(obj):
                    return "infinity"
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        return json.dumps(state, default=convert, indent=2)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_cycle9():
    """Full demonstration of Cycle 9"""

    print("=" * 70)
    print("CYCLE 9: SPHERE SIGNAL PROPAGATION")
    print("=" * 70)

    # 1. Lucas number
    print("\n1. LUCAS NUMBER L₉")
    print("-" * 40)
    L9_computed = round(PHI**9 + PHI**-9)
    print(f"L₉ = φ⁹ + φ⁻⁹ = {L9_computed}")
    print(f"Lucas sequence: ...{LUCAS_SEQUENCE[7]}, {LUCAS_SEQUENCE[8]}, {LUCAS_SEQUENCE[9]}...")

    # 2. RRRR-Lattice
    print("\n2. RRRR-LATTICE STRUCTURE")
    print("-" * 40)
    print("Phase boundaries (2π intervals):")
    for i, (r_name, boundary) in enumerate(zip(RRRR_ROTATIONS, PHASE_BOUNDARIES[1:])):
        print(f"  {r_name}: {PHASE_BOUNDARIES[i]:.4f} → {boundary:.4f}")

    # 3. BFADGS Operators
    print("\n3. BFADGS OPERATOR SYSTEM")
    print("-" * 40)
    operators = BFADGSOperatorSystem()
    print(f"  B = {operators.B.name} (entropy bound)")
    print(f"  F = {operators.F.name} (information flow)")
    print(f"  A = {operators.A.name} (holographic surface)")
    print(f"  D = {operators.D.name} (reduction)")
    print(f"  G = {operators.G.name} (entropic force)")
    print(f"  S = {operators.S.name} (audio mapping)")

    # 4. Sphere Signal
    print("\n4. SPHERE SIGNAL")
    print("-" * 40)
    sphere = SphereSignal()
    print(f"Radius: {sphere.radius:.4f} (= 8π)")
    print(f"Angular resolution: {sphere.n_theta} × {sphere.n_phi}")

    # Sample at a point
    sample = sphere.compute_signal_at_point(
        r=4*np.pi, theta=np.pi/4, phi=np.pi/2, time=0.5
    )
    print(f"\nSample at (r=4π, θ=π/4, φ=π/2, t=0.5):")
    print(f"  Phase: {sample['phase']}")
    print(f"  Entropy: {sample['entropy']:.6f}")
    print(f"  Redshift: {sample['redshift']:.6f}")
    print(f"  Audio sample: {sample['audio_sample']:.6f}")

    # 5. Black Hole Sonification
    print("\n5. BLACK HOLE SONIFICATION")
    print("-" * 40)
    sonifier = BlackHoleSonifier(mass=1.0)
    print(f"Mass: {sonifier.mass} (solar masses)")
    print(f"Schwarzschild radius: {sonifier.schwarzschild_radius:.6f}")
    print(f"Hawking temperature: {sonifier.hawking_temperature:.6e}")
    print(f"Photon sphere: {sonifier.photon_sphere_radius:.6f}")
    print(f"ISCO: {sonifier.isco_radius:.6f}")

    # Generate short audio sample
    print("\nGenerating 0.1s audio...")
    audio = sonifier.generate_audio_buffer(0.1)
    print(f"Audio buffer shape: {audio.shape}")
    print(f"Max amplitude: {np.max(np.abs(audio)):.4f}")

    # 6. Cycle 9 State
    print("\n6. CYCLE 9 STATE")
    print("-" * 40)
    state = Cycle9State()
    state.step(1.0)

    unified = state.unified_state()
    print(f"Time: {unified['time']:.2f}")
    print(f"L₉ = {unified['L9']}")
    print(f"Total rotation: {unified['total_rotation']}")

    print("\nPhase statistics:")
    for phase, stats in unified['phase_statistics'].items():
        print(f"  {phase}: entropy={stats['mean_entropy']:.4f}, audio={stats['mean_audio']:.4f}")

    print("\n" + "=" * 70)
    print("9π COMPLETE: SPHERE SIGNAL PROPAGATES THROUGH RRRR-LATTICE")
    print("=" * 70)

    return state


if __name__ == "__main__":
    state = demonstrate_cycle9()
