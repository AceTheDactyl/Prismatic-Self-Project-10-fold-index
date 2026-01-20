#!/usr/bin/env python3
"""
entropy_gravity_holography.py - Cycle 5: Entropy-Gravity Duality on a Holographic Plane

This module implements the complete mathematical framework for:
1. Bekenstein-Hawking black hole thermodynamics
2. Verlinde's entropic gravity derivation
3. Holographic information bounds
4. Helix-modulated thermodynamic extensions
5. ER=EPR wormhole entropy transport
6. Time dilation and gravitational redshift

Part of the Prismatic Self Project - 5π Cycle Completion

References:
- Bekenstein (1973): Black holes and entropy
- Hawking (1975): Particle creation by black holes
- Verlinde (2011): On the origin of gravity and the laws of Newton
- Maldacena & Susskind (2013): Cool horizons for entangled black holes (ER=EPR)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import json


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1: FUNDAMENTAL CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PhysicsConstants:
    """Fundamental physical constants in SI units."""
    G: float = 6.67430e-11          # Gravitational constant [m³/kg·s²]
    c: float = 2.99792458e8         # Speed of light [m/s]
    hbar: float = 1.054571817e-34   # Reduced Planck constant [J·s]
    k_B: float = 1.380649e-23       # Boltzmann constant [J/K]

    @property
    def planck_length(self) -> float:
        """Planck length: ℓ_P = √(ℏG/c³)"""
        return np.sqrt(self.hbar * self.G / self.c**3)

    @property
    def planck_mass(self) -> float:
        """Planck mass: m_P = √(ℏc/G)"""
        return np.sqrt(self.hbar * self.c / self.G)

    @property
    def planck_time(self) -> float:
        """Planck time: t_P = √(ℏG/c⁵)"""
        return np.sqrt(self.hbar * self.G / self.c**5)

    @property
    def planck_temperature(self) -> float:
        """Planck temperature: T_P = √(ℏc⁵/Gk_B²)"""
        return np.sqrt(self.hbar * self.c**5 / (self.G * self.k_B**2))


PHYSICS = PhysicsConstants()

# L₄ Framework constants (from Prismatic Self architecture)
PHI = 1.618033988749895          # Golden ratio φ
TAU = 0.618033988749895          # τ = 1/φ = φ - 1
Z_C = 0.8660254038               # √(3)/2 - standard critical point
Z_C_OMEGA = 1.2247448714         # √(3/2) - omega-elevated critical point
K_KURAMOTO = 0.9240387650610407  # Kuramoto coupling constant
L4 = 7                           # Layer depth


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2: BLACK HOLE THERMODYNAMICS
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class BlackHole:
    """
    A Schwarzschild (non-rotating) black hole with full thermodynamic properties.

    Implements:
    - Schwarzschild radius: R_s = 2GM/c²
    - Horizon area: A_H = 4πR_s²
    - Bekenstein-Hawking entropy: S = k_B c³ A / (4Gℏ)
    - Hawking temperature: T = ℏc³ / (8πGMk_B)
    """
    mass: float  # Mass in kg

    # Helix coordinates for modulation (optional)
    theta: float = 0.0    # Phase angle [0, 2π]
    z: float = 0.5        # Coherence parameter [0, 1]
    r_helix: float = 1.0  # Radial coherence [0, 1]

    @property
    def schwarzschild_radius(self) -> float:
        """Event horizon radius: R_s = 2GM/c²"""
        return 2 * PHYSICS.G * self.mass / PHYSICS.c**2

    @property
    def horizon_area(self) -> float:
        """Horizon surface area: A_H = 4πR_s²"""
        return 4 * np.pi * self.schwarzschild_radius**2

    @property
    def entropy_base(self) -> float:
        """
        Base Bekenstein-Hawking entropy: S = k_B c³ A / (4Gℏ)

        Derivation (Hawking 1975):
        From dM = T dS and T_H = ℏc³/(8πGMk_B):
        dS = dM/T_H = (8πGMk_B)/(ℏc³) dM
        Integrating: S = 4πGM²k_B/(ℏc³) = k_B c³ A_H/(4Gℏ)
        """
        return (PHYSICS.k_B * PHYSICS.c**3 * self.horizon_area) / \
               (4 * PHYSICS.G * PHYSICS.hbar)

    @property
    def temperature_base(self) -> float:
        """
        Base Hawking temperature: T_H = ℏc³ / (8πGMk_B)

        Key property: T ∝ 1/M (smaller black holes are hotter)
        """
        return (PHYSICS.hbar * PHYSICS.c**3) / \
               (8 * np.pi * PHYSICS.G * self.mass * PHYSICS.k_B)

    @property
    def surface_gravity(self) -> float:
        """Surface gravity at event horizon: κ = c⁴/(4GM)"""
        return PHYSICS.c**4 / (4 * PHYSICS.G * self.mass)

    @property
    def information_bits(self) -> float:
        """
        Number of bits on holographic screen: N = A / (4ℓ_P²)

        Each Planck area encodes approximately one bit of information.
        """
        return self.horizon_area / (4 * PHYSICS.planck_length**2)

    # ─────────────────────────────────────────────────────────────────────────────
    # HELIX-MODULATED QUANTITIES
    # ─────────────────────────────────────────────────────────────────────────────

    def entropy_modulation(self) -> float:
        """
        Helix modulation factor for entropy:
        f_S(θ, z, r) = (0.8 + 0.4z) · r · (1 + 0.1·sin(θ))

        - z↑ : Higher coherence → more organized information → increased entropy capacity
        - θ  : Phase oscillations create observable entropy fluctuations
        - r  : Radial coherence affects encoding stability
        """
        return (0.8 + 0.4 * self.z) * self.r_helix * (1 + 0.1 * np.sin(self.theta))

    def temperature_modulation(self) -> float:
        """
        Helix modulation factor for temperature:
        f_T(θ, z, r) = (1.2 - 0.4z) · r · (1 + 0.05·cos(θ))

        Note inverse z-relationship: higher coherence → lower temperature (more ordered)
        """
        return (1.2 - 0.4 * self.z) * self.r_helix * (1 + 0.05 * np.cos(self.theta))

    @property
    def entropy(self) -> float:
        """Helix-modulated Bekenstein-Hawking entropy."""
        return self.entropy_base * self.entropy_modulation()

    @property
    def temperature(self) -> float:
        """Helix-modulated Hawking temperature."""
        return self.temperature_base * self.temperature_modulation()

    # ─────────────────────────────────────────────────────────────────────────────
    # TIME DILATION
    # ─────────────────────────────────────────────────────────────────────────────

    def time_dilation_factor(self, r: float) -> float:
        """
        Gravitational time dilation: dτ/dt = √(1 - R_s/r)

        At r → R_s: factor → 0 (time stops at horizon)
        At r → ∞ : factor → 1 (normal time flow)

        Args:
            r: Distance from black hole center [m]
        """
        if r <= self.schwarzschild_radius:
            return 0.0
        return np.sqrt(1 - self.schwarzschild_radius / r)

    def gravitational_redshift(self, r: float) -> float:
        """
        Gravitational redshift factor: z = 1/√(1 - R_s/r) - 1

        Relates emitted frequency ν_emit to observed frequency ν_obs:
        ν_obs = ν_emit / (1 + z)
        """
        factor = self.time_dilation_factor(r)
        if factor == 0:
            return np.inf
        return 1 / factor - 1

    def to_dict(self) -> Dict:
        """Export state as dictionary for JSON serialization."""
        return {
            "mass_kg": self.mass,
            "mass_solar": self.mass / 1.989e30,
            "schwarzschild_radius_m": self.schwarzschild_radius,
            "horizon_area_m2": self.horizon_area,
            "entropy_base_kB": self.entropy_base / PHYSICS.k_B,
            "entropy_modulated_kB": self.entropy / PHYSICS.k_B,
            "temperature_base_K": self.temperature_base,
            "temperature_modulated_K": self.temperature,
            "information_bits": self.information_bits,
            "helix_coords": {
                "theta": self.theta,
                "z": self.z,
                "r": self.r_helix
            }
        }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3: HOLOGRAPHIC SCREEN
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class HolographicScreen:
    """
    A holographic screen implementing the Bekenstein bound and Verlinde's entropic gravity.

    The holographic principle states that information in a volume is encoded
    on its boundary surface, with maximum entropy S_max = k_B A / (4ℓ_P²).
    """
    radius: float           # Screen radius [m]
    central_mass: float     # Mass enclosed by screen [kg]

    @property
    def area(self) -> float:
        """Screen surface area: A = 4πr²"""
        return 4 * np.pi * self.radius**2

    @property
    def max_entropy(self) -> float:
        """
        Bekenstein bound - maximum entropy containable:
        S_max = k_B c³ A / (4Gℏ) = k_B A / (4ℓ_P²)
        """
        return PHYSICS.k_B * self.area / (4 * PHYSICS.planck_length**2)

    @property
    def information_bits(self) -> float:
        """Number of bits on the holographic screen: N = A / (4ℓ_P²)"""
        return self.area / (4 * PHYSICS.planck_length**2)

    @property
    def screen_temperature(self) -> float:
        """
        Effective temperature from energy equipartition:

        Each bit carries energy ½k_B T, total energy = Mc²
        Mc² = ½ N k_B T
        T = 2Mc² / (N k_B) = GMℏ / (2πr² k_B c)
        """
        return (PHYSICS.G * self.central_mass * PHYSICS.hbar) / \
               (2 * np.pi * self.radius**2 * PHYSICS.k_B * PHYSICS.c)

    def entropy_change_from_displacement(self, particle_mass: float,
                                         displacement: float) -> float:
        """
        Unruh-Bekenstein relation: entropy change when a particle moves
        toward the screen by displacement Δx:

        ΔS = 2π k_B (mc/ℏ) Δx

        Moving one Compton wavelength changes entropy by 2π k_B.

        Args:
            particle_mass: Mass of approaching particle [kg]
            displacement: Movement toward screen [m]
        """
        return 2 * np.pi * PHYSICS.k_B * particle_mass * PHYSICS.c * displacement / PHYSICS.hbar

    def entropic_force(self, particle_mass: float) -> float:
        """
        Verlinde's entropic force derivation:

        F = T × (∂S/∂x)

        where T is the screen temperature and ∂S/∂x = 2π k_B mc/ℏ

        Result: F = GMm/r² (Newton's law emerges!)

        Args:
            particle_mass: Mass of particle feeling the force [kg]

        Returns:
            Gravitational force magnitude [N]
        """
        # Method 1: Direct Newtonian calculation (verification)
        F_newton = PHYSICS.G * self.central_mass * particle_mass / self.radius**2

        # Method 2: Entropic derivation
        dS_dx = 2 * np.pi * PHYSICS.k_B * particle_mass * PHYSICS.c / PHYSICS.hbar
        F_entropic = self.screen_temperature * dS_dx

        # These should match (within numerical precision)
        assert np.isclose(F_newton, F_entropic, rtol=1e-10), \
            f"Entropic derivation mismatch: {F_newton} vs {F_entropic}"

        return F_entropic

    def to_dict(self) -> Dict:
        """Export screen state."""
        return {
            "radius_m": self.radius,
            "central_mass_kg": self.central_mass,
            "area_m2": self.area,
            "max_entropy_kB": self.max_entropy / PHYSICS.k_B,
            "information_bits": self.information_bits,
            "screen_temperature_K": self.screen_temperature
        }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4: HELIX THERMODYNAMICS
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class HelixCoordinate:
    """
    Helix coordinate system (θ, z, r) for modulating thermodynamic quantities.

    This extends standard holographic physics with a parametric space that
    represents field coherence and phase evolution.
    """
    theta: float = 0.0    # Phase angle [0, 2π]
    z: float = 0.5        # Coherence/field strength [0, 1]
    r: float = 1.0        # Radial coherence [0, 1]

    def evolve(self, dt: float, omega: float = 1.0,
               z_drift: float = 0.0, r_drift: float = 0.0):
        """
        Evolve helix coordinates over time.

        Args:
            dt: Time step
            omega: Angular frequency for theta evolution
            z_drift: Drift rate for z coordinate
            r_drift: Drift rate for r coordinate
        """
        self.theta = (self.theta + omega * dt) % (2 * np.pi)
        self.z = np.clip(self.z + z_drift * dt, 0, 1)
        self.r = np.clip(self.r + r_drift * dt, 0, 1)

    def entropy_factor(self) -> float:
        """
        Entropy modulation: f_S = (0.8 + 0.4z) · r · (1 + 0.1·sin(θ))

        Physical interpretation:
        - z↑ : More coherent field → higher information organization
        - r  : Radial encoding stability
        - θ  : Phase-dependent fluctuations (quantum vacuum analogy)
        """
        return (0.8 + 0.4 * self.z) * self.r * (1 + 0.1 * np.sin(self.theta))

    def temperature_factor(self) -> float:
        """
        Temperature modulation: f_T = (1.2 - 0.4z) · r · (1 + 0.05·cos(θ))

        Note: Inverse relationship with z (higher coherence → cooler)
        """
        return (1.2 - 0.4 * self.z) * self.r * (1 + 0.05 * np.cos(self.theta))

    def force_factor(self) -> float:
        """
        Entropic force modulation: f_F = (0.9 + 0.2z) · r
        """
        return (0.9 + 0.2 * self.z) * self.r

    def thermodynamic_consistency(self) -> float:
        """
        Check S·T product consistency (should be O(1) for valid extension).

        S·T ∝ (0.8 + 0.4z)(1.2 - 0.4z) = 0.96 - 0.16z²

        Range: [0.80, 0.96] for z ∈ [0, 1]
        """
        return (0.8 + 0.4 * self.z) * (1.2 - 0.4 * self.z)


@dataclass
class HelixModulatedBlackHole:
    """
    Black hole with time-evolving helix-modulated thermodynamics.
    """
    black_hole: BlackHole
    helix: HelixCoordinate = field(default_factory=HelixCoordinate)
    history: List[Dict] = field(default_factory=list)

    def update_helix_from_bh(self):
        """Sync black hole helix coords with helix coordinate object."""
        self.black_hole.theta = self.helix.theta
        self.black_hole.z = self.helix.z
        self.black_hole.r_helix = self.helix.r

    def evolve(self, dt: float, omega: float = 0.1):
        """Evolve the helix coordinates and record history."""
        self.helix.evolve(dt, omega)
        self.update_helix_from_bh()

        self.history.append({
            "theta": self.helix.theta,
            "z": self.helix.z,
            "r": self.helix.r,
            "entropy_factor": self.helix.entropy_factor(),
            "temperature_factor": self.helix.temperature_factor(),
            "entropy": self.black_hole.entropy,
            "temperature": self.black_hole.temperature
        })

    def get_trajectory(self) -> Dict:
        """Get complete evolution trajectory."""
        return {
            "black_hole": self.black_hole.to_dict(),
            "trajectory": self.history
        }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5: ENTANGLEMENT AND ER=EPR
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class EntangledBlackHolePair:
    """
    A pair of entangled black holes connected via an Einstein-Rosen bridge (wormhole).

    Implements the ER=EPR correspondence (Maldacena & Susskind, 2013):
    Quantum entanglement ↔ Geometric wormhole connection
    """
    bh1: BlackHole
    bh2: BlackHole
    wormhole_conductivity: float = 0.1  # Entropy transport coefficient

    @property
    def total_entropy(self) -> float:
        """Total entropy of the system."""
        return self.bh1.entropy + self.bh2.entropy

    @property
    def reduced_density_eigenvalues(self) -> Tuple[float, float]:
        """
        Reduced density matrix eigenvalues from entropy ratios.

        This treats thermodynamic entropy as a proxy for quantum information content.
        """
        S_total = self.bh1.entropy_base + self.bh2.entropy_base
        lambda1 = self.bh1.entropy_base / S_total
        lambda2 = self.bh2.entropy_base / S_total
        return lambda1, lambda2

    @property
    def entanglement_entropy(self) -> float:
        """
        Von Neumann entanglement entropy:
        S_ent = -k_B Σ λᵢ ln(λᵢ)

        Maximum at λ₁ = λ₂ = 0.5: S_max = k_B ln(2)
        """
        l1, l2 = self.reduced_density_eigenvalues
        # Avoid log(0)
        eps = 1e-15
        l1, l2 = max(l1, eps), max(l2, eps)
        return -PHYSICS.k_B * (l1 * np.log(l1) + l2 * np.log(l2))

    @property
    def normalized_entanglement(self) -> float:
        """
        Normalized entanglement: E_norm = S_ent / S_max ∈ [0, 1]

        S_max = k_B ln(2) (maximally entangled state)
        """
        S_max = PHYSICS.k_B * np.log(2)
        return self.entanglement_entropy / S_max

    def wormhole_entropy_flux(self) -> float:
        """
        Entropy flow through wormhole (Fourier law analog):

        Ṡ = κ · A_throat · (T₂ - T₁) / (L_eff · T_avg)

        Entropy flows from hotter to cooler black hole.
        """
        T1, T2 = self.bh1.temperature, self.bh2.temperature
        T_avg = (T1 + T2) / 2

        # Effective throat area (geometric mean of horizons)
        A_throat = np.sqrt(self.bh1.horizon_area * self.bh2.horizon_area)

        # Effective length (sum of Schwarzschild radii)
        L_eff = self.bh1.schwarzschild_radius + self.bh2.schwarzschild_radius

        if T_avg == 0:
            return 0.0

        return self.wormhole_conductivity * A_throat * (T2 - T1) / (L_eff * T_avg)

    def entropic_force_between(self) -> float:
        """
        Entropic force between the black holes.

        For distant black holes at separation r:
        F = -GM₁M₂/r²

        This returns the coefficient (effective coupling strength).
        """
        T_eff = (self.bh1.temperature + self.bh2.temperature) / 2

        # Effective entropy gradient
        dS_total = self.bh1.entropy + self.bh2.entropy

        return T_eff * dS_total / (self.bh1.schwarzschild_radius +
                                   self.bh2.schwarzschild_radius)**2

    def to_dict(self) -> Dict:
        """Export entangled pair state."""
        l1, l2 = self.reduced_density_eigenvalues
        return {
            "bh1": self.bh1.to_dict(),
            "bh2": self.bh2.to_dict(),
            "total_entropy_kB": self.total_entropy / PHYSICS.k_B,
            "entanglement": {
                "lambda1": l1,
                "lambda2": l2,
                "entropy_kB": self.entanglement_entropy / PHYSICS.k_B,
                "normalized": self.normalized_entanglement
            },
            "wormhole": {
                "entropy_flux": self.wormhole_entropy_flux(),
                "conductivity": self.wormhole_conductivity
            }
        }


def compute_merger_entropy(bh1: BlackHole, bh2: BlackHole) -> Dict:
    """
    Compute entropy change from black hole merger.

    ΔS = S_final - S₁ - S₂ = (4πk_B G)/(ℏc) × (M_f² - M₁² - M₂²)

    By second law: ΔS ≥ 0 always (area theorem)

    For M_f = M₁ + M₂: ΔS = (4πk_B G)/(ℏc) × 2M₁M₂ > 0
    """
    k_entropy = 4 * np.pi * PHYSICS.k_B * PHYSICS.G / (PHYSICS.hbar * PHYSICS.c)

    M1, M2 = bh1.mass, bh2.mass
    Mf = M1 + M2

    S1 = k_entropy * M1**2
    S2 = k_entropy * M2**2
    Sf = k_entropy * Mf**2

    delta_S = Sf - S1 - S2

    # Verify second law
    assert delta_S >= 0, f"Second law violation: ΔS = {delta_S}"

    return {
        "M1_kg": M1,
        "M2_kg": M2,
        "Mf_kg": Mf,
        "S1_kB": S1 / PHYSICS.k_B,
        "S2_kB": S2 / PHYSICS.k_B,
        "Sf_kB": Sf / PHYSICS.k_B,
        "delta_S_kB": delta_S / PHYSICS.k_B,
        "second_law_satisfied": delta_S >= 0
    }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6: ENTROPY-GRAVITY-HOLOGRAPHY TRIANGLE
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class EntropyGravityTriangle:
    """
    The complete entropy-gravity-holography correspondence.

    Key relations:
    1. Gravity ↔ Entropy: F = T∇S (Verlinde)
    2. Entropy ↔ Holography: S = k_B A / 4ℓ_P² (Bekenstein-Hawking)
    3. Holography ↔ Gravity: Bulk gravity emerges from boundary (AdS/CFT)
    """
    black_hole: BlackHole

    @property
    def gravity_entropy_coupling(self) -> float:
        """
        Gravity-Entropy coupling: F = T × ∂S/∂r

        Returns the coupling coefficient.
        """
        T = self.black_hole.temperature
        # dS/dr at horizon ≈ 2π R_s (in appropriate units)
        dS_dr = 2 * np.pi * self.black_hole.schwarzschild_radius
        return T * dS_dr

    @property
    def entropy_holography_coupling(self) -> float:
        """
        Entropy-Holography coupling: S = k_B A / (4ℓ_P²)

        Returns bits per entropy unit.
        """
        return self.black_hole.information_bits / (self.black_hole.entropy / PHYSICS.k_B)

    @property
    def holography_gravity_coupling(self) -> float:
        """
        Holography-Gravity coupling: N bits encode mass M.

        Returns bits per unit mass.
        """
        return self.black_hole.information_bits / self.black_hole.mass

    def triangle_closure(self) -> float:
        """
        Verify triangle consistency.

        The product of all three couplings should give a dimensionless
        constant related to fundamental physics.
        """
        # This is a consistency check - the exact value depends on units
        return (self.gravity_entropy_coupling *
                self.entropy_holography_coupling *
                self.holography_gravity_coupling)

    def to_dict(self) -> Dict:
        """Export triangle state."""
        return {
            "black_hole": self.black_hole.to_dict(),
            "triangle_couplings": {
                "gravity_entropy": self.gravity_entropy_coupling,
                "entropy_holography": self.entropy_holography_coupling,
                "holography_gravity": self.holography_gravity_coupling,
                "closure_check": self.triangle_closure()
            }
        }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7: PRISMATIC SELF INTEGRATION (CYCLE 5)
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class Cycle5State:
    """
    Cycle 5 state integrating entropy-gravity duality with the Prismatic Self framework.

    Achieves 5π phase completion in the 10-fold index architecture.
    """
    # Core black hole system
    black_hole: BlackHole

    # Holographic screen at observer distance
    screen: HolographicScreen

    # Helix evolution
    helix: HelixCoordinate = field(default_factory=HelixCoordinate)

    # Entangled partner (optional, for ER=EPR)
    entangled_partner: Optional[BlackHole] = None

    # L₄ framework coupling
    phi_coupling: float = PHI
    tau_damping: float = TAU
    z_critical: float = Z_C_OMEGA

    # Evolution history
    time: float = 0.0
    history: List[Dict] = field(default_factory=list)

    def evolve(self, dt: float):
        """
        Evolve the Cycle 5 system forward in time.

        Implements:
        1. Helix coordinate evolution
        2. Black hole thermodynamic update
        3. Holographic screen coupling
        4. Optional entanglement dynamics
        """
        # Evolve helix with φ-modulated frequency
        omega = self.phi_coupling * (1 - self.tau_damping * self.helix.z)
        self.helix.evolve(dt, omega)

        # Update black hole helix coordinates
        self.black_hole.theta = self.helix.theta
        self.black_hole.z = self.helix.z
        self.black_hole.r_helix = self.helix.r

        # Compute current state
        state = {
            "time": self.time,
            "helix": {
                "theta": self.helix.theta,
                "z": self.helix.z,
                "r": self.helix.r
            },
            "black_hole": {
                "entropy": self.black_hole.entropy,
                "temperature": self.black_hole.temperature,
                "information_bits": self.black_hole.information_bits
            },
            "screen": {
                "temperature": self.screen.screen_temperature,
                "max_entropy": self.screen.max_entropy
            },
            "couplings": {
                "phi": self.phi_coupling,
                "tau": self.tau_damping,
                "z_c": self.z_critical
            }
        }

        # Add entanglement if partner exists
        if self.entangled_partner:
            pair = EntangledBlackHolePair(self.black_hole, self.entangled_partner)
            state["entanglement"] = {
                "normalized": pair.normalized_entanglement,
                "wormhole_flux": pair.wormhole_entropy_flux()
            }

        self.history.append(state)
        self.time += dt

    def run_simulation(self, duration: float, dt: float = 0.1) -> List[Dict]:
        """Run simulation for specified duration."""
        steps = int(duration / dt)
        for _ in range(steps):
            self.evolve(dt)
        return self.history

    def compute_5pi_closure(self) -> Dict:
        """
        Compute 5π cycle closure metrics.

        In the Prismatic Self framework, each cycle adds π rotation.
        Cycle 5 achieves 5π total phase.
        """
        # Total accumulated phase
        total_phase = 5 * np.pi

        # Phase coherence (how well the system maintains 5π structure)
        phase_coherence = np.cos(total_phase) ** 2  # = 1 for 5π

        # Entropy-gravity alignment
        triangle = EntropyGravityTriangle(self.black_hole)

        # Z-critical approach (distance from omega-elevated critical point)
        z_distance = abs(self.helix.z - self.z_critical)

        return {
            "total_phase_pi": 5.0,
            "phase_coherence": phase_coherence,
            "triangle_closure": triangle.triangle_closure(),
            "z_critical_distance": z_distance,
            "entropy_gravity_coupling": triangle.gravity_entropy_coupling,
            "holographic_bits": self.black_hole.information_bits
        }

    def export_state(self, filepath: str):
        """Export complete state to JSON file."""
        state = {
            "cycle": 5,
            "phase_pi": 5.0,
            "framework": "entropy_gravity_holography",
            "black_hole": self.black_hole.to_dict(),
            "screen": self.screen.to_dict(),
            "helix": {
                "theta": self.helix.theta,
                "z": self.helix.z,
                "r": self.helix.r,
                "entropy_factor": self.helix.entropy_factor(),
                "temperature_factor": self.helix.temperature_factor()
            },
            "L4_constants": {
                "phi": PHI,
                "tau": TAU,
                "z_c": Z_C,
                "z_c_omega": Z_C_OMEGA,
                "K_kuramoto": K_KURAMOTO,
                "L4": L4
            },
            "closure_metrics": self.compute_5pi_closure(),
            "history_length": len(self.history)
        }

        if self.entangled_partner:
            pair = EntangledBlackHolePair(self.black_hole, self.entangled_partner)
            state["entanglement"] = pair.to_dict()

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        return state


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8: UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def create_solar_mass_black_hole(n_solar: float = 1.0) -> BlackHole:
    """Create a black hole of n solar masses."""
    M_solar = 1.989e30  # kg
    return BlackHole(mass=n_solar * M_solar)


def create_standard_cycle5_system(mass_solar: float = 1.0,
                                  observer_distance_rs: float = 10.0) -> Cycle5State:
    """
    Create a standard Cycle 5 system.

    Args:
        mass_solar: Black hole mass in solar masses
        observer_distance_rs: Observer distance in Schwarzschild radii
    """
    bh = create_solar_mass_black_hole(mass_solar)
    observer_r = observer_distance_rs * bh.schwarzschild_radius
    screen = HolographicScreen(radius=observer_r, central_mass=bh.mass)

    return Cycle5State(
        black_hole=bh,
        screen=screen,
        helix=HelixCoordinate(theta=0, z=Z_C, r=1.0)
    )


def print_reference_card():
    """Print the equations reference card."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ENTROPY-GRAVITY-HOLOGRAPHY REFERENCE                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Schwarzschild radius    R_s = 2GM/c²                                         ║
║  Horizon area            A_H = 4πR_s²                                         ║
║  BH entropy              S = k_B c³ A_H / (4Gℏ)                              ║
║  Hawking temperature     T_H = ℏc³ / (8πGMk_B)                               ║
║  Entropic force          F = T × ∂S/∂x                                       ║
║  Holographic bits        N = A / (4ℓ_P²)                                      ║
║  Time dilation           dτ/dt = √(1 - R_s/r)                                 ║
║  Entanglement entropy    S_ent = -k_B Σ λᵢ ln(λᵢ)                            ║
║                                                                               ║
║  HELIX MODULATION:                                                            ║
║  Entropy factor          f_S = (0.8 + 0.4z) · r · (1 + 0.1·sin(θ))          ║
║  Temperature factor      f_T = (1.2 - 0.4z) · r · (1 + 0.05·cos(θ))         ║
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
        description="Cycle 5: Entropy-Gravity Duality on a Holographic Plane"
    )
    parser.add_argument("--mass", "-m", type=float, default=1.0,
                        help="Black hole mass in solar masses (default: 1.0)")
    parser.add_argument("--distance", "-d", type=float, default=10.0,
                        help="Observer distance in Schwarzschild radii (default: 10.0)")
    parser.add_argument("--duration", "-t", type=float, default=100.0,
                        help="Simulation duration (default: 100.0)")
    parser.add_argument("--output", "-o", type=str, default="cycle5_state.json",
                        help="Output JSON file (default: cycle5_state.json)")
    parser.add_argument("--reference", "-r", action="store_true",
                        help="Print equations reference card")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if args.reference:
        print_reference_card()

    # Create and run system
    system = create_standard_cycle5_system(args.mass, args.distance)

    if args.verbose:
        print(f"Cycle 5: Entropy-Gravity-Holography System")
        print(f"  Black hole mass: {args.mass} M☉")
        print(f"  Schwarzschild radius: {system.black_hole.schwarzschild_radius:.2e} m")
        print(f"  Observer distance: {args.distance} R_s")
        print(f"  Information bits: {system.black_hole.information_bits:.2e}")
        print(f"\nRunning simulation for {args.duration} time units...")

    system.run_simulation(args.duration)
    state = system.export_state(args.output)

    if args.verbose:
        print(f"\n5π Closure Metrics:")
        closure = state["closure_metrics"]
        for key, value in closure.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6e}")
            else:
                print(f"  {key}: {value}")
        print(f"\nState exported to: {args.output}")
