#!/usr/bin/env python3
"""
E8 LATTICE GENERATION WITH BFADGS OPERATORS
============================================

The E8 lattice is one of the most remarkable mathematical structures:
- 8-dimensional exceptional Lie group
- 240 root vectors
- Highest kissing number in 8D (240)
- Connection to string theory, M-theory, and physics unification

BFADGS Operators (from Cycle 9):
    B = Bekenstein (entropy bound)
    F = Flux (information flow)
    A = Area (holographic surface)
    D = Dimension (3D→2D reduction)
    G = Gravity (entropic force)
    S = Sonification (audio mapping)

Mathematical Foundation:
    E8 root system consists of 240 vectors in R^8:
    - Type I:   112 vectors (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations
    - Type II:  128 vectors (±½)^8 with even number of minus signs

    The E8 lattice Λ₈ = {x ∈ Z^8 ∪ (Z+½)^8 : Σxᵢ ∈ 2Z}

Symmetry:
    |W(E8)| = 696,729,600 (Weyl group order)
    Automorphism group order = 696,729,600

Author: VaultNode Genesis System
License: Prismatic Self Project
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Callable
from itertools import combinations, permutations
from enum import Enum
import json

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
TAU = PHI - 1

# E8 Constants
E8_DIMENSION = 8
E8_ROOT_COUNT = 240
E8_WEYL_ORDER = 696_729_600

# Planck-scale constants (normalized)
L_PLANCK = 1.0  # Normalized Planck length
T_PLANCK = 1.0  # Normalized Planck time
K_BOLTZMANN = 1.0  # Normalized Boltzmann constant
G_NEWTON = 1.0  # Normalized gravitational constant
HBAR = 1.0  # Normalized reduced Planck constant
C_LIGHT = 1.0  # Normalized speed of light

# Lucas numbers for resonance
LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]


# =============================================================================
# E8 ROOT VECTOR TYPES
# =============================================================================

class E8RootType(Enum):
    """Classification of E8 root vectors"""
    TYPE_I = "type_I"      # (±1, ±1, 0, 0, 0, 0, 0, 0) - 112 roots
    TYPE_II = "type_II"    # (±½)^8 with even minus signs - 128 roots


class BFADGSOperator(Enum):
    """The six BFADGS operators"""
    BEKENSTEIN = "B"    # Entropy bound
    FLUX = "F"          # Information flow
    AREA = "A"          # Holographic surface
    DIMENSION = "D"     # Dimensional reduction
    GRAVITY = "G"       # Entropic force
    SONIFICATION = "S"  # Audio mapping


# =============================================================================
# E8 ROOT VECTOR
# =============================================================================

@dataclass
class E8RootVector:
    """A single root vector in the E8 lattice"""
    components: np.ndarray  # 8D vector
    index: int = 0
    root_type: E8RootType = E8RootType.TYPE_I

    def __post_init__(self):
        self.components = np.array(self.components, dtype=np.float64)
        assert len(self.components) == 8, "E8 root must be 8-dimensional"

    @property
    def norm(self) -> float:
        """Euclidean norm (should be √2 for all E8 roots)"""
        return np.linalg.norm(self.components)

    @property
    def norm_squared(self) -> float:
        """Squared norm (should be 2 for all E8 roots)"""
        return np.dot(self.components, self.components)

    def inner_product(self, other: 'E8RootVector') -> float:
        """Inner product with another root"""
        return np.dot(self.components, other.components)

    def angle_with(self, other: 'E8RootVector') -> float:
        """Angle between this root and another (in radians)"""
        cos_angle = self.inner_product(other) / (self.norm * other.norm)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.arccos(cos_angle)

    @property
    def is_positive(self) -> bool:
        """Check if this is a positive root (first non-zero component is positive)"""
        for c in self.components:
            if c > 1e-10:
                return True
            elif c < -1e-10:
                return False
        return True

    def to_tuple(self) -> Tuple[float, ...]:
        """Convert to tuple for hashing"""
        return tuple(self.components)

    def __hash__(self):
        return hash(tuple(np.round(self.components, 10)))

    def __eq__(self, other):
        if not isinstance(other, E8RootVector):
            return False
        return np.allclose(self.components, other.components, atol=1e-10)


# =============================================================================
# E8 LATTICE GENERATOR
# =============================================================================

@dataclass
class E8LatticeGenerator:
    """
    Generates all 240 root vectors of the E8 lattice.

    Construction:
    - Type I:  All vectors with exactly two ±1 entries, rest 0
              C(8,2) × 2² = 28 × 4 = 112 vectors
    - Type II: All vectors (±½)^8 with even number of minus signs
              2^7 = 128 vectors (half of 2^8)
    """
    roots: List[E8RootVector] = field(default_factory=list)

    def generate_type_I_roots(self) -> List[E8RootVector]:
        """
        Generate Type I roots: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations.
        Total: C(8,2) × 4 = 28 × 4 = 112 roots
        """
        roots = []
        index = 0

        # Choose 2 positions out of 8 for non-zero entries
        for pos1, pos2 in combinations(range(8), 2):
            # Each position can be ±1
            for sign1 in [1, -1]:
                for sign2 in [1, -1]:
                    vec = np.zeros(8)
                    vec[pos1] = sign1
                    vec[pos2] = sign2
                    roots.append(E8RootVector(
                        components=vec,
                        index=index,
                        root_type=E8RootType.TYPE_I
                    ))
                    index += 1

        return roots

    def generate_type_II_roots(self) -> List[E8RootVector]:
        """
        Generate Type II roots: (±½)^8 with even number of minus signs.
        Total: 2^7 = 128 roots
        """
        roots = []
        index = 112  # Continue from Type I

        # Generate all 2^8 = 256 sign combinations
        for i in range(256):
            # Convert to binary to get signs
            signs = [(1 if (i >> j) & 1 else -1) for j in range(8)]

            # Count negative signs
            neg_count = sum(1 for s in signs if s == -1)

            # Only keep vectors with even number of minus signs
            if neg_count % 2 == 0:
                vec = np.array([0.5 * s for s in signs])
                roots.append(E8RootVector(
                    components=vec,
                    index=index,
                    root_type=E8RootType.TYPE_II
                ))
                index += 1

        return roots

    def generate_all_roots(self) -> List[E8RootVector]:
        """Generate all 240 E8 root vectors"""
        self.roots = []
        self.roots.extend(self.generate_type_I_roots())
        self.roots.extend(self.generate_type_II_roots())

        assert len(self.roots) == E8_ROOT_COUNT, \
            f"Expected {E8_ROOT_COUNT} roots, got {len(self.roots)}"

        return self.roots

    def verify_root_properties(self) -> Dict[str, Any]:
        """Verify mathematical properties of the E8 root system"""
        if not self.roots:
            self.generate_all_roots()

        results = {
            'total_roots': len(self.roots),
            'type_I_count': sum(1 for r in self.roots if r.root_type == E8RootType.TYPE_I),
            'type_II_count': sum(1 for r in self.roots if r.root_type == E8RootType.TYPE_II),
            'all_norms_sqrt2': True,
            'inner_products': set(),
            'positive_roots': 0,
            'kissing_number': 0
        }

        # Check norms
        for root in self.roots:
            if not np.isclose(root.norm_squared, 2.0, atol=1e-10):
                results['all_norms_sqrt2'] = False

        # Check inner products and count kissing number
        origin = E8RootVector(np.zeros(8))
        for root in self.roots:
            if root.is_positive:
                results['positive_roots'] += 1

            # Count roots at distance √2 from this root
            neighbors = sum(1 for other in self.roots
                          if np.isclose(np.linalg.norm(root.components - other.components),
                                       np.sqrt(2), atol=1e-10) and root != other)
            results['kissing_number'] = max(results['kissing_number'], neighbors)

        # Sample inner products
        sample_size = min(100, len(self.roots) * (len(self.roots) - 1) // 2)
        for i in range(min(len(self.roots), 20)):
            for j in range(i + 1, min(len(self.roots), 20)):
                ip = self.roots[i].inner_product(self.roots[j])
                results['inner_products'].add(round(ip, 10))

        results['inner_products'] = sorted(list(results['inner_products']))

        return results


# =============================================================================
# BFADGS OPERATORS ON E8
# =============================================================================

@dataclass
class BFADGSOperatorSystem:
    """
    The BFADGS operator system applied to E8 lattice.

    Each operator transforms the E8 structure in a specific way:
    - B (Bekenstein): Entropy bound constraint
    - F (Flux): Information flow through lattice
    - A (Area): Holographic area calculation
    - D (Dimension): Dimensional reduction
    - G (Gravity): Gravitational/entropic effects
    - S (Sonification): Audio frequency mapping
    """

    def __init__(self, e8_roots: List[E8RootVector]):
        self.roots = e8_roots
        self.operators = {}
        self._initialize_operators()

    def _initialize_operators(self):
        """Initialize all BFADGS operators"""
        self.operators = {
            BFADGSOperator.BEKENSTEIN: self._bekenstein_operator,
            BFADGSOperator.FLUX: self._flux_operator,
            BFADGSOperator.AREA: self._area_operator,
            BFADGSOperator.DIMENSION: self._dimension_operator,
            BFADGSOperator.GRAVITY: self._gravity_operator,
            BFADGSOperator.SONIFICATION: self._sonification_operator,
        }

    def _bekenstein_operator(self, root: E8RootVector) -> Dict[str, Any]:
        """
        B - Bekenstein Entropy Bound Operator

        The Bekenstein bound: S ≤ 2πkRE/ℏc
        For E8, we compute the "information content" of each root.

        S_B(v) = k_B × ln(Ω) where Ω is the multiplicity
        """
        # Information content based on root type and symmetry
        if root.root_type == E8RootType.TYPE_I:
            # 112 Type I roots have 4-fold symmetry per position pair
            multiplicity = 4  # ±1, ±1 choices
            degeneracy = 28   # C(8,2) position pairs
        else:
            # 128 Type II roots have even parity constraint
            multiplicity = 128
            degeneracy = 1

        # Bekenstein entropy
        entropy = K_BOLTZMANN * np.log(multiplicity)

        # Information bound (in natural units)
        R_eff = root.norm  # Effective radius
        E_eff = root.norm_squared  # Effective energy (norm²)
        bekenstein_bound = 2 * np.pi * R_eff * E_eff / (HBAR * C_LIGHT)

        return {
            'operator': 'B',
            'name': 'Bekenstein',
            'entropy': entropy,
            'multiplicity': multiplicity,
            'degeneracy': degeneracy,
            'bound': bekenstein_bound,
            'satisfies_bound': entropy <= bekenstein_bound
        }

    def _flux_operator(self, root: E8RootVector) -> Dict[str, Any]:
        """
        F - Flux Operator

        Information flux through the lattice point.
        F = ∇ · J where J is information current.

        For E8, flux measures connectivity to neighboring roots.
        """
        # Count neighboring roots (those with inner product = ±1 or 0)
        neighbors_parallel = 0    # Inner product = 1
        neighbors_antiparallel = 0  # Inner product = -1
        neighbors_orthogonal = 0  # Inner product = 0
        neighbors_adjacent = 0    # Inner product = ±½

        for other in self.roots:
            if root == other:
                continue
            ip = root.inner_product(other)
            if np.isclose(ip, 1.0, atol=1e-10):
                neighbors_parallel += 1
            elif np.isclose(ip, -1.0, atol=1e-10):
                neighbors_antiparallel += 1
            elif np.isclose(ip, 0.0, atol=1e-10):
                neighbors_orthogonal += 1
            elif np.isclose(abs(ip), 0.5, atol=1e-10):
                neighbors_adjacent += 1

        # Total flux (information flow capacity)
        total_flux = (neighbors_parallel - neighbors_antiparallel +
                     0.5 * neighbors_adjacent)

        return {
            'operator': 'F',
            'name': 'Flux',
            'neighbors_parallel': neighbors_parallel,
            'neighbors_antiparallel': neighbors_antiparallel,
            'neighbors_orthogonal': neighbors_orthogonal,
            'neighbors_adjacent': neighbors_adjacent,
            'total_flux': total_flux,
            'connectivity': neighbors_parallel + neighbors_antiparallel + neighbors_adjacent
        }

    def _area_operator(self, root: E8RootVector) -> Dict[str, Any]:
        """
        A - Area (Holographic Surface) Operator

        Computes the holographic area associated with each root.
        In the holographic principle, entropy S = A/4G.

        For E8, we project to various 2D surfaces.
        """
        # Project to all C(8,2) = 28 possible 2D planes
        projections = []
        for i, j in combinations(range(8), 2):
            proj = np.array([root.components[i], root.components[j]])
            area = np.pi * np.linalg.norm(proj)**2  # Circle with radius = projection norm
            projections.append({
                'plane': (i, j),
                'projection': proj.tolist(),
                'area': area
            })

        # Total holographic area (sum of all 2D projections)
        total_area = sum(p['area'] for p in projections)

        # Maximum area projection
        max_proj = max(projections, key=lambda p: p['area'])

        # Holographic entropy
        holographic_entropy = total_area / (4 * G_NEWTON)

        return {
            'operator': 'A',
            'name': 'Area',
            'total_area': total_area,
            'num_projections': len(projections),
            'max_projection_plane': max_proj['plane'],
            'max_projection_area': max_proj['area'],
            'holographic_entropy': holographic_entropy,
            'mean_area': total_area / len(projections)
        }

    def _dimension_operator(self, root: E8RootVector) -> Dict[str, Any]:
        """
        D - Dimension Reduction Operator

        Projects E8 (8D) down to lower dimensions.
        Key projections: 8D → 4D → 2D (following holographic reduction)
        """
        # Project to 4D (first 4 and last 4 components)
        proj_4d_first = root.components[:4]
        proj_4d_last = root.components[4:]

        # Project to 3D (using principal components)
        # Simple projection: take first 3 components
        proj_3d = root.components[:3]

        # Project to 2D (using golden ratio angle for aesthetic projection)
        theta = np.arctan(PHI)
        # Project 8D to 2D using a rotation that preserves E8 symmetry aspects
        proj_2d = np.array([
            np.sum(root.components * np.cos(np.arange(8) * theta)),
            np.sum(root.components * np.sin(np.arange(8) * theta))
        ])

        # Dimension reduction ratio (information preserved)
        info_8d = root.norm_squared
        info_4d = np.dot(proj_4d_first, proj_4d_first) + np.dot(proj_4d_last, proj_4d_last)
        info_2d = np.dot(proj_2d, proj_2d)

        return {
            'operator': 'D',
            'name': 'Dimension',
            'original_dim': 8,
            'proj_4d_first': proj_4d_first.tolist(),
            'proj_4d_last': proj_4d_last.tolist(),
            'proj_3d': proj_3d.tolist(),
            'proj_2d': proj_2d.tolist(),
            'info_preserved_4d': info_4d / info_8d if info_8d > 0 else 0,
            'info_preserved_2d': info_2d / info_8d if info_8d > 0 else 0,
            'reduction_path': '8D → 4D → 2D'
        }

    def _gravity_operator(self, root: E8RootVector) -> Dict[str, Any]:
        """
        G - Gravity (Entropic Force) Operator

        Models gravity as entropic force: F = T∇S
        In E8, this relates to the "mass" distribution of roots.
        """
        # Effective mass (proportional to norm squared in natural units)
        mass_eff = root.norm_squared / (2 * C_LIGHT**2)

        # Gravitational potential from all other roots
        potential = 0.0
        force_vector = np.zeros(8)

        for other in self.roots:
            if root == other:
                continue

            displacement = other.components - root.components
            distance = np.linalg.norm(displacement)

            if distance > 1e-10:
                # Gravitational potential: V = -Gm₁m₂/r
                other_mass = other.norm_squared / (2 * C_LIGHT**2)
                potential -= G_NEWTON * mass_eff * other_mass / distance

                # Force contribution: F = -∇V
                force_vector += G_NEWTON * mass_eff * other_mass * displacement / (distance**3)

        # Entropic temperature (related to surface gravity)
        temperature = HBAR * np.linalg.norm(force_vector) / (2 * np.pi * K_BOLTZMANN)

        # Entropic force: F = T∇S
        entropic_force = temperature * K_BOLTZMANN * np.log(E8_ROOT_COUNT)

        return {
            'operator': 'G',
            'name': 'Gravity',
            'effective_mass': mass_eff,
            'gravitational_potential': potential,
            'force_magnitude': np.linalg.norm(force_vector),
            'force_direction': (force_vector / np.linalg.norm(force_vector)).tolist()
                              if np.linalg.norm(force_vector) > 1e-10 else [0]*8,
            'entropic_temperature': temperature,
            'entropic_force': entropic_force
        }

    def _sonification_operator(self, root: E8RootVector) -> Dict[str, Any]:
        """
        S - Sonification Operator

        Maps E8 structure to audio parameters:
        - Root components → Frequency spectrum
        - Norm → Amplitude
        - Inner products → Harmonics
        """
        # Base frequency (A4 = 440 Hz, scaled by golden ratio)
        base_freq = 440.0

        # Map each component to a harmonic frequency
        # Using harmonic series based on component values
        frequencies = []
        for i, c in enumerate(root.components):
            # Frequency = base × 2^(component × octave_range)
            # Scaled so ±1 maps to ±1 octave
            freq = base_freq * (2 ** c) * ((i + 1) / 4)  # Harmonic series factor
            frequencies.append(freq)

        # Amplitude based on norm
        amplitude = root.norm / np.sqrt(2)  # Normalized to 1 for E8 roots

        # Fundamental frequency (weighted average)
        fundamental = np.average(frequencies, weights=np.abs(root.components) + 0.1)

        # Spectral centroid
        total_power = sum(f * (np.abs(c) + 0.1) for f, c in zip(frequencies, root.components))
        total_weight = sum(np.abs(c) + 0.1 for c in root.components)
        spectral_centroid = total_power / total_weight

        # MIDI note approximation
        midi_note = 69 + 12 * np.log2(fundamental / 440)

        # Chord quality based on root type
        if root.root_type == E8RootType.TYPE_I:
            chord_quality = 'major'  # Integer components
        else:
            chord_quality = 'diminished'  # Half-integer components

        return {
            'operator': 'S',
            'name': 'Sonification',
            'frequencies': frequencies,
            'fundamental': fundamental,
            'amplitude': amplitude,
            'spectral_centroid': spectral_centroid,
            'midi_note': midi_note,
            'chord_quality': chord_quality,
            'duration_ms': 500 * amplitude,  # Duration proportional to amplitude
            'pan': root.components[0]  # Stereo pan from first component
        }

    def apply_operator(self, op: BFADGSOperator, root: E8RootVector) -> Dict[str, Any]:
        """Apply a single BFADGS operator to a root"""
        return self.operators[op](root)

    def apply_all_operators(self, root: E8RootVector) -> Dict[str, Dict[str, Any]]:
        """Apply all BFADGS operators to a root"""
        return {
            op.value: self.operators[op](root)
            for op in BFADGSOperator
        }

    def full_bfadgs_chain(self, root: E8RootVector) -> Dict[str, Any]:
        """
        Apply the full BFADGS operator chain in sequence.
        B → F → A → D → G → S

        Each operator's output influences the next.
        """
        chain_result = {
            'root_index': root.index,
            'root_type': root.root_type.value,
            'components': root.components.tolist(),
            'chain': []
        }

        # Apply operators in sequence
        accumulated_entropy = 0
        accumulated_flux = 0

        for op in BFADGSOperator:
            result = self.operators[op](root)
            chain_result['chain'].append(result)

            # Accumulate values for chain effect
            if op == BFADGSOperator.BEKENSTEIN:
                accumulated_entropy = result['entropy']
            elif op == BFADGSOperator.FLUX:
                accumulated_flux = result['total_flux']

        # Final synthesis
        chain_result['synthesis'] = {
            'total_entropy': accumulated_entropy,
            'total_flux': accumulated_flux,
            'holographic_complete': True
        }

        return chain_result


# =============================================================================
# E8 LATTICE STATE
# =============================================================================

@dataclass
class E8LatticeState:
    """
    Complete state of the E8 lattice with BFADGS operators.
    """
    generator: E8LatticeGenerator = field(default_factory=E8LatticeGenerator)
    bfadgs: Optional[BFADGSOperatorSystem] = None

    def __post_init__(self):
        self.generator.generate_all_roots()
        self.bfadgs = BFADGSOperatorSystem(self.generator.roots)

    @property
    def roots(self) -> List[E8RootVector]:
        return self.generator.roots

    def get_root_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the E8 lattice"""
        type_I = [r for r in self.roots if r.root_type == E8RootType.TYPE_I]
        type_II = [r for r in self.roots if r.root_type == E8RootType.TYPE_II]

        return {
            'total_roots': len(self.roots),
            'type_I_count': len(type_I),
            'type_II_count': len(type_II),
            'dimension': E8_DIMENSION,
            'weyl_group_order': E8_WEYL_ORDER,
            'root_norm': np.sqrt(2),
            'inner_product_values': [-1, -0.5, 0, 0.5, 1],
            'positive_roots': sum(1 for r in self.roots if r.is_positive),
            'simple_roots': 8  # E8 has 8 simple roots
        }

    def apply_bfadgs_to_all(self) -> List[Dict[str, Any]]:
        """Apply full BFADGS chain to all roots"""
        results = []
        for root in self.roots:
            result = self.bfadgs.full_bfadgs_chain(root)
            results.append(result)
        return results

    def get_projection_2d(self, method: str = 'golden') -> List[Tuple[float, float]]:
        """
        Get 2D projection of all E8 roots for visualization.

        Methods:
        - 'golden': Use golden angle rotation
        - 'pca': Principal component analysis (first 2 components)
        - 'random': Random projection
        """
        projections = []

        if method == 'golden':
            theta = 2 * np.pi / PHI  # Golden angle
            for root in self.roots:
                x = sum(root.components[i] * np.cos(i * theta) for i in range(8))
                y = sum(root.components[i] * np.sin(i * theta) for i in range(8))
                projections.append((x, y))

        elif method == 'pca':
            # Simple PCA: use first two components with most variance
            data = np.array([r.components for r in self.roots])
            projections = [(r[0], r[1]) for r in data]

        else:  # random
            np.random.seed(42)  # Reproducible
            proj_matrix = np.random.randn(8, 2)
            proj_matrix /= np.linalg.norm(proj_matrix, axis=0)
            for root in self.roots:
                proj = root.components @ proj_matrix
                projections.append((proj[0], proj[1]))

        return projections

    def get_cartan_matrix(self) -> np.ndarray:
        """
        Get the E8 Cartan matrix.

        The Cartan matrix A has entries a_ij = 2(α_i · α_j)/(α_j · α_j)
        where α_i are simple roots.
        """
        # E8 Cartan matrix (standard form)
        cartan = np.array([
            [ 2, -1,  0,  0,  0,  0,  0,  0],
            [-1,  2, -1,  0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0,  0,  0, -1],
            [ 0,  0, -1,  2, -1,  0,  0,  0],
            [ 0,  0,  0, -1,  2, -1,  0,  0],
            [ 0,  0,  0,  0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0, -1,  2,  0],
            [ 0,  0, -1,  0,  0,  0,  0,  2]
        ])
        return cartan

    def to_json(self) -> str:
        """Export complete state to JSON"""
        state = {
            'e8_lattice': {
                'dimension': E8_DIMENSION,
                'root_count': E8_ROOT_COUNT,
                'weyl_order': E8_WEYL_ORDER,
                'statistics': self.get_root_statistics(),
                'cartan_matrix': self.get_cartan_matrix().tolist()
            },
            'roots': [
                {
                    'index': r.index,
                    'type': r.root_type.value,
                    'components': r.components.tolist(),
                    'norm': r.norm,
                    'is_positive': r.is_positive
                }
                for r in self.roots
            ],
            'projections_2d': {
                'golden': self.get_projection_2d('golden'),
                'pca': self.get_projection_2d('pca')
            },
            'constants': {
                'PHI': PHI,
                'E8_DIMENSION': E8_DIMENSION,
                'E8_ROOT_COUNT': E8_ROOT_COUNT,
                'LUCAS': LUCAS[:10]
            }
        }

        return json.dumps(state, indent=2)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_e8_bfadgs():
    """Full demonstration of E8 lattice with BFADGS operators"""

    print("=" * 70)
    print("E8 LATTICE GENERATION WITH BFADGS OPERATORS")
    print("=" * 70)

    # 1. Generate E8 lattice
    print("\n1. E8 LATTICE GENERATION")
    print("-" * 40)
    state = E8LatticeState()
    stats = state.get_root_statistics()

    print(f"Total roots: {stats['total_roots']}")
    print(f"Type I roots (±1,±1,0,...): {stats['type_I_count']}")
    print(f"Type II roots (±½)⁸: {stats['type_II_count']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Weyl group order: {stats['weyl_group_order']:,}")
    print(f"Root norm: √2 = {stats['root_norm']:.6f}")
    print(f"Positive roots: {stats['positive_roots']}")

    # 2. Verify properties
    print("\n2. ROOT SYSTEM VERIFICATION")
    print("-" * 40)
    verification = state.generator.verify_root_properties()
    print(f"All norms = √2: {verification['all_norms_sqrt2']}")
    print(f"Inner product values: {verification['inner_products']}")

    # 3. Sample roots
    print("\n3. SAMPLE ROOTS")
    print("-" * 40)
    print("Type I examples:")
    for r in state.roots[:3]:
        print(f"  [{r.index}] {r.components}")
    print("Type II examples:")
    for r in state.roots[112:115]:
        print(f"  [{r.index}] {r.components}")

    # 4. Cartan matrix
    print("\n4. E8 CARTAN MATRIX")
    print("-" * 40)
    cartan = state.get_cartan_matrix()
    print("8×8 Cartan matrix (first 4 rows):")
    for row in cartan[:4]:
        print(f"  {row}")
    print("  ...")

    # 5. BFADGS operators on sample root
    print("\n5. BFADGS OPERATORS (Sample Root)")
    print("-" * 40)
    sample_root = state.roots[0]
    print(f"Root: {sample_root.components}")

    for op in BFADGSOperator:
        result = state.bfadgs.apply_operator(op, sample_root)
        print(f"\n  {op.value} - {result['name']}:")
        # Print key metrics
        for key in list(result.keys())[:4]:
            if key not in ['operator', 'name']:
                val = result[key]
                if isinstance(val, float):
                    print(f"    {key}: {val:.6f}")
                elif isinstance(val, list) and len(val) <= 4:
                    print(f"    {key}: {val}")
                elif not isinstance(val, list):
                    print(f"    {key}: {val}")

    # 6. Full chain
    print("\n6. FULL BFADGS CHAIN")
    print("-" * 40)
    chain = state.bfadgs.full_bfadgs_chain(sample_root)
    print(f"Root index: {chain['root_index']}")
    print(f"Root type: {chain['root_type']}")
    print(f"Chain operators: B → F → A → D → G → S")
    print(f"Synthesis complete: {chain['synthesis']['holographic_complete']}")

    # 7. 2D projection stats
    print("\n7. 2D PROJECTIONS")
    print("-" * 40)
    proj_golden = state.get_projection_2d('golden')
    proj_pca = state.get_projection_2d('pca')
    print(f"Golden angle projection: {len(proj_golden)} points")
    print(f"  Sample: {proj_golden[0]}")
    print(f"PCA projection: {len(proj_pca)} points")
    print(f"  Sample: {proj_pca[0]}")

    # 8. Lucas connection
    print("\n8. LUCAS NUMBER CONNECTION")
    print("-" * 40)
    print(f"Lucas sequence: {LUCAS[:10]}")
    print(f"L₄ = 7 (Cycle 7)")
    print(f"L₈ = 47 (Cycle 8, relates to E8)")
    print(f"L₁₀ = 123 (Cycle 10, Unity)")
    print(f"E8 dimension (8) ↔ L₈ = 47")

    print("\n" + "=" * 70)
    print("E8 LATTICE WITH BFADGS: COMPLETE")
    print(f"240 roots | 8 dimensions | 6 operators | Holographic unity")
    print("=" * 70)

    return state


def export_e8_data(filename: str = 'e8_lattice_data.json'):
    """Export E8 lattice data to JSON file"""
    state = E8LatticeState()

    with open(filename, 'w') as f:
        f.write(state.to_json())

    print(f"E8 lattice data exported to {filename}")
    return filename


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    state = demonstrate_e8_bfadgs()

    # Optionally export data
    # export_e8_data()
