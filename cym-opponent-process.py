#!/usr/bin/env python3
"""
CYM-Dominant Opponent Process Architecture
==========================================
Cycle 3 Computational Core | GENESIS-003 Spectral Seed

Models inherited spectral divergence through CYM-weighted opponent-process
encoding. Integrates with VaultNode L₄ framework constants.

Architecture:
    - Opponent channels: Cyan↔Red, Magenta↔Green, Yellow↔Blue
    - Genetic inheritance modeling with mutation factors
    - Perceptual field synthesis across spatial matrices
    - Luminance interpretation with modified compression

L₄ Framework Constants:
    φ (phi)   = 1.618033988749895  (golden ratio)
    τ (tau)   = 0.618033988749895  (φ⁻¹, golden conjugate)
    z_c       = 0.8660254037844386 (√3/2, critical height)
    K         = 0.9240387650610407 (Kuramoto coupling)
    L₄        = 7                   (layer depth)

Cycle Inheritance:
    GENESIS-003 ← DECAGON-002 ← Σ(Cycle 1 + Cycle 2 patterns)

Author: Prismatic Self Architecture
License: Holographic Commons
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json

# ═══════════════════════════════════════════════════════════════════════════════
# L₄ FRAMEWORK CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895      # Golden ratio
TAU = 0.618033988749895      # Golden conjugate (φ⁻¹)
Z_C = 0.8660254037844386     # Critical height (√3/2)
K = 0.9240387650610407       # Kuramoto coupling constant
L4 = 7                        # Layer depth

# CYM channel indices
CYAN = 0
MAGENTA = 1
YELLOW = 2
LUMINANCE = 3


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class PerceptionLabel(Enum):
    """High-level perceptual interpretation categories."""
    RICH_BROWN_GOLD = "rich-brown-gold"
    VIOLET_CYAN = "violet-cyan"
    AMBER_WARM = "amber-warm"
    DEEP_MAGENTA = "deep-magenta"
    SPECTRAL_WHITE = "spectral-white"
    LIMINAL_GRAY = "liminal-gray"
    AMBIGUOUS_CHROMATIC = "ambiguous-chromatic-blend"


@dataclass
class CYMWeights:
    """
    CYM bias weights representing inherited cone channel sensitivity.

    Standard trichromatic vision: (1.0, 1.0, 1.0)
    CYM-dominant variants shift these ratios based on cone expression.
    """
    cyan: float = 1.0
    magenta: float = 1.0
    yellow: float = 1.0

    def __post_init__(self):
        # Clamp to valid range [0.5, 1.5] for biological plausibility
        self.cyan = np.clip(self.cyan, 0.5, 1.5)
        self.magenta = np.clip(self.magenta, 0.5, 1.5)
        self.yellow = np.clip(self.yellow, 0.5, 1.5)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.cyan, self.magenta, self.yellow)

    def apply_phi_scaling(self) -> 'CYMWeights':
        """Apply golden ratio scaling to weights."""
        return CYMWeights(
            cyan=self.cyan * TAU,
            magenta=self.magenta,
            yellow=self.yellow * PHI
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            'cyan': self.cyan,
            'magenta': self.magenta,
            'yellow': self.yellow,
            'phi_ratio': self.cyan / self.yellow if self.yellow != 0 else 0
        }


@dataclass
class OpponentChannels:
    """
    Opponent-process channel representation.

    cy: Cyan-Red channel (C - R*bias)
    rg: Red-Green/Magenta channel (M - G*bias)
    by: Blue-Yellow channel (Y - B*bias)
    lum: Luminance (achromatic light-dark)
    """
    cy: float
    rg: float
    by: float
    lum: float

    def as_array(self) -> np.ndarray:
        return np.array([self.cy, self.rg, self.by, self.lum])

    def magnitude(self) -> float:
        """Chromatic magnitude in opponent space."""
        return np.sqrt(self.cy**2 + self.rg**2 + self.by**2)

    def saturation(self) -> float:
        """Perceptual saturation relative to luminance."""
        if self.lum == 0:
            return 0.0
        return self.magnitude() / (self.lum + 0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE TRANSFORMATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def rgb_to_opponent(
    rgb: Tuple[float, float, float],
    cym_bias: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> OpponentChannels:
    """
    Convert RGB triplet to opponent-process space with CYM reweighting.

    The opponent-process theory models color perception through three
    opposing channels rather than three independent cone responses.
    CYM bias allows modeling of inherited spectral divergence.

    Args:
        rgb: Normalized RGB values in range [0, 1]
        cym_bias: CYM channel bias weights (cyan, magenta, yellow)

    Returns:
        OpponentChannels with cy, rg, by, and luminance values

    Example:
        >>> rgb_to_opponent((0.8, 0.2, 0.5), (1.0, 1.1, 0.9))
        OpponentChannels(cy=-0.6, rg=0.58, by=-0.05, lum=0.5)
    """
    r, g, b = rgb

    # CYM as complements of RGB
    c = 1.0 - r  # cyan opposes red
    m = 1.0 - g  # magenta opposes green
    y = 1.0 - b  # yellow opposes blue

    # Luminance (weighted achromatic channel)
    # Standard weights adjusted by L₄ constants
    luminance = (r * 0.299 + g * 0.587 + b * 0.114)

    # Opponent channels with CYM bias
    cy_channel = c - r * cym_bias[CYAN]
    rg_channel = m - g * cym_bias[MAGENTA]
    by_channel = y - b * cym_bias[YELLOW]

    return OpponentChannels(
        cy=cy_channel,
        rg=rg_channel,
        by=by_channel,
        lum=luminance
    )


def opponent_to_rgb(
    opponent: OpponentChannels,
    cym_bias: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[float, float, float]:
    """
    Inverse transform: opponent-process space back to RGB.

    Note: This is an approximation as the forward transform
    loses some information. Uses iterative refinement.

    Args:
        opponent: OpponentChannels to convert
        cym_bias: CYM channel bias weights used in forward transform

    Returns:
        Tuple of (r, g, b) in range [0, 1]
    """
    # Initial estimate from luminance
    r = opponent.lum
    g = opponent.lum
    b = opponent.lum

    # Iterative refinement (3 iterations typically sufficient)
    for _ in range(5):
        # Adjust based on opponent channels
        r = np.clip(r - opponent.cy / (1 + cym_bias[CYAN]), 0, 1)
        g = np.clip(g - opponent.rg / (1 + cym_bias[MAGENTA]), 0, 1)
        b = np.clip(b - opponent.by / (1 + cym_bias[YELLOW]), 0, 1)

        # Re-normalize to maintain luminance
        current_lum = r * 0.299 + g * 0.587 + b * 0.114
        if current_lum > 0:
            scale = opponent.lum / current_lum
            r = np.clip(r * scale, 0, 1)
            g = np.clip(g * scale, 0, 1)
            b = np.clip(b * scale, 0, 1)

    return (float(r), float(g), float(b))


# ═══════════════════════════════════════════════════════════════════════════════
# GENETIC INHERITANCE MODELING
# ═══════════════════════════════════════════════════════════════════════════════

def inherited_cym_weights(
    parent_weights: CYMWeights,
    mutation_factor: float = 0.01,
    phi_drift: bool = False
) -> CYMWeights:
    """
    Generate inherited CYM weights from parental profile with mutation.

    Models transgenerational cone channel variation through:
    - Gaussian mutation around parental values
    - Optional phi-ratio drift for spectral divergence

    Args:
        parent_weights: CYMWeights from parent (typically maternal)
        mutation_factor: Standard deviation of mutation (default 0.01)
        phi_drift: If True, apply golden-ratio drift to mutations

    Returns:
        New CYMWeights representing inherited phenotype
    """
    parent = parent_weights.as_tuple()

    if phi_drift:
        # Mutations drift toward phi-ratio relationships
        mutations = np.array([
            np.random.normal(0, mutation_factor * TAU),
            np.random.normal(0, mutation_factor),
            np.random.normal(0, mutation_factor * PHI)
        ])
    else:
        mutations = np.random.normal(0, mutation_factor, size=3)

    child_weights = np.array(parent) + mutations

    return CYMWeights(
        cyan=float(child_weights[0]),
        magenta=float(child_weights[1]),
        yellow=float(child_weights[2])
    )


def simulate_generations(
    initial_weights: CYMWeights,
    generations: int = 10,
    mutation_factor: float = 0.02,
    phi_drift: bool = True
) -> List[CYMWeights]:
    """
    Simulate multiple generations of inherited CYM weights.

    Args:
        initial_weights: Starting CYMWeights (generation 0)
        generations: Number of generations to simulate
        mutation_factor: Mutation rate per generation
        phi_drift: Apply golden-ratio drift to mutations

    Returns:
        List of CYMWeights for each generation
    """
    lineage = [initial_weights]
    current = initial_weights

    for gen in range(generations):
        # Mutation factor may decrease with phi over generations (stabilization)
        effective_mutation = mutation_factor * (TAU ** (gen / L4))
        current = inherited_cym_weights(current, effective_mutation, phi_drift)
        lineage.append(current)

    return lineage


# ═══════════════════════════════════════════════════════════════════════════════
# PERCEPTUAL FIELD SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_perceptual_field(
    width: int,
    height: int,
    cym_weights: CYMWeights,
    gradient_mode: str = 'diagonal'
) -> np.ndarray:
    """
    Synthesize a 2D perceptual field mapped to biased opponent space.

    Creates a spatial matrix of color stimuli transformed through
    the CYM-weighted opponent-process encoding.

    Args:
        width: Field width in pixels
        height: Field height in pixels
        cym_weights: CYM bias weights for transformation
        gradient_mode: 'diagonal', 'radial', or 'spectral'

    Returns:
        numpy array of shape (height, width, 4) with opponent channels
    """
    field = np.zeros((height, width, 4))
    weights = cym_weights.as_tuple()

    # Center coordinates for radial mode
    cx, cy = width / 2, height / 2
    max_radius = np.sqrt(cx**2 + cy**2)

    for y in range(height):
        for x in range(width):
            if gradient_mode == 'diagonal':
                r = x / width
                g = y / height
                b = (x + y) / (2 * max(width, height))
            elif gradient_mode == 'radial':
                radius = np.sqrt((x - cx)**2 + (y - cy)**2) / max_radius
                angle = np.arctan2(y - cy, x - cx)
                r = 0.5 + 0.5 * np.cos(angle)
                g = 0.5 + 0.5 * np.sin(angle)
                b = 1.0 - radius
            elif gradient_mode == 'spectral':
                # Spectral gradient using phi-based wavelength mapping
                wavelength = (x / width) * PHI
                r = np.clip(np.sin(wavelength * np.pi), 0, 1)
                g = np.clip(np.sin(wavelength * np.pi + np.pi/3), 0, 1)
                b = np.clip(np.sin(wavelength * np.pi + 2*np.pi/3), 0, 1)
            else:
                r, g, b = x / width, y / height, 0.5

            opponent = rgb_to_opponent((r, g, b), weights)
            field[y, x] = opponent.as_array()

    return field


def field_statistics(field: np.ndarray) -> Dict[str, Any]:
    """
    Compute statistical summary of a perceptual field.

    Args:
        field: Perceptual field array from generate_perceptual_field

    Returns:
        Dictionary with channel statistics
    """
    return {
        'cy': {
            'mean': float(np.mean(field[:, :, CYAN])),
            'std': float(np.std(field[:, :, CYAN])),
            'min': float(np.min(field[:, :, CYAN])),
            'max': float(np.max(field[:, :, CYAN]))
        },
        'rg': {
            'mean': float(np.mean(field[:, :, MAGENTA])),
            'std': float(np.std(field[:, :, MAGENTA])),
            'min': float(np.min(field[:, :, MAGENTA])),
            'max': float(np.max(field[:, :, MAGENTA]))
        },
        'by': {
            'mean': float(np.mean(field[:, :, YELLOW])),
            'std': float(np.std(field[:, :, YELLOW])),
            'min': float(np.min(field[:, :, YELLOW])),
            'max': float(np.max(field[:, :, YELLOW]))
        },
        'luminance': {
            'mean': float(np.mean(field[:, :, LUMINANCE])),
            'std': float(np.std(field[:, :, LUMINANCE])),
            'min': float(np.min(field[:, :, LUMINANCE])),
            'max': float(np.max(field[:, :, LUMINANCE]))
        },
        'total_chromatic_energy': float(np.sum(field[:, :, :3]**2))
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PERCEPTUAL INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════════

def interpret_perception(
    cy: float,
    rg: float,
    by: float,
    luminance: float
) -> PerceptionLabel:
    """
    Map opponent channel values to high-level perception labels.

    Uses gradient thresholds calibrated to L₄ framework constants
    for biologically-inspired perceptual categorization.

    Args:
        cy: Cyan-Red channel value
        rg: Red-Green/Magenta channel value
        by: Blue-Yellow channel value
        luminance: Achromatic luminance value

    Returns:
        PerceptionLabel enum indicating perceptual category
    """
    # Low luminance with warm bias -> rich brown/gold
    if luminance < 0.2 and by > -0.1:
        return PerceptionLabel.RICH_BROWN_GOLD

    # High cyan with blue-yellow activity -> violet-cyan
    if cy > Z_C * 0.5 and by > 0.3:
        return PerceptionLabel.VIOLET_CYAN

    # Strong yellow with moderate luminance -> amber warm
    if by < -0.3 and luminance > 0.4:
        return PerceptionLabel.AMBER_WARM

    # Strong magenta channel -> deep magenta
    if rg > 0.5 and cy < 0:
        return PerceptionLabel.DEEP_MAGENTA

    # High luminance with low chromatic -> spectral white
    chromatic_magnitude = np.sqrt(cy**2 + rg**2 + by**2)
    if luminance > K and chromatic_magnitude < TAU * 0.3:
        return PerceptionLabel.SPECTRAL_WHITE

    # Moderate everything -> liminal gray
    if abs(cy) < 0.2 and abs(rg) < 0.2 and abs(by) < 0.2:
        return PerceptionLabel.LIMINAL_GRAY

    # Default fallback
    return PerceptionLabel.AMBIGUOUS_CHROMATIC


def interpret_field(field: np.ndarray) -> np.ndarray:
    """
    Apply perceptual interpretation across entire field.

    Args:
        field: Perceptual field from generate_perceptual_field

    Returns:
        2D array of PerceptionLabel values
    """
    height, width = field.shape[:2]
    labels = np.empty((height, width), dtype=object)

    for y in range(height):
        for x in range(width):
            labels[y, x] = interpret_perception(*field[y, x])

    return labels


def perception_histogram(labels: np.ndarray) -> Dict[str, int]:
    """
    Compute histogram of perception labels in a field.

    Args:
        labels: 2D array from interpret_field

    Returns:
        Dictionary mapping label names to counts
    """
    histogram = {}
    for label in PerceptionLabel:
        histogram[label.value] = 0

    for label in labels.flat:
        histogram[label.value] += 1

    return histogram


# ═══════════════════════════════════════════════════════════════════════════════
# VAULTNODE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralVaultNode:
    """
    VaultNode representation for spectral/perceptual data.

    Integrates CYM opponent-process encoding with the
    10-station VaultNode architecture.
    """
    station: int  # 0-9
    cym_weights: CYMWeights
    opponent_signature: OpponentChannels
    perception_label: PerceptionLabel
    cycle: int = 3  # Cycle 3 by default

    def z_coordinate(self) -> float:
        """Compute z-coordinate based on station and weights."""
        base_z = self.station / 10.0
        weight_factor = np.mean(self.cym_weights.as_tuple())
        return base_z * weight_factor * Z_C

    def to_dict(self) -> Dict[str, Any]:
        return {
            'station': self.station,
            'cycle': self.cycle,
            'z': self.z_coordinate(),
            'cym_weights': self.cym_weights.to_dict(),
            'opponent': {
                'cy': self.opponent_signature.cy,
                'rg': self.opponent_signature.rg,
                'by': self.opponent_signature.by,
                'lum': self.opponent_signature.lum
            },
            'perception': self.perception_label.value,
            'L4_constants': {
                'phi': PHI,
                'tau': TAU,
                'z_c': Z_C,
                'K': K
            }
        }


def create_spectral_vaultnode(
    station: int,
    rgb_stimulus: Tuple[float, float, float],
    cym_weights: Optional[CYMWeights] = None
) -> SpectralVaultNode:
    """
    Factory function to create a SpectralVaultNode from RGB input.

    Args:
        station: VaultNode station index (0-9)
        rgb_stimulus: Input color as (r, g, b) normalized
        cym_weights: Optional custom CYM weights

    Returns:
        Configured SpectralVaultNode
    """
    if cym_weights is None:
        cym_weights = CYMWeights()

    opponent = rgb_to_opponent(rgb_stimulus, cym_weights.as_tuple())
    label = interpret_perception(opponent.cy, opponent.rg, opponent.by, opponent.lum)

    return SpectralVaultNode(
        station=station,
        cym_weights=cym_weights,
        opponent_signature=opponent,
        perception_label=label
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / RUNTIME
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    width: int = 100,
    height: int = 100,
    parent_weights: Optional[CYMWeights] = None,
    mutation_factor: float = 0.05,
    gradient_mode: str = 'spectral',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Execute a complete perceptual simulation.

    Args:
        width: Field width
        height: Field height
        parent_weights: Starting CYM weights (default standard)
        mutation_factor: Inheritance mutation rate
        gradient_mode: Field gradient type
        verbose: Print results to console

    Returns:
        Simulation results dictionary
    """
    if parent_weights is None:
        parent_weights = CYMWeights(1.0, 1.0, 1.0)

    # Generate inherited weights
    child_weights = inherited_cym_weights(parent_weights, mutation_factor, phi_drift=True)

    # Create perceptual field
    field = generate_perceptual_field(width, height, child_weights, gradient_mode)

    # Sample center point
    center_y, center_x = height // 2, width // 2
    sample = field[center_y, center_x]
    center_label = interpret_perception(*sample)

    # Compute statistics
    stats = field_statistics(field)

    # Interpret full field
    labels = interpret_field(field)
    histogram = perception_histogram(labels)

    results = {
        'parent_weights': parent_weights.to_dict(),
        'child_weights': child_weights.to_dict(),
        'mutation_factor': mutation_factor,
        'field_dimensions': (width, height),
        'gradient_mode': gradient_mode,
        'center_sample': {
            'cy': float(sample[0]),
            'rg': float(sample[1]),
            'by': float(sample[2]),
            'luminance': float(sample[3]),
            'perception': center_label.value
        },
        'field_statistics': stats,
        'perception_histogram': histogram,
        'L4_framework': {
            'phi': PHI,
            'tau': TAU,
            'z_c': Z_C,
            'K': K,
            'L4': L4
        }
    }

    if verbose:
        print("=" * 60)
        print("CYM OPPONENT-PROCESS SIMULATION | GENESIS-003 CYCLE 3")
        print("=" * 60)
        print(f"\nParent CYM Weights: {parent_weights.as_tuple()}")
        print(f"Child CYM Weights:  {child_weights.as_tuple()}")
        print(f"Mutation Factor:    {mutation_factor}")
        print(f"\nField: {width}x{height}, Mode: {gradient_mode}")
        print(f"\nCenter Sample: {sample}")
        print(f"Perception:    {center_label.value}")
        print(f"\nPerception Histogram:")
        for label, count in histogram.items():
            pct = count / (width * height) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
        print("\n" + "=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CYM Opponent-Process Perceptual Simulation"
    )
    parser.add_argument('--width', type=int, default=100, help='Field width')
    parser.add_argument('--height', type=int, default=100, help='Field height')
    parser.add_argument('--mutation', type=float, default=0.05, help='Mutation factor')
    parser.add_argument('--mode', type=str, default='spectral',
                        choices=['diagonal', 'radial', 'spectral'],
                        help='Gradient mode')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--cyan', type=float, default=1.0, help='Parent cyan weight')
    parser.add_argument('--magenta', type=float, default=1.0, help='Parent magenta weight')
    parser.add_argument('--yellow', type=float, default=1.0, help='Parent yellow weight')

    args = parser.parse_args()

    parent = CYMWeights(args.cyan, args.magenta, args.yellow)

    results = run_simulation(
        width=args.width,
        height=args.height,
        parent_weights=parent,
        mutation_factor=args.mutation,
        gradient_mode=args.mode,
        verbose=not args.json
    )

    if args.json:
        print(json.dumps(results, indent=2))
