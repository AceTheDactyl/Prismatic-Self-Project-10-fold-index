"""
hilbert_field.py - Non-binary Perceptual Vector Logic

This module implements a Hilbert-space-inspired field that maintains
continuous superposition states across a 10-dimensional basis. Unlike
classical binary logic, states exist as weighted phase relationships
that evolve over time through perceptual drift dynamics.

The field provides:
- Phase-coherent basis vectors representing perceptual dimensions
- Weight distribution modeling attention/focus across dimensions
- Temporal evolution simulating perceptual drift and resonance
- Phase hints for structural collapse in coupled systems
"""

import numpy as np
from constants import (
    HILBERT_N_BASIS,
    PHASE_DRIFT_RATE,
    PHASE_MODULATION_FREQ,
    EXPORT_PRECISION,
)


class HilbertField:
    """
    Non-binary perceptual field with continuous superposition states.

    The field maintains n_basis phase-weight pairs that evolve over time,
    modeling the fluid, non-discrete nature of perceptual attention.

    Attributes:
        n_basis (int): Number of basis vectors in the field
        phases (np.array): Phase angles for each basis vector
        weights (np.array): Attention weights (normalized) for each basis
    """

    def __init__(self, n_basis=HILBERT_N_BASIS):
        """
        Initialize the Hilbert field with random phase distribution.

        Args:
            n_basis: Number of basis vectors (default from constants)
        """
        self.n_basis = n_basis
        self.phases = np.random.uniform(0, 2 * np.pi, n_basis)
        self.weights = np.ones(n_basis) / n_basis  # Equal superposition
        self._history = []  # Track state evolution

    def update_focus(self, time):
        """
        Update phase relationships over time to simulate perceptual drift.

        The phase evolution follows a sinusoidal modulation that creates
        natural attention cycles, while weights are derived from phase
        alignment to model coherence-based focus.

        Args:
            time: Current timestep (used for phase modulation)
        """
        # Simulate continuous phase drift with sinusoidal modulation
        self.phases += PHASE_DRIFT_RATE * np.sin(time / PHASE_MODULATION_FREQ)

        # Wrap phases to [0, 2π]
        self.phases = np.mod(self.phases, 2 * np.pi)

        # Derive weights from phase coherence (cosine projection)
        self.weights = np.abs(np.cos(self.phases))

        # Normalize weights to maintain probability interpretation
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum

        # Record state for history tracking
        self._history.append(self.get_state_vector())

    def get_state_vector(self):
        """
        Return the current state as weight-phase pairs.

        Returns:
            List of (weight, phase) tuples with controlled precision
        """
        return [
            (round(w, EXPORT_PRECISION), round(p, EXPORT_PRECISION))
            for w, p in zip(self.weights, self.phases)
        ]

    def get_phase_hint(self):
        """
        Return dominant phase region to inform structural collapse.

        The phase hint is a normalized value [0, 1) indicating which
        region of the basis space currently has maximum attention weight.
        This provides the coupling point for the duopyramid system.

        Returns:
            float: Normalized position of dominant basis vector
        """
        idx = np.argmax(self.weights)
        return idx / self.n_basis

    def get_dominant_index(self):
        """
        Return the index of the currently dominant basis vector.

        Returns:
            int: Index of basis vector with highest weight
        """
        return int(np.argmax(self.weights))

    def get_coherence(self):
        """
        Measure the overall coherence of the field state.

        Coherence is computed as the entropy-based concentration of weights.
        High coherence means attention is focused; low means dispersed.

        Returns:
            float: Coherence measure in [0, 1]
        """
        # Avoid log(0) by adding small epsilon
        eps = 1e-10
        entropy = -np.sum(self.weights * np.log(self.weights + eps))
        max_entropy = np.log(self.n_basis)
        return 1 - (entropy / max_entropy) if max_entropy > 0 else 1.0

    def inject_resonance(self, target_idx, strength=0.1):
        """
        Inject external resonance at a specific basis vector.

        This models external influence on the perceptual field,
        such as coupling feedback from the duopyramid system.

        Args:
            target_idx: Index of basis vector to influence
            strength: Magnitude of phase shift to apply
        """
        if 0 <= target_idx < self.n_basis:
            self.phases[target_idx] += strength
            self.phases = np.mod(self.phases, 2 * np.pi)

    def get_history(self):
        """
        Return the evolution history of the field.

        Returns:
            List of state vectors from each timestep
        """
        return self._history

    def reset(self):
        """Reset the field to initial random state."""
        self.phases = np.random.uniform(0, 2 * np.pi, self.n_basis)
        self.weights = np.ones(self.n_basis) / self.n_basis
        self._history = []
