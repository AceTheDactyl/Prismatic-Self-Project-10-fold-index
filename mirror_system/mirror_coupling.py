"""
mirror_coupling.py - Bridge Between Systems A and B

This module implements the bidirectional coupling between the Hilbert field
(continuous perceptual space) and the Duopyramid system (discrete symbolic
structure). The mirror coupling creates mutual modulation where states in
one system influence the evolution of the other.

The coupling provides:
- Bidirectional state transfer between continuous and discrete domains
- Resonance detection for coherent state alignment
- Modulation strength based on coupling coefficient
- State synchronization tracking for stability analysis
"""

import numpy as np
from constants import COUPLING_STRENGTH, RESONANCE_THRESHOLD, EXPORT_PRECISION


class MirrorCoupling:
    """
    Bidirectional bridge between Hilbert field and Duopyramid system.

    The coupling implements a mirror-like relationship where each system
    reflects aspects of the other. The Hilbert field's continuous phases
    modulate the duopyramid's discrete activations, while the duopyramid's
    node activity patterns influence the field's phase evolution.

    Attributes:
        a (DuopyramidSystem): The discrete symbolic system
        b (HilbertField): The continuous perceptual field
        coupling_strength (float): Bidirectional influence coefficient
        resonance_history (list): Track resonance over time
    """

    def __init__(self, system_a, system_b, coupling_strength=COUPLING_STRENGTH):
        """
        Initialize the mirror coupling between two systems.

        Args:
            system_a: DuopyramidSystem instance (discrete structure)
            system_b: HilbertField instance (continuous field)
            coupling_strength: Magnitude of mutual influence
        """
        self.a = system_a  # Duopyramid (discrete)
        self.b = system_b  # Hilbert (continuous)
        self.coupling_strength = coupling_strength
        self.resonance_history = []
        self._sync_events = []

    def couple_states(self):
        """
        Create mutual modulation between systems.

        The coupling works bidirectionally:
        1. Duopyramid → Hilbert: Node activity creates weighted phase shifts
        2. Hilbert → Duopyramid: (Implicit via phase_hint in resolve_input)

        This method handles the discrete-to-continuous direction.
        """
        activity = self.a.get_node_activity()
        activity_sum = sum(activity)

        if activity_sum > 0:
            # Compute activity-weighted centroid
            weighted_phase_shift = sum(
                w * i for i, w in enumerate(activity)
            ) / activity_sum

            # Apply modulated phase shift to each Hilbert basis vector
            for i in range(self.b.n_basis):
                phase_delta = self.coupling_strength * np.sin(i - weighted_phase_shift)
                self.b.phases[i] += phase_delta

            # Ensure phases stay in valid range
            self.b.phases = np.mod(self.b.phases, 2 * np.pi)

        # Track resonance state
        resonance = self.compute_resonance()
        self.resonance_history.append(resonance)

        # Detect synchronization events
        if resonance > RESONANCE_THRESHOLD:
            self._sync_events.append({
                'resonance': resonance,
                'hilbert_dominant': self.b.get_dominant_index(),
                'duopyramid_dominant': self.a.get_dominant_node(),
            })

    def compute_resonance(self):
        """
        Compute the resonance measure between coupled systems.

        Resonance indicates how well the continuous and discrete systems
        are aligned. High resonance means the dominant Hilbert basis
        corresponds to the most active duopyramid node.

        Returns:
            float: Resonance measure in [0, 1]
        """
        hilbert_dominant = self.b.get_dominant_index()
        duopyramid_activity = self.a.get_node_activity()

        if sum(duopyramid_activity) == 0:
            return 0.0

        # Measure alignment between Hilbert focus and duopyramid activity
        normalized_activity = np.array(duopyramid_activity) / sum(duopyramid_activity)
        return float(normalized_activity[hilbert_dominant])

    def get_coupling_state(self):
        """
        Return the current state of the coupling.

        Returns:
            dict: Coupling state including resonance and alignment info
        """
        return {
            'resonance': round(self.compute_resonance(), EXPORT_PRECISION),
            'hilbert_dominant': self.b.get_dominant_index(),
            'duopyramid_dominant': self.a.get_dominant_node(),
            'aligned': self.b.get_dominant_index() == self.a.get_dominant_node(),
            'coupling_strength': self.coupling_strength,
        }

    def get_resonance_history(self):
        """
        Return the history of resonance values.

        Returns:
            list: Resonance values for each coupling step
        """
        return [round(r, EXPORT_PRECISION) for r in self.resonance_history]

    def get_sync_events(self):
        """
        Return detected synchronization events.

        Returns:
            list: Events where resonance exceeded threshold
        """
        return self._sync_events

    def get_mirror_state(self):
        """
        Return the complete mirror state showing both systems.

        This provides a unified view of the coupled system state,
        useful for visualization and LLM interpretation.

        Returns:
            dict: Complete mirror state
        """
        return {
            'hilbert': {
                'state_vector': self.b.get_state_vector(),
                'dominant_index': self.b.get_dominant_index(),
                'coherence': round(self.b.get_coherence(), EXPORT_PRECISION),
            },
            'duopyramid': {
                'node_activity': self.a.get_node_activity(),
                'dominant_node': self.a.get_dominant_node(),
                'dominant_glyph': self.a.get_dominant_glyph(),
                'poles': self.a.get_pole_states(),
            },
            'coupling': self.get_coupling_state(),
        }

    def force_alignment(self, target_idx):
        """
        Force both systems to align at a specific index.

        This is useful for resetting to a known state or testing.

        Args:
            target_idx: Index to align both systems toward
        """
        # Influence Hilbert field
        self.b.inject_resonance(target_idx, strength=0.5)

        # Influence duopyramid
        hint = target_idx / self.a.n_nodes
        self.a.resolve_input(hint)

    def adjust_coupling_strength(self, new_strength):
        """
        Dynamically adjust the coupling strength.

        Args:
            new_strength: New coupling coefficient
        """
        self.coupling_strength = new_strength

    def reset(self):
        """Reset coupling history while preserving system states."""
        self.resonance_history = []
        self._sync_events = []
