"""
duopyramid_system.py - Closed 40-fold Symmetry Logic

This module implements a duopyramid structure representing closed symbolic
space with 10-node decagonal symmetry and 2 polar vertices (alpha/omega).
The total 40-fold symmetry emerges from the 10 nodes × 4 coupling modes
between the poles and the decagonal ring.

The structure provides:
- Fixed-state resolution from continuous hints
- Decay dynamics preventing state lock-in
- Polar modulation for upper/lower field influence
- Activity patterns that feed back into the Hilbert field
"""

import numpy as np
from constants import (
    DUOPYRAMID_N_NODES,
    NODE_DECAY_RATE,
    NODE_ACTIVATION_RATE,
    EXPORT_PRECISION,
    NODE_GLYPHS,
    POLE_GLYPHS,
)


class DuopyramidSystem:
    """
    Closed symbolic structure with decagonal base and bipolar vertices.

    The duopyramid models a discrete symbolic manifold where continuous
    perceptual states collapse into specific node activations. The 10
    decagonal nodes represent symbolic positions, while alpha/omega poles
    represent projection origin and collapse attractor respectively.

    Attributes:
        n_nodes (int): Number of decagonal nodes
        alpha (float): Upper pole activity
        omega (float): Lower pole activity
        node_states (np.array): Activity levels for each decagonal node
    """

    def __init__(self, n_nodes=DUOPYRAMID_N_NODES):
        """
        Initialize the duopyramid with zeroed state.

        Args:
            n_nodes: Number of nodes in decagonal ring (default from constants)
        """
        self.n_nodes = n_nodes
        self.alpha = 0.0  # Upper pole
        self.omega = 0.0  # Lower pole
        self.node_states = np.zeros(self.n_nodes)
        self._focus_history = []

    def resolve_input(self, hint):
        """
        Collapse toward one of the 10 symbolic node positions.

        The continuous hint value [0, 1) is mapped to a discrete node
        index. Existing states decay while the focused node activates,
        creating a soft attention mechanism with memory.

        Args:
            hint: Normalized position [0, 1) from Hilbert field phase hint
        """
        # Map continuous hint to discrete node index
        focus_idx = int(round(hint * self.n_nodes)) % self.n_nodes

        # Apply decay to all nodes (prevents lock-in)
        self.node_states *= NODE_DECAY_RATE

        # Activate the focused node
        self.node_states[focus_idx] += NODE_ACTIVATION_RATE

        # Update poles based on node activity distribution
        self._update_poles()

        # Track focus history
        self._focus_history.append(focus_idx)

    def _update_poles(self):
        """
        Update polar vertices based on node activity patterns.

        Alpha (upper) responds to ascending activity gradients,
        Omega (lower) responds to descending/convergent patterns.
        """
        # Alpha responds to activity spread (divergence)
        activity_variance = np.var(self.node_states)
        self.alpha = 0.9 * self.alpha + 0.1 * activity_variance

        # Omega responds to activity concentration (convergence)
        max_activity = np.max(self.node_states)
        self.omega = 0.9 * self.omega + 0.1 * max_activity

    def get_node_activity(self):
        """
        Return current activity levels for all nodes.

        Returns:
            List of activity values with controlled precision
        """
        return [round(val, EXPORT_PRECISION) for val in self.node_states]

    def get_dominant_node(self):
        """
        Return the index of the most active node.

        Returns:
            int: Index of node with highest activity
        """
        return int(np.argmax(self.node_states))

    def get_dominant_glyph(self):
        """
        Return the symbolic glyph of the most active node.

        Returns:
            str: Unicode glyph representing dominant symbolic position
        """
        return NODE_GLYPHS.get(self.get_dominant_node(), "?")

    def get_pole_states(self):
        """
        Return the current polar vertex states.

        Returns:
            dict: Alpha and omega pole activity levels
        """
        return {
            "alpha": round(self.alpha, EXPORT_PRECISION),
            "omega": round(self.omega, EXPORT_PRECISION),
        }

    def get_resonance_pattern(self):
        """
        Compute the resonance pattern across the decagonal ring.

        Resonance is measured as the harmonic relationship between
        adjacent nodes, indicating structural coherence.

        Returns:
            float: Resonance measure in [0, 1]
        """
        if np.sum(self.node_states) == 0:
            return 0.0

        # Compute circular autocorrelation
        normalized = self.node_states / (np.sum(self.node_states) + 1e-10)
        correlation = np.correlate(normalized, normalized, mode='full')

        # Return normalized resonance strength
        return float(np.max(correlation[:self.n_nodes]))

    def get_symbolic_state(self):
        """
        Return the full symbolic state as glyph-activity pairs.

        Returns:
            dict: Mapping from glyphs to activity levels
        """
        return {
            NODE_GLYPHS[i]: round(self.node_states[i], EXPORT_PRECISION)
            for i in range(self.n_nodes)
        }

    def get_40_fold_state(self):
        """
        Return the complete 40-fold symmetry state.

        The 40 dimensions arise from 10 nodes × 4 coupling modes:
        - Node-to-alpha coupling
        - Node-to-omega coupling
        - Node-to-clockwise-neighbor coupling
        - Node-to-counter-clockwise-neighbor coupling

        Returns:
            dict: Complete 40-dimensional state representation
        """
        state = {}
        for i in range(self.n_nodes):
            base = self.node_states[i]
            state[f"node_{i}_alpha"] = round(base * self.alpha, EXPORT_PRECISION)
            state[f"node_{i}_omega"] = round(base * self.omega, EXPORT_PRECISION)
            state[f"node_{i}_cw"] = round(
                base * self.node_states[(i + 1) % self.n_nodes], EXPORT_PRECISION
            )
            state[f"node_{i}_ccw"] = round(
                base * self.node_states[(i - 1) % self.n_nodes], EXPORT_PRECISION
            )
        return state

    def inject_polar_influence(self, pole, strength=0.1):
        """
        Inject influence from a polar vertex into the node ring.

        Args:
            pole: Either 'alpha' or 'omega'
            strength: Magnitude of influence to propagate
        """
        if pole == 'alpha':
            # Alpha distributes evenly (divergent)
            self.node_states += strength / self.n_nodes
        elif pole == 'omega':
            # Omega concentrates at dominant (convergent)
            dominant = self.get_dominant_node()
            self.node_states[dominant] += strength

    def reset(self):
        """Reset all states to zero."""
        self.alpha = 0.0
        self.omega = 0.0
        self.node_states = np.zeros(self.n_nodes)
        self._focus_history = []
