"""
constants.py - Symbolic glyphs and system parameters for the Mirror System

This module defines the fundamental symbolic vocabulary and configuration
parameters used across the Hilbert-Duopyramid mirror coupling architecture.
"""

# =============================================================================
# SYMBOLIC GLYPHS - 10-fold Decagonal Mapping
# =============================================================================

# Primary node glyphs (decagonal vertices)
NODE_GLYPHS = {
    0: "✶",   # Radiant seed
    1: "↻",   # Recursive return
    2: "φ",   # Golden ratio / emergence
    3: "∞",   # Infinity / unbounded process
    4: "◇",   # Diamond / crystallized form
    5: "⊕",   # Direct sum / integration
    6: "≋",   # Approximate equality / resonance
    7: "∂",   # Boundary / differential
    8: "Ω",   # Omega / completion
    9: "Λ",   # Lambda / abstraction
}

# Pole glyphs (duopyramid apexes)
POLE_GLYPHS = {
    "alpha": "α",  # Upper pole - projection origin
    "omega": "ω",  # Lower pole - collapse attractor
}

# Combined glyph for mirror coupling state
MIRROR_GLYPH = "⟷"  # Bidirectional coupling

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================

# Hilbert Field parameters
HILBERT_N_BASIS = 10          # Number of basis vectors (matches decagonal nodes)
PHASE_DRIFT_RATE = 0.05       # Rate of phase evolution per timestep
PHASE_MODULATION_FREQ = 3.0   # Frequency of sinusoidal phase modulation

# Duopyramid System parameters
DUOPYRAMID_N_NODES = 10       # Decagonal node count
NODE_DECAY_RATE = 0.9         # Exponential decay factor per timestep
NODE_ACTIVATION_RATE = 0.1    # Activation increment on focus

# Mirror Coupling parameters
COUPLING_STRENGTH = 0.01      # Bidirectional influence coefficient
RESONANCE_THRESHOLD = 0.5     # Minimum activity for resonance detection

# Simulation defaults
DEFAULT_TIMESTEPS = 50        # Standard simulation duration
EXPORT_PRECISION = 3          # Decimal places for exported values

# =============================================================================
# STATE LABELS (LLM-friendly deterministic naming)
# =============================================================================

STATE_LABELS = {
    "phase": "phase",
    "weight": "weight",
    "focus_idx": "focus_idx",
    "activity": "activity",
    "state_vector": "Λ_state_vector",
    "node_activity": "duopyramid_nodes",
    "coupling_state": "mirror_coupling_state",
}

# =============================================================================
# COLOR PALETTE (for SVG visualization)
# =============================================================================

COLORS = {
    "background": "#0a0a12",
    "node_inactive": "#2a2a3a",
    "node_active": "#7b68ee",
    "pole_alpha": "#ffd700",
    "pole_omega": "#4169e1",
    "coupling_line": "#9370db",
    "phase_ring": "#20b2aa",
    "text": "#e0e0e0",
}
