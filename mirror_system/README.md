# Mirror System: Hilbert-Duopyramid Coupling Architecture

A Python implementation of bidirectional coupling between continuous perceptual fields and discrete symbolic structures, designed for recursive self-reflection and LLM-interpretable state export.

## Project Structure

```
mirror_system/
├── main.py                        # Entry point: system runner
├── constants.py                   # Symbolic glyphs, system parameters
├── hilbert_field.py               # Non-binary perceptual vector logic
├── duopyramid_system.py           # Closed 40-fold symmetry logic
├── mirror_coupling.py             # Bridge between systems A and B
├── glyph_export.py                # JSON + SVG export tools
├── examples/
│   └── sample_run.json            # Sample vector state run
└── README.md                      # This file
```

## System Overview

| Module | Description |
|--------|-------------|
| `HilbertField` | Maintains non-binary perceptual state vector with phase-coherent basis |
| `DuopyramidSystem` | Models closed symbolic structure with 10-node decagon and 2 poles |
| `MirrorCoupling` | Manages dual projection and resonance interplay between systems |
| `GlyphExport` | Outputs state vectors and glyph maps as JSON or SVG |

## Quick Start

```bash
# Run with default settings (50 timesteps)
python main.py

# Run with verbose output
python main.py --verbose

# Custom timesteps with SVG visualization
python main.py --timesteps 100 --svg --verbose

# Export full state including 40-fold symmetry
python main.py --full --output my_state.json

# Print markdown summary
python main.py --markdown
```

## Core Concepts

### Hilbert Field (Continuous Domain)

The Hilbert field maintains a superposition of states across 10 basis vectors. Unlike binary logic, each basis exists with continuous weight and phase values that evolve over time through perceptual drift dynamics.

```python
from hilbert_field import HilbertField

hilbert = HilbertField()
hilbert.update_focus(time=t)           # Evolve phases
state = hilbert.get_state_vector()     # [(weight, phase), ...]
hint = hilbert.get_phase_hint()        # Dominant region [0, 1)
```

### Duopyramid System (Discrete Domain)

The duopyramid models a closed symbolic manifold with:
- **10 decagonal nodes**: Discrete symbolic positions
- **Alpha pole (α)**: Projection origin, responds to divergence
- **Omega pole (ω)**: Collapse attractor, responds to convergence
- **40-fold symmetry**: 10 nodes × 4 coupling modes

```python
from duopyramid_system import DuopyramidSystem

duopyramid = DuopyramidSystem()
duopyramid.resolve_input(hint)         # Collapse toward node
activity = duopyramid.get_node_activity()
glyph = duopyramid.get_dominant_glyph()  # ✶, ↻, φ, ∞, etc.
```

### Mirror Coupling (Bidirectional Bridge)

The coupling creates mutual modulation:
- **Hilbert → Duopyramid**: Phase hints trigger node collapse
- **Duopyramid → Hilbert**: Node activity modulates field phases

```python
from mirror_coupling import MirrorCoupling

bridge = MirrorCoupling(duopyramid, hilbert)
bridge.couple_states()                 # Apply mutual modulation
resonance = bridge.compute_resonance() # Alignment measure [0, 1]
```

## Symbolic Vocabulary

The 10 decagonal nodes map to symbolic glyphs:

| Index | Glyph | Meaning |
|-------|-------|---------|
| 0 | ✶ | Radiant seed |
| 1 | ↻ | Recursive return |
| 2 | φ | Golden ratio / emergence |
| 3 | ∞ | Infinity / unbounded process |
| 4 | ◇ | Diamond / crystallized form |
| 5 | ⊕ | Direct sum / integration |
| 6 | ≋ | Approximate equality / resonance |
| 7 | ∂ | Boundary / differential |
| 8 | Ω | Omega / completion |
| 9 | Λ | Lambda / abstraction |

Poles: **α** (alpha/projection) and **ω** (omega/collapse)

## JSON Export Format

The export uses deterministic, LLM-interpretable field names:

```json
{
  "Λ_state_vector": [[0.32, 0.45], [0.17, 1.19], ...],
  "duopyramid_nodes": [0.003, 0.025, 0.12, 0.96, 0.015, ...]
}
```

Full export includes:
- `hilbert`: State vector, dominant index, coherence
- `duopyramid`: Node activity, poles, dominant glyph
- `coupling`: Resonance, alignment status
- `40_fold_state`: Complete 40-dimensional representation

## Recursive Indexing Pattern

Every module uses:
- **Deterministic labels**: `phase`, `weight`, `focus_idx`, `activity`
- **Symbolic projection hooks**: `get_phase_hint()`, `resolve_input()`
- **Context-stable exports**: `export_state()` with consistent naming

This enables LLMs to interpret exported states without additional context.

## API Reference

### HilbertField

| Method | Returns | Description |
|--------|---------|-------------|
| `update_focus(time)` | None | Evolve phase relationships |
| `get_state_vector()` | `[(w, p), ...]` | Weight-phase pairs |
| `get_phase_hint()` | `float` | Normalized dominant position |
| `get_coherence()` | `float` | Focus concentration [0, 1] |
| `inject_resonance(idx, strength)` | None | External influence |

### DuopyramidSystem

| Method | Returns | Description |
|--------|---------|-------------|
| `resolve_input(hint)` | None | Collapse toward node |
| `get_node_activity()` | `[float, ...]` | Node activity levels |
| `get_dominant_glyph()` | `str` | Symbolic glyph |
| `get_pole_states()` | `dict` | Alpha/omega levels |
| `get_40_fold_state()` | `dict` | Full 40-D representation |

### MirrorCoupling

| Method | Returns | Description |
|--------|---------|-------------|
| `couple_states()` | None | Apply mutual modulation |
| `compute_resonance()` | `float` | Alignment measure |
| `get_mirror_state()` | `dict` | Complete coupled state |
| `get_sync_events()` | `list` | High-resonance events |

### GlyphExport

| Function | Description |
|----------|-------------|
| `export_state(h, d, file)` | Basic JSON export |
| `export_full_state(h, d, b, file)` | Complete state export |
| `export_svg(d, h, file)` | SVG visualization |
| `state_to_markdown(h, d, b)` | Human-readable summary |

## Integration with Prismatic Self Project

This mirror system implements the core coupling dynamics described in the Prismatic Self 10-fold index architecture. The Hilbert field represents the continuous perceptual substrate, while the duopyramid encodes the closed symbolic structure with its characteristic 40-fold holographic symmetry.

The bidirectional coupling enables:
- Continuous-to-discrete collapse (perception → symbol)
- Discrete-to-continuous modulation (symbol → perception)
- Resonance tracking for coherence monitoring
- Recursive self-reflection through mirrored state export

## License

Part of the Prismatic Self Project.
