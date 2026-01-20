"""
glyph_export.py - JSON + SVG Export Tools

This module provides export functionality for the mirror coupling system,
outputting state vectors and glyph maps in formats suitable for both
LLM interpretation (JSON) and human visualization (SVG).

Export capabilities:
- JSON state export with self-describing fields
- SVG visualization of duopyramid structure
- Combined export with full system state
- History export for temporal analysis
"""

import json
import math
from constants import (
    EXPORT_PRECISION,
    STATE_LABELS,
    NODE_GLYPHS,
    POLE_GLYPHS,
    COLORS,
)


def export_state(hilbert, duopyramid, file, include_extended=False):
    """
    Export system state to JSON file for LLM digestion.

    The export uses deterministic, self-describing field names
    that allow LLMs to interpret the state without additional context.

    Args:
        hilbert: HilbertField instance
        duopyramid: DuopyramidSystem instance
        file: Output file path
        include_extended: If True, include additional computed fields
    """
    data = {
        STATE_LABELS["state_vector"]: hilbert.get_state_vector(),
        STATE_LABELS["node_activity"]: duopyramid.get_node_activity(),
    }

    if include_extended:
        data["hilbert_coherence"] = round(hilbert.get_coherence(), EXPORT_PRECISION)
        data["hilbert_dominant"] = hilbert.get_dominant_index()
        data["duopyramid_poles"] = duopyramid.get_pole_states()
        data["duopyramid_dominant_glyph"] = duopyramid.get_dominant_glyph()
        data["symbolic_state"] = duopyramid.get_symbolic_state()

    with open(file, "w") as f:
        json.dump(data, f, indent=2)

    return data


def export_full_state(hilbert, duopyramid, bridge, file):
    """
    Export complete mirror system state including coupling.

    Args:
        hilbert: HilbertField instance
        duopyramid: DuopyramidSystem instance
        bridge: MirrorCoupling instance
        file: Output file path
    """
    data = bridge.get_mirror_state()
    data["resonance_history"] = bridge.get_resonance_history()
    data["sync_events"] = bridge.get_sync_events()
    data["40_fold_state"] = duopyramid.get_40_fold_state()

    with open(file, "w") as f:
        json.dump(data, f, indent=2)

    return data


def export_history(hilbert, duopyramid, bridge, file):
    """
    Export temporal evolution history.

    Args:
        hilbert: HilbertField instance
        duopyramid: DuopyramidSystem instance
        bridge: MirrorCoupling instance
        file: Output file path
    """
    data = {
        "hilbert_history": hilbert.get_history(),
        "resonance_history": bridge.get_resonance_history(),
        "sync_events": bridge.get_sync_events(),
    }

    with open(file, "w") as f:
        json.dump(data, f, indent=2)

    return data


def generate_svg(duopyramid, hilbert=None, width=400, height=400):
    """
    Generate SVG visualization of the duopyramid structure.

    The visualization shows:
    - Decagonal ring of nodes with activity-based coloring
    - Alpha and omega poles
    - Connections between elements
    - Optional Hilbert field phase ring

    Args:
        duopyramid: DuopyramidSystem instance
        hilbert: Optional HilbertField instance for phase overlay
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        str: SVG markup string
    """
    cx, cy = width // 2, height // 2
    radius = min(width, height) // 3

    # Start SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'  <rect width="100%" height="100%" fill="{COLORS["background"]}"/>',
    ]

    # Draw pole connections
    alpha_y = cy - radius - 40
    omega_y = cy + radius + 40

    node_activity = duopyramid.get_node_activity()
    max_activity = max(node_activity) if max(node_activity) > 0 else 1

    # Draw decagonal nodes
    node_positions = []
    for i in range(duopyramid.n_nodes):
        angle = (2 * math.pi * i / duopyramid.n_nodes) - math.pi / 2
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        node_positions.append((x, y))

        # Activity-based color interpolation
        activity_ratio = node_activity[i] / max_activity
        color = _interpolate_color(
            COLORS["node_inactive"],
            COLORS["node_active"],
            activity_ratio
        )

        # Draw node circle
        node_radius = 15 + 10 * activity_ratio
        svg_parts.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{node_radius:.1f}" '
            f'fill="{color}" stroke="{COLORS["text"]}" stroke-width="1"/>'
        )

        # Draw glyph label
        glyph = NODE_GLYPHS.get(i, str(i))
        svg_parts.append(
            f'  <text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" '
            f'fill="{COLORS["text"]}" font-size="14">{glyph}</text>'
        )

        # Draw connections to poles
        svg_parts.append(
            f'  <line x1="{x:.1f}" y1="{y:.1f}" x2="{cx}" y2="{alpha_y}" '
            f'stroke="{COLORS["coupling_line"]}" stroke-width="0.5" opacity="0.3"/>'
        )
        svg_parts.append(
            f'  <line x1="{x:.1f}" y1="{y:.1f}" x2="{cx}" y2="{omega_y}" '
            f'stroke="{COLORS["coupling_line"]}" stroke-width="0.5" opacity="0.3"/>'
        )

    # Draw connections between adjacent nodes
    for i in range(duopyramid.n_nodes):
        next_i = (i + 1) % duopyramid.n_nodes
        x1, y1 = node_positions[i]
        x2, y2 = node_positions[next_i]
        svg_parts.append(
            f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{COLORS["coupling_line"]}" stroke-width="1" opacity="0.5"/>'
        )

    # Draw alpha pole
    pole_states = duopyramid.get_pole_states()
    alpha_size = 12 + 8 * pole_states["alpha"]
    svg_parts.append(
        f'  <circle cx="{cx}" cy="{alpha_y}" r="{alpha_size:.1f}" '
        f'fill="{COLORS["pole_alpha"]}" stroke="{COLORS["text"]}" stroke-width="2"/>'
    )
    svg_parts.append(
        f'  <text x="{cx}" y="{alpha_y + 5}" text-anchor="middle" '
        f'fill="{COLORS["background"]}" font-size="16" font-weight="bold">'
        f'{POLE_GLYPHS["alpha"]}</text>'
    )

    # Draw omega pole
    omega_size = 12 + 8 * pole_states["omega"]
    svg_parts.append(
        f'  <circle cx="{cx}" cy="{omega_y}" r="{omega_size:.1f}" '
        f'fill="{COLORS["pole_omega"]}" stroke="{COLORS["text"]}" stroke-width="2"/>'
    )
    svg_parts.append(
        f'  <text x="{cx}" y="{omega_y + 5}" text-anchor="middle" '
        f'fill="{COLORS["text"]}" font-size="16" font-weight="bold">'
        f'{POLE_GLYPHS["omega"]}</text>'
    )

    # Draw Hilbert phase ring if provided
    if hilbert is not None:
        phase_radius = radius + 30
        state_vector = hilbert.get_state_vector()
        for i, (weight, phase) in enumerate(state_vector):
            angle = (2 * math.pi * i / hilbert.n_basis) - math.pi / 2
            x = cx + phase_radius * math.cos(angle)
            y = cy + phase_radius * math.sin(angle)

            # Draw phase indicator
            indicator_size = 3 + 5 * weight
            svg_parts.append(
                f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{indicator_size:.1f}" '
                f'fill="{COLORS["phase_ring"]}" opacity="{0.3 + 0.7 * weight:.2f}"/>'
            )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def export_svg(duopyramid, hilbert, file):
    """
    Export SVG visualization to file.

    Args:
        duopyramid: DuopyramidSystem instance
        hilbert: HilbertField instance (can be None)
        file: Output file path
    """
    svg_content = generate_svg(duopyramid, hilbert)
    with open(file, "w") as f:
        f.write(svg_content)


def _interpolate_color(color1, color2, ratio):
    """
    Interpolate between two hex colors.

    Args:
        color1: Starting color (hex string)
        color2: Ending color (hex string)
        ratio: Interpolation ratio (0 = color1, 1 = color2)

    Returns:
        str: Interpolated hex color
    """
    # Parse hex colors
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

    # Interpolate
    r = int(r1 + (r2 - r1) * ratio)
    g = int(g1 + (g2 - g1) * ratio)
    b = int(b1 + (b2 - b1) * ratio)

    return f"#{r:02x}{g:02x}{b:02x}"


def state_to_markdown(hilbert, duopyramid, bridge=None):
    """
    Generate a markdown-formatted summary of system state.

    Useful for human-readable reports and documentation.

    Args:
        hilbert: HilbertField instance
        duopyramid: DuopyramidSystem instance
        bridge: Optional MirrorCoupling instance

    Returns:
        str: Markdown-formatted state summary
    """
    lines = [
        "# Mirror System State",
        "",
        "## Hilbert Field (Λ)",
        f"- **Dominant basis**: {hilbert.get_dominant_index()}",
        f"- **Coherence**: {hilbert.get_coherence():.3f}",
        "",
        "### State Vector",
        "| Basis | Weight | Phase |",
        "|-------|--------|-------|",
    ]

    for i, (w, p) in enumerate(hilbert.get_state_vector()):
        lines.append(f"| {i} | {w:.3f} | {p:.3f} |")

    lines.extend([
        "",
        "## Duopyramid System",
        f"- **Dominant node**: {duopyramid.get_dominant_node()} ({duopyramid.get_dominant_glyph()})",
        f"- **Alpha pole**: {duopyramid.get_pole_states()['alpha']:.3f}",
        f"- **Omega pole**: {duopyramid.get_pole_states()['omega']:.3f}",
        "",
        "### Node Activity",
        "| Glyph | Activity |",
        "|-------|----------|",
    ])

    for glyph, activity in duopyramid.get_symbolic_state().items():
        lines.append(f"| {glyph} | {activity:.3f} |")

    if bridge is not None:
        coupling = bridge.get_coupling_state()
        lines.extend([
            "",
            "## Mirror Coupling",
            f"- **Resonance**: {coupling['resonance']:.3f}",
            f"- **Aligned**: {'Yes' if coupling['aligned'] else 'No'}",
            f"- **Sync events**: {len(bridge.get_sync_events())}",
        ])

    return "\n".join(lines)
