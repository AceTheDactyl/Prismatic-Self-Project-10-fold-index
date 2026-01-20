#!/usr/bin/env python3
"""
main.py - Entry point: System Runner

This is the primary entry point for the Hilbert-Duopyramid Mirror Coupling System.
It orchestrates the interaction between the continuous perceptual field (HilbertField)
and the discrete symbolic structure (DuopyramidSystem) through bidirectional coupling.

Usage:
    python main.py                    # Run with defaults
    python main.py --timesteps 100    # Custom timestep count
    python main.py --output state.json # Custom output file
    python main.py --svg              # Also generate SVG visualization
"""

import argparse
import os
from hilbert_field import HilbertField
from duopyramid_system import DuopyramidSystem
from mirror_coupling import MirrorCoupling
from glyph_export import export_state, export_full_state, export_svg, state_to_markdown
from constants import DEFAULT_TIMESTEPS


def run_simulation(timesteps=DEFAULT_TIMESTEPS, verbose=False):
    """
    Execute the mirror coupling simulation.

    Args:
        timesteps: Number of evolution steps to run
        verbose: If True, print progress updates

    Returns:
        tuple: (hilbert, duopyramid, bridge) system instances
    """
    # Initialize both systems
    hilbert = HilbertField()
    duopyramid = DuopyramidSystem()
    bridge = MirrorCoupling(duopyramid, hilbert)

    if verbose:
        print(f"Initialized Mirror System")
        print(f"  Hilbert basis vectors: {hilbert.n_basis}")
        print(f"  Duopyramid nodes: {duopyramid.n_nodes}")
        print(f"  Coupling strength: {bridge.coupling_strength}")
        print(f"\nRunning {timesteps} timesteps...")

    # Step through vector evolution
    for t in range(timesteps):
        # Update Hilbert field phase relationships
        hilbert.update_focus(time=t)

        # Collapse duopyramid based on Hilbert resonance
        duopyramid.resolve_input(hilbert.get_phase_hint())

        # Apply bidirectional mirror coupling
        bridge.couple_states()

        if verbose and (t + 1) % 10 == 0:
            coupling = bridge.get_coupling_state()
            print(f"  t={t+1}: resonance={coupling['resonance']:.3f}, "
                  f"aligned={coupling['aligned']}, "
                  f"glyph={duopyramid.get_dominant_glyph()}")

    if verbose:
        print(f"\nSimulation complete.")
        print(f"  Final resonance: {bridge.compute_resonance():.3f}")
        print(f"  Sync events: {len(bridge.get_sync_events())}")

    return hilbert, duopyramid, bridge


def main():
    """Main entry point with CLI argument handling."""
    parser = argparse.ArgumentParser(
        description="Hilbert-Duopyramid Mirror Coupling System"
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Number of simulation timesteps (default: {DEFAULT_TIMESTEPS})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="examples/sample_run.json",
        help="Output JSON file path (default: examples/sample_run.json)"
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also generate SVG visualization"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Export full state including coupling and 40-fold symmetry"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress updates during simulation"
    )
    parser.add_argument(
        "--markdown", "-m",
        action="store_true",
        help="Print markdown summary to stdout"
    )

    args = parser.parse_args()

    # Run simulation
    hilbert, duopyramid, bridge = run_simulation(
        timesteps=args.timesteps,
        verbose=args.verbose
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export state
    if args.full:
        data = export_full_state(hilbert, duopyramid, bridge, args.output)
        if args.verbose:
            print(f"\nExported full state to {args.output}")
    else:
        data = export_state(hilbert, duopyramid, args.output)
        if args.verbose:
            print(f"\nExported state to {args.output}")

    # Generate SVG if requested
    if args.svg:
        svg_path = args.output.replace('.json', '.svg')
        export_svg(duopyramid, hilbert, svg_path)
        if args.verbose:
            print(f"Exported SVG to {svg_path}")

    # Print markdown summary if requested
    if args.markdown:
        print("\n" + state_to_markdown(hilbert, duopyramid, bridge))

    return hilbert, duopyramid, bridge


if __name__ == "__main__":
    main()
