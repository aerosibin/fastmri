"""
run_visualization.py
Master runner and Command Line Interface for Calgary-Campinas MRI Visualization.

Usage examples:
  1. Generate all presentation-ready figures (saved to reports/figures/):
     python run_visualization.py

  2. Launch live interactive explorer GUI with slice slider and acceleration toggle:
     python run_visualization.py --interactive

  3. Inspect a specific subject or slice:
     python run_visualization.py --subject e14155s3_P69120 --slice 85

  4. List all available downloaded subjects:
     python run_visualization.py --list-subjects
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.mri_processor import (
    find_all_subjects,
    load_subject_kspace,
    reconstruct_slice
)
from visualization.figure_generator import (
    generate_physics_pipeline_figure,
    generate_accelerated_mri_problem_figure,
    generate_orthogonal_views_figure,
    generate_multislice_montage_figure
)
from visualization.interactive_viewer import launch_viewer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calgary-Campinas Brain MR Dataset: Visualization & Mentor Presentation Suite"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="CC359/Raw-data/Single-channel",
        help="Path to Single-channel dataset root folder (default: CC359/Raw-data/Single-channel)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Directory where presentation figures will be saved (default: reports/figures)"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Specific subject ID or filename prefix to visualize. If None, picks the first validation subject."
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="Slice index to visualize (default: center slice of subject volume, typically slice 85)."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive slice-by-slice GUI viewer."
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="List all downloaded and extracted subjects across Train and Val."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    print("=" * 70)
    print(" Calgary-Campinas Brain MRI (CC359) Visualization Suite")
    print("=" * 70)

    # 1. Discover subjects
    try:
        subjects = find_all_subjects(data_root)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    if not subjects:
        print(f"[Error] No .npy files found under {data_root.resolve()}.")
        sys.exit(1)

    # If user wants subject listing:
    if args.list_subjects:
        print(f"\nFound {len(subjects)} subjects across Train and Val sets:")
        print(f"{'Split':<12} | {'Group':<14} | {'Subject ID':<22} | {'Size (MB)':<10}")
        print("-" * 65)
        for s in subjects:
            print(f"{s['split']:<12} | {s['group']:<14} | {s['subject_id']:<22} | {s['size_mb']:<10.1f}")
        return

    # Select target subject
    selected_subject = None
    if args.subject:
        for s in subjects:
            if args.subject in s["subject_id"] or args.subject in s["filename"]:
                selected_subject = s
                break
        if selected_subject is None:
            print(f"[Error] Subject '{args.subject}' not found. Available subjects count: {len(subjects)}.")
            sys.exit(1)
    else:
        # Default to first Validation subject
        val_subjects = [s for s in subjects if s["split"] == "Validation"]
        selected_subject = val_subjects[0] if val_subjects else subjects[0]

    print(f"\nTarget Subject : {selected_subject['subject_id']} ({selected_subject['split']}, {selected_subject['group']})")
    print(f"File Path      : {selected_subject['filepath']}")

    # Interactive mode
    if args.interactive:
        print("\nLaunching Interactive GUI Viewer...")
        print("  - Use the slider to scrub through slices")
        print("  - Select acceleration factor (R=2x, R=4x, R=8x)")
        print("  - Use Left/Right arrow keys on keyboard to navigate slices")
        launch_viewer(selected_subject["filepath"], subject_id=selected_subject["subject_id"])
        return

    # Batch figure generation mode
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output Folder  : {out_dir.resolve()}\n")

    print("[1/4] Loading complex k-space...")
    complex_kspace = load_subject_kspace(selected_subject["filepath"], mmap=True)
    n_slices = complex_kspace.shape[0]
    target_slice = args.slice if args.slice is not None else (n_slices // 2)

    if target_slice < 0 or target_slice >= n_slices:
        print(f"[Warning] Slice {target_slice} out of bounds (0..{n_slices-1}). Defaulting to {n_slices // 2}.")
        target_slice = n_slices // 2

    kspace_slice = complex_kspace[target_slice]

    print(f"[2/4] Generating Figure 1: Raw k-Space Physics Pipeline...")
    fig1_path = out_dir / "01_kspace_physics_pipeline.png"
    generate_physics_pipeline_figure(kspace_slice, selected_subject["subject_id"], target_slice, fig1_path)
    print(f"      Saved -> {fig1_path}")

    print(f"[3/4] Generating Figure 2: Accelerated MRI Undersampling Problem (R=1, 2, 4, 8x)...")
    fig2_path = out_dir / "02_accelerated_mri_problem.png"
    generate_accelerated_mri_problem_figure(kspace_slice, selected_subject["subject_id"], target_slice, fig2_path)
    print(f"      Saved -> {fig2_path}")

    print(f"[4/4] Generating Figure 3: 3D Multi-Planar Reformatting (Sagittal, Coronal, Axial)...")
    fig3_path = out_dir / "03_orthogonal_brain_views.png"
    generate_orthogonal_views_figure(complex_kspace, selected_subject["subject_id"], fig3_path)
    print(f"      Saved -> {fig3_path}")

    print(f"[Bonus] Generating Figure 4: Whole-Brain Multi-Slice Montage...")
    fig4_path = out_dir / "04_multislice_montage.png"
    generate_multislice_montage_figure(complex_kspace, selected_subject["subject_id"], fig4_path)
    print(f"      Saved -> {fig4_path}")

    print("\n" + "=" * 70)
    print(" SUCCESS! All 4 presentation-ready figures are generated.")
    print(f" Location: {out_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
