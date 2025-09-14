"""
Legacy compatibility script for breaking trees by generation.

This script provides a simple interface for the generation-based
tree analysis functionality from the coordinate refactor.
"""

import argparse
import os
from main_lable import process_pulmonary_tree, break_by_generation
from coordinate_utils import load_nifti
from detect_tree import tree_detection


def main():
    """Main entry point for break by generation script."""
    parser = argparse.ArgumentParser(
        description="Break pulmonary tree by generation (legacy compatibility)"
    )
    
    parser.add_argument(
        'segmentation_path',
        type=str,
        help='Path to segmentation .nii or .nii.gz file'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./generation_results',
        help='Output directory for generation results'
    )
    
    parser.add_argument(
        '--min_branch_length', 
        type=float, 
        default=2.0,
        help='Minimum branch length in mm'
    )
    
    args = parser.parse_args()
    
    print(f"Processing: {args.segmentation_path}")
    
    # Load segmentation
    data, affine, spacing = load_nifti(args.segmentation_path, as_canonical=True)
    binary_seg = (data > 0).astype('uint8')
    
    # Perform tree detection
    tree_result = tree_detection(
        segmentation=binary_seg,
        spacing=spacing,
        min_branch_length=args.min_branch_length,
        use_isotropic_skeletonization=True
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract base name
    base_name = os.path.splitext(os.path.basename(args.segmentation_path))[0]
    if base_name.endswith('.nii'):
        base_name = base_name[:-4]
    
    # Break by generation
    break_by_generation(tree_result, args.output_dir, base_name)
    
    print(f"Generation analysis saved to: {args.output_dir}")


if __name__ == '__main__':
    main()