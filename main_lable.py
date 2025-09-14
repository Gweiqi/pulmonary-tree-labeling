"""
Main labeling script for pulmonary tree analysis.

This script demonstrates the coordinate system refactor by integrating
tree detection, angle calculation, and coordinate transformations.
"""

import os
import argparse
import numpy as np
import json
from typing import Dict, Optional

from coordinate_utils import load_nifti, validate_coordinate_consistency
from detect_tree import tree_detection, get_tree_length, extract_centerline_points
from airway_area_utils import (
    calculate_bifurcation_angles, 
    determine_left_right_lung,
    calculate_airway_metrics,
    export_centerline_csv
)


def process_pulmonary_tree(segmentation_path: str,
                          output_dir: str,
                          min_branch_length: float = 2.0,
                          use_isotropic_skeletonization: bool = True,
                          export_formats: list = None) -> Dict:
    """
    Complete pulmonary tree processing pipeline with coordinate system refactor.
    
    Args:
        segmentation_path: Path to segmentation .nii or .nii.gz file
        output_dir: Output directory for results
        min_branch_length: Minimum branch length in physical units
        use_isotropic_skeletonization: Whether to use isotropic skeletonization
        export_formats: List of export formats ('csv', 'json')
        
    Returns:
        results: Dictionary containing all analysis results
    """
    if export_formats is None:
        export_formats = ['csv', 'json']
    
    print(f"Loading segmentation from: {segmentation_path}")
    
    # Load segmentation with proper coordinate handling
    data, affine, spacing = load_nifti(segmentation_path, as_canonical=True)
    
    print(f"Data shape: {data.shape}")
    print(f"Spacing: {spacing}")
    print(f"Affine matrix:\n{affine}")
    
    # Convert to binary segmentation
    binary_seg = (data > 0).astype(np.uint8)
    
    print("Performing tree detection...")
    
    # Perform tree detection with proper spacing
    tree_result = tree_detection(
        segmentation=binary_seg,
        spacing=spacing,
        min_branch_length=min_branch_length,
        use_isotropic_skeletonization=use_isotropic_skeletonization
    )
    
    print(f"Found {len(tree_result['skeleton_points'])} skeleton points")
    print(f"Found {len(tree_result['branch_points'])} branch points")
    print(f"Found {len(tree_result['endpoint_points'])} endpoint points")
    
    # Validate coordinate consistency
    print("Validating coordinate consistency...")
    is_valid, max_distance = validate_coordinate_consistency(
        tree_result['skeleton_points'], binary_seg, spacing
    )
    
    print(f"Coordinate validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"Maximum distance from skeleton to segmentation: {max_distance:.3f} mm")
    
    # Calculate tree metrics
    print("Calculating tree metrics...")
    
    total_length = get_tree_length(
        tree_result['graph'], 
        tree_result['skeleton_points'], 
        spacing
    )
    
    # Calculate bifurcation angles
    bifurcation_angles = calculate_bifurcation_angles(tree_result)
    
    # Determine left/right lung classification
    lr_labels = determine_left_right_lung(
        tree_result['skeleton_points'], 
        spacing, 
        affine
    )
    
    # Calculate comprehensive metrics
    airway_metrics = calculate_airway_metrics(tree_result, binary_seg)
    
    # Extract centerline points with metadata
    centerline_points = extract_centerline_points(tree_result, order_by_distance=True)
    
    # Compile results
    results = {
        'segmentation_path': segmentation_path,
        'data_shape': data.shape,
        'spacing': spacing.tolist(),
        'affine': affine.tolist(),
        'validation': {
            'is_valid': is_valid,
            'max_distance_to_segmentation': max_distance
        },
        'tree_structure': {
            'num_skeleton_points': len(tree_result['skeleton_points']),
            'num_branch_points': len(tree_result['branch_points']),
            'num_endpoint_points': len(tree_result['endpoint_points']),
            'total_length_mm': total_length,
            'branch_points': tree_result['branch_points'],
            'endpoint_points': tree_result['endpoint_points']
        },
        'bifurcation_angles': bifurcation_angles,
        'left_right_labels': lr_labels.tolist(),
        'airway_metrics': airway_metrics,
        'centerline_points': centerline_points
    }
    
    # Export results
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(segmentation_path))[0]
    if base_name.endswith('.nii'):
        base_name = base_name[:-4]  # Remove .nii from .nii.gz
    
    if 'json' in export_formats:
        json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
        print(f"Saving JSON results to: {json_path}")
        
        # Convert numpy types for JSON serialization
        json_results = _convert_for_json(results)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    if 'csv' in export_formats:
        csv_path = os.path.join(output_dir, f"{base_name}_centerline.csv")
        print(f"Saving CSV centerline to: {csv_path}")
        
        export_centerline_csv(
            tree_result, 
            csv_path, 
            affine=affine, 
            include_world_coords=True
        )
    
    print("Analysis completed successfully!")
    return results


def _convert_for_json(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {key: _convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def break_by_generation(tree_result: Dict, output_dir: str, base_name: str) -> None:
    """
    Break tree into generation-based segments (legacy compatibility function).
    
    Args:
        tree_result: Result from tree_detection()
        output_dir: Output directory
        base_name: Base filename for outputs
    """
    from airway_area_utils import bfs_label_airways
    
    # Label airways by generation
    generation_labels = bfs_label_airways(tree_result, generation_labels=True)
    
    # Group points by generation
    generations = {}
    for point_idx, generation in generation_labels.items():
        if generation not in generations:
            generations[generation] = []
        generations[generation].append(point_idx)
    
    # Export generation information
    gen_info = {
        'generations': generations,
        'num_generations': len(generations),
        'points_per_generation': {gen: len(points) for gen, points in generations.items()}
    }
    
    output_path = os.path.join(output_dir, f"{base_name}_generations.json")
    with open(output_path, 'w') as f:
        json.dump(_convert_for_json(gen_info), f, indent=2)
    
    print(f"Generation analysis saved to: {output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Pulmonary tree labeling with coordinate system refactor"
    )
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True,
        help='Directory containing segmentation files'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--min_branch_length', 
        type=float, 
        default=2.0,
        help='Minimum branch length in mm'
    )
    
    parser.add_argument(
        '--use_isotropic', 
        action='store_true',
        help='Use isotropic skeletonization'
    )
    
    parser.add_argument(
        '--export_formats', 
        nargs='+', 
        choices=['csv', 'json'], 
        default=['csv', 'json'],
        help='Export formats'
    )
    
    args = parser.parse_args()
    
    # Find all segmentation files
    input_dir = args.input_dir
    supported_extensions = ('.nii', '.nii.gz')
    
    segmentation_files = []
    for file in os.listdir(input_dir):
        if file.endswith(supported_extensions):
            segmentation_files.append(os.path.join(input_dir, file))
    
    if not segmentation_files:
        print(f"No segmentation files found in {input_dir}")
        return
    
    print(f"Found {len(segmentation_files)} segmentation files")
    
    # Process each file
    for seg_file in segmentation_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(seg_file)}")
        print(f"{'='*60}")
        
        try:
            results = process_pulmonary_tree(
                segmentation_path=seg_file,
                output_dir=args.output_dir,
                min_branch_length=args.min_branch_length,
                use_isotropic_skeletonization=args.use_isotropic,
                export_formats=args.export_formats
            )
            
            # Additional legacy compatibility: break by generation
            base_name = os.path.splitext(os.path.basename(seg_file))[0]
            if base_name.endswith('.nii'):
                base_name = base_name[:-4]
            
            # Extract tree_result from processing for generation analysis
            data, affine, spacing = load_nifti(seg_file, as_canonical=True)
            binary_seg = (data > 0).astype(np.uint8)
            tree_result = tree_detection(
                segmentation=binary_seg,
                spacing=spacing,
                min_branch_length=args.min_branch_length,
                use_isotropic_skeletonization=args.use_isotropic
            )
            
            break_by_generation(tree_result, args.output_dir, base_name)
            
        except Exception as e:
            print(f"Error processing {seg_file}: {str(e)}")
            continue
    
    print(f"\nProcessing completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()