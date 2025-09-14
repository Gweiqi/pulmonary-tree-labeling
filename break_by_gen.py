"""
Break By Generation - Example Usage Script

This script demonstrates how to use the new coordinate-fixed airway tree
analysis pipeline. It loads a NIfTI segmentation, performs tree detection,
and outputs generation/segment analysis.

Usage:
    python break_by_gen.py <input_nii_path> [output_csv_path]

Author: Generated for issue fix
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any

# Import our new modules
from coordinate_utils import load_nii, get_spacing_from_affine
from detect_tree import tree_detection
from airway_area_utils import (
    get_voxel_by_generation, 
    get_voxel_by_segment_no,
    calculate_bifurcation_angles_world,
    assign_left_right_world,
    compute_voxel_volumes,
    validate_bfs_labeling
)


def main():
    parser = argparse.ArgumentParser(description='Pulmonary tree analysis with coordinate fixes')
    parser.add_argument('input_nii', help='Path to input NIfTI segmentation file')
    parser.add_argument('--output_csv', help='Path to output CSV file for bifurcation angles')
    parser.add_argument('--output_dir', help='Directory to save all outputs', default='.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold')
    parser.add_argument('--max_distance', type=float, default=5.0, help='Max distance for nearby nodes')
    parser.add_argument('--branch_penalty', type=float, default=16.0, help='Branch penalty for tree construction')
    
    args = parser.parse_args()
    
    # Load NIfTI data
    print(f"Loading NIfTI file: {args.input_nii}")
    try:
        data, affine, nii_img = load_nii(args.input_nii)
        print(f"Data shape: {data.shape}")
        print(f"Affine matrix:\n{affine}")
        
        # Get spacing
        spacing = get_spacing_from_affine(affine)
        print(f"Voxel spacing: {spacing} mm")
        
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return 1
    
    # Perform tree detection
    print("Performing tree detection...")
    try:
        connection_dict = tree_detection(
            data, 
            affine, 
            threshold=args.threshold,
            max_nearby_distance=args.max_distance,
            branch_penalty=args.branch_penalty
        )
        
        if not connection_dict:
            print("Warning: No tree structure detected")
            return 1
        
        print(f"Detected {len(connection_dict)} nodes")
        
    except Exception as e:
        print(f"Error in tree detection: {e}")
        return 1
    
    # Assign left/right lung labels
    print("Assigning left/right lung labels...")
    try:
        assign_left_right_world(connection_dict, affine)
    except Exception as e:
        print(f"Warning: Could not assign left/right labels: {e}")
    
    # Generate BFS labeling
    print("Generating BFS voxel labeling...")
    try:
        # Convert to binary for BFS
        binary_seg = (data > args.threshold).astype(np.uint8)
        
        generation_array = get_voxel_by_generation(connection_dict, binary_seg)
        segment_array = get_voxel_by_segment_no(connection_dict, binary_seg)
        
        # Validate labeling
        validate_bfs_labeling(binary_seg, generation_array, "generation")
        validate_bfs_labeling(binary_seg, segment_array, "segment")
        
    except Exception as e:
        print(f"Error in BFS labeling: {e}")
        return 1
    
    # Calculate volumes
    print("Computing volumes...")
    try:
        generation_volumes, segment_volumes = compute_voxel_volumes(
            generation_array, segment_array, spacing
        )
        
        print(f"Generation volumes: {dict(list(generation_volumes.items())[:5])}...")
        print(f"Segment volumes: {dict(list(segment_volumes.items())[:5])}...")
        
    except Exception as e:
        print(f"Error computing volumes: {e}")
        return 1
    
    # Calculate bifurcation angles
    print("Calculating bifurcation angles...")
    try:
        bifurcation_angles = calculate_bifurcation_angles_world(connection_dict)
        
        print(f"Found {len(bifurcation_angles)} bifurcations")
        
        if bifurcation_angles:
            # Print sample angles
            sample = bifurcation_angles[0]
            print(f"Sample bifurcation: ID={sample['bifurcation_id']}, "
                  f"angles=({sample['angle_parent_child1']:.1f}°, "
                  f"{sample['angle_parent_child2']:.1f}°, "
                  f"{sample['angle_between_children']:.1f}°)")
    
    except Exception as e:
        print(f"Error calculating angles: {e}")
        return 1
    
    # Save outputs
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save bifurcation angles CSV
    if bifurcation_angles:
        csv_path = args.output_csv or os.path.join(output_dir, 'bifurcation_angles.csv')
        df = pd.DataFrame(bifurcation_angles)
        df.to_csv(csv_path, index=False)
        print(f"Saved bifurcation angles to: {csv_path}")
    
    # Save tree structure summary
    summary_path = os.path.join(output_dir, 'tree_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Tree Analysis Summary\n")
        f.write(f"====================\n\n")
        f.write(f"Input file: {args.input_nii}\n")
        f.write(f"Data shape: {data.shape}\n")
        f.write(f"Voxel spacing: {spacing} mm\n")
        f.write(f"Total nodes: {len(connection_dict)}\n")
        f.write(f"Bifurcations: {len(bifurcation_angles)}\n")
        f.write(f"Generations: {len(generation_volumes)}\n")
        f.write(f"Segments: {len(segment_volumes)}\n\n")
        
        # Node statistics
        generations = [node['generation'] for node in connection_dict.values() if 'generation' in node]
        if generations:
            f.write(f"Generation range: {min(generations)} to {max(generations)}\n")
        
        sides = [node.get('side', 'unknown') for node in connection_dict.values()]
        side_counts = {side: sides.count(side) for side in set(sides)}
        f.write(f"Side distribution: {side_counts}\n")
    
    print(f"Saved tree summary to: {summary_path}")
    
    # Print final statistics
    print("\n=== Analysis Complete ===")
    print(f"Total nodes: {len(connection_dict)}")
    print(f"Bifurcations: {len(bifurcation_angles)}")
    print(f"Valid angles: {sum(1 for b in bifurcation_angles if not np.isnan(b['angle_between_children']))}")
    print(f"Output directory: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())