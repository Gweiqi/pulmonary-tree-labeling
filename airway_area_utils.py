"""
Airway Area Utilities and BFS Labeling

This module provides utilities for airway area computation, BFS voxel labeling,
and world-space bifurcation angle calculation.

Key Features:
- Fixed BFS labeling (removes -2 offset issue)
- Generation and segment number assignment to voxels
- World-space bifurcation angle calculation
- Left/right lung assignment

Author: Generated for issue fix
"""

import numpy as np
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
import nibabel as nib
from coordinate_utils import (
    angle_between, 
    compute_parent_direction, 
    compute_child_direction,
    get_axis_codes
)


def get_voxel_by_generation(connection_dict: Dict[int, Dict[str, Any]], 
                          seg_result: np.ndarray) -> np.ndarray:
    """
    Assign generation numbers to voxels using BFS from skeleton nodes.
    
    Fixed version that properly initializes with -1 and removes the -2 offset bug.
    
    Args:
        connection_dict: Tree structure with generation numbers
        seg_result: Binary segmentation array
        
    Returns:
        Array with generation numbers assigned to each voxel (-1 for background)
    """
    # Initialize result array with -1 (background)
    ret = np.full(seg_result.shape, -1, dtype=np.int32)
    
    # Create queue for BFS: (z, y, x, generation)
    queue = deque()
    
    # Seed BFS with skeleton nodes
    for node_id, node_data in connection_dict.items():
        if 'loc' in node_data and 'generation' in node_data:
            z, y, x = map(int, node_data['loc'])
            generation = node_data['generation']
            
            # Check bounds
            if (0 <= z < seg_result.shape[0] and 
                0 <= y < seg_result.shape[1] and 
                0 <= x < seg_result.shape[2]):
                # Only seed if it's foreground and not already assigned
                if seg_result[z, y, x] > 0 and ret[z, y, x] == -1:
                    ret[z, y, x] = generation
                    queue.append((z, y, x, generation))
    
    # BFS to propagate generation labels
    directions = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    
    while queue:
        z, y, x, generation = queue.popleft()
        
        # Check all 6-connected neighbors
        for dz, dy, dx in directions:
            nz, ny, nx = z + dz, y + dy, x + dx
            
            # Check bounds
            if (0 <= nz < seg_result.shape[0] and 
                0 <= ny < seg_result.shape[1] and 
                0 <= nx < seg_result.shape[2]):
                
                # Only assign if foreground and unassigned
                if seg_result[nz, ny, nx] > 0 and ret[nz, ny, nx] == -1:
                    ret[nz, ny, nx] = generation
                    queue.append((nz, ny, nx, generation))
    
    return ret


def get_voxel_by_segment_no(connection_dict: Dict[int, Dict[str, Any]], 
                           seg_result: np.ndarray) -> np.ndarray:
    """
    Assign segment numbers to voxels using BFS from skeleton nodes.
    
    Fixed version that properly initializes with -1 and removes the -2 offset bug.
    
    Args:
        connection_dict: Tree structure with segment numbers
        seg_result: Binary segmentation array
        
    Returns:
        Array with segment numbers assigned to each voxel (-1 for background)
    """
    # Initialize result array with -1 (background)
    ret = np.full(seg_result.shape, -1, dtype=np.int32)
    
    # Create queue for BFS: (z, y, x, segment_no)
    queue = deque()
    
    # Seed BFS with skeleton nodes
    for node_id, node_data in connection_dict.items():
        if 'loc' in node_data and 'segment_no' in node_data:
            z, y, x = map(int, node_data['loc'])
            segment_no = node_data['segment_no']
            
            # Check bounds
            if (0 <= z < seg_result.shape[0] and 
                0 <= y < seg_result.shape[1] and 
                0 <= x < seg_result.shape[2]):
                # Only seed if it's foreground and not already assigned
                if seg_result[z, y, x] > 0 and ret[z, y, x] == -1:
                    ret[z, y, x] = segment_no
                    queue.append((z, y, x, segment_no))
    
    # BFS to propagate segment labels
    directions = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    
    while queue:
        z, y, x, segment_no = queue.popleft()
        
        # Check all 6-connected neighbors
        for dz, dy, dx in directions:
            nz, ny, nx = z + dz, y + dy, x + dx
            
            # Check bounds
            if (0 <= nz < seg_result.shape[0] and 
                0 <= ny < seg_result.shape[1] and 
                0 <= nx < seg_result.shape[2]):
                
                # Only assign if foreground and unassigned
                if seg_result[nz, ny, nx] > 0 and ret[nz, ny, nx] == -1:
                    ret[nz, ny, nx] = segment_no
                    queue.append((nz, ny, nx, segment_no))
    
    return ret


def get_voxel_by_generation_without_bfs(connection_dict: Dict[int, Dict[str, Any]], 
                                       seg_result: np.ndarray) -> np.ndarray:
    """
    Assign generation numbers without BFS propagation (skeleton nodes only).
    
    Args:
        connection_dict: Tree structure with generation numbers
        seg_result: Binary segmentation array
        
    Returns:
        Array with generation numbers at skeleton locations only
    """
    # Initialize result array with -1 (background)
    ret = np.full(seg_result.shape, -1, dtype=np.int32)
    
    # Assign generation numbers only at skeleton nodes
    for node_id, node_data in connection_dict.items():
        if 'loc' in node_data and 'generation' in node_data:
            z, y, x = map(int, node_data['loc'])
            generation = node_data['generation']
            
            # Check bounds
            if (0 <= z < seg_result.shape[0] and 
                0 <= y < seg_result.shape[1] and 
                0 <= x < seg_result.shape[2]):
                ret[z, y, x] = generation
    
    return ret


def get_voxel_by_segment_no_without_bfs(connection_dict: Dict[int, Dict[str, Any]], 
                                       seg_result: np.ndarray) -> np.ndarray:
    """
    Assign segment numbers without BFS propagation (skeleton nodes only).
    
    Args:
        connection_dict: Tree structure with segment numbers
        seg_result: Binary segmentation array
        
    Returns:
        Array with segment numbers at skeleton locations only
    """
    # Initialize result array with -1 (background)
    ret = np.full(seg_result.shape, -1, dtype=np.int32)
    
    # Assign segment numbers only at skeleton nodes
    for node_id, node_data in connection_dict.items():
        if 'loc' in node_data and 'segment_no' in node_data:
            z, y, x = map(int, node_data['loc'])
            segment_no = node_data['segment_no']
            
            # Check bounds
            if (0 <= z < seg_result.shape[0] and 
                0 <= y < seg_result.shape[1] and 
                0 <= x < seg_result.shape[2]):
                ret[z, y, x] = segment_no
    
    return ret


def calculate_bifurcation_angles_world(connection_dict: Dict[int, Dict[str, Any]], 
                                     forward_steps: int = 4, 
                                     back_steps: int = 4,
                                     affine: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """
    Calculate bifurcation angles in world coordinate space.
    
    This function identifies bifurcation nodes and computes angles between
    parent and child branches using world coordinates.
    
    Args:
        connection_dict: Tree structure with world_loc coordinates
        forward_steps: Number of steps forward along children for direction calculation
        back_steps: Number of steps backward along parent for direction calculation
        affine: Affine matrix (not needed if world_loc already present)
        
    Returns:
        List of bifurcation angle dictionaries with keys:
        - bifurcation_id: Node ID of bifurcation
        - parent_id: Parent node ID (if exists)
        - child1_id, child2_id: Child node IDs
        - generation: Generation number
        - angle_parent_child1: Angle between parent and child1 (degrees)
        - angle_parent_child2: Angle between parent and child2 (degrees)
        - angle_between_children: Angle between child1 and child2 (degrees)
        - world_x, world_y, world_z: World coordinates of bifurcation
        - segment_no: Segment number
    """
    bifurcation_angles = []
    
    for node_id, node_data in connection_dict.items():
        children = node_data.get('children', [])
        
        # Only process bifurcation nodes (2+ children)
        if len(children) < 2:
            continue
        
        # Get world coordinates
        if 'world_loc' not in node_data:
            continue
        
        world_loc = node_data['world_loc']
        
        # Process first two children (binary bifurcation assumption)
        child1_id = children[0]
        child2_id = children[1]
        
        # Compute direction vectors
        parent_direction = compute_parent_direction(connection_dict, node_id, back_steps)
        child1_direction = compute_child_direction(connection_dict, node_id, child1_id, forward_steps)
        child2_direction = compute_child_direction(connection_dict, node_id, child2_id, forward_steps)
        
        # Initialize angles to NaN
        angle_parent_child1 = np.nan
        angle_parent_child2 = np.nan
        angle_between_children = np.nan
        
        # Calculate angles if directions are valid
        if parent_direction is not None and child1_direction is not None:
            angle_parent_child1 = np.degrees(angle_between(parent_direction, child1_direction))
        
        if parent_direction is not None and child2_direction is not None:
            angle_parent_child2 = np.degrees(angle_between(parent_direction, child2_direction))
        
        if child1_direction is not None and child2_direction is not None:
            angle_between_children = np.degrees(angle_between(child1_direction, child2_direction))
        
        # Create bifurcation angle record
        bifurcation_record = {
            'bifurcation_id': node_id,
            'parent_id': node_data.get('parent'),
            'child1_id': child1_id,
            'child2_id': child2_id,
            'generation': node_data.get('generation', -1),
            'angle_parent_child1': angle_parent_child1,
            'angle_parent_child2': angle_parent_child2,
            'angle_between_children': angle_between_children,
            'world_x': world_loc[0],
            'world_y': world_loc[1],
            'world_z': world_loc[2],
            'segment_no': node_data.get('segment_no', -1)
        }
        
        bifurcation_angles.append(bifurcation_record)
    
    return bifurcation_angles


def assign_left_right_world(connection_dict: Dict[int, Dict[str, Any]], 
                           affine: np.ndarray) -> None:
    """
    Assign left/right lung labels based on anatomical orientation.
    
    This function:
    1. Finds the carina (first generation=1 bifurcation)
    2. Uses world X coordinates and axis codes to determine anatomical sides
    3. Propagates side labels through the tree
    
    Args:
        connection_dict: Tree structure with world coordinates
        affine: Affine matrix for axis orientation
    """
    # Get axis orientation codes
    axis_codes = get_axis_codes(affine)
    x_axis_code = axis_codes[0]  # 'L', 'R', 'P', 'A', 'I', or 'S'
    
    # Find carina (first bifurcation at generation 1)
    carina_id = None
    for node_id, node_data in connection_dict.items():
        if (node_data.get('generation') == 1 and 
            len(node_data.get('children', [])) >= 2):
            carina_id = node_id
            break
    
    if carina_id is None:
        print("Warning: No carina bifurcation found (generation 1 with 2+ children)")
        return
    
    carina_data = connection_dict[carina_id]
    children = carina_data['children'][:2]  # Take first two children
    
    # Get world X coordinates of children
    child1_id, child2_id = children[0], children[1]
    child1_x = connection_dict[child1_id]['world_loc'][0]
    child2_x = connection_dict[child2_id]['world_loc'][0]
    
    # Determine anatomical sides based on X axis orientation
    if x_axis_code == 'L':  # Left-to-right axis (L = negative, R = positive)
        left_child_id = child1_id if child1_x < child2_x else child2_id
        right_child_id = child2_id if child1_x < child2_x else child1_id
    elif x_axis_code == 'R':  # Right-to-left axis (R = negative, L = positive)
        left_child_id = child1_id if child1_x > child2_x else child2_id
        right_child_id = child2_id if child1_x > child2_x else child1_id
    else:
        print(f"Warning: Unexpected X axis code '{x_axis_code}'. Using coordinate comparison.")
        # Fallback: assume standard orientation
        left_child_id = child1_id if child1_x < child2_x else child2_id
        right_child_id = child2_id if child1_x < child2_x else child1_id
    
    # Propagate side labels through subtrees
    def propagate_side(node_id: int, side: str) -> None:
        connection_dict[node_id]['side'] = side
        for child_id in connection_dict[node_id].get('children', []):
            propagate_side(child_id, side)
    
    propagate_side(left_child_id, 'left')
    propagate_side(right_child_id, 'right')
    
    # Mark carina and root
    connection_dict[carina_id]['side'] = 'carina'
    root_id = carina_data.get('parent')
    if root_id is not None:
        connection_dict[root_id]['side'] = 'trachea'


# LEGACY FUNCTIONS (marked for backward compatibility)
def calculate_bifurcation_angles_legacy(connection_dict: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    LEGACY: Old bifurcation angle calculation with index-space vectors.
    
    This function is kept for backward compatibility but should not be used
    in new code. Use calculate_bifurcation_angles_world instead.
    
    Args:
        connection_dict: Tree structure
        
    Returns:
        List of angle records (may have incorrect/biased angles)
    """
    print("WARNING: Using legacy bifurcation angle calculation")
    
    # Placeholder implementation with basic structure
    bifurcation_angles = []
    
    for node_id, node_data in connection_dict.items():
        children = node_data.get('children', [])
        if len(children) >= 2:
            # LEGACY: Would use incorrect axis permutations and index-space vectors
            bifurcation_record = {
                'bifurcation_id': node_id,
                'parent_id': node_data.get('parent'),
                'child1_id': children[0],
                'child2_id': children[1],
                'generation': node_data.get('generation', -1),
                'angle_parent_child1': 90.0,  # Placeholder
                'angle_parent_child2': 90.0,  # Placeholder
                'angle_between_children': 90.0,  # Placeholder
                # LEGACY: Incorrect axis ordering exports
                'x': node_data['loc'][1] if 'loc' in node_data else 0,  # Wrong axis
                'y': node_data['loc'][2] if 'loc' in node_data else 0,  # Wrong axis
                'z': node_data['loc'][0] if 'loc' in node_data else 0,  # Wrong axis
                'segment_no': node_data.get('segment_no', -1)
            }
            bifurcation_angles.append(bifurcation_record)
    
    return bifurcation_angles


def compute_voxel_volumes(generation_array: np.ndarray, 
                         segment_array: np.ndarray,
                         spacing: np.ndarray) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute volumes for each generation and segment in world units.
    
    Args:
        generation_array: Array with generation labels
        segment_array: Array with segment labels
        spacing: Voxel spacing [x, y, z] in world units
        
    Returns:
        Tuple of (generation_volumes, segment_volumes) dictionaries
    """
    voxel_volume = np.prod(spacing)
    
    # Compute generation volumes
    generation_volumes = {}
    unique_generations = np.unique(generation_array)
    for gen in unique_generations:
        if gen >= 0:  # Exclude background (-1)
            voxel_count = np.sum(generation_array == gen)
            generation_volumes[gen] = voxel_count * voxel_volume
    
    # Compute segment volumes
    segment_volumes = {}
    unique_segments = np.unique(segment_array)
    for seg in unique_segments:
        if seg >= 0:  # Exclude background (-1)
            voxel_count = np.sum(segment_array == seg)
            segment_volumes[seg] = voxel_count * voxel_volume
    
    return generation_volumes, segment_volumes


def validate_bfs_labeling(seg_result: np.ndarray, 
                         labeled_array: np.ndarray,
                         label_name: str = "labels") -> bool:
    """
    Validate that BFS labeling covers all foreground voxels.
    
    Args:
        seg_result: Original binary segmentation
        labeled_array: BFS-labeled array
        label_name: Name for error reporting
        
    Returns:
        True if validation passes, False otherwise
    """
    # Check that all foreground voxels are labeled
    foreground_mask = seg_result > 0
    unlabeled_foreground = np.logical_and(foreground_mask, labeled_array == -1)
    
    unlabeled_count = np.sum(unlabeled_foreground)
    total_foreground = np.sum(foreground_mask)
    
    if unlabeled_count > 0:
        print(f"Warning: {unlabeled_count}/{total_foreground} foreground voxels unlabeled in {label_name}")
        return False
    
    print(f"BFS labeling validation passed: all {total_foreground} foreground voxels labeled in {label_name}")
    return True