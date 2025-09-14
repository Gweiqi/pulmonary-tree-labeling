"""
Coordinate Utilities for Pulmonary Tree Labeling

This module provides centralized coordinate handling functions to fix spatial/angular
deviations between extracted airway centerlines and original NIfTI segmentation.

Key Features:
- NIfTI loading with proper affine handling
- Voxel to world coordinate conversion
- World coordinate augmentation for connection dictionaries
- Angle computation utilities in world space

Author: Generated for issue fix
"""

import numpy as np
import nibabel as nib
from typing import Tuple, Dict, Any, Optional, List


def load_nii(path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """
    Load NIfTI file and return data, affine matrix, and nibabel image object.
    
    Args:
        path: Path to .nii or .nii.gz file
        
    Returns:
        Tuple of (data array, affine matrix, nibabel image)
    """
    nii_img = nib.load(path)
    data = nii_img.get_fdata()
    affine = nii_img.affine
    return data, affine, nii_img


def get_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """
    Extract voxel spacing from affine matrix using column norms.
    
    Args:
        affine: 4x4 affine transformation matrix
        
    Returns:
        Array of [x_spacing, y_spacing, z_spacing]
    """
    # Calculate column norms of the 3x3 rotation/scaling part
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return spacing


def voxel_to_world(ijk_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert voxel coordinates to world coordinates using affine matrix.
    
    Args:
        ijk_coords: Nx3 array of voxel coordinates [i, j, k] (can be single point)
        affine: 4x4 affine transformation matrix
        
    Returns:
        Nx3 array of world coordinates [x, y, z]
    """
    # Ensure input is 2D array
    ijk_coords = np.atleast_2d(ijk_coords)
    
    # Add homogeneous coordinate
    homogeneous = np.ones((ijk_coords.shape[0], 4))
    homogeneous[:, :3] = ijk_coords
    
    # Apply affine transformation
    world_coords = (affine @ homogeneous.T).T
    
    # Return only x, y, z coordinates (drop homogeneous)
    return world_coords[:, :3]


def single_voxel_to_world(i: float, j: float, k: float, affine: np.ndarray) -> np.ndarray:
    """
    Convert single voxel coordinate to world coordinate.
    
    Args:
        i, j, k: Voxel coordinates
        affine: 4x4 affine transformation matrix
        
    Returns:
        Array of world coordinates [x, y, z]
    """
    return voxel_to_world(np.array([[i, j, k]]), affine)[0]


def add_world_loc_to_connection_dict(connection_dict: Dict[int, Dict[str, Any]], 
                                   affine: np.ndarray) -> None:
    """
    Add world coordinates to each node in connection_dict.
    
    Assumes connection_dict nodes have 'loc' key with [z, y, x] voxel coordinates.
    Adds 'world_loc' key with [x, y, z] world coordinates.
    
    Args:
        connection_dict: Dictionary mapping node_id -> node_data
        affine: 4x4 affine transformation matrix
    """
    for node_id, node_data in connection_dict.items():
        if 'loc' not in node_data:
            raise ValueError(f"Node {node_id} missing 'loc' field")
        
        # Convert from [z, y, x] voxel to world coordinates
        z, y, x = node_data['loc']
        # Note: nibabel/NIfTI convention expects [i, j, k] = [x, y, z] ordering for affine
        world_loc = single_voxel_to_world(x, y, z, affine)
        node_data['world_loc'] = world_loc


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculate angle between two vectors in radians.
    
    Args:
        u, v: 3D vectors
        
    Returns:
        Angle in radians (0 to π)
    """
    # Normalize vectors
    u_norm = u / np.linalg.norm(u)
    v_norm = v / np.linalg.norm(v)
    
    # Calculate angle using dot product
    cos_angle = np.clip(np.dot(u_norm, v_norm), -1.0, 1.0)
    return np.arccos(cos_angle)


def compute_branch_direction(connection_dict: Dict[int, Dict[str, Any]], 
                           start_node_id: int, 
                           end_node_id: int,
                           steps: int = 4) -> Optional[np.ndarray]:
    """
    Compute direction vector from start_node along branch toward end_node.
    
    Collects several nodes along the path and computes average direction.
    
    Args:
        connection_dict: Node dictionary with world_loc
        start_node_id: Starting node ID
        end_node_id: Target node ID (child or parent)
        steps: Number of steps to look ahead for direction calculation
        
    Returns:
        Direction vector or None if path not found
    """
    if start_node_id not in connection_dict or end_node_id not in connection_dict:
        return None
    
    # Simple implementation: direct vector between nodes
    # TODO: Implement path-following with multiple steps for smoothing
    start_world = connection_dict[start_node_id]['world_loc']
    end_world = connection_dict[end_node_id]['world_loc']
    
    direction = end_world - start_world
    if np.linalg.norm(direction) < 1e-10:
        return None
    
    return direction / np.linalg.norm(direction)


def compute_parent_direction(connection_dict: Dict[int, Dict[str, Any]], 
                           node_id: int,
                           steps: int = 4) -> Optional[np.ndarray]:
    """
    Compute direction vector from node toward its parent.
    
    Args:
        connection_dict: Node dictionary
        node_id: Node ID
        steps: Number of steps for direction calculation
        
    Returns:
        Direction vector toward parent or None if no parent
    """
    if node_id not in connection_dict:
        return None
    
    node_data = connection_dict[node_id]
    if 'parent' not in node_data or node_data['parent'] is None:
        return None
    
    parent_id = node_data['parent']
    return compute_branch_direction(connection_dict, node_id, parent_id, steps)


def compute_child_direction(connection_dict: Dict[int, Dict[str, Any]], 
                          node_id: int, 
                          child_id: int,
                          steps: int = 4) -> Optional[np.ndarray]:
    """
    Compute direction vector from node toward specified child.
    
    Args:
        connection_dict: Node dictionary
        node_id: Parent node ID
        child_id: Child node ID
        steps: Number of steps for direction calculation
        
    Returns:
        Direction vector toward child or None if not valid child
    """
    if node_id not in connection_dict or child_id not in connection_dict:
        return None
    
    return compute_branch_direction(connection_dict, node_id, child_id, steps)


def get_axis_codes(affine: np.ndarray) -> Tuple[str, str, str]:
    """
    Get axis orientation codes from affine matrix.
    
    Args:
        affine: 4x4 affine transformation matrix
        
    Returns:
        Tuple of (x_axis_code, y_axis_code, z_axis_code)
        Each code is one of: 'L'/'R' (left/right), 'P'/'A' (posterior/anterior), 'I'/'S' (inferior/superior)
    """
    axis_codes = nib.aff2axcodes(affine)
    return axis_codes


def calculate_world_distance(world_loc1: np.ndarray, world_loc2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two world coordinate points.
    
    Args:
        world_loc1, world_loc2: World coordinate arrays [x, y, z]
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(world_loc2 - world_loc1)


def validate_connection_dict_world_coords(connection_dict: Dict[int, Dict[str, Any]]) -> bool:
    """
    Validate that all nodes in connection_dict have valid world_loc coordinates.
    
    Args:
        connection_dict: Node dictionary
        
    Returns:
        True if all nodes have valid world_loc, False otherwise
    """
    for node_id, node_data in connection_dict.items():
        if 'world_loc' not in node_data:
            print(f"Node {node_id} missing world_loc")
            return False
        
        world_loc = node_data['world_loc']
        if not isinstance(world_loc, np.ndarray) or world_loc.shape != (3,):
            print(f"Node {node_id} has invalid world_loc shape: {world_loc.shape if hasattr(world_loc, 'shape') else type(world_loc)}")
            return False
        
        if not np.all(np.isfinite(world_loc)):
            print(f"Node {node_id} has non-finite world_loc: {world_loc}")
            return False
    
    return True