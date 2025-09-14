"""
Coordinate utilities for pulmonary tree labeling.

This module provides functions for handling coordinate transformations, 
NIfTI loading with proper spacing, and physical distance calculations.
Solves coordinate system inconsistencies by providing unified interfaces.
"""

import numpy as np
import nibabel as nib
from scipy import ndimage
from typing import Tuple, Optional, Union


def load_nifti(path: str, as_canonical: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load NIfTI file with proper coordinate handling.
    
    Args:
        path: Path to .nii or .nii.gz file
        as_canonical: If True, reorient to canonical RAS+ orientation
        
    Returns:
        data: Image data array with shape (i, j, k) corresponding to (x, y, z)
        affine: 4x4 affine transformation matrix
        spacing: Voxel spacing array (sx, sy, sz) corresponding to (x, y, z) directions
    """
    img = nib.load(path)
    
    if as_canonical:
        # Reorient to canonical RAS+ if needed
        img = nib.as_closest_canonical(img)
    
    data = img.get_fdata()
    affine = img.affine
    
    # Extract spacing from affine matrix
    # The spacing is the length of each column vector in the rotation part
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    
    return data, affine, spacing


def to_world(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert voxel indices to world coordinates using affine transformation.
    
    Args:
        indices: Array of voxel indices with shape (..., 3) in (i, j, k) order
        affine: 4x4 affine transformation matrix
        
    Returns:
        world_coords: World coordinates with shape (..., 3) in (x, y, z) order
    """
    original_shape = indices.shape
    indices_flat = indices.reshape(-1, 3)
    
    # Add homogeneous coordinate
    indices_homog = np.column_stack([indices_flat, np.ones(indices_flat.shape[0])])
    
    # Apply affine transformation
    world_coords = (affine @ indices_homog.T).T[:, :3]
    
    return world_coords.reshape(original_shape)


def to_voxel(world_points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert world coordinates to voxel indices using inverse affine transformation.
    
    Args:
        world_points: Array of world coordinates with shape (..., 3) in (x, y, z) order
        affine: 4x4 affine transformation matrix
        
    Returns:
        indices: Voxel indices with shape (..., 3) in (i, j, k) order
    """
    original_shape = world_points.shape
    world_flat = world_points.reshape(-1, 3)
    
    # Add homogeneous coordinate
    world_homog = np.column_stack([world_flat, np.ones(world_flat.shape[0])])
    
    # Apply inverse affine transformation
    affine_inv = np.linalg.inv(affine)
    voxel_coords = (affine_inv @ world_homog.T).T[:, :3]
    
    return voxel_coords.reshape(original_shape)


def make_isotropic(data: np.ndarray, spacing: np.ndarray, 
                   target_spacing: Optional[float] = None,
                   order: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample anisotropic data to isotropic spacing.
    
    Args:
        data: Input data array
        spacing: Current voxel spacing (sx, sy, sz)
        target_spacing: Target isotropic spacing. If None, uses minimum of current spacing
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns:
        resampled_data: Resampled data array
        new_spacing: New isotropic spacing array
        scale_factors: Scaling factors applied to each dimension
    """
    if target_spacing is None:
        target_spacing = np.min(spacing)
    
    # Calculate scale factors
    scale_factors = spacing / target_spacing
    
    # Resample data
    resampled_data = ndimage.zoom(data, scale_factors, order=order)
    
    # New spacing
    new_spacing = np.array([target_spacing, target_spacing, target_spacing])
    
    return resampled_data, new_spacing, scale_factors


def physical_vector(loc_a: np.ndarray, loc_b: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """
    Compute physical vector between two voxel locations.
    
    Args:
        loc_a: Source location in voxel coordinates (i, j, k)
        loc_b: Target location in voxel coordinates (i, j, k)
        spacing: Voxel spacing (sx, sy, sz)
        
    Returns:
        physical_vec: Physical vector in world units
    """
    voxel_diff = np.array(loc_b) - np.array(loc_a)
    return voxel_diff * spacing


def physical_distance(loc_a: np.ndarray, loc_b: np.ndarray, spacing: np.ndarray) -> float:
    """
    Compute physical distance between two voxel locations.
    
    Args:
        loc_a: Source location in voxel coordinates (i, j, k)
        loc_b: Target location in voxel coordinates (i, j, k)
        spacing: Voxel spacing (sx, sy, sz)
        
    Returns:
        distance: Physical distance in world units
    """
    physical_vec = physical_vector(loc_a, loc_b, spacing)
    return np.linalg.norm(physical_vec)


def physical_distance_matrix(locations: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """
    Compute pairwise physical distance matrix between locations.
    
    Args:
        locations: Array of locations with shape (N, 3) in voxel coordinates
        spacing: Voxel spacing (sx, sy, sz)
        
    Returns:
        dist_matrix: Symmetric distance matrix with shape (N, N)
    """
    N = len(locations)
    dist_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            dist = physical_distance(locations[i], locations[j], spacing)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


def get_world_x_coordinate(voxel_coords: np.ndarray, spacing: np.ndarray, 
                          affine: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Get world X coordinate for left/right lung determination.
    
    Args:
        voxel_coords: Voxel coordinates with shape (..., 3) in (i, j, k) order
        spacing: Voxel spacing (sx, sy, sz)
        affine: Optional affine matrix. If None, uses simple scaling
        
    Returns:
        world_x: World X coordinates for left/right determination
    """
    if affine is not None:
        # Use proper affine transformation
        world_coords = to_world(voxel_coords, affine)
        return world_coords[..., 0]  # X coordinate
    else:
        # Fallback: simple scaling (assumes canonical orientation)
        return voxel_coords[..., 0] * spacing[0]


def validate_coordinate_consistency(skeleton_points: np.ndarray, 
                                   segmentation: np.ndarray, 
                                   spacing: np.ndarray,
                                   tolerance_factor: float = 0.5) -> Tuple[bool, float]:
    """
    Validate that skeleton points are within reasonable distance of segmentation.
    
    Args:
        skeleton_points: Array of skeleton points in voxel coordinates (N, 3)
        segmentation: Binary segmentation array
        spacing: Voxel spacing (sx, sy, sz)
        tolerance_factor: Tolerance as fraction of minimum spacing
        
    Returns:
        is_valid: True if skeleton is consistent with segmentation
        max_distance: Maximum distance found between skeleton and segmentation
    """
    tolerance = tolerance_factor * np.min(spacing)
    
    # Find segmentation boundary points
    seg_coords = np.column_stack(np.where(segmentation > 0))
    
    max_distance = 0.0
    
    for point in skeleton_points:
        # Find nearest segmentation voxel
        distances = np.array([physical_distance(point, seg_coord, spacing) 
                             for seg_coord in seg_coords])
        min_distance = np.min(distances)
        max_distance = max(max_distance, min_distance)
        
        if min_distance > tolerance:
            return False, max_distance
    
    return True, max_distance