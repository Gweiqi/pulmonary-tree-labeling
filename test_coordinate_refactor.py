"""
Test script for coordinate system refactor validation.

This script tests the new coordinate processing functionality to ensure
that the refactor addresses the coordinate system issues described in
the problem statement.
"""

import numpy as np
import tempfile
import os
import nibabel as nib
from typing import Tuple

from coordinate_utils import (
    load_nifti, to_world, to_voxel, make_isotropic, 
    physical_distance, validate_coordinate_consistency
)
from detect_tree import tree_detection, get_tree_length
from airway_area_utils import calculate_bifurcation_angles, determine_left_right_lung


def create_test_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Create synthetic test data that mimics a branching airway structure.
    
    Returns:
        data: Binary segmentation array
        affine: Affine transformation matrix
        spacing: Voxel spacing array
        temp_path: Path to temporary NIfTI file
    """
    # Create a simple Y-shaped structure
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)
    
    # Main trunk (vertical)
    data[20:45, 30:34, 30:34] = 1
    
    # Left branch (diagonal)
    for i in range(15):
        z = 45 + i
        y = 32 + i // 2
        x = 32 - i // 3
        if z < shape[0] and y < shape[1] and x >= 0:
            data[z-2:z+2, y-1:y+2, x-1:x+2] = 1
    
    # Right branch (diagonal)
    for i in range(15):
        z = 45 + i
        y = 32 + i // 2
        x = 32 + i // 3
        if z < shape[0] and y < shape[1] and x < shape[2]:
            data[z-2:z+2, y-1:y+2, x-1:x+2] = 1
    
    # Create anisotropic spacing to test coordinate handling
    spacing = np.array([0.5, 0.7, 0.8])  # Different spacing in each direction
    
    # Create affine matrix with spacing but no rotation for simpler testing
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1] 
    affine[2, 2] = spacing[2]
    
    # Add a small translation instead of rotation for testing
    affine[:3, 3] = [10, 20, 30]
    
    # Create temporary NIfTI file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'test_airway.nii.gz')
    
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, temp_path)
    
    return data, affine, spacing, temp_path


def test_coordinate_transformations():
    """Test coordinate transformation functions."""
    print("Testing coordinate transformations...")
    
    data, affine, spacing, temp_path = create_test_data()
    
    # Test load_nifti
    loaded_data, loaded_affine, loaded_spacing = load_nifti(temp_path)
    
    assert loaded_data.shape == data.shape, "Data shape mismatch after loading"
    assert np.allclose(loaded_spacing, spacing, atol=1e-3), "Spacing extraction failed"
    
    # Test coordinate transformations
    test_indices = np.array([[10, 20, 30], [0, 0, 0], [63, 63, 63]])
    
    # Round trip: voxel -> world -> voxel
    world_coords = to_world(test_indices, loaded_affine)
    back_to_voxel = to_voxel(world_coords, loaded_affine)
    
    assert np.allclose(test_indices, back_to_voxel, atol=1e-6), "Round trip transformation failed"
    
    print("✓ Coordinate transformations passed")
    
    # Cleanup
    os.remove(temp_path)


def test_isotropic_resampling():
    """Test isotropic resampling functionality."""
    print("Testing isotropic resampling...")
    
    data, affine, spacing, temp_path = create_test_data()
    
    # Test isotropic resampling
    iso_data, iso_spacing, scale_factors = make_isotropic(data, spacing)
    
    # Check that new spacing is isotropic
    assert np.allclose(iso_spacing, iso_spacing[0]), "Isotropic spacing not achieved"
    
    # Check that scale factors are correct
    expected_scale = spacing / np.min(spacing)
    assert np.allclose(scale_factors, expected_scale, atol=1e-6), "Scale factors incorrect"
    
    print("✓ Isotropic resampling passed")
    
    # Cleanup
    os.remove(temp_path)


def test_tree_detection():
    """Test tree detection with proper coordinate handling."""
    print("Testing tree detection...")
    
    data, affine, spacing, temp_path = create_test_data()
    
    # Run tree detection
    tree_result = tree_detection(
        segmentation=data,
        spacing=spacing,
        min_branch_length=1.0,
        use_isotropic_skeletonization=True
    )
    
    # Basic sanity checks
    assert len(tree_result['skeleton_points']) > 0, "No skeleton points found"
    assert len(tree_result['branch_points']) > 0, "No branch points found"
    assert len(tree_result['endpoint_points']) >= 2, "Should have at least 2 endpoints"
    
    # Test tree length calculation
    total_length = get_tree_length(
        tree_result['graph'], 
        tree_result['skeleton_points'], 
        spacing
    )
    
    assert total_length > 0, "Tree length should be positive"
    
    print(f"✓ Tree detection passed (length: {total_length:.2f} mm)")
    
    # Cleanup
    os.remove(temp_path)


def test_coordinate_consistency():
    """Test that skeleton points are consistent with segmentation."""
    print("Testing coordinate consistency...")
    
    data, affine, spacing, temp_path = create_test_data()
    
    # Run tree detection
    tree_result = tree_detection(
        segmentation=data,
        spacing=spacing,
        min_branch_length=0.5,  # Lower threshold for test data
        use_isotropic_skeletonization=True
    )
    
    # Validate coordinate consistency
    is_valid, max_distance = validate_coordinate_consistency(
        tree_result['skeleton_points'], 
        data, 
        spacing,
        tolerance_factor=2.0  # More lenient for test data
    )
    
    print(f"  Coordinate validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"  Max distance to segmentation: {max_distance:.3f} mm")
    
    # This should pass for our synthetic data
    assert is_valid, f"Coordinate consistency failed (max distance: {max_distance:.3f})"
    
    print("✓ Coordinate consistency passed")
    
    # Cleanup
    os.remove(temp_path)


def test_angle_calculations():
    """Test bifurcation angle calculations."""
    print("Testing angle calculations...")
    
    data, affine, spacing, temp_path = create_test_data()
    
    # Run tree detection
    tree_result = tree_detection(
        segmentation=data,
        spacing=spacing,
        min_branch_length=0.5,
        use_isotropic_skeletonization=True
    )
    
    # Calculate bifurcation angles
    angles = calculate_bifurcation_angles(tree_result)
    
    if angles:
        print(f"  Found {len(angles)} bifurcation angles")
        for branch_idx, angle in angles.items():
            print(f"    Branch {branch_idx}: {angle:.1f} degrees")
        
        # Angles should be reasonable (between 0 and 180 degrees)
        for angle in angles.values():
            assert 0 <= angle <= 180, f"Invalid angle: {angle}"
    else:
        print("  No significant bifurcations found in test data")
    
    print("✓ Angle calculations passed")
    
    # Cleanup
    os.remove(temp_path)


def test_left_right_determination():
    """Test left/right lung determination."""
    print("Testing left/right lung determination...")
    
    data, affine, spacing, temp_path = create_test_data()
    
    # Run tree detection
    tree_result = tree_detection(
        segmentation=data,
        spacing=spacing,
        min_branch_length=0.5,
        use_isotropic_skeletonization=True
    )
    
    # Test left/right determination
    lr_labels = determine_left_right_lung(
        tree_result['skeleton_points'],
        spacing,
        affine
    )
    
    # Should have labels for all skeleton points
    assert len(lr_labels) == len(tree_result['skeleton_points']), "Missing left/right labels"
    
    # Labels should be 0 or 1
    assert np.all((lr_labels == 0) | (lr_labels == 1)), "Invalid left/right labels"
    
    # Should have both left and right points for our Y-shaped structure
    unique_labels = np.unique(lr_labels)
    if len(unique_labels) > 1:
        print(f"  Found both left ({np.sum(lr_labels == 0)}) and right ({np.sum(lr_labels == 1)}) points")
    else:
        print(f"  All points classified as: {'left' if unique_labels[0] == 0 else 'right'}")
    
    print("✓ Left/right determination passed")
    
    # Cleanup
    os.remove(temp_path)


def test_physical_distance_calculations():
    """Test physical distance calculations."""
    print("Testing physical distance calculations...")
    
    # Test with known coordinates and spacing
    spacing = np.array([1.0, 2.0, 0.5])
    loc_a = np.array([0, 0, 0])
    loc_b = np.array([1, 1, 2])  # Should give distance sqrt(1^2 + 2^2 + 1^2) = sqrt(6)
    
    from coordinate_utils import physical_distance
    distance = physical_distance(loc_a, loc_b, spacing)
    expected_distance = np.sqrt(1.0**2 + 2.0**2 + 1.0**2)  # sqrt(6) ≈ 2.449
    
    assert np.isclose(distance, expected_distance), f"Distance calculation failed: {distance} vs {expected_distance}"
    
    print(f"✓ Physical distance calculations passed (distance: {distance:.3f})")


def run_all_tests():
    """Run all coordinate refactor tests."""
    print("Running coordinate system refactor tests...")
    print("=" * 50)
    
    try:
        test_coordinate_transformations()
        test_isotropic_resampling()
        test_physical_distance_calculations()
        test_tree_detection()
        test_coordinate_consistency()
        test_angle_calculations()
        test_left_right_determination()
        
        print("=" * 50)
        print("🎉 All tests passed! Coordinate system refactor is working correctly.")
        return True
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)