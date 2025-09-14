"""
Airway area utilities for pulmonary tree labeling.

This module provides unified functions for angle calculations, left/right lung
determination, and other airway analysis tasks with proper coordinate handling.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
from coordinate_utils import physical_vector, physical_distance, get_world_x_coordinate


def calculate_bifurcation_angles(tree_result: Dict, 
                                connection_dict: Optional[Dict] = None,
                                angle_threshold: float = 30.0) -> Dict[int, float]:
    """
    Calculate bifurcation angles at branch points with proper coordinate handling.
    
    Args:
        tree_result: Result from tree_detection() containing graph and coordinates
        connection_dict: Optional connection dictionary (legacy compatibility)
        angle_threshold: Minimum angle in degrees to consider as significant bifurcation
        
    Returns:
        angles: Dictionary mapping branch point indices to their bifurcation angles
    """
    graph = tree_result['graph']
    skeleton_coords = tree_result['skeleton_points']
    spacing = tree_result['spacing']
    branch_points = tree_result['branch_points']
    
    angles = {}
    
    for branch_idx in branch_points:
        # Get all neighbors of this branch point
        neighbors = list(graph.neighbors(branch_idx))
        
        if len(neighbors) < 2:
            continue  # Not a true bifurcation
        
        branch_coord = skeleton_coords[branch_idx]
        
        # Calculate vectors from branch point to each neighbor in physical space
        vectors = []
        for neighbor_idx in neighbors:
            neighbor_coord = skeleton_coords[neighbor_idx]
            vector = physical_vector(branch_coord, neighbor_coord, spacing)
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vectors.append(vector / norm)
        
        if len(vectors) < 2:
            continue
        
        # Calculate all pairwise angles between branch vectors
        branch_angles = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                # Calculate angle between vectors
                dot_product = np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0)
                angle_rad = np.arccos(dot_product)
                angle_deg = np.degrees(angle_rad)
                branch_angles.append(angle_deg)
        
        if branch_angles:
            # Use the minimum angle as the bifurcation angle
            min_angle = min(branch_angles)
            if min_angle >= angle_threshold:
                angles[branch_idx] = min_angle
    
    return angles


def determine_left_right_lung(skeleton_coords: np.ndarray,
                             spacing: np.ndarray,
                             affine: Optional[np.ndarray] = None,
                             midline_threshold: float = 0.0) -> np.ndarray:
    """
    Determine left/right lung classification for skeleton points.
    
    Args:
        skeleton_coords: Array of skeleton coordinates in voxel space (N, 3)
        spacing: Voxel spacing (sx, sy, sz)
        affine: Optional affine matrix for proper world coordinate transformation
        midline_threshold: World X coordinate threshold for left/right separation
        
    Returns:
        labels: Array of labels (0=left, 1=right) for each skeleton point
    """
    # Get world X coordinates for left/right determination
    world_x = get_world_x_coordinate(skeleton_coords, spacing, affine)
    
    if midline_threshold == 0.0:
        # Use median as automatic midline
        midline_threshold = np.median(world_x)
    
    # Classify based on world X coordinate
    # Assuming RAS+ orientation: left is higher X, right is lower X
    labels = (world_x < midline_threshold).astype(int)  # 0=left, 1=right
    
    return labels


def bfs_label_airways(tree_result: Dict, 
                     root_points: Optional[List[int]] = None,
                     generation_labels: bool = True) -> Dict[int, int]:
    """
    Label airway segments using breadth-first search from root points.
    
    Args:
        tree_result: Result from tree_detection()
        root_points: List of root point indices. If None, uses endpoints
        generation_labels: If True, labels represent generation; if False, segment IDs
        
    Returns:
        labels: Dictionary mapping point indices to their labels
    """
    graph = tree_result['graph']
    skeleton_coords = tree_result['skeleton_points']
    
    if root_points is None:
        # Use endpoints as roots
        root_points = tree_result['endpoint_points']
    
    if not root_points:
        return {}
    
    labels = {}
    visited = set()
    
    if generation_labels:
        # BFS for generation labeling
        queue = deque([(root_idx, 0) for root_idx in root_points])
        
        while queue:
            current_idx, generation = queue.popleft()
            
            if current_idx in visited:
                continue
            
            visited.add(current_idx)
            labels[current_idx] = generation
            
            # Add unvisited neighbors with incremented generation
            for neighbor_idx in graph.neighbors(current_idx):
                if neighbor_idx not in visited:
                    queue.append((neighbor_idx, generation + 1))
    
    else:
        # BFS for segment labeling
        segment_id = 0
        
        for root_idx in root_points:
            if root_idx in visited:
                continue
            
            # BFS from this root
            queue = deque([root_idx])
            current_segment_points = []
            
            while queue:
                current_idx = queue.popleft()
                
                if current_idx in visited:
                    continue
                
                visited.add(current_idx)
                current_segment_points.append(current_idx)
                
                # Add unvisited neighbors
                for neighbor_idx in graph.neighbors(current_idx):
                    if neighbor_idx not in visited:
                        queue.append(neighbor_idx)
            
            # Assign segment ID to all points in this connected component
            for point_idx in current_segment_points:
                labels[point_idx] = segment_id
            
            segment_id += 1
    
    return labels


def extract_branch_segments(tree_result: Dict) -> List[Dict]:
    """
    Extract individual branch segments between branch points and endpoints.
    
    Args:
        tree_result: Result from tree_detection()
        
    Returns:
        segments: List of segment dictionaries with metadata
    """
    graph = tree_result['graph']
    skeleton_coords = tree_result['skeleton_points']
    spacing = tree_result['spacing']
    branch_points = set(tree_result['branch_points'])
    endpoint_points = set(tree_result['endpoint_points'])
    
    segments = []
    visited_edges = set()
    
    # Find all simple paths between critical points (branch points and endpoints)
    critical_points = list(branch_points | endpoint_points)
    
    for start_point in critical_points:
        for neighbor in graph.neighbors(start_point):
            edge = tuple(sorted([start_point, neighbor]))
            if edge in visited_edges:
                continue
            
            # Trace path until reaching another critical point
            path = [start_point]
            current = neighbor
            prev = start_point
            
            while current not in critical_points:
                path.append(current)
                # Find next node (should be unique for tree structure)
                next_nodes = [n for n in graph.neighbors(current) if n != prev]
                if not next_nodes:
                    break
                prev = current
                current = next_nodes[0]
            
            # Add final critical point
            path.append(current)
            
            # Mark all edges in this path as visited
            for i in range(len(path) - 1):
                edge = tuple(sorted([path[i], path[i + 1]]))
                visited_edges.add(edge)
            
            # Calculate segment properties
            segment_coords = skeleton_coords[path]
            
            # Calculate segment length
            segment_length = 0.0
            for i in range(len(path) - 1):
                coord_a = skeleton_coords[path[i]]
                coord_b = skeleton_coords[path[i + 1]]
                segment_length += physical_distance(coord_a, coord_b, spacing)
            
            segment_info = {
                'path_indices': path,
                'coordinates': segment_coords,
                'length': segment_length,
                'start_point': path[0],
                'end_point': path[-1],
                'start_is_branch': path[0] in branch_points,
                'end_is_branch': path[-1] in branch_points,
                'start_is_endpoint': path[0] in endpoint_points,
                'end_is_endpoint': path[-1] in endpoint_points
            }
            
            segments.append(segment_info)
    
    return segments


def calculate_airway_metrics(tree_result: Dict, 
                           segmentation: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate comprehensive airway metrics.
    
    Args:
        tree_result: Result from tree_detection()
        segmentation: Optional original segmentation for cross-sectional area calculation
        
    Returns:
        metrics: Dictionary containing various airway measurements
    """
    graph = tree_result['graph']
    skeleton_coords = tree_result['skeleton_points']
    spacing = tree_result['spacing']
    
    # Basic tree metrics
    total_length = sum(physical_distance(skeleton_coords[i], skeleton_coords[j], spacing)
                      for i, j in graph.edges())
    
    num_branches = len(tree_result['branch_points'])
    num_endpoints = len(tree_result['endpoint_points'])
    num_segments = len(extract_branch_segments(tree_result))
    
    # Calculate bifurcation angles
    bifurcation_angles = calculate_bifurcation_angles(tree_result)
    avg_bifurcation_angle = np.mean(list(bifurcation_angles.values())) if bifurcation_angles else 0.0
    
    # Calculate segment lengths
    segments = extract_branch_segments(tree_result)
    segment_lengths = [seg['length'] for seg in segments]
    
    metrics = {
        'total_length': total_length,
        'num_branches': num_branches,
        'num_endpoints': num_endpoints,
        'num_segments': num_segments,
        'avg_bifurcation_angle': avg_bifurcation_angle,
        'bifurcation_angles': bifurcation_angles,
        'segment_lengths': segment_lengths,
        'avg_segment_length': np.mean(segment_lengths) if segment_lengths else 0.0,
        'max_segment_length': np.max(segment_lengths) if segment_lengths else 0.0,
        'min_segment_length': np.min(segment_lengths) if segment_lengths else 0.0
    }
    
    return metrics


def export_centerline_csv(tree_result: Dict, 
                         filename: str,
                         affine: Optional[np.ndarray] = None,
                         include_world_coords: bool = True) -> None:
    """
    Export centerline points to CSV with both voxel and world coordinates.
    
    Args:
        tree_result: Result from tree_detection()
        filename: Output CSV filename
        affine: Optional affine matrix for world coordinate calculation
        include_world_coords: If True, include world coordinates in output
    """
    import csv
    from coordinate_utils import to_world
    
    skeleton_coords = tree_result['skeleton_points']
    spacing = tree_result['spacing']
    branch_points = set(tree_result['branch_points'])
    endpoint_points = set(tree_result['endpoint_points'])
    
    # Calculate world coordinates if requested
    world_coords = None
    if include_world_coords and affine is not None:
        world_coords = to_world(skeleton_coords, affine)
    
    # Prepare header
    header = ['point_id', 'voxel_x', 'voxel_y', 'voxel_z', 'is_branch', 'is_endpoint']
    if include_world_coords:
        header.extend(['world_x', 'world_y', 'world_z'])
    
    # Write CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for i, coord in enumerate(skeleton_coords):
            row = [
                i,  # point_id
                coord[0], coord[1], coord[2],  # voxel coordinates
                i in branch_points,  # is_branch
                i in endpoint_points  # is_endpoint
            ]
            
            if include_world_coords and world_coords is not None:
                row.extend([world_coords[i, 0], world_coords[i, 1], world_coords[i, 2]])
            
            writer.writerow(row)