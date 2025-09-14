"""
Tree Detection and Construction for Pulmonary Airways

This module implements tree detection and construction algorithms for pulmonary
airway centerline extraction with proper coordinate handling.

Key Features:
- Regularized tree construction using MST/Prim-like algorithm
- World coordinate integration
- Deterministic root selection
- Generation and segment number assignment

Author: Generated for issue fix
"""

import numpy as np
import heapq
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict, deque
from coordinate_utils import (
    add_world_loc_to_connection_dict, 
    calculate_world_distance,
    validate_connection_dict_world_coords
)


def get_skeleton_from_segmentation(segmentation: np.ndarray, 
                                 threshold: float = 0.5) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    Extract skeleton centerline from binary segmentation.
    
    This is a placeholder implementation. In practice, this would use
    skeletonization algorithms like skimage.morphology.skeletonize_3d.
    
    Args:
        segmentation: 3D binary array
        threshold: Threshold for binary conversion
        
    Returns:
        Tuple of (skeleton array, initial center_dict with voxel coordinates)
    """
    from skimage.morphology import skeletonize
    
    # Convert to binary
    binary_seg = segmentation > threshold
    
    # Skeletonize (works for 3D data)
    skeleton = skeletonize(binary_seg)
    
    # Extract centerline points
    center_points = np.argwhere(skeleton > 0)
    
    # Create initial center_dict with [z, y, x] ordering as per legacy format
    center_dict = {}
    for i, (z, y, x) in enumerate(center_points):
        center_dict[i] = {
            'loc': [z, y, x],
            'radius': 1.0,  # Placeholder radius
            'id': i
        }
    
    return skeleton, center_dict


def create_nearby_dict(center_dict: Dict[int, Dict[str, Any]], 
                      max_distance: float = 5.0) -> Dict[int, List[int]]:
    """
    Create dictionary of nearby nodes for each node based on Euclidean distance.
    
    Args:
        center_dict: Dictionary of centerline nodes with 'loc' or 'world_loc'
        max_distance: Maximum distance to consider as "nearby"
        
    Returns:
        Dictionary mapping node_id -> list of nearby node_ids
    """
    nearby_dict = defaultdict(list)
    node_ids = list(center_dict.keys())
    
    for i, node_id1 in enumerate(node_ids):
        for node_id2 in node_ids[i+1:]:
            node1 = center_dict[node_id1]
            node2 = center_dict[node_id2]
            
            # Use world coordinates if available, otherwise voxel coordinates
            if 'world_loc' in node1 and 'world_loc' in node2:
                dist = calculate_world_distance(node1['world_loc'], node2['world_loc'])
            else:
                # Fallback to voxel coordinates (less accurate)
                loc1 = np.array(node1['loc'])
                loc2 = np.array(node2['loc'])
                dist = np.linalg.norm(loc2 - loc1)
            
            if dist <= max_distance:
                nearby_dict[node_id1].append(node_id2)
                nearby_dict[node_id2].append(node_id1)
    
    return dict(nearby_dict)


def build_tree_regularized(center_dict: Dict[int, Dict[str, Any]], 
                         nearby_dict: Dict[int, List[int]], 
                         affine: np.ndarray,
                         branch_penalty: float = 16.0,
                         max_children: int = 2) -> Dict[int, Dict[str, Any]]:
    """
    Build regularized tree structure using MST/Prim-like algorithm.
    
    This implements a deterministic tree construction with:
    - Root selection by minimal z-coordinate (superior/inlet assumption)
    - Min-heap based on squared world distance
    - Branch penalty to prefer elongation before branching
    - Child limit enforcement
    
    Args:
        center_dict: Node dictionary with locations
        nearby_dict: Nearby nodes for each node
        affine: Affine transformation matrix for world coordinates
        branch_penalty: Additional cost penalty for creating new branches
        max_children: Maximum children per node (typically 2 for binary tree)
        
    Returns:
        Updated connection_dict with parent/children relationships
    """
    if not center_dict:
        return {}
    
    # Ensure world coordinates are available
    if 'world_loc' not in next(iter(center_dict.values())):
        add_world_loc_to_connection_dict(center_dict, affine)
    
    # Initialize connection_dict as copy of center_dict
    connection_dict = {k: dict(v) for k, v in center_dict.items()}
    
    # Initialize tree structure fields
    for node_id in connection_dict:
        connection_dict[node_id]['parent'] = None
        connection_dict[node_id]['children'] = []
        connection_dict[node_id]['generation'] = -1
        connection_dict[node_id]['segment_no'] = -1
    
    # Select root: node with minimal z-coordinate (superior assumption)
    # TODO: Add option for root selection by largest local radius
    root_id = min(connection_dict.keys(), 
                 key=lambda nid: connection_dict[nid]['world_loc'][2])  # z-coordinate
    
    connection_dict[root_id]['generation'] = 0
    connection_dict[root_id]['segment_no'] = 0
    
    # Priority queue: (cost, parent_id, child_id)
    pq = []
    visited = {root_id}
    
    # Add initial edges from root
    for neighbor_id in nearby_dict.get(root_id, []):
        if neighbor_id not in visited:
            world_dist = calculate_world_distance(
                connection_dict[root_id]['world_loc'],
                connection_dict[neighbor_id]['world_loc']
            )
            cost = world_dist ** 2  # Squared distance
            heapq.heappush(pq, (cost, root_id, neighbor_id))
    
    # Prim-like algorithm with regularization
    while pq:
        cost, parent_id, child_id = heapq.heappop(pq)
        
        # Skip if child already visited
        if child_id in visited:
            continue
        
        # Check if parent already has max children
        if len(connection_dict[parent_id]['children']) >= max_children:
            # Apply branch penalty and re-add to queue
            penalized_cost = cost + branch_penalty
            heapq.heappush(pq, (penalized_cost, parent_id, child_id))
            continue
        
        # Add edge to tree
        connection_dict[parent_id]['children'].append(child_id)
        connection_dict[child_id]['parent'] = parent_id
        visited.add(child_id)
        
        # Add new edges from this child
        for neighbor_id in nearby_dict.get(child_id, []):
            if neighbor_id not in visited:
                world_dist = calculate_world_distance(
                    connection_dict[child_id]['world_loc'],
                    connection_dict[neighbor_id]['world_loc']
                )
                cost = world_dist ** 2
                # Add small penalty if this node already has children (encourage elongation)
                if connection_dict[child_id]['children']:
                    cost += branch_penalty * 0.1
                heapq.heappush(pq, (cost, child_id, neighbor_id))
    
    # Assign generation and segment numbers via DFS
    _assign_generation_segment_dfs(connection_dict, root_id)
    
    return connection_dict


def _assign_generation_segment_dfs(connection_dict: Dict[int, Dict[str, Any]], 
                                 root_id: int) -> None:
    """
    Assign generation and segment numbers using depth-first search.
    
    Args:
        connection_dict: Tree structure dictionary
        root_id: Root node ID
    """
    segment_counter = 0
    
    def dfs(node_id: int, generation: int, segment_no: int) -> None:
        nonlocal segment_counter
        
        connection_dict[node_id]['generation'] = generation
        connection_dict[node_id]['segment_no'] = segment_no
        
        children = connection_dict[node_id]['children']
        
        if len(children) == 0:
            # Leaf node, no further processing
            return
        elif len(children) == 1:
            # Continuation, same segment
            dfs(children[0], generation, segment_no)
        else:
            # Bifurcation, new segments for children
            for child_id in children:
                segment_counter += 1
                dfs(child_id, generation + 1, segment_counter)
    
    dfs(root_id, 0, 0)


def tree_detection(segmentation: np.ndarray, 
                  affine: np.ndarray,
                  threshold: float = 0.5,
                  max_nearby_distance: float = 5.0,
                  branch_penalty: float = 16.0) -> Dict[int, Dict[str, Any]]:
    """
    Main tree detection function.
    
    Args:
        segmentation: 3D binary segmentation array
        affine: 4x4 affine transformation matrix from NIfTI
        threshold: Binary threshold for segmentation
        max_nearby_distance: Maximum distance for nearby node detection
        branch_penalty: Penalty for branching in tree construction
        
    Returns:
        connection_dict with tree structure and world coordinates
    """
    # Extract skeleton centerline
    skeleton, center_dict = get_skeleton_from_segmentation(segmentation, threshold)
    
    if not center_dict:
        return {}
    
    # Create nearby relationships
    nearby_dict = create_nearby_dict(center_dict, max_nearby_distance)
    
    # Build regularized tree
    connection_dict = build_tree_regularized(center_dict, nearby_dict, affine, branch_penalty)
    
    # Validate world coordinates
    if not validate_connection_dict_world_coords(connection_dict):
        raise ValueError("Tree construction failed: invalid world coordinates")
    
    return connection_dict


# LEGACY FUNCTIONS (for backward compatibility)
def get_connection_dict_legacy(center_dict: Dict[int, Dict[str, Any]], 
                             nearby_dict: Dict[int, List[int]]) -> Dict[int, Dict[str, Any]]:
    """
    LEGACY: Old tree construction method.
    
    This function is kept for backward compatibility but should not be used
    in new code. Use build_tree_regularized instead.
    
    Args:
        center_dict: Node dictionary
        nearby_dict: Nearby nodes dictionary
        
    Returns:
        Connection dictionary with basic tree structure
    """
    # LEGACY implementation placeholder
    # This would contain the old problematic tree construction logic
    print("WARNING: Using legacy tree construction method")
    
    # Initialize with basic parent/child structure
    connection_dict = {k: dict(v) for k, v in center_dict.items()}
    for node_id in connection_dict:
        connection_dict[node_id]['parent'] = None
        connection_dict[node_id]['children'] = []
        connection_dict[node_id]['generation'] = 0
        connection_dict[node_id]['segment_no'] = 0
    
    return connection_dict


def calculate_segment_lengths_world(connection_dict: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
    """
    Calculate segment lengths using world coordinates.
    
    Args:
        connection_dict: Tree structure with world_loc
        
    Returns:
        Dictionary mapping segment_no -> total length in world units
    """
    segment_lengths = defaultdict(float)
    
    for node_id, node_data in connection_dict.items():
        if node_data['parent'] is not None:
            parent_id = node_data['parent']
            parent_data = connection_dict[parent_id]
            
            # Calculate distance between consecutive nodes
            distance = calculate_world_distance(
                node_data['world_loc'], 
                parent_data['world_loc']
            )
            
            segment_no = node_data['segment_no']
            segment_lengths[segment_no] += distance
    
    return dict(segment_lengths)


def find_root_node(connection_dict: Dict[int, Dict[str, Any]]) -> Optional[int]:
    """
    Find the root node (node with no parent).
    
    Args:
        connection_dict: Tree structure dictionary
        
    Returns:
        Root node ID or None if not found
    """
    for node_id, node_data in connection_dict.items():
        if node_data.get('parent') is None:
            return node_id
    return None


def get_leaf_nodes(connection_dict: Dict[int, Dict[str, Any]]) -> List[int]:
    """
    Get all leaf nodes (nodes with no children).
    
    Args:
        connection_dict: Tree structure dictionary
        
    Returns:
        List of leaf node IDs
    """
    leaf_nodes = []
    for node_id, node_data in connection_dict.items():
        if not node_data.get('children', []):
            leaf_nodes.append(node_id)
    return leaf_nodes


def get_bifurcation_nodes(connection_dict: Dict[int, Dict[str, Any]]) -> List[int]:
    """
    Get all bifurcation nodes (nodes with 2+ children).
    
    Args:
        connection_dict: Tree structure dictionary
        
    Returns:
        List of bifurcation node IDs
    """
    bifurcation_nodes = []
    for node_id, node_data in connection_dict.items():
        if len(node_data.get('children', [])) >= 2:
            bifurcation_nodes.append(node_id)
    return bifurcation_nodes