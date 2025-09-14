"""
Tree detection utilities for pulmonary tree labeling.

This module provides tree detection and skeletonization functions with proper
coordinate system handling and physical distance calculations.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from typing import Dict, List, Tuple, Optional
import networkx as nx
from coordinate_utils import physical_distance, physical_distance_matrix, make_isotropic


def tree_detection(segmentation: np.ndarray, 
                  spacing: np.ndarray,
                  min_branch_length: float = 2.0,
                  use_isotropic_skeletonization: bool = True,
                  connection_radius: float = 3.0,
                  branch_penalty_factor: float = 1.5) -> Dict:
    """
    Detect tree structure from binary segmentation with proper coordinate handling.
    
    Args:
        segmentation: Binary segmentation array (0=background, >0=foreground)
        spacing: Voxel spacing array (sx, sy, sz) corresponding to axes
        min_branch_length: Minimum branch length in physical units
        use_isotropic_skeletonization: If True, perform skeletonization on isotropic data
        connection_radius: Maximum connection radius in physical units
        branch_penalty_factor: Penalty factor for branch connections in graph construction
        
    Returns:
        result: Dictionary containing:
            - skeleton_points: Array of skeleton points in original voxel coordinates
            - connections: List of connections between skeleton points
            - graph: NetworkX graph representation
            - branch_points: Indices of branch points
            - endpoint_points: Indices of endpoint points
    """
    # Ensure binary segmentation
    binary_seg = (segmentation > 0).astype(np.uint8)
    
    if use_isotropic_skeletonization:
        # Perform skeletonization on isotropic data to avoid distortion
        iso_data, iso_spacing, scale_factors = make_isotropic(binary_seg, spacing)
        skeleton_iso = skeletonize(iso_data)
        
        # Convert skeleton back to original coordinate system
        skeleton_coords_iso = np.column_stack(np.where(skeleton_iso))
        skeleton_coords_orig = skeleton_coords_iso / scale_factors
        
        # Round to nearest integer coordinates
        skeleton_coords_orig = np.round(skeleton_coords_orig).astype(int)
        
        # Remove duplicates and ensure coordinates are within bounds
        skeleton_coords_orig = _filter_valid_coordinates(skeleton_coords_orig, binary_seg.shape)
        
    else:
        # Direct skeletonization (may be distorted for anisotropic voxels)
        skeleton = skeletonize(binary_seg)
        skeleton_coords_orig = np.column_stack(np.where(skeleton))
    
    # Build connectivity graph using physical distances
    graph, connections = _build_connectivity_graph(
        skeleton_coords_orig, spacing, connection_radius, branch_penalty_factor
    )
    
    # Identify branch and endpoint nodes
    branch_points, endpoint_points = _identify_branch_endpoints(graph)
    
    # Filter short branches
    if min_branch_length > 0:
        graph, skeleton_coords_orig, connections = _filter_short_branches(
            graph, skeleton_coords_orig, spacing, min_branch_length
        )
        # Recompute branch and endpoint points after filtering
        branch_points, endpoint_points = _identify_branch_endpoints(graph)
    
    return {
        'skeleton_points': skeleton_coords_orig,
        'connections': connections,
        'graph': graph,
        'branch_points': branch_points,
        'endpoint_points': endpoint_points,
        'spacing': spacing
    }


def _filter_valid_coordinates(coords: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Filter coordinates to ensure they are within volume bounds and remove duplicates."""
    # Remove coordinates outside bounds
    valid_mask = np.all((coords >= 0) & (coords < np.array(shape)), axis=1)
    coords = coords[valid_mask]
    
    # Remove duplicates
    coords = np.unique(coords, axis=0)
    
    return coords


def _build_connectivity_graph(skeleton_coords: np.ndarray, 
                             spacing: np.ndarray,
                             connection_radius: float,
                             branch_penalty_factor: float) -> Tuple[nx.Graph, List]:
    """
    Build connectivity graph using physical distances and Prim-style minimum spanning tree.
    """
    n_points = len(skeleton_coords)
    
    # Compute distance matrix in physical coordinates
    dist_matrix = physical_distance_matrix(skeleton_coords, spacing)
    
    # Create graph
    graph = nx.Graph()
    
    # Add nodes
    for i in range(n_points):
        graph.add_node(i, pos=skeleton_coords[i])
    
    # Add edges based on connection radius
    connections = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            distance = dist_matrix[i, j]
            if distance <= connection_radius:
                # Apply branch penalty for nodes that would create high-degree vertices
                weight = distance
                if graph.degree(i) > 1 or graph.degree(j) > 1:
                    weight *= branch_penalty_factor
                
                graph.add_edge(i, j, weight=weight, distance=distance)
                connections.append((i, j, distance))
    
    # Use minimum spanning tree to avoid cycles while preserving tree structure
    if graph.number_of_edges() > 0:
        mst = nx.minimum_spanning_tree(graph, weight='weight')
        
        # Update connections to only include MST edges
        connections = []
        for edge in mst.edges(data=True):
            i, j, data = edge
            connections.append((i, j, data['distance']))
        
        graph = mst
    
    return graph, connections


def _identify_branch_endpoints(graph: nx.Graph) -> Tuple[List[int], List[int]]:
    """Identify branch points (degree > 2) and endpoints (degree = 1)."""
    branch_points = []
    endpoint_points = []
    
    for node in graph.nodes():
        degree = graph.degree(node)
        if degree > 2:
            branch_points.append(node)
        elif degree == 1:
            endpoint_points.append(node)
    
    return branch_points, endpoint_points


def _filter_short_branches(graph: nx.Graph, 
                          skeleton_coords: np.ndarray,
                          spacing: np.ndarray,
                          min_length: float) -> Tuple[nx.Graph, np.ndarray, List]:
    """Remove branches shorter than minimum length."""
    filtered_graph = graph.copy()
    
    # Find all simple paths between endpoints and branch points
    endpoints = [n for n in graph.nodes() if graph.degree(n) == 1]
    branch_points = [n for n in graph.nodes() if graph.degree(n) > 2]
    
    # Identify short branches to remove
    nodes_to_remove = set()
    
    for endpoint in endpoints:
        # Find path from endpoint to first branch point
        try:
            # Find nearest branch point or another endpoint
            targets = branch_points + [e for e in endpoints if e != endpoint]
            if not targets:
                continue
                
            shortest_paths = []
            for target in targets:
                try:
                    path = nx.shortest_path(graph, endpoint, target)
                    if len(path) > 1:  # Valid path
                        shortest_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
            
            if not shortest_paths:
                continue
            
            # Take the shortest path
            path = min(shortest_paths, key=len)
            
            # Calculate path length in physical units
            path_length = 0.0
            for i in range(len(path) - 1):
                coord_a = skeleton_coords[path[i]]
                coord_b = skeleton_coords[path[i + 1]]
                path_length += physical_distance(coord_a, coord_b, spacing)
            
            # Remove if too short
            if path_length < min_length:
                # Remove all nodes in the path except the last one (which is a branch point)
                for node in path[:-1]:
                    nodes_to_remove.add(node)
                    
        except (nx.NetworkXNoPath, IndexError):
            continue
    
    # Remove identified nodes
    filtered_graph.remove_nodes_from(nodes_to_remove)
    
    # Update skeleton coordinates and connections
    remaining_nodes = list(filtered_graph.nodes())
    remaining_coords = skeleton_coords[remaining_nodes]
    
    # Create mapping from old to new indices
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_nodes)}
    
    # Update connections
    new_connections = []
    for edge in filtered_graph.edges(data=True):
        i, j, data = edge
        new_i = node_mapping[i]
        new_j = node_mapping[j]
        new_connections.append((new_i, new_j, data['distance']))
    
    # Create new graph with updated indices
    new_graph = nx.Graph()
    for new_idx, coord in enumerate(remaining_coords):
        new_graph.add_node(new_idx, pos=coord)
    
    for i, j, distance in new_connections:
        new_graph.add_edge(i, j, distance=distance)
    
    return new_graph, remaining_coords, new_connections


def get_tree_length(graph: nx.Graph, skeleton_coords: np.ndarray, spacing: np.ndarray) -> float:
    """
    Calculate total tree length using physical distances.
    
    Args:
        graph: NetworkX graph of tree structure
        skeleton_coords: Array of skeleton coordinates
        spacing: Voxel spacing array
        
    Returns:
        total_length: Total tree length in physical units
    """
    total_length = 0.0
    
    for edge in graph.edges():
        i, j = edge
        coord_a = skeleton_coords[i]
        coord_b = skeleton_coords[j]
        total_length += physical_distance(coord_a, coord_b, spacing)
    
    return total_length


def extract_centerline_points(tree_result: Dict, 
                             order_by_distance: bool = True) -> List[Dict]:
    """
    Extract centerline points with metadata.
    
    Args:
        tree_result: Result dictionary from tree_detection()
        order_by_distance: If True, order points by distance from root
        
    Returns:
        centerline_points: List of dictionaries with point information
    """
    skeleton_coords = tree_result['skeleton_points']
    graph = tree_result['graph']
    spacing = tree_result['spacing']
    branch_points = set(tree_result['branch_points'])
    endpoint_points = set(tree_result['endpoint_points'])
    
    centerline_points = []
    
    for i, coord in enumerate(skeleton_coords):
        point_info = {
            'voxel_coords': coord,
            'point_index': i,
            'is_branch': i in branch_points,
            'is_endpoint': i in endpoint_points,
            'degree': graph.degree(i) if i in graph else 0
        }
        centerline_points.append(point_info)
    
    if order_by_distance and len(centerline_points) > 0:
        # Order by distance from first endpoint (arbitrary root)
        endpoints = [p for p in centerline_points if p['is_endpoint']]
        if endpoints:
            root_coord = endpoints[0]['voxel_coords']
            for point in centerline_points:
                point['distance_from_root'] = physical_distance(
                    root_coord, point['voxel_coords'], spacing
                )
            centerline_points.sort(key=lambda x: x['distance_from_root'])
    
    return centerline_points