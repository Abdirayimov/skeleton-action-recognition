"""COCO-17 skeleton graph adjacency utilities.

The 17 COCO keypoints are connected by 19 edges (head, shoulders, arms,
hips, legs). For ST-GCN we build three partition matrices following the
original paper:

    A_self        - identity (self-loops)
    A_centripetal - edges toward the body centroid
    A_centrifugal - edges away from the body centroid

The centroid for COCO is the midpoint of the shoulders + hips (joints
5, 6, 11, 12). Distances are measured in graph hops.
"""

from __future__ import annotations

import numpy as np

# (parent, child) - directed away from the body centroid for clarity.
COCO17_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (11, 12), (5, 11), (6, 12),     # torso
    (0, 5), (0, 6), (3, 4),                 # head <-> shoulders, ear-ear
]
NUM_NODES = 17
ROOT_JOINTS = (5, 6, 11, 12)


def build_adjacency(num_nodes: int = NUM_NODES) -> np.ndarray:
    """Symmetric binary adjacency for the COCO-17 skeleton.

    Returns:
        np.ndarray of shape (num_nodes, num_nodes) with 1.0 on edges
        and 0.0 elsewhere. The diagonal is zero (self-loops live in
        their own partition).
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in COCO17_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def shortest_path_distances(A: np.ndarray) -> np.ndarray:
    """Floyd-Warshall over the unweighted graph defined by `A`."""
    n = A.shape[0]
    D = np.full((n, n), np.inf, dtype=np.float32)
    D[A > 0] = 1.0
    np.fill_diagonal(D, 0.0)
    for k in range(n):
        D = np.minimum(D, D[:, k : k + 1] + D[k : k + 1, :])
    return D


def normalize(A: np.ndarray) -> np.ndarray:
    """Symmetric normalization D^{-1/2} A D^{-1/2}."""
    deg = A.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    Dn = np.diag(inv_sqrt)
    return Dn @ A @ Dn


def partition_adjacency(num_nodes: int = NUM_NODES) -> np.ndarray:
    """Three-way (self / centripetal / centrifugal) partitioning.

    Following Yan et al., a directed edge (i, j) is centripetal when j
    is closer to the centroid than i, and centrifugal otherwise.

    Returns:
        np.ndarray of shape (3, num_nodes, num_nodes), each slice
        symmetrically normalized.
    """
    A = build_adjacency(num_nodes)
    distances_to_centroid = np.min(
        shortest_path_distances(A)[:, list(ROOT_JOINTS)], axis=1
    )

    A_self = np.eye(num_nodes, dtype=np.float32)
    A_in = np.zeros_like(A)
    A_out = np.zeros_like(A)

    for i, j in zip(*np.where(A > 0)):
        if distances_to_centroid[j] < distances_to_centroid[i]:
            A_in[i, j] = 1.0
        elif distances_to_centroid[j] > distances_to_centroid[i]:
            A_out[i, j] = 1.0
        else:
            # Same hop count from centroid: split evenly.
            A_in[i, j] = 0.5
            A_out[i, j] = 0.5

    return np.stack(
        [normalize(A_self), normalize(A_in), normalize(A_out)],
        axis=0,
    )
