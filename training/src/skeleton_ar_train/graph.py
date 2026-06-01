"""Skeleton graph adjacency utilities for ST-GCN.

Two skeleton topologies are supported:

* ``coco``        - the 17-keypoint COCO layout produced by RTMPose, used
                    by the live C++ video pipeline.
* ``ntu-rgb+d``   - the 25-joint Kinect-v2 layout of the NTU-RGB+D
                    dataset, which is the topology the original ST-GCN
                    paper trained on. The bundled NTU training uses this.

For each layout we build three partition matrices following Yan et al.:

    A_self        - identity (self-loops)
    A_centripetal - edges toward the body centroid
    A_centrifugal - edges away from the body centroid

Distances are measured in graph hops to a per-layout centroid joint.
"""

from __future__ import annotations

import numpy as np

# COCO-17 (RTMPose). (parent, child) directed away from the centroid.
COCO17_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 7), (7, 9), (6, 8), (8, 10),         # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (11, 12), (5, 11), (6, 12),      # torso
    (0, 5), (0, 6), (3, 4),                  # head <-> shoulders, ear-ear
]

# NTU-RGB+D 25-joint Kinect v2 skeleton (0-indexed; the canonical bone
# list from the NTU/ST-GCN/CTR-GCN literature, shifted from 1-based).
NTU25_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 20), (2, 20), (3, 2),
    (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10),
    (12, 0), (13, 12), (14, 13), (15, 14),
    (16, 0), (17, 16), (18, 17), (19, 18),
    (21, 22), (22, 7), (23, 24), (24, 11),
]

# layout -> (num_nodes, edges, centroid/root joints)
_LAYOUTS: dict[str, tuple[int, list[tuple[int, int]], tuple[int, ...]]] = {
    "coco": (17, COCO17_EDGES, (5, 6, 11, 12)),
    "ntu-rgb+d": (25, NTU25_EDGES, (20,)),  # joint 20 = spine ("center")
    "ntu": (25, NTU25_EDGES, (20,)),
}

# Backwards-compatible defaults (COCO).
NUM_NODES = 17
ROOT_JOINTS = (5, 6, 11, 12)


def layout_num_nodes(layout: str = "coco") -> int:
    return _LAYOUTS[layout][0]


def build_adjacency(layout: str = "coco") -> np.ndarray:
    """Symmetric binary adjacency for the given skeleton layout.

    Returns:
        np.ndarray of shape (V, V) with 1.0 on edges and 0.0 elsewhere.
        The diagonal is zero (self-loops live in their own partition).
    """
    num_nodes, edges, _ = _LAYOUTS[layout]
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
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


def partition_adjacency(layout: str = "coco") -> np.ndarray:
    """Three-way (self / centripetal / centrifugal) partitioning.

    Following Yan et al., a directed edge (i, j) is centripetal when j
    is closer to the centroid than i, and centrifugal otherwise.

    Args:
        layout: "coco" (17 joints) or "ntu-rgb+d" (25 joints).

    Returns:
        np.ndarray of shape (3, V, V), each slice symmetrically
        normalized.
    """
    _, _, root_joints = _LAYOUTS[layout]
    A = build_adjacency(layout)
    distances_to_centroid = np.min(
        shortest_path_distances(A)[:, list(root_joints)], axis=1
    )

    A_self = np.eye(A.shape[0], dtype=np.float32)
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
