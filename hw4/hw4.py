import csv
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data(filepath: str) -> List[Dict[str, str]]:
    """Read CSV file and return a list of dictionaries, one per row."""
    rows = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def calc_features(row: Dict[str, str]) -> np.ndarray:
    """Given one row (dict), return a numpy array of shape (9,) 
        containing the 9 country features
    """
    keys = [
        "child_mort",
        "exports",
        "health",
        "imports",
        "income",
        "inflation",
        "life_expec",
        "total_fer",
        "gdpp",
    ]
    vals = []
    for k in keys:
        v = row[k]
        fv = float(v)
        vals.append(fv)
    arr = np.array(vals, dtype=np.float64)
    arr = arr.reshape((9,))
    return arr


def hac(features: List[np.ndarray], linkage_type: str) -> np.ndarray:
    """Perform hierarchical agglomerative clustering (single or complete linkage).

    Args:
        features: list of numpy arrays shape (9,)
        linkage_type: "single" or "complete"

    Returns:
        Z: numpy array shape (n-1, 4) where columns are [idx1, idx2, dist, size]
    """
    if linkage_type not in ("single", "complete"):
        raise ValueError("linkage_type must be 'single' or 'complete'")

    n = len(features)
    if n == 0:
        return np.zeros((0, 4))

    # Precompute pairwise Euclidean distances between original points
    X = np.vstack(features)  # shape (n, 9)
    # distance_matrix between original points
    from scipy.spatial import distance
    pdist = distance.squareform(distance.pdist(X, metric='euclidean'))

    # We'll maintain a distance matrix for clusters up to size 2n-1
    max_clusters = 2 * n - 1
    dist_mat = np.full((max_clusters, max_clusters), np.inf, dtype=np.float64)

    # fill distances between original points
    dist_mat[:n, :n] = pdist
    # diagonal to inf
    for i in range(n):
        dist_mat[i, i] = np.inf

    # active set: True for cluster indices currently present
    active = [False] * max_clusters
    for i in range(n):
        active[i] = True

    # cluster sizes
    sizes = [0] * max_clusters
    for i in range(n):
        sizes[i] = 1

    Z = np.zeros((n - 1, 4), dtype=np.float64)

    next_cluster_idx = n

    # Perform n-1 merges
    for merge_i in range(n - 1):
        # find pair of active clusters with minimum distance
        best_dist = np.inf
        best_i = -1
        best_j = -1
        # tie-breaking: iterate i ascending, j ascending ensures smallest i then smallest j
        for i in range(max_clusters):
            if not active[i]:
                continue
            # j > i
            for j in range(i + 1, max_clusters):
                if not active[j]:
                    continue
                d = dist_mat[i, j]
                if d < best_dist:
                    best_dist = d
                    best_i = i
                    best_j = j
        # best_i, best_j found
        if best_i == -1:
            raise RuntimeError("No clusters to merge")

        # ensure ordering (best_i < best_j) - already by loop
        i_idx, j_idx = best_i, best_j

        # record merge
        Z[merge_i, 0] = float(min(i_idx, j_idx))
        Z[merge_i, 1] = float(max(i_idx, j_idx))
        Z[merge_i, 2] = float(best_dist)
        new_size = sizes[i_idx] + sizes[j_idx]
        Z[merge_i, 3] = float(new_size)

        # create new cluster at next_cluster_idx
        new_idx = next_cluster_idx
        next_cluster_idx += 1

        # update distances between new cluster and all other active clusters using
        # single: min(d(A,K), d(B,K))
        # complete: max(d(A,K), d(B,K))
        for k in range(max_clusters):
            if k == i_idx or k == j_idx:
                continue
            if not active[k]:
                continue
            if linkage_type == "single":
                d_new = min(dist_mat[i_idx, k], dist_mat[j_idx, k])
            else:  # complete
                d_new = max(dist_mat[i_idx, k], dist_mat[j_idx, k])
            dist_mat[new_idx, k] = d_new
            dist_mat[k, new_idx] = d_new

        # set new cluster active
        active[new_idx] = True
        sizes[new_idx] = new_size

        # mark old clusters inactive
        active[i_idx] = False
        active[j_idx] = False

        # set distances from merged clusters to inf to avoid reusing
        dist_mat[i_idx, :] = np.inf
        dist_mat[:, i_idx] = np.inf
        dist_mat[j_idx, :] = np.inf
        dist_mat[:, j_idx] = np.inf

        # set self-distance
        dist_mat[new_idx, new_idx] = np.inf

    return Z


def fig_hac(Z: np.ndarray, names: List[str]):
    """Create a matplotlib figure visualizing the hierarchical clustering using SciPy's dendrogram.

    Returns the matplotlib.figure.Figure object.
    """
    fig = plt.figure()
    # SciPy dendrogram expects a linkage matrix similar to Z
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(Z, labels=names, leaf_rotation=90)
    fig.tight_layout()
    return fig


def normalize_features(features: List[np.ndarray]) -> List[np.ndarray]:
    """
        Normalize features column-wise (z-score) using mean and std computed across the
        provided list 

        fertility rate feature is usually between 0-7 but income is usually between 0-100000
        so we want to normalize them to have mean 0 and std 1 that way no feature dominates 
    """
    if len(features) == 0:
        return []
    X = np.vstack(features)  # shape (n, 9)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # avoid division by zero -- if sigma==0 set to 1 so values become 0 after centering
    sigma_fixed = np.where(sigma == 0, 1.0, sigma)
    X_norm = (X - mu) / sigma_fixed

    # return as list of arrays dtype float64
    return [np.array(X_norm[i], dtype=np.float64) for i in range(X_norm.shape[0])]

if __name__ == "__main__":
    data = load_data("Country-data.csv")
    features = [calc_features(row) for row in data]
    names = [row["country"] for row in data]
    features_normalized = normalize_features(features)
    np.savetxt("output.txt", features_normalized)
    n = 20
    Z = hac(features[:n], linkage_type="complete")
    fig = fig_hac(Z, names[:n])
    plt.show()