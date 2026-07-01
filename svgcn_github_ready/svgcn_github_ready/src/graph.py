import torch
from sklearn.neighbors import NearestNeighbors


def make_knn_edges_from_scaled_features(X_scaled, k: int = 15):
    """Build bidirectional KNN edges using cosine distance."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nbrs.fit(X_scaled)
    _, indices = nbrs.kneighbors(X_scaled)

    edge_set = set()
    for i in range(X_scaled.shape[0]):
        for j in indices[i][1:]:
            edge_set.add((i, j))
            edge_set.add((j, i))

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    return edge_index
