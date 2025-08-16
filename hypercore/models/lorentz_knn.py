from typing import Optional

import torch
import numpy as np
from geoopt import Manifold

class LorentzKNeighborsClassifier:
    def __init__(self, manifold: Manifold, n_neighbors: int = 5):
        self.manifold = manifold
        self.n_neighbors = n_neighbors
        self.X_fit_: Optional[torch.Tensor] = None # (N, d+1) on hyperboloid
        self.y_fit_: Optional[np.ndarray] = None

    def fit(self, X, y):
        X = torch.as_tensor(X, dtype=torch.float32)
        # ensure points are on the hyperboloid (canonicalized)
        self.X_fit_ = self.manifold.projx(X)
        self.y_fit_ = np.asarray(y)
        return self

    @torch.no_grad()
    def kneighbors(self, X, n_neighbors: Optional[int] = None, return_distance: bool = True):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        Xq = self.manifold.projx(torch.as_tensor(X, dtype=torch.float32))  # (Q, d+1)
        # pairwise Lorentz geodesic distances (Q, N)
        dists = self.manifold.dist(Xq.unsqueeze(1), self.X_fit_.unsqueeze(0))
        vals, idx = torch.topk(dists, k=n_neighbors, largest=False, dim=1)
        if return_distance:
            return vals.cpu().numpy(), idx.cpu().numpy()
        return idx.cpu().numpy()

    def predict(self, X):
        _, idx = self.kneighbors(X, return_distance=True)
        neighbor_labels = self.y_fit_[idx]
        # majority vote
        preds = [np.bincount(row, minlength=int(self.y_fit_.max())+1).argmax() for row in neighbor_labels]
        return np.asarray(preds)