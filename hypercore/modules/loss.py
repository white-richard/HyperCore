import torch
from pytorch_metric_learning.distances import BaseDistance
from pytorch_metric_learning import losses, miners, reducers
from hypercore.utils.manifold_distance import ManifoldDistance
from hypercore.manifolds.lorentzian import Lorentz

class LorentzTripletLoss(torch.nn.Module):
    """
    Triplet loss in the Lorentz model of hyperbolic space.
    Args:
        manifold: instance of a Lorentz manifold class from hypercore.manifolds
        margin: margin for the triplet loss
        type_of_triplets: one of "all", "hard", "semihard", "easy", or None
            (if None, no mining is done)
    """
    def __init__(self, manifold:Lorentz, margin=1.0, type_of_triplets="semihard", normalize_embeddings=True):
        super().__init__()
        self.manifold = manifold
        self.margin = float(margin)
        distance = ManifoldDistance(manifold, normalize_embeddings=normalize_embeddings)
        # MeanReducer more stable norm than AverageNonZeroReducer
        # when number of triplets varies between batches?
        reducer = reducers.MeanReducer()
        self.loss = losses.TripletMarginLoss(
            margin=margin, 
            distance=distance, 
            reducer=reducer
            )
        self.miner = None
        if type_of_triplets is not None:
            self.miner = miners.TripletMarginMiner(
                margin=margin, type_of_triplets=type_of_triplets, distance=distance
            )
   
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: in the Lorentz model, shape (B, D+1)
            labels: shape (B,)
        Returns:
            loss value
        """
        if self.miner is not None:
            a, p, n = self.miner(embeddings, labels)
            return self.loss(embeddings, labels, (a, p, n)), len(a)
        else:
            return self.loss(embeddings, labels)
