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

    def __init__(
        self,
        manifold: Lorentz,
        margin=0.05,
        type_of_triplets="batchhard",
        type_of_reducer="AvgNonZeroReducer",
        normalize_embeddings=False,
        use_soft_margin=True,
        swap=False,
    ):
        super().__init__()
        self.manifold = manifold
        self.margin = float(margin)
        self.miner = None

        if type_of_reducer == "MeanReducer":
            reducer = reducers.MeanReducer()
        elif type_of_reducer == "AvgNonZeroReducer":
            reducer = reducers.AvgNonZeroReducer()
        else:
            raise NotImplemented

        distance = ManifoldDistance(manifold, normalize_embeddings=normalize_embeddings)

        self.loss = losses.TripletMarginLoss(
            margin=margin,
            distance=distance,
            reducer=reducer,
            smooth_loss=use_soft_margin,
            swap=swap,
        )

        if type_of_triplets is not None:
            if type_of_triplets == "batchhard":
                self.miner = miners.BatchHardMiner(distance=distance)
            else:
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
