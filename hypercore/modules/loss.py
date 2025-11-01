import torch
from pytorch_metric_learning.distances import BaseDistance
from pytorch_metric_learning import losses, miners


class ManifoldDistance(BaseDistance):
    """
    Wraps manifold.dist(x, y, keepdim=False, dim=-1) so it can be used
    with pytorch-metric-learning losses and miners.
    """
    def __init__(self, manifold, scale=0.0, **kwargs):
        super().__init__(normalize_embeddings=False, is_inverted=False, **kwargs)
        self.manifold = manifold
        self.scale = scale

    def compute_mat(self, query_emb, ref_emb):
        query_emb = query_emb.to(torch.float64)
        ref_emb = ref_emb.to(torch.float64)
        if ref_emb is None:
            ref_emb = query_emb
        mat = self.pairwise_distance(query_emb, ref_emb) # [N,M]
        return mat.to(torch.float32)
    
    def pairwise_distance(self, query_emb, ref_emb):
        dist = self.manifold.pairwise_distance(query_emb, ref_emb, keepdim=False, dim=-1)  # [N]
        if self.scale and self.scale != 0.0:
            dist = self.scale * dist
        return dist.to(torch.float32)


class LorentzTripletLoss(torch.nn.Module):
    """
    Triplet loss in the Lorentz model of hyperbolic space.
    Args:
        manifold: instance of a Lorentz manifold class from hypercore.manifolds
        margin: margin for the triplet loss
        scale: scaling factor for distances (default 0.0, i.e. no scaling)
        type_of_triplets: one of "all", "hard", "semihard", "easy", or None
            (if None, no mining is done)
    """
    def __init__(self, manifold, margin=1.0, scale=0.0, type_of_triplets="semihard"):
        super().__init__()
        self.manifold = manifold
        self.margin = float(margin)
        self.scale = float(scale)
        self.dist = ManifoldDistance(manifold, scale=0.0)
        self.loss = losses.TripletMarginLoss(margin=margin, distance=self.dist)
        self.miner = None
        if type_of_triplets is not None:
            self.miner = miners.TripletMarginMiner(
                margin=margin, type_of_triplets=type_of_triplets, distance=self.dist
            )
   
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: in the Lorentz model, shape (B, D+1)
            labels: shape (B,)
        Returns:
            loss value
        """
        if hasattr(self, 'miner'):
            hard_pairs = self.miner(embeddings, labels)
            return self.loss(embeddings, labels, hard_pairs)
        else:
            return self.loss(embeddings, labels)
    
class LorentzArcFaceLoss(torch.nn.Module):
    """
    ArcFace loss in the Lorentz model of hyperbolic space, using pytorch-metric-learning.
    Args:
        manifold: instance of a Lorentz manifold class from hypercore.manifolds
        scale: scaling factor for distances (default 0.0, i.e. no scaling)
        margin: angular margin for ArcFace
    """
    def __init__(self, manifold, num_classes, embedding_size, scale=0.0, margin=0.5):
        super().__init__()
        self.manifold = manifold
        self.scale = float(scale)
        self.margin = float(margin)
        self.dist = ManifoldDistance(manifold)

        self.loss = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=57.3 * margin,
            scale=scale,
        )

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: in the Lorentz model, shape (B, D+1)
            labels: shape (B,)
        Returns:
            loss value
        """
        loss = self.loss(embeddings, labels)
        return loss