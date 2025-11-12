import torch
from pytorch_metric_learning.distances import BaseDistance
from hypercore.manifolds.lorentzian import Lorentz


class ManifoldDistance(BaseDistance):
    """
    Wraps geodesic distance computations on a given Lorentz manifold to be compatible
    with pytorch-metric-learning losses and miners.
    Args:
        manifold: An instance of the Lorentz manifold from hypercore.
        distance: The type of distance to compute. ('geodesic' | 'lorentz')
    """
    def __init__(self, manifold: Lorentz, distance:str='geodesic', **kwargs):
        super().__init__(is_inverted=False, **kwargs)
        self.manifold = manifold
        self.distance = distance
        self.collect_stats = False 
        assert not self.normalize_embeddings
        assert self.power == 1
        assert self.is_inverted is False

    def forward(self, query_emb, ref_emb=None):
        self.reset_stats()
        self.check_shapes(query_emb, ref_emb)
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        self.set_default_stats(
            query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
        )
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        # if self.power != 1:
        #     mat = mat**self.power
        assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        return mat

    def compute_mat(self, query_emb, ref_emb):
        dtype = query_emb.dtype
        if ref_emb is None:
            ref_emb = query_emb
        
        query_emb = query_emb.to(torch.float64)
        ref_emb = ref_emb.to(torch.float64)

        mat = self.pairwise_distance(query_emb, ref_emb)  # [N,M]
        return mat.to(dtype)
    
    def pairwise_distance(self, query_emb, ref_emb):
        return self.manifold.pairwise_distance(query_emb, ref_emb, keepdim=False, dim=-1, distance=self.distance)  # [N]

    def smallest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.max(*args, **kwargs)
        return torch.min(*args, **kwargs)

    def largest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.min(*args, **kwargs)
        return torch.max(*args, **kwargs)

    # This measures the margin between x and y
    def margin(self, x, y):
        if self.is_inverted:
            return y - x
        return x - y

    def normalize(self, embeddings, dim=1, **kwargs):
        return self.manifold.normalize(embeddings, dim=dim, **kwargs)

    def maybe_normalize(self, embeddings, dim=1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings

    def get_norm(self, embeddings, dim=1, **kwargs):
        raise NotImplementedError
        return self.manifold.norm(embeddings, dim=dim, **kwargs)

    def set_default_stats(
        self, query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
    ):
        if self.collect_stats:
            raise NotImplementedError
            with torch.no_grad():
                self.initial_avg_query_norm = torch.mean(
                    self.get_norm(query_emb)
                ).item()
                self.initial_avg_ref_norm = torch.mean(self.get_norm(ref_emb)).item()
                self.final_avg_query_norm = torch.mean(
                    self.get_norm(query_emb_normalized)
                ).item()
                self.final_avg_ref_norm = torch.mean(
                    self.get_norm(ref_emb_normalized)
                ).item()

    def check_shapes(self, query_emb, ref_emb):
        if query_emb.ndim != 2 or (ref_emb is not None and ref_emb.ndim != 2):
            raise ValueError(
                "embeddings must be a 2D tensor of shape (batch_size, embedding_size)"
            )