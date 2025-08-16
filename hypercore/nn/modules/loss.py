from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from geoopt import Manifold

import hypercore.nn as hnn

def dist_matrix(manifold: Manifold,
                u: torch.Tensor,
                v: torch.Tensor | None = None,
                chunk: int = 2048) -> torch.Tensor:
    if v is None:
        v = u
    N, M = u.size(0), v.size(0)
    out = u.new_empty(N, M)
    for i in range(0, N, chunk):
        ui = u[i:i+chunk]  # (ni, d+1)
        # broadcasted pairwise geodesic distances
        Dij = manifold.dist(ui.unsqueeze(1), v.unsqueeze(0), keepdim=False, dim=-1)
        out[i:i+chunk] = Dij
    return out

def lift_spatial_to_hyperboloid_with_linner(x_spatial, c, l_inner):
    """
    x_spatial: (..., d)  channel-last.
    l_inner:   your Lorentz inner; must accept (..., 1+d) inputs.
    """
    zeros_t = torch.zeros_like(x_spatial[..., :1])                               # (..., 1)
    tmp = torch.cat([zeros_t, x_spatial], dim=-1)                                # (..., 1+d) with time=0
    r2 = l_inner(tmp, tmp, keep_dim=True)                                         # (..., 1) == ||x_spatial||^2
    r2 = torch.clamp(r2, min=0)                                                  # numeric safety
    c_t = torch.as_tensor(c, dtype=x_spatial.dtype, device=x_spatial.device)
    x0 = torch.sqrt(1.0 / c_t + r2)
    return torch.cat([x0, x_spatial], dim=-1)

class LorentzHIERLoss(nn.Module):
    """
    LHIER loss: HIER on the hyperboloid.
    Assumes inputs z_s_h are already Lorentzian (on the hyperboloid, shape (B, d+1)).
    Proxies (LCAs) are manifold parameters optimized on the hyperboloid.
    """
    def __init__(self, manifold:Manifold, nb_proxies, sz_embed, mrg=0.1, tau=0.1, clip_r=2.3, sim_scale=1.0):
        super().__init__()
        self.manifold = manifold
        self.nb_proxies = nb_proxies
        self.sz_embed = sz_embed # == d+1
        self.tau = tau
        self.hyp_c = self.manifold.c
        self.mrg = mrg
        # self.clip_r = clip_r
        self.sim_scale = sim_scale
        
        # Initialize LCAs directly on the hyperboloid (ambient dim = d+1)
        with torch.no_grad():
            x_spatial = torch.randn(self.nb_proxies, self.sz_embed, device="cuda" if torch.cuda.is_available() else "cpu")
            self.lcas = lift_spatial_to_hyperboloid_with_linner(x_spatial, self.hyp_c, self.manifold.l_inner)
            # self.lcas: (P, d+1) == ambient_dim
        self.lcas = torch.nn.Parameter(self.lcas)
        # self.lcas = torch.randn(self.nb_proxies, self.sz_embed).cuda()
        # self.lcas = self.lcas / math.sqrt(self.sz_embed) * clip_r * 0.9
        # self.lcas = torch.nn.Parameter(self.lcas)
                
        if self.hyp_c > 0:
            self.dist_f = lambda x, y: dist_matrix(self.manifold, x, y)
        else:
            raise NotImplemented
            # Euclidean fallback stays the same (or switch to torch.cdist if you prefer)
            self.dist_f = lambda x, y: 2 - 2 * F.linear(x, y)
    
    def compute_gHHC(self, z_s_h, lcas_h, dist_matrix, indices_tuple, sim_matrix):
        i, j, k = indices_tuple
        cp_dist = dist_matrix # (N, P)

        max_dists_ij  = torch.maximum(cp_dist[i], cp_dist[j])
        lca_ij_prob   = F.gumbel_softmax(-max_dists_ij / self.tau, dim=1, hard=True)
        lca_ij_idx    = lca_ij_prob.argmax(-1)

        max_dists_ijk = torch.maximum(cp_dist[k], max_dists_ij)
        lca_ijk_prob  = F.gumbel_softmax(-max_dists_ijk / self.tau, dim=1, hard=True)
        lca_ijk_idx   = lca_ijk_prob.argmax(-1)

        dist_i_lca_ij,  dist_i_lca_ijk  = (cp_dist[i] * lca_ij_prob).sum(1),  (cp_dist[i] * lca_ijk_prob).sum(1)
        dist_j_lca_ij,  dist_j_lca_ijk  = (cp_dist[j] * lca_ij_prob).sum(1),  (cp_dist[j] * lca_ijk_prob).sum(1)
        dist_k_lca_ij,  dist_k_lca_ijk  = (cp_dist[k] * lca_ij_prob).sum(1),  (cp_dist[k] * lca_ijk_prob).sum(1)

        hc_loss = (
            torch.relu(dist_i_lca_ij -  dist_i_lca_ijk + self.mrg) +
            torch.relu(dist_j_lca_ij -  dist_j_lca_ijk + self.mrg) +
            torch.relu(dist_k_lca_ijk - dist_k_lca_ij  + self.mrg)
        )
        hc_loss = hc_loss * (lca_ij_idx != lca_ijk_idx).float()
        return hc_loss.mean()
        
    def get_reciprocal_triplets(self, sim_matrix, topk=20, t_per_anchor=100):
        N = sim_matrix.size(0)
        if N < 3:  # not enough to form triplets
            device = sim_matrix.device
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty

        k = max(1, min(topk, N - 1))
        topk_index = torch.topk(sim_matrix, k, dim=1).indices

        nn_matrix = torch.zeros_like(sim_matrix).scatter_(1, topk_index, 1)
        sim_matrix = ((nn_matrix + nn_matrix.t()) / 2).float()
        sim_matrix = sim_matrix.fill_diagonal_(-1)

        anchor_idx, positive_idx, negative_idx = [], [], []
        for i in range(N):
            pos_pool = torch.nonzero(sim_matrix[i] == 1, as_tuple=False).squeeze(-1).cpu().numpy()
            neg_pool = torch.nonzero(sim_matrix[i] <  1, as_tuple=False).squeeze(-1).cpu().numpy()
            if pos_pool.size <= 1 or neg_pool.size == 0:
                continue
            pair_idxs1 = np.random.choice(pos_pool, t_per_anchor, replace=True)
            pair_idxs2 = np.random.choice(neg_pool, t_per_anchor, replace=True)
            positive_idx.append(pair_idxs1)
            negative_idx.append(pair_idxs2)
            anchor_idx.append(np.full(t_per_anchor, i, dtype=np.int64))

        if len(anchor_idx) == 0:
            device = sim_matrix.device
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty

        device = sim_matrix.device
        anchor_idx   = torch.from_numpy(np.concatenate(anchor_idx)).to(device)
        positive_idx = torch.from_numpy(np.concatenate(positive_idx)).to(device)
        negative_idx = torch.from_numpy(np.concatenate(negative_idx)).to(device)
        return anchor_idx, positive_idx, negative_idx

    # def get_reciprocal_triplets(self, sim_matrix, topk=20, t_per_anchor=100):
    #     anchor_idx, positive_idx, negative_idx = [], [], []

    #     topk_index = torch.topk(sim_matrix, topk)[1]
    #     nn_matrix = torch.zeros_like(sim_matrix).scatter_(1, topk_index, torch.ones_like(sim_matrix))
    #     sim_matrix = ((nn_matrix + nn_matrix.t()) / 2).float()
    #     sim_matrix = sim_matrix.fill_diagonal_(-1)

    #     for i in range(len(sim_matrix)):
    #         if len(torch.nonzero(sim_matrix[i] == 1)) <= 1:
    #             continue
    #         pos_pool = torch.nonzero(sim_matrix[i] == 1).squeeze().cpu().numpy()
    #         neg_pool = torch.nonzero(sim_matrix[i] <  1).squeeze().cpu().numpy()

    #         pair_idxs1 = np.random.choice(pos_pool, t_per_anchor, replace=True)
    #         pair_idxs2 = np.random.choice(neg_pool, t_per_anchor, replace=True)

    #         positive_idx.append(pair_idxs1)
    #         negative_idx.append(pair_idxs2)
    #         anchor_idx.append(np.ones(t_per_anchor) * i)

    #     anchor_idx = np.concatenate(anchor_idx)
    #     positive_idx = np.concatenate(positive_idx)
    #     negative_idx = np.concatenate(negative_idx)
    #     return anchor_idx, positive_idx, negative_idx
    
    def forward(self, z_s_h, y, topk=30):
        """
        z_s_h: (B, d+1) hyperboloid embeddings (already Lorentzian).
        y : (B,) labels (only used to gently upweight same-class pairs in mining).
        """
        # ensure parameters & inputs share device
        device = z_s_h.device
        if self.lcas.device != device:
            self.lcas.data = self.lcas.data.to(device)

        B = z_s_h.size(0)
        lcas_h = self.lcas # already on manifold (keep with manifold optimizer)

        # Build Lorentzian pairwise distances once
        all_nodes = torch.cat([z_s_h, lcas_h], dim=0) # (B+P, d+1)
        D = self.dist_f(all_nodes, all_nodes) # (B+P, B+P)

        # Similarities for mining (with optional scale)
        sim_matrix  = torch.exp(-self.sim_scale * D[:B, :B]).detach()
        sim_matrix[(y.unsqueeze(1) == y.unsqueeze(0))] += 1
        sim_matrix2 = torch.exp(-self.sim_scale * D[B:, B:]).detach()

        # data -> lcas consistency
        idx1 = self.get_reciprocal_triplets(sim_matrix, topk=topk, t_per_anchor=50)
        loss  = self.compute_gHHC(z_s_h, lcas_h, D[:B, B:], idx1, sim_matrix)

        # lcas -> lcas consistency (treat proxies as points as well)
        idx2 = self.get_reciprocal_triplets(sim_matrix2, topk=topk, t_per_anchor=50)
        loss += self.compute_gHHC(lcas_h, lcas_h, D[B:, B:], idx2, sim_matrix2)

        return loss
    
# Circle Loss
# NOTE Probably broken

def convert_label_to_similarity_lorentz(manifold: Manifold, hyper_feat: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Lorentzian inner product (negative of squared distance for normalized embeddings)
    sim = manifold.inner(hyper_feat, hyper_feat)  # shape (N, N)

    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    sim = sim.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)

    return sim[positive_matrix], sim[negative_matrix]

class LorentzianCircleLoss(nn.Module):
    def __init__(self, manifold: Manifold ,m: float, gamma: float) -> None:
        super().__init__()
        self.manifold = manifold
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        # Detach prevents gradients flowing into similarity computation
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, 0.)
        an = torch.clamp_min(sn.detach() + self.m, 0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss
