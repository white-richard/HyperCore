import torch

def lift_spatial_with_projx(manifold, x_spatial, eps=1e-9):
    """
    Given points in the spatial coordinates of the Lorentz model,
    lift them to the full (d+1)-dimensional coordinates using projx.
    Args:
        manifold: instance of a Lorentz manifold class from hypercore.manifolds
        x_spatial: tensor of shape (..., d) representing points in spatial coordinates
        eps: small value to avoid numerical issues when computing sqrt
    Returns:
        x: tensor of shape (..., d+1) representing points in full Lorentz coordinates
    """
    # squared Euclidean norm ||x||^2
    r2 = (x_spatial * x_spatial).sum(dim=-1, keepdim=True)
    # 1/c as tensor
    inv_c = torch.as_tensor(1.0 / manifold.c, dtype=x_spatial.dtype, device=x_spatial.device)
    # x0 = sqrt(1/c + ||x||^2)
    x0 = torch.sqrt(torch.clamp(inv_c + r2, min=eps))
    x = torch.cat([x0, x_spatial], dim=-1) # (..., d+1)
    x = manifold.projx(x, dim=-1)
    return x

def lock_curvature_to_one(manifold, k=1.0):
    """
    Set the curvature of a Lorentz manifold to k and disable gradients.
    Args:
        manifold: instance of a Lorentz manifold class from hypercore.manifolds
        k: curvature value to set (default 1.0)
    """
    # Support both "k" and "c" naming
    for attr in ["k", "c"]:
        if hasattr(manifold, attr):
            val = getattr(manifold, attr)
            # If it's a Parameter, fix value and disable grad
            if isinstance(val, torch.nn.Parameter):
                with torch.no_grad():
                    val.fill_(k)
                val.requires_grad_(False)
            else:
                # Could be a tensor buffer, just overwrite
                setattr(manifold, attr, torch.as_tensor(1.0, dtype=val.dtype, device=val.device))
            print(f"Locked {attr} to 1.0 and disabled gradients.")