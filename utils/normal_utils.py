
import torch
import numpy as np
from torch.nn import functional as F


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.zeros((batch_size, 3,3)).cuda()

    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]

    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)

    return R

def scale_to_mat(scales):
    """Create scaling matrices."""
    N = scales.size(0)
    scale_mat = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).to(scales.device)
    scale_mat[:, 0, 0] = scales[:, 0]
    scale_mat[:, 1, 1] = scales[:, 1]
    scale_mat[:, 2, 2] = 1.0  # No scaling in z-direction
    return scale_mat

def compute_normal_world_space(quaternions, scales, viewmat, points_world):
    # BIG TODO normals+multipliers should be taken from rasterizer! If you need this, please, for now fix it by yourself.
    """Compute normal vectors from quaternions and scaling factors.
    Exactly the same implementation in rasterizer."""
    W = viewmat[:3,:3].clone()
    cam_pos = viewmat[3,:3].clone()
    p_view = torch.matmul(W.T, points_world.T).T + cam_pos
    R = quat_to_rot(quaternions)
    S = scale_to_mat(scales)
    RS = torch.bmm(R, S)
    tn = RS[:, :,2]
    tn_w = torch.matmul(W.T,RS)[:, :,2]
    cos = torch.sum(-tn_w * p_view, dim=1)
    multiplier = torch.full_like(cos, -1)
    multiplier[cos >= 0] = 1
    tn*=multiplier.unsqueeze(1)
    normal_vectors = tn / (tn.norm(dim=1, keepdim=True)+0.000001)
    return normal_vectors, multiplier


# ---- Relaxed Manhattan World Prior ----
# Manhattan axes for Z-up coordinate system
_MANHATTAN_AXES = None  # lazily created on correct device

def _get_manhattan_axes(device):
    """Return [3, 3] tensor of Manhattan world axes (rows = axes), cached per device."""
    global _MANHATTAN_AXES
    if _MANHATTAN_AXES is None or _MANHATTAN_AXES.device != device:
        _MANHATTAN_AXES = torch.eye(3, device=device)  # [[1,0,0],[0,1,0],[0,0,1]]
    return _MANHATTAN_AXES


def manhattan_loss(normals: torch.Tensor) -> torch.Tensor:
    """Relaxed Manhattan world prior on surfel normals.

    Encourages each normal to align with its nearest Manhattan axis (X, Y, or Z).
    For each normal n_i the loss contribution is  1 - max_k (n_i · a_k)^2,
    which is zero when the normal is perfectly axis-aligned and 1 at 45°.

    Args:
        normals: [N, 3] unit normals in world space (gradients flow through
                 quaternion / scale via compute_normal_world_space).

    Returns:
        Scalar mean loss (already reduced).
    """
    axes = _get_manhattan_axes(normals.device)  # [3, 3]
    # dots[i, k] = n_i · a_k   →  [N, 3]
    dots = normals @ axes.T
    # Best squared alignment per Gaussian
    max_cos2, _ = (dots * dots).max(dim=1)  # [N]
    return (1.0 - max_cos2).mean()

