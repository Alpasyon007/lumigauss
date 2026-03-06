#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def _axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vector [3] to 3x3 rotation matrix via Rodrigues' formula."""
    angle = torch.norm(axis_angle)
    if angle < 1e-8:
        return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    k = axis_angle / angle
    K = torch.zeros(3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[0, 1] = -k[2]; K[0, 2] = k[1]
    K[1, 0] = k[2];  K[1, 2] = -k[0]
    K[2, 0] = -k[1]; K[2, 1] = k[0]
    R = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype) + \
        torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", mask = None,
                 sun_direction = None, sun_elevation = None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        # Sun direction and elevation for this camera/image (can be None if not using sun model)
        self.sun_direction = None
        if sun_direction is not None:
            self.sun_direction = torch.tensor(sun_direction, dtype=torch.float32)

        self.sun_elevation = None  # Elevation in degrees
        if sun_elevation is not None:
            self.sun_elevation = float(sun_elevation)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.mask = mask.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # Camera calibration refinement: small learnable deltas (disabled by default)
        self._cam_cal_enabled = False
        self.delta_rot = nn.Parameter(torch.zeros(3, device="cuda"), requires_grad=False)    # axis-angle
        self.delta_trans = nn.Parameter(torch.zeros(3, device="cuda"), requires_grad=False)   # translation

        # Sun direction calibration: small learnable delta (disabled by default)
        self._sun_cal_enabled = False
        self.delta_sun_dir = nn.Parameter(torch.zeros(3, device="cuda"), requires_grad=False)

    def enable_cam_cal(self):
        """Enable learnable camera pose refinement."""
        self._cam_cal_enabled = True
        self.delta_rot.requires_grad_(True)
        self.delta_trans.requires_grad_(True)

    def get_cam_cal_params(self):
        """Return list of learnable camera calibration parameters."""
        return [self.delta_rot, self.delta_trans]

    def enable_sun_cal(self):
        """Enable learnable sun direction refinement."""
        self._sun_cal_enabled = True
        self.delta_sun_dir.requires_grad_(True)

    def get_sun_cal_params(self):
        """Return list of learnable sun calibration parameters."""
        return [self.delta_sun_dir]

    def get_adjusted_sun_direction(self):
        """Return the sun direction with the learnable delta applied (DIFFERENTIABLE).

        adjusted = normalize(original_sun_dir + delta_sun_dir)

        Gradients flow through the addition and normalisation back to
        delta_sun_dir. The returned tensor is on CUDA with requires_grad.
        """
        if self.sun_direction is None:
            return None
        if not self._sun_cal_enabled:
            return self.sun_direction.to("cuda")
        orig = self.sun_direction.to("cuda")
        adjusted = orig + self.delta_sun_dir
        return torch.nn.functional.normalize(adjusted, dim=0)

    def apply_cam_cal(self):
        """Recompute view transforms from the original pose + learnable deltas.

        Updates the camera matrices (DETACHED - rasterizer can't differentiate
        through them anyway). The actual gradient path is through
        transform_means3D() which transforms gaussian positions differentiably.
        """
        if not self._cam_cal_enabled:
            return

        with torch.no_grad():
            # Original world-to-view in column-major (transposed from row-major)
            W2V_orig = torch.tensor(
                getWorld2View2(self.R, self.T, self.trans, self.scale),
                dtype=torch.float32, device="cuda"
            ).transpose(0, 1)  # column-major [4, 4]

            # Build delta transform (small rotation + translation in camera frame)
            dR = _axis_angle_to_rotation_matrix(self.delta_rot)  # [3, 3]
            delta_mat = torch.eye(4, device="cuda", dtype=torch.float32)
            delta_mat[:3, :3] = dR
            delta_mat[3, :3] = self.delta_trans  # column-major: translation in last row

            # Apply: W2V' = W2V_orig @ delta  (post-multiply = camera-frame perturbation)
            W2V_new = W2V_orig @ delta_mat

            self.world_view_transform = W2V_new
            self.full_proj_transform = (W2V_new.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center = W2V_new.inverse()[3, :3]

    def transform_means3D(self, means3D: torch.Tensor) -> torch.Tensor:
        """Transform gaussian world positions by the camera pose delta (DIFFERENTIABLE).

        This is the gradient path for camera calibration. Instead of trying to
        differentiate through the rasterizer's viewmatrix (not supported), we
        transform the gaussian world positions such that rendering with the
        ORIGINAL camera gives the same result as rendering with the delta-perturbed
        camera. The rasterizer IS differentiable w.r.t. means3D, so gradients
        flow: loss → rasterizer → means3D_transformed → delta_rot/delta_trans.

        The camera delta is a post-multiplication in camera frame:
            W2V' = W2V @ delta_cam
        This is equivalent to pre-transforming world points:
            p_cam' = (p_world @ delta_world) @ W2V
        where delta_world = W2V^{-1} @ delta_cam^{-1} ... but this is complex.

        Simpler: we treat the delta as applied in camera space after the view
        transform. The equivalent is: transform world points by delta_cam
        (a tiny rotation+translation), THEN apply the original view matrix.
        In column-major: p_cam_new = p_world_homo @ W2V_orig @ delta_cam
                                   = (p_world @ delta_R.T + delta_t) ... NO,
        we need to think in terms of column-major row-vector convention.

        In column-major with row vectors:
            p_cam = p_world_homo @ W2V
            p_cam_new = p_world_homo @ W2V @ delta_cam
                      = p_cam @ delta_cam
            But when using world-space transform on means3D:
            p_cam_new = (p_world_new)_homo @ W2V
            So p_world_new @ W2V[:3,:3] + W2V[3,:3] = p_world @ W2V[:3,:3] + W2V[3,:3]) @ delta_cam[:3,:3] + delta_cam[3,:3]

        For simplicity, we apply the delta directly as a world-space perturbation:
            means3D_new = means3D @ dR + dt

        This is a small rigid-body transformation of the scene, equivalent to
        a small camera pose change. The optimizer will learn deltas that minimize
        the photometric loss.

        Args:
            means3D: Gaussian positions [N, 3]

        Returns:
            Transformed positions [N, 3] with gradients through delta_rot/delta_trans.
            Returns means3D unchanged if cam_cal is disabled.
        """
        if not self._cam_cal_enabled:
            return means3D

        dR = _axis_angle_to_rotation_matrix(self.delta_rot)  # [3, 3]
        # means3D @ dR applies the delta rotation in world frame (row-vector convention)
        return means3D @ dR + self.delta_trans.unsqueeze(0)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

