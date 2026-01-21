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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.mlp import MLP
from utils.sh_utils import *
from utils.sun_utils import SunModel
from collections import OrderedDict


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        if self.use_sun:
            # Sun model will be initialized later when n_images is known
            # Call setup_sun_model() explicitly after Scene is created
            pass
        elif self.with_mlp:
            self.setup_mlp()
        else:
            self.setup_env_params()


    def __init__(self, sh_degree : int, with_mlp: bool = False, mlp_W=128, mlp_D=4, N_a = 32,
                 use_sun: bool = False, n_images: int = None):

        # We only implement deg 2 for SH_gauss and SH_env.
        # More on environment map degree: https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
        assert sh_degree == 2

        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc_positive = torch.empty(0)
        self._features_rest_positive = torch.empty(0)
        self._features_dc_negative = torch.empty(0)
        self._features_rest_negative = torch.empty(0)
        self._albedo = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._casts_shadow = torch.empty(0)  # Per-gaussian shadow casting flag (1=casts shadow, 0=sky/transparent)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.with_mlp = with_mlp
        self.mlp_W = mlp_W
        self.mlp_D = mlp_D
        self.N_a = N_a

        # Sun model parameters (explicit directional lighting, no SH for environment)
        self.use_sun = use_sun
        self.n_images = n_images
        self.sun_model = None

        self.setup_functions()




    def setup_mlp(self):
        N_vocab=1700
        self.embedding = torch.nn.Embedding(N_vocab, self.N_a).cuda()
        self.mlp = MLP(2, self.mlp_W, self.mlp_D, self.N_a).cuda()

        self.correct_gaussians_with_mlp = False

    def setup_env_params(self):
        # init values from osr
        self.env_params = nn.ParameterDict(OrderedDict(
            [(str(x), nn.Parameter(
                torch.tensor([
                    #version of osr default map, rotated
                    [ 2.9861000e+00,  3.4646001e+00,  3.9559000e+00],
                    [ 8.2520002e-01,  5.2737999e-01,  9.7384997e-02],
                    [ 1.0013000e-01, -6.7589000e-02, -3.1161001e-01],
                    [ 2.2311001e-03,  4.3553002e-03,  4.9501001e-03],
                    [-6.5793000e-02, -4.3269999e-02,  1.7002000e-01],
                    [-1.1078000e-01,  6.0607001e-02,  1.9541000e-01],
                    [-3.3267748e-01, -4.2370442e-01, -4.7939608e-01],
                    [-6.4355000e-03,  9.7476002e-03, -2.3863001e-02],
                    [-7.2156233e-01, -6.4352357e-01, -3.7317836e-01]
            ],dtype=torch.float32).T


            )) for x in range(1700)])).cuda()

    def setup_sun_model(self):
        """
        Initialize the sun model for explicit directional lighting (no SH).

        This model keeps sun as an explicit directional light for:
        - Sharp shadow boundaries (computed separately using geometry)
        - Accurate Lambert shading: albedo * sun_intensity * max(0, N·L) + ambient
        - Residual SH for sky gradients and indirect lighting

        Sun direction is obtained from Camera objects at runtime.
        The model learns:
        - sun_intensity: Per-image sun intensity [n_images, 3] for RGB channels
        - ambient_color: Per-image ambient color [n_images, 3] for RGB channels
        - residual_sh: Per-image residual SH for environment details

        Note: Shadowing is handled separately using geometry-based shadow computation.
        """
        if self.n_images is None:
            raise ValueError("n_images must be provided when use_sun=True")

        self.sun_model = SunModel(
            n_images=self.n_images,
            device="cuda"
        )
        print(f"Initialized SunModel for {self.n_images} images")


    def capture(self): #MLP and embedding saved separately.
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc_positive,
            self._features_rest_positive,
            self._features_dc_negative,
            self._features_rest_negative,
            self._albedo,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc_positive,
        self._features_rest_positive,
        self._features_dc_negative,
        self._features_rest_negative,
        self._albedo,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    # @property
    def get_features(self, multiplier):
        """2DGS flips Gaussians to always face the camera by multiplying their normals by 1 or -1.
        For each Gaussian, two sets of SH_gauss are kept based on the normal direction:
        one for the default direction and another for the opposite direction."""

        features_dc_pos = self._features_dc_positive
        features_rest_pos = self._features_rest_positive
        features_positive = torch.cat((features_dc_pos, features_rest_pos), dim=1)

        features_dc_neg = self._features_dc_negative
        features_rest_neg = self._features_rest_negative
        features_negative = torch.cat((features_dc_neg, features_rest_neg), dim=1)
        mask = multiplier == 1
        # Select features based on the mask
        features = torch.where(mask.view(-1,1,1), features_positive, features_negative)
        return features

    @property
    def get_albedo(self):
        return torch.clamp(SH2RGB(self._albedo), 0.0)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_casts_shadow(self):
        """Get per-gaussian shadow casting flag (1=casts shadow, 0=sky/transparent)"""
        return self._casts_shadow

    def compute_embedding(self, emb_idx):
        return self.embedding(torch.full((1,),emb_idx).cuda())


    def compute_gaussian_rgb(self, sh_scene, shadowed=True, multiplier=None, normal_vectors=None, env_hemisphere_lightning=True):
        #Computation of RGB could be implemented in CUDA. If you need it, please take care of it yourself.
        assert shadowed or (not shadowed and torch.is_tensor(normal_vectors))
        assert not shadowed or (shadowed and torch.is_tensor(multiplier))
        assert sh_scene.shape[-1] == (self.max_sh_degree+1)**2

        albedo = self.get_albedo
        if shadowed:
            shs_gauss = self.get_features(multiplier).transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
            sh2intensity = eval_sh_shadowed(shs_gauss[:,0:1,:], sh_scene) #shs_gauss[:,0:1,:] for no color bleeding
        else:
            if env_hemisphere_lightning:
                sh2intensity = eval_sh_hemisphere(normal_vectors, sh_scene) #Expected output [B, 3]. Normals have to be unit.
            else:
                sh2intensity = eval_sh_point(normal_vectors, sh_scene) #Expected output [B, 3]. Normals have to be unit.

        intensity_hdr = torch.clamp_min(sh2intensity, 0.00001)
        intensity = intensity_hdr**(1 / 2.2)  # linear to srgb
        rgb = torch.clamp(intensity*albedo, 0.0)

        return rgb, intensity

    def compute_directional_rgb(self, emb_idx, normal_vectors, sun_direction):
        """
        Compute RGB using explicit directional sun lighting (no SH).

        This implements proper Lambert shading without shadows:
            L = albedo * (sun_intensity * max(0, N·L) + ambient + residual_sh)

        Note: Shadows should be applied externally by multiplying the direct component
        with a geometry-based shadow mask.

        Args:
            emb_idx: Image embedding index (for per-image lighting parameters)
            normal_vectors: Surface normals [N, 3]
            sun_direction: Sun direction vector [3] from camera

        Returns:
            rgb: Final RGB values [N, 3]
            intensity: Intensity values [N, 3]
            sun_direction: Sun direction vector [3] (normalized)
            lighting_components: Dict with 'direct', 'ambient', 'residual' for debugging
        """
        assert self.use_sun and self.sun_model is not None, \
            "compute_directional_rgb requires use_sun=True"

        albedo = self.get_albedo  # [N, 3]

        # Compute directional lighting (unshadowed)
        intensity_hdr, sun_dir, components = self.sun_model(
            emb_idx, normal_vectors, sun_direction=sun_direction
        )

        # Apply gamma correction (linear to sRGB)
        intensity_hdr = torch.clamp_min(intensity_hdr, 0.00001)
        intensity = intensity_hdr ** (1 / 2.2)

        # Final RGB = albedo * intensity
        rgb = torch.clamp(intensity * albedo, 0.0)

        return rgb, intensity, sun_dir, components


    def update_sky_gaussians(self, cameras, sky_masks, sky_vote_threshold=0.5):
        """
        Update _casts_shadow based on sky mask voting across multiple camera views.

        For each gaussian, project into camera views and check if it falls in sky region.
        If a gaussian is in the sky region in more than sky_vote_threshold fraction of
        visible views, mark it as non-shadow-casting.

        Args:
            cameras: List of Camera objects with projection matrices
            sky_masks: Dict mapping image_name to sky mask tensor [H, W] where 0=sky, 1=not sky
            sky_vote_threshold: Fraction of views where gaussian must be in sky to be marked (default 0.5)
        """
        N = self._xyz.shape[0]
        positions = self._xyz.detach()  # [N, 3]

        sky_votes = torch.zeros(N, device="cuda", dtype=torch.float32)
        visible_count = torch.zeros(N, device="cuda", dtype=torch.float32)

        num_cams_with_masks = 0
        total_sky_pixels_sampled = 0

        for cam in cameras:
            if cam.image_name not in sky_masks:
                continue

            sky_mask = sky_masks[cam.image_name]  # [H, W], 0=sky, 1=not sky
            if sky_mask is None:
                continue

            num_cams_with_masks += 1
            H, W = sky_mask.shape

            # Project gaussian centers to camera image coordinates
            # Add homogeneous coordinate
            ones = torch.ones(N, 1, device="cuda", dtype=positions.dtype)
            positions_homo = torch.cat([positions, ones], dim=1)  # [N, 4]

            # Apply full projection transform
            clip_coords = positions_homo @ cam.full_proj_transform  # [N, 4]

            # Perspective divide
            w = clip_coords[:, 3:4]
            ndc = clip_coords[:, :3] / (w + 1e-8)  # [N, 3]

            # Check if in front of camera (w > 0)
            in_front = (w.squeeze() > 0)

            # NDC to pixel coordinates
            # NDC is in [-1, 1], convert to [0, W-1] and [0, H-1]
            # Note: Y is flipped in image coordinates (0 at top, H-1 at bottom)
            px = ((ndc[:, 0] + 1) * 0.5 * (W - 1)).long()
            py = ((1 - ndc[:, 1]) * 0.5 * (H - 1)).long()  # Flip Y for image coordinates

            # Check if in image bounds
            in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)

            # Combined visibility mask
            visible = in_front & in_bounds

            # Update visible count
            visible_count += visible.float()

            # Check sky mask for visible gaussians
            # sky_mask: 0=sky (black), 1=not sky (after thresholding, these are floats)
            valid_px = px.clamp(0, W - 1)
            valid_py = py.clamp(0, H - 1)

            # Sample sky mask at projected positions - use < 0.5 for sky detection
            sampled_mask_values = sky_mask[valid_py, valid_px]
            is_sky = (sampled_mask_values < 0.5)  # < 0.5 = sky region (was black in original mask)

            # Count sky pixels for this camera
            sky_in_this_cam = (visible & is_sky).sum().item()
            total_sky_pixels_sampled += sky_in_this_cam

            # Add sky votes for visible gaussians that are in sky region
            sky_votes += (visible & is_sky).float()

        # Compute sky fraction
        sky_fraction = torch.where(
            visible_count > 0,
            sky_votes / visible_count,
            torch.zeros_like(sky_votes)
        )

        # Mark gaussians as non-shadow-casting if sky_fraction > threshold
        is_sky_gaussian = sky_fraction > sky_vote_threshold
        self._casts_shadow = torch.where(is_sky_gaussian, torch.zeros_like(self._casts_shadow), torch.ones_like(self._casts_shadow))

        num_sky = is_sky_gaussian.sum().item()
        num_with_votes = (visible_count > 0).sum().item()
        max_sky_fraction = sky_fraction.max().item() if sky_fraction.numel() > 0 else 0
        print(f"[Sky Classification] {num_sky}/{N} marked sky | {num_with_votes} visible | max_frac={max_sky_fraction:.3f} | cams_w_masks={num_cams_with_masks} | total_sky_samples={total_sky_pixels_sampled}")


    def compute_env_sh(self, emb_idx):
        if self.use_sun:
            raise RuntimeError("compute_env_sh() should not be called when use_sun=True. Use compute_directional_rgb() instead.")
        elif self.with_mlp:
            return self.mlp(self.compute_embedding(emb_idx))
        else:
            return self.env_params[str(emb_idx)]


    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features_positive = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features_negative = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        albedo = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # init SHgauss with 0.1
        features_positive[:, :3, 0 ] = 0.01
        features_positive[:, :3, 1:] = 0.01
        features_negative[:, :3, 0 ] = 0.01
        features_negative[:, :3, 1:] = 0.01

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc_positive = nn.Parameter(features_positive[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_positive = nn.Parameter(features_positive[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc_negative = nn.Parameter(features_negative[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_negative = nn.Parameter(features_negative[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._albedo = nn.Parameter(albedo.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # Initialize all gaussians as shadow-casting (will be updated by sky mask classification)
        self._casts_shadow = torch.ones((fused_point_cloud.shape[0],), dtype=torch.float32, device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc_positive], 'lr': training_args.feature_lr, "name": "f_dc_positive"},
            {'params': [self._features_rest_positive], 'lr': training_args.feature_lr / 1.0, "name": "f_rest_positive"},
            {'params': [self._features_dc_negative], 'lr': training_args.feature_lr, "name": "f_dc_negative"},
            {'params': [self._features_rest_negative], 'lr': training_args.feature_lr / 1.0, "name": "f_rest_negative"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        if self.use_sun:
            # Sun model: explicit directional lighting + residual SH
            # Note: Shadow handling is done separately using geometry-based shadow computation
            l_env = [
                {'params': [self.sun_model.sun_intensity], 'lr': training_args.env_lr * 2.0, "name": "sun_intensity"},
                {'params': [self.sun_model.ambient_color], 'lr': training_args.env_lr * 2.0, "name": "ambient_color"},
            ]
            # Add residual SH parameters if enabled
            if self.sun_model.use_residual_sh:
                l_env.append({'params': [self.sun_model.residual_sh], 'lr': training_args.env_lr, "name": "residual_sh"})
        elif self.with_mlp:
            l_env = [{'params': [*self.embedding.parameters()], 'lr': training_args.embedding_lr, "name": "embedding"},
                     {'params': [*self.mlp.parameters()], 'lr': training_args.mlp_lr, "name": "mlp"},
                    ]
        else:
            l_env = [{'params': [*self.env_params.parameters()], 'lr': training_args.env_lr, "name": "env_params_lr"},]

        self.optimizer_env = torch.optim.Adam(l_env, lr=0.0, eps=1e-15)
        print('Env optimizer has parameters: ', [p["name"] for p in self.optimizer_env.param_groups])

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc_positive.shape[1]*self._features_dc_positive.shape[2]):
            l.append('f_dc_positive_{}'.format(i))
        for i in range(self._features_rest_positive.shape[1]*self._features_rest_positive.shape[2]):
            l.append('f_rest_positive_{}'.format(i))
        for i in range(self._features_dc_negative.shape[1]*self._features_dc_negative.shape[2]):
            l.append('f_dc_negative_{}'.format(i))
        for i in range(self._features_rest_negative.shape[1]*self._features_rest_negative.shape[2]):
            l.append('f_rest_negative_{}'.format(i))
        for i in range(self._albedo.shape[1]):
            l.append('albedo_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc_positive = self._features_dc_positive.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_positive = self._features_rest_positive.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc_negative = self._features_dc_negative.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_negative = self._features_rest_negative.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc_positive, f_rest_positive, f_dc_negative, f_rest_negative, albedo, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc_positive = np.zeros((xyz.shape[0], 3, 1))
        features_dc_positive[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_positive_0"])
        features_dc_positive[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_positive_1"])
        features_dc_positive[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_positive_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_positive_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra_positive = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra_positive[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra_positive = features_extra_positive.reshape((features_extra_positive.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        features_dc_negative = np.zeros((xyz.shape[0], 3, 1))
        features_dc_negative[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_negative_0"])
        features_dc_negative[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_negative_1"])
        features_dc_negative[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_negative_2"])

        extra_f_names_negative = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_negative_")]
        extra_f_names_negative = sorted(extra_f_names_negative, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names_negative)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra_negative = np.zeros((xyz.shape[0], len(extra_f_names_negative)))
        for idx, attr_name in enumerate(extra_f_names_negative):
            features_extra_negative[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra_negative = features_extra_negative.reshape((features_extra_negative.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        albedo_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("albedo_")]
        albedo_names = sorted(albedo_names, key = lambda x: int(x.split('_')[-1]))
        albedo = np.zeros((xyz.shape[0], len(albedo_names)))
        for idx, attr_name in enumerate(albedo_names):
            albedo[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc_positive = nn.Parameter(torch.tensor(features_dc_positive, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_positive = nn.Parameter(torch.tensor(features_extra_positive, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc_negative = nn.Parameter(torch.tensor(features_dc_negative, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_negative = nn.Parameter(torch.tensor(features_extra_negative, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc_positive = optimizable_tensors["f_dc_positive"]
        self._features_rest_positive = optimizable_tensors["f_rest_positive"]
        self._features_dc_negative = optimizable_tensors["f_dc_negative"]
        self._features_rest_negative = optimizable_tensors["f_rest_negative"]
        self._opacity = optimizable_tensors["opacity"]
        self._albedo = optimizable_tensors["albedo"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._casts_shadow = self._casts_shadow[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc_positive, new_features_rest_positive,
                              new_features_dc_negative, new_features_rest_negative,
                              new_albedo, new_opacities, new_scaling, new_rotation, new_casts_shadow=None):
        d = {"xyz": new_xyz,
        "f_dc_positive": new_features_dc_positive,
        "f_rest_positive": new_features_rest_positive,
        "f_dc_negative": new_features_dc_negative,
        "f_rest_negative": new_features_rest_negative,
        "albedo": new_albedo,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc_positive = optimizable_tensors["f_dc_positive"]
        self._features_rest_positive = optimizable_tensors["f_rest_positive"]
        self._features_dc_negative = optimizable_tensors["f_dc_negative"]
        self._features_rest_negative = optimizable_tensors["f_rest_negative"]
        self._albedo = optimizable_tensors["albedo"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Handle casts_shadow - new gaussians inherit from parent or default to 1
        if new_casts_shadow is not None:
            self._casts_shadow = torch.cat([self._casts_shadow, new_casts_shadow], dim=0)
        else:
            # Default new gaussians to shadow-casting
            self._casts_shadow = torch.cat([self._casts_shadow, torch.ones(new_xyz.shape[0], device="cuda")], dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc_positive = self._features_dc_positive[selected_pts_mask].repeat(N,1,1)
        new_features_rest_positive = self._features_rest_positive[selected_pts_mask].repeat(N,1,1)
        new_features_dc_negative = self._features_dc_negative[selected_pts_mask].repeat(N,1,1)
        new_features_rest_negative = self._features_rest_negative[selected_pts_mask].repeat(N,1,1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_casts_shadow = self._casts_shadow[selected_pts_mask].repeat(N)  # Inherit shadow-casting from parent

        self.densification_postfix(new_xyz, new_features_dc_positive, new_features_rest_positive,new_features_dc_negative, new_features_rest_negative, new_albedo, new_opacity, new_scaling, new_rotation, new_casts_shadow)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc_positive = self._features_dc_positive[selected_pts_mask]
        new_features_rest_positive = self._features_rest_positive[selected_pts_mask]
        new_features_dc_negative = self._features_dc_negative[selected_pts_mask]
        new_features_rest_negative = self._features_rest_negative[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_casts_shadow = self._casts_shadow[selected_pts_mask]  # Inherit shadow-casting from parent

        self.densification_postfix(new_xyz, new_features_dc_positive, new_features_rest_positive,new_features_dc_negative, new_features_rest_negative, new_albedo, new_opacities, new_scaling, new_rotation, new_casts_shadow)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

