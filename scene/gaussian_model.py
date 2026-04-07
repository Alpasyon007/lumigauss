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
import math
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
                 use_sun: bool = False, n_images: int = None, use_residual_sh: bool = True,
                 full_pbr: bool = False, scene_lighting_sh: bool = False,
                 sky_sh_degree: int = 1):

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
        self._roughness = torch.empty(0)
        self._metallic = torch.empty(0)
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
        self.full_pbr = bool(full_pbr and use_sun)
        if full_pbr and not use_sun:
            print("Warning: --full_pbr requires --use_sun. Disabling full_pbr.")
        self.n_images = n_images
        self.use_residual_sh = use_residual_sh  # Whether to use residual SH for environment details
        self.sky_sh_degree = sky_sh_degree  # Degree of residual sky SH
        self.scene_lighting_sh = scene_lighting_sh  # Scene-global SH for sun params vs per-image
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
        Initialize the sun model for explicit directional lighting.

        This model keeps sun as an explicit directional light for:
        - Sharp shadow boundaries (computed separately using geometry)
        - Accurate Lambert shading: albedo * sun_intensity * max(0, N·L) + ambient
        - Residual SH for sky gradients and indirect lighting

        Sun direction is obtained from Camera objects at runtime.
        All lighting parameters (intensity, colour correction, ambient) are
        scene-global SH functions of the sun direction — not per-image.

        Note: Shadowing is handled separately using geometry-based shadow computation.
        """
        if self.n_images is None:
            raise ValueError("n_images must be provided when use_sun=True")

        self.sun_model = SunModel(
            n_images=self.n_images,
            device="cuda",
            use_residual_sh=self.use_residual_sh,
            sh_degree=self.sky_sh_degree,
            scene_sh=self.scene_lighting_sh
        )
        mode_str = "scene-global SH" if self.scene_lighting_sh else "per-image"
        print(f"Initialized SunModel for {self.n_images} images, {mode_str} mode (use_residual_sh={self.use_residual_sh})")


    def capture(self): #MLP and embedding saved separately.
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc_positive,
            self._features_rest_positive,
            self._features_dc_negative,
            self._features_rest_negative,
            self._albedo,
            self._roughness,
            self._metallic,
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
        if len(model_args) >= 17:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc_positive,
            self._features_rest_positive,
            self._features_dc_negative,
            self._features_rest_negative,
            self._albedo,
            self._roughness,
            self._metallic,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
        else:
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
            self._roughness = nn.Parameter(
                torch.full((self._xyz.shape[0], 1), 0.6, device="cuda", dtype=torch.float32).requires_grad_(self.full_pbr)
            )
            self._metallic = nn.Parameter(
                torch.zeros((self._xyz.shape[0], 1), device="cuda", dtype=torch.float32).requires_grad_(self.full_pbr)
            )

        # Ensure PBR material params are trainable only when full_pbr is enabled.
        self._roughness = nn.Parameter(self._roughness.detach().requires_grad_(self.full_pbr))
        self._metallic = nn.Parameter(self._metallic.detach().requires_grad_(self.full_pbr))
        # Reinitialize _casts_shadow as learnable parameter with correct size (default: non-shadow-casting)
        # Will be initialized from sky masks and further optimized via sky mask loss during training
        self._casts_shadow = nn.Parameter(torch.zeros(self._xyz.shape[0], device="cuda", dtype=torch.float32).requires_grad_(True))
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
    def get_base_color(self):
        return self.get_albedo

    @property
    def get_roughness(self):
        return torch.clamp(self._roughness, 0.04, 1.0)

    @property
    def get_metallic(self):
        return torch.clamp(self._metallic, 0.0, 1.0)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_casts_shadow(self):
        """Get per-gaussian shadow casting flag (1=casts shadow, 0=sky/transparent)"""
        # Clamp to [0, 1] since it's now a learnable parameter
        return torch.clamp(self._casts_shadow, 0.0, 1.0)

    def init_casts_shadow_from_sky_masks(self, cameras, sky_masks):
        """Initialize _casts_shadow by projecting gaussians into views and sampling sky masks.

        Gaussians that project into non-sky regions (mask=1) in ANY view are
        marked as shadow-casting (1.0).  The rest stay at 0.0.

        Args:
            cameras: List of training Camera objects.
            sky_masks: Dict mapping image_name → [H, W] tensor (0=sky, 1=not sky).
        """
        import torch.nn.functional as F

        xyz = self._xyz.detach()  # [N, 3]
        N = xyz.shape[0]
        votes = torch.zeros(N, device="cuda")

        cams = [c for c in cameras if c.image_name in sky_masks]

        ones = torch.ones(N, 1, device="cuda")
        xyz_h = torch.cat([xyz, ones], dim=-1)  # [N, 4]

        for cam in cams:
            mask = sky_masks[cam.image_name]  # [H, W], 0=sky, 1=not-sky
            H, W = mask.shape

            # Project: clip coords = xyz_h @ full_proj_transform
            proj = xyz_h @ cam.full_proj_transform  # [N, 4]
            w = proj[:, 3:4].clamp(min=1e-6)
            ndc = proj[:, :3] / w  # [N, 3]

            # NDC → pixel coords
            px = ((ndc[:, 0] + 1.0) * 0.5 * W).long()
            py = ((ndc[:, 1] + 1.0) * 0.5 * H).long()

            # Valid: in front of camera and inside image bounds
            valid = (w.squeeze(-1) > 0.1) & (px >= 0) & (px < W) & (py >= 0) & (py < H)

            # Sample sky mask at projected positions
            px_valid = px[valid]
            py_valid = py[valid]
            mask_vals = mask[py_valid, px_valid]  # 1 = not sky

            # Any view saying "not sky" → this gaussian casts shadow
            votes[valid] += mask_vals

        # Mark as shadow-casting if any view voted not-sky
        init_val = (votes > 0).float()

        with torch.no_grad():
            self._casts_shadow.data.copy_(init_val)

        n_shadow = int(init_val.sum().item())
        print(f"[casts_shadow init] {n_shadow}/{N} gaussians marked as shadow-casting "
              f"from {len(cams)} camera views")

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

    def compute_directional_rgb(self, emb_idx, normal_vectors, sun_direction, sun_elevation=None, normal_multiplier=None):
        """
        Compute RGB using explicit directional sun lighting with sun color prior.

        This implements proper Lambert shading without shadows:
            L = albedo * (sun_color(elev) * intensity * max(0, N·L) + ambient + sky_sh)

        Note: Shadows should be applied externally by multiplying the direct component
        with a geometry-based shadow mask.

        Args:
            emb_idx: Image embedding index (for per-image lighting parameters)
            normal_vectors: Surface normals [N, 3]
            sun_direction: Sun direction vector [3] from camera
            sun_elevation: Sun elevation angle in degrees (for color prior)
            normal_multiplier: Per-gaussian sign [N] from compute_normal_world_space
                for correcting camera-flipped normals in N·L computation.

        Returns:
            rgb: Final RGB values [N, 3]
            intensity: Intensity values [N, 3]
            sun_direction: Sun direction vector [3] (normalized)
            lighting_components: Dict with 'direct', 'ambient', 'residual' for debugging
        """
        assert self.use_sun and self.sun_model is not None, \
            "compute_directional_rgb requires use_sun=True"

        albedo = self.get_albedo  # [N, 3]

        # Compute directional lighting (unshadowed) with sun color prior
        intensity_hdr, sun_dir, components = self.sun_model(
            emb_idx, normal_vectors, sun_direction=sun_direction,
            sun_elevation=sun_elevation, normal_multiplier=normal_multiplier
        )

        # Sky gaussians (casts_shadow < 0.5) are emissive — not affected by sun direction or shadows.
        # Override their lighting to use flat sun intensity (no N·L modulation).
        # By zeroing ambient/residual and setting direct = sun_int, the manual shadowed
        # recomposition (direct * shadow_mask + ambient + residual) also stays correct
        # because shadow_mask = 1 for sky gaussians (set in compute_shadows_for_gaussians).
        casts_shadow = self.get_casts_shadow  # [N]
        is_sky = (casts_shadow < 0.5).unsqueeze(-1)  # [N, 1]
        sun_int_flat = components['sun_color'].unsqueeze(0).expand_as(components['direct'])  # [N, 3]
        components['direct'] = torch.where(is_sky, sun_int_flat, components['direct'])
        components['ambient'] = torch.where(is_sky, torch.zeros_like(components['ambient']), components['ambient'])
        components['residual'] = torch.where(is_sky, torch.zeros_like(components['residual']), components['residual'])
        components['sky_sh'] = components['residual']  # Keep in sync

        # Recompute total intensity with sky-overridden components
        intensity_hdr = components['direct'] + components['ambient'] + components['residual']
        intensity_hdr = torch.clamp(intensity_hdr, min=0.0)

        # Apply gamma correction (linear to sRGB)
        intensity_hdr = torch.clamp_min(intensity_hdr, 0.00001)
        intensity = intensity_hdr ** (1 / 2.2)

        # Final RGB = albedo * intensity
        rgb = torch.clamp(intensity * albedo, 0.0)

        return rgb, intensity, sun_dir, components

    def compute_directional_pbr(self, emb_idx, normal_vectors, sun_direction, camera_center,
                                sun_elevation=None, shadow_mask=None):
        """
        Full PBR shading with Cook-Torrance microfacet BRDF (guarded by full_pbr).

        Args:
            emb_idx: Image embedding index.
            normal_vectors: Surface normals [N, 3].
            sun_direction: Sun direction [3].
            camera_center: Camera position [3].
            sun_elevation: Sun elevation angle in degrees.
            shadow_mask: Optional per-gaussian shadow mask [N, 1] where 1=lit, 0=shadow.

        Returns:
            rgb, intensity, sun_dir, components
        """
        assert self.full_pbr and self.use_sun and self.sun_model is not None, \
            "compute_directional_pbr requires full_pbr=True and use_sun=True"

        base_color = self.get_base_color
        roughness = self.get_roughness
        metallic = self.get_metallic

        intensity_hdr_lambert, sun_dir, components = self.sun_model(
            emb_idx, normal_vectors, sun_direction=sun_direction, sun_elevation=sun_elevation
        )

        sun_int = torch.clamp(components['sun_color'], min=1e-6).unsqueeze(0)
        ambient_light = components['ambient']
        residual_light = components['residual']

        normals_norm = normal_vectors / (torch.norm(normal_vectors, dim=-1, keepdim=True) + 1e-8)

        if not isinstance(camera_center, torch.Tensor):
            camera_center = torch.tensor(camera_center, dtype=torch.float32, device=self._xyz.device)
        camera_center = camera_center.to(self._xyz.device)

        view_dirs = camera_center.unsqueeze(0) - self.get_xyz
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)

        light_dirs = sun_dir.unsqueeze(0).expand_as(view_dirs)
        half_vec = view_dirs + light_dirs
        half_vec = half_vec / (torch.norm(half_vec, dim=-1, keepdim=True) + 1e-8)

        n_dot_l = torch.clamp((normals_norm * light_dirs).sum(dim=-1, keepdim=True), min=0.0)
        n_dot_v = torch.clamp((normals_norm * view_dirs).sum(dim=-1, keepdim=True), min=1e-4)
        n_dot_h = torch.clamp((normals_norm * half_vec).sum(dim=-1, keepdim=True), min=1e-4)
        v_dot_h = torch.clamp((view_dirs * half_vec).sum(dim=-1, keepdim=True), min=0.0)

        alpha = torch.clamp(roughness ** 2, min=1e-3)
        alpha2 = alpha ** 2
        denom = (n_dot_h ** 2) * (alpha2 - 1.0) + 1.0
        D = alpha2 / (math.pi * denom ** 2 + 1e-6)

        k = ((roughness + 1.0) ** 2) / 8.0
        G_v = n_dot_v / (n_dot_v * (1.0 - k) + k + 1e-6)
        G_l = n_dot_l / (n_dot_l * (1.0 - k) + k + 1e-6)
        G = G_v * G_l

        F0 = 0.04 * (1.0 - metallic) + base_color * metallic
        F = F0 + (1.0 - F0) * ((1.0 - v_dot_h) ** 5)

        specular = (D * G) * F / torch.clamp(4.0 * n_dot_v * n_dot_l, min=1e-4)
        k_d = (1.0 - F) * (1.0 - metallic)
        diffuse = k_d * base_color / math.pi

        direct_brdf = diffuse + specular
        direct = direct_brdf * sun_int * n_dot_l

        if shadow_mask is not None:
            direct = direct * shadow_mask

        indirect = base_color * (ambient_light + residual_light) * (1.0 - metallic)
        indirect_spec = F0 * (ambient_light + residual_light) * 0.25

        intensity_hdr = torch.clamp(direct + indirect + indirect_spec, min=0.0)
        intensity = torch.clamp_min(intensity_hdr, 1e-5) ** (1 / 2.2)
        rgb = torch.clamp(intensity, 0.0)

        # Sky gaussians (casts_shadow < 0.5) are emissive — bypass all BRDF/lighting modulation.
        # Their final color is just base_color with gamma correction, independent of sun direction.
        casts_shadow_flag = self.get_casts_shadow  # [N]
        is_sky = (casts_shadow_flag < 0.5).unsqueeze(-1)  # [N, 1]
        sky_intensity_hdr = base_color  # Emissive: use base_color as HDR intensity
        sky_intensity = torch.clamp_min(sky_intensity_hdr, 1e-5) ** (1 / 2.2)
        sky_rgb = torch.clamp(sky_intensity, 0.0)
        intensity_hdr = torch.where(is_sky, sky_intensity_hdr, intensity_hdr)
        intensity = torch.where(is_sky, sky_intensity, intensity)
        rgb = torch.where(is_sky, sky_rgb, rgb)

        # Override components so sky gaussians have zero direct (shadow_mask irrelevant)
        # and all energy in indirect (direction-independent)
        direct = torch.where(is_sky, torch.zeros_like(direct), direct)

        components['direct_pbr'] = direct
        components['indirect_pbr'] = torch.where(is_sky, sky_intensity_hdr, indirect + indirect_spec)
        components['lambert_intensity'] = intensity_hdr_lambert

        return rgb, intensity, sun_dir, components


    def update_sky_gaussians(self, cameras, sky_masks, sky_vote_threshold=0.5):
        """
        DEPRECATED: Old voting-based approach.
        Now we use compute_sky_mask_loss() to optimize _casts_shadow via rendering loss.
        """
        pass

    def compute_sky_mask_loss(self, viewpoint_cam, sky_mask, render_func, pipe, background, override_xyz=None):
        """
        Compute loss between rendered casts_shadow values and sky mask.

        Renders the scene with gaussian colors = casts_shadow value (grayscale),
        then computes L1 loss against the sky mask.

        Args:
            viewpoint_cam: Camera to render from
            sky_mask: [H, W] tensor where 0=sky (black), 1=not sky (white)
            render_func: The render function to use
            pipe: Pipeline parameters
            background: Background color tensor
            override_xyz: Optional [N, 3] transformed positions for cam_cal

        Returns:
            sky_mask_loss: L1 loss between rendered casts_shadow and sky mask
            rendered_casts_shadow: The rendered image for visualization
        """
        import torch.nn.functional as F

        # Get casts_shadow values and expand to RGB (grayscale)
        casts_shadow = self.get_casts_shadow  # [N]
        casts_shadow_rgb = casts_shadow.unsqueeze(-1).expand(-1, 3)  # [N, 3]

        # Render with black background so we can see the mask clearly
        black_bg = torch.zeros(3, device="cuda")
        render_pkg = render_func(viewpoint_cam, self, pipe, black_bg, override_color=casts_shadow_rgb, override_xyz=override_xyz)
        rendered_casts_shadow = render_pkg["render"]  # [3, H, W]

        # Convert to grayscale (all channels should be same, take first)
        rendered_gray = rendered_casts_shadow[0:1]  # [1, H, W]

        # Resize sky mask to match rendered size if needed
        H_render, W_render = rendered_casts_shadow.shape[1], rendered_casts_shadow.shape[2]
        H_mask, W_mask = sky_mask.shape

        if H_mask != H_render or W_mask != W_render:
            sky_mask_resized = F.interpolate(
                sky_mask.unsqueeze(0).unsqueeze(0),
                size=(H_render, W_render),
                mode='nearest'
            ).squeeze()  # [H, W]
        else:
            sky_mask_resized = sky_mask

        # Compute L1 loss: rendered should match sky mask
        # sky_mask: 0=sky (should be black/0 in render = non-shadow-casting)
        #           1=not sky (should be white/1 in render = shadow-casting)
        sky_mask_loss = F.l1_loss(rendered_gray.squeeze(), sky_mask_resized)

        return sky_mask_loss, rendered_casts_shadow


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
        self._roughness = nn.Parameter(torch.full((fused_point_cloud.shape[0], 1), 0.6, device="cuda", dtype=torch.float32).requires_grad_(self.full_pbr))
        self._metallic = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 1), device="cuda", dtype=torch.float32).requires_grad_(self.full_pbr))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # Initialize all gaussians as non-shadow-casting (0.0); will be set from sky masks
        self._casts_shadow = nn.Parameter(torch.zeros((fused_point_cloud.shape[0],), dtype=torch.float32, device="cuda").requires_grad_(True))
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
            {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
            {'params': [self._roughness], 'lr': (training_args.albedo_lr * 0.5) if self.full_pbr else 0.0, "name": "roughness"},
            {'params': [self._metallic], 'lr': (training_args.albedo_lr * 0.5) if self.full_pbr else 0.0, "name": "metallic"},
            {'params': [self._casts_shadow], 'lr': training_args.opacity_lr, "name": "casts_shadow"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        if self.use_sun:
            if self.sun_model.scene_sh:
                # Scene-global SH functions of sun direction + global sky SH
                l_env = [
                    {'params': [self.sun_model.intensity_sh], 'lr': training_args.env_lr * 2.0, "name": "sun_intensity"},
                    {'params': [self.sun_model.color_correction_sh], 'lr': training_args.env_lr * 0.5, "name": "sun_color_correction"},
                    {'params': [self.sun_model.ambient_sh], 'lr': training_args.env_lr * 2.0, "name": "ambient_color"},
                ]
            else:
                # Original per-image parameters
                l_env = [
                    {'params': [self.sun_model.sun_intensity_multiplier], 'lr': training_args.env_lr * 2.0, "name": "sun_intensity"},
                    {'params': [self.sun_model.sun_color_correction], 'lr': training_args.env_lr * 0.5, "name": "sun_color_correction"},
                    {'params': [self.sun_model.ambient_color], 'lr': training_args.env_lr * 2.0, "name": "ambient_color"},
                ]
            # Add global sky SH parameters if enabled
            if self.sun_model.use_residual_sh:
                l_env.append({'params': [self.sun_model.sky_sh], 'lr': training_args.env_lr, "name": "sky_sh"})
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
        l.append('roughness_0')
        l.append('metallic_0')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('casts_shadow')
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
        roughness = self._roughness.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        casts_shadow = self._casts_shadow.detach().cpu().numpy().reshape(-1, 1)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc_positive, f_rest_positive, f_dc_negative, f_rest_negative, albedo, roughness, metallic, opacities, scale, rotation, casts_shadow), axis=1)
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

        roughness_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("roughness_")]
        roughness = np.zeros((xyz.shape[0], 1)) + 0.6
        if len(roughness_names) > 0:
            roughness_names = sorted(roughness_names, key=lambda x: int(x.split('_')[-1]))
            for idx, attr_name in enumerate(roughness_names[:1]):
                roughness[:, idx] = np.asarray(plydata.elements[0][attr_name])

        metallic_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("metallic_")]
        metallic = np.zeros((xyz.shape[0], 1))
        if len(metallic_names) > 0:
            metallic_names = sorted(metallic_names, key=lambda x: int(x.split('_')[-1]))
            for idx, attr_name in enumerate(metallic_names[:1]):
                metallic[:, idx] = np.asarray(plydata.elements[0][attr_name])

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
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(self.full_pbr))
        self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(self.full_pbr))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # Load _casts_shadow from PLY if present, otherwise default to all shadow-casting
        casts_shadow_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("casts_shadow")]
        if casts_shadow_names:
            casts_shadow_vals = np.asarray(plydata.elements[0]["casts_shadow"])
            self._casts_shadow = nn.Parameter(torch.tensor(casts_shadow_vals, dtype=torch.float, device="cuda").requires_grad_(False))
            n_sky = int((casts_shadow_vals < 0.5).sum())
            print(f"Loaded _casts_shadow from PLY: {n_sky}/{len(casts_shadow_vals)} sky gaussians")
        else:
            self._casts_shadow = nn.Parameter(torch.ones((xyz.shape[0],), dtype=torch.float, device="cuda").requires_grad_(False))
            print("Warning: _casts_shadow not found in PLY, defaulting to all shadow-casting")

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
        if "roughness" in optimizable_tensors:
            self._roughness = optimizable_tensors["roughness"]
        if "metallic" in optimizable_tensors:
            self._metallic = optimizable_tensors["metallic"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._casts_shadow = optimizable_tensors["casts_shadow"]

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
                              new_albedo, new_opacities, new_scaling, new_rotation, new_casts_shadow=None,
                              new_roughness=None, new_metallic=None):
        # Default new gaussians to non-shadow-casting (0.0); sky mask loss will refine
        if new_casts_shadow is None:
            new_casts_shadow = torch.zeros(new_xyz.shape[0], device="cuda", dtype=torch.float32)
        if new_roughness is None:
            new_roughness = torch.full((new_xyz.shape[0], 1), 0.6, device="cuda", dtype=torch.float32)
        if new_metallic is None:
            new_metallic = torch.zeros((new_xyz.shape[0], 1), device="cuda", dtype=torch.float32)
        new_roughness = new_roughness.requires_grad_(self.full_pbr)
        new_metallic = new_metallic.requires_grad_(self.full_pbr)

        d = {"xyz": new_xyz,
        "f_dc_positive": new_features_dc_positive,
        "f_rest_positive": new_features_rest_positive,
        "f_dc_negative": new_features_dc_negative,
        "f_rest_negative": new_features_rest_negative,
        "albedo": new_albedo,
        "roughness": new_roughness,
        "metallic": new_metallic,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "casts_shadow": new_casts_shadow}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc_positive = optimizable_tensors["f_dc_positive"]
        self._features_rest_positive = optimizable_tensors["f_rest_positive"]
        self._features_dc_negative = optimizable_tensors["f_dc_negative"]
        self._features_rest_negative = optimizable_tensors["f_rest_negative"]
        self._albedo = optimizable_tensors["albedo"]
        if "roughness" in optimizable_tensors:
            self._roughness = optimizable_tensors["roughness"]
        if "metallic" in optimizable_tensors:
            self._metallic = optimizable_tensors["metallic"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._casts_shadow = optimizable_tensors["casts_shadow"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def adaptive_densify_from_loss_grid(self, loss_grid, grid_counts, grid_min, grid_max,
                                          loss_thresh_quantile=0.7, count_thresh=2,
                                          max_new_gaussians=512):
        """
        Spawn new Gaussians in voxel cells that have high accumulated loss but
        few existing Gaussians, filling in empty / under-represented regions.

        Args:
            loss_grid:  [Gx, Gy, Gz] tensor – accumulated per-cell photometric loss
            grid_counts: [Gx, Gy, Gz] tensor – number of Gaussians currently in each cell
            grid_min:   [3] tensor – world-space min corner of the grid (AABB)
            grid_max:   [3] tensor – world-space max corner of the grid (AABB)
            loss_thresh_quantile: float – only cells whose loss is above this quantile
                                  are considered (e.g. 0.7 = top 30%)
            count_thresh: int   – cells with fewer than this many Gaussians are "sparse"
            max_new_gaussians: int – upper cap on the number of Gaussians added in one call
        """
        device = self.get_xyz.device
        G = loss_grid.shape  # (Gx, Gy, Gz)
        cell_size = (grid_max - grid_min) / torch.tensor(G, device=device, dtype=torch.float32)

        # ---- Identify candidate cells: observed, sparse, and high-loss ----
        observed_mask = loss_grid > 0
        if observed_mask.sum() == 0:
            return  # nothing observed yet

        sparsity_deficit = (float(count_thresh) - grid_counts).clamp_min(0.0)
        sparse_mask = sparsity_deficit > 0
        candidate_mask = observed_mask & sparse_mask
        if candidate_mask.sum() == 0:
            return

        sparse_losses = loss_grid[candidate_mask]
        loss_threshold = torch.quantile(sparse_losses, loss_thresh_quantile)
        candidate_mask = candidate_mask & (loss_grid >= loss_threshold)

        # Fallback: if quantile is too strict, keep all observed sparse cells.
        if candidate_mask.sum() == 0:
            candidate_mask = observed_mask & sparse_mask

        candidate_indices = candidate_mask.nonzero(as_tuple=False)  # [K, 3]

        if candidate_indices.shape[0] == 0:
            return  # no candidates

        # ---- Determine how many Gaussians to place per cell ----
        n_cells = candidate_indices.shape[0]
        # Weight by loss so worse cells get more Gaussians.
        # Allocate *exactly* max_new_gaussians samples while ensuring broad coverage.
        cell_losses = loss_grid[candidate_mask].clamp_min(0)
        cell_sparse = sparsity_deficit[candidate_mask]
        if cell_losses.sum() <= 0:
            return

        # Score combines loss and sparsity pressure so truly empty cells are preferred.
        cell_scores = cell_losses * (1.0 + 2.0 * cell_sparse / (float(count_thresh) + 1e-6))
        weights = cell_scores / cell_scores.sum().clamp_min(1e-12)

        budget = int(max_new_gaussians)
        budget = max(budget, 0)
        if budget == 0:
            return

        # Guarantee at least 1 gaussian in the top cells (up to budget)
        top_k = min(n_cells, budget)
        order = torch.argsort(cell_scores, descending=True)
        per_cell_budget = torch.zeros(n_cells, device=device, dtype=torch.long)
        per_cell_budget[order[:top_k]] = 1
        remaining = budget - top_k

        if remaining > 0:
            sampled = torch.multinomial(weights, remaining, replacement=True)
            per_cell_budget += torch.bincount(sampled, minlength=n_cells)

        # ---- Sample positions uniformly inside each candidate cell ----
        new_xyz_list = []
        for i in range(n_cells):
            idx = candidate_indices[i]  # (ix, iy, iz)
            n_pts = per_cell_budget[i].item()
            # Cell world-space bounds
            cell_lo = grid_min + idx.float() * cell_size
            cell_hi = cell_lo + cell_size
            pts = torch.rand(n_pts, 3, device=device) * (cell_hi - cell_lo) + cell_lo
            new_xyz_list.append(pts)

        new_xyz = torch.cat(new_xyz_list, dim=0)
        n_new = new_xyz.shape[0]

        if n_new == 0:
            return

        # ---- Initialise attributes for the new Gaussians ----
        # Use scene-wide median / mean as sensible defaults so new Gaussians
        # blend in and can be optimised quickly.

        # Scale: keep newborn Gaussians small enough to fit target cells.
        median_scale_lin = self.get_scaling.median(dim=0).values  # [2], linear space
        cell_scale_lin = cell_size[:2].clamp_min(1e-6) * 0.5
        target_scale_lin = torch.minimum(median_scale_lin, cell_scale_lin)
        new_scaling = self.scaling_inverse_activation(target_scale_lin).unsqueeze(0).expand(n_new, -1).clone()

        # Rotation: random quaternions
        new_rotation = torch.randn(n_new, 4, device=device)
        new_rotation = torch.nn.functional.normalize(new_rotation, dim=-1)

        # Opacity: start above prune threshold so they survive long enough to learn.
        new_opacity = self.inverse_opacity_activation(
            0.12 * torch.ones(n_new, 1, device=device)
        )

        # SH features: initialise near zero (neutral grey)
        sh_dim = (self.max_sh_degree + 1) ** 2
        new_features_dc_pos = torch.zeros(n_new, 1, 3, device=device) + 0.02
        new_features_rest_pos = torch.zeros(n_new, sh_dim - 1, 3, device=device) + 0.01
        new_features_dc_neg = torch.zeros(n_new, 1, 3, device=device) + 0.02
        new_features_rest_neg = torch.zeros(n_new, sh_dim - 1, 3, device=device) + 0.01

        # Albedo: initialise to mean albedo
        mean_albedo = self._albedo.mean(dim=0, keepdim=True)
        new_albedo = mean_albedo.expand(n_new, -1).clone()
        mean_roughness = self._roughness.mean(dim=0, keepdim=True)
        mean_metallic = self._metallic.mean(dim=0, keepdim=True)
        new_roughness = mean_roughness.expand(n_new, -1).clone()
        new_metallic = mean_metallic.expand(n_new, -1).clone()

        # Shadow casting: default on
        new_casts_shadow = torch.ones(n_new, device=device)

        self.densification_postfix(
            new_xyz, new_features_dc_pos, new_features_rest_pos,
            new_features_dc_neg, new_features_rest_neg,
            new_albedo, new_opacity, new_scaling, new_rotation,
            new_casts_shadow, new_roughness=new_roughness, new_metallic=new_metallic
        )

        print(f"[Adaptive Densification] Added {n_new} Gaussians across "
              f"{n_cells} sparse high-loss cells "
              f"(total now: {self.get_xyz.shape[0]})")

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
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
        new_metallic = self._metallic[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_casts_shadow = self._casts_shadow[selected_pts_mask].repeat(N)  # Inherit shadow-casting from parent

        self.densification_postfix(new_xyz, new_features_dc_positive, new_features_rest_positive,new_features_dc_negative, new_features_rest_negative, new_albedo, new_opacity, new_scaling, new_rotation, new_casts_shadow, new_roughness=new_roughness, new_metallic=new_metallic)

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
        new_roughness = self._roughness[selected_pts_mask]
        new_metallic = self._metallic[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_casts_shadow = self._casts_shadow[selected_pts_mask]  # Inherit shadow-casting from parent

        self.densification_postfix(new_xyz, new_features_dc_positive, new_features_rest_positive,new_features_dc_negative, new_features_rest_negative, new_albedo, new_opacities, new_scaling, new_rotation, new_casts_shadow, new_roughness=new_roughness, new_metallic=new_metallic)

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

