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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 2 # degree for SH_gauss and SH_env
        self._source_path = ""
        self.eval_file = ""
        self._model_path = "output"
        self._images = "images"
        self._resolution = 2
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.with_mlp = False
        self.mlp_W = 64
        self.mlp_D = 3
        self.N_a = 24
        # Sun position parameters - explicit directional lighting with sun color prior
        self.use_sun = False  # Enable physical sun model with explicit directional lighting
        self.full_pbr = False  # Enable full PBR shading path (guarded, use with --use_sun)
        self.sun_json_path = ""  # Path to JSON file with sun positions per image
        self.sky_mask_path = ""  # Path to folder with sky masks (black=sky, white=not sky)
        self.use_residual_sh = True  # Use global sky SH for environment (enables relighting)
        # Camera calibration refinement (jointly optimise camera poses during training)
        self.use_cam_cal = False  # Enable learnable camera pose refinement
        # Sun direction calibration (jointly optimise per-image sun directions during training)
        self.use_sun_cal = False  # Enable learnable sun direction refinement
        # Shadow computation method: 'none', 'shadow_map', 'ray_march', 'voxel'
        # Progressive resolution: train at low-res first, switch to high-res for detail
        self.progressive_resolution = False  # Enable progressive resolution training
        self.progressive_switch_iter = 15000  # Iteration to switch from low-res to full-res
        self.shadow_method = "shadow_map"
        self.shadow_map_resolution = 512  # Resolution for shadow mapping
        self.shadow_bias = 0.1  # Depth bias for shadow comparison
        self.ray_march_steps = 64  # Number of steps for ray marching
        self.voxel_resolution = 128  # Resolution for voxel grid
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.use_gaussians = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 40_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.002 # for SH_gauss
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.opacity_cull = 0.05
        self.lambda_dist = 0.01
        self.lambda_normal = 0.05

        self.start_shadowed = 20500
        self.warmup = 20000

        self.start_regularization = 6000
        self.densification_interval = 500
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        # Adaptive grid-based densification (populates empty high-loss regions)
        self.use_adaptive_dens = False
        # More aggressive defaults to better fill empty/under-represented regions.
        self.adaptive_dens_grid_res = 48             # Higher: finer/smaller cells (more localized placement, noisier); Lower: coarser/larger cells (smoother, less precise)
        self.adaptive_dens_interval = 250            # Higher: trigger less often (slower growth); Lower: trigger more often (faster growth, more compute/memory)
        self.adaptive_dens_from_iter = 500           # Higher: starts later (safer/stabler early training); Lower: starts earlier (fills gaps sooner)
        self.adaptive_dens_until_iter = 30000        # Higher: keep adaptive densification active longer; Lower: stop earlier
        self.adaptive_dens_loss_thresh = 0.01         # Higher (e.g. 0.8): only very top-loss cells selected (more selective); Lower (e.g. 0.2): many cells selected (more aggressive spread)
        self.adaptive_dens_max_gaussians = 4096      # Higher: more new gaussians per trigger (stronger effect, higher VRAM/time); Lower: fewer additions
        self.adaptive_dens_count_thresh = 12         # Higher: more cells treated as sparse (more filling); Lower: only near-empty cells treated as sparse
        # When depth is 0 (often indicates holes / empty regions), optionally sample along the camera ray
        # within the scene AABB so loss can still be attributed to 3D space.
        self.adaptive_dens_fill_empty = True
        self.adaptive_dens_zero_depth_max_pixels = 16384  # Higher: use more zero-depth pixels for hole filling (stronger but slower); Lower: faster but less coverage
        self.adaptive_dens_zero_depth_samples = 2         # Higher: more samples along each zero-depth ray (better volume coverage, more compute); Lower: cheaper
        self.adaptive_dens_surface_samples = 2            # Higher: more samples around valid-depth surfaces (thicker coverage); Lower: thinner/surface-only coverage
        self.adaptive_dens_surface_jitter = 20           # Higher: broader depth neighborhood around surface (more exploratory); Lower: tighter around rendered surface
        self.adaptive_dens_ema_decay = 0.8                # Higher: slower/stabler adaptation to new loss (more memory); Lower: faster reaction to recent errors
        self.adaptive_dens_use_highfreq = True            # Enable high-frequency (edge/detail) checks to prioritize structurally important regions
        self.adaptive_dens_hf_boost = 0.75                # Additional score multiplier for high-frequency pixels
        self.adaptive_dens_hf_quantile = 0.6              # Quantile threshold for high-frequency pixel detection
        self.adaptive_dens_hole_score_quantile = 0.5      # For depth==0 pixels, keep only higher-scoring half by default
        self.adaptive_dens_vis_interval = 100             # TensorBoard logging interval for adaptive densification maps
        self.adaptive_dens_center_radius = -1.0           # If >0, only densify within this distance from mean camera center
        self.dens_everything = False                      # Debug: force adaptive densification to populate all grid cells
        self.dens_everything_per_cell = 1                 # Debug: target number of new gaussians per grid cell
        self.dens_everything_max_gaussians = 50000        # Debug safety cap to avoid OOM when densifying every cell

        # Monocular depth estimation regularisation (penalises Gaussians far from estimated surfaces)
        self.use_depth_est = False                        # Enable depth estimation surface regularisation
        self.depth_est_lambda = 0.1                       # Higher: stronger depth regularisation (surfaces snap to estimate); Lower: weaker constraint
        self.depth_est_from_iter = 1000                   # Higher: start regularising later (geometry settles first); Lower: start earlier
        self.depth_est_model = "Intel/dpt-hybrid-midas"   # HuggingFace model id for monocular depth estimation
        # Depth-guided densification (spawns Gaussians in sparse areas using mono-depth)
        self.depth_est_densify = True                     # Enable depth-guided densification (requires --use_depth_est)
        self.depth_est_densify_interval = 500             # Higher: trigger less often; Lower: more frequent fills
        self.depth_est_densify_from_iter = 1500           # Higher: start later (geometry must settle first); Lower: start earlier
        self.depth_est_densify_until_iter = 15000         # Higher: keep filling longer; Lower: stop sooner
        self.depth_est_densify_max_new = 1024             # Higher: more new Gaussians per trigger (stronger fill, more VRAM); Lower: fewer
        self.depth_est_densify_alpha_thresh = 0.3         # Higher: stricter (more area considered sparse); Lower: only truly empty pixels
        self.depth_est_densify_loss_quantile = 0.5        # Higher: only highest-loss sparse pixels get filled; Lower: more permissive

        self.env_lr = 0.02
        self.mlp_lr = 0.002
        self.mlp_lr_final_ratio = 10.0
        self.embedding_lr = 0.002
        self.embedding_lr_final_ratio = 10
        self.albedo_lr= 0.0025
        # Sun direction calibration learning rates
        #self.sun_cal_lr = 0.001         # Learning rate for sun direction deltas
        self.sun_cal_lr = 0.1
        self.sun_cal_from_iter = 10000    # Start sun calibration after this iteration
        self.sun_cal_until_iter = 30000 # Stop sun calibration at this iteration
        self.sun_cal_reg_lambda = 0.0001  # L2 regularization pulling delta_sun_dir toward zero (higher = tighter to input)

        self.gauss_loss_lambda = 0.001
        self.env_loss_lambda = 0.05
        self.consistency_loss_lambda_init = 1.0
        self.consistency_loss_lambda_final_ratio = 1.0
        self.shadow_loss_lambda=10.0
        self.sky_mask_loss_weight = 0.5

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
