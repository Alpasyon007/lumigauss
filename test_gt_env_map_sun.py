#
# Test script for sun-based methods (--use_sun, --full_pbr, etc.)
#
# Equivalent to test_gt_env_map.py but for models trained with explicit
# directional sun lighting and shadow maps. Enables fair comparison of
# metrics (PSNR, SSIM, MSE, MAE) between the original SH/MLP approach
# and the newer sun-based approach.
#
# Instead of rotating an SH environment map, this script sweeps the sun
# azimuth over the angle range specified in the test config, computes
# shadow maps for each candidate direction, and selects the best match
# to the ground-truth image (SOLNERF protocol).
#

import cv2
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import numpy as np
import sys
import importlib
import json
from utils.normal_utils import compute_normal_world_space
from utils.shadow_utils import compute_shadows_for_gaussians
from utils.loss_utils import mse2psnr, img2mae, img2mse
from skimage.metrics import structural_similarity as ssim_skimage


TINY_NUMBER = 1e-6


def sun_direction_from_azimuth(azimuth_rad, elevation_rad=None, base_sun_direction=None):
    """
    Rotate a sun direction around the vertical (Y) axis by a given azimuth angle.

    If base_sun_direction is given, rotates it. Otherwise constructs a direction
    from azimuth and elevation angles.

    Args:
        azimuth_rad: Azimuth rotation in radians (around Y axis).
        elevation_rad: Elevation angle in radians (from horizon). Used only when
                       base_sun_direction is None.
        base_sun_direction: Optional [3] base direction to rotate.

    Returns:
        Rotated sun direction [3] tensor on CUDA.
    """
    cos_a = np.cos(azimuth_rad)
    sin_a = np.sin(azimuth_rad)

    # Y-axis rotation matrix
    rot_y = torch.tensor([
        [cos_a,  0, sin_a],
        [0,      1, 0    ],
        [-sin_a, 0, cos_a]
    ], dtype=torch.float32)

    if base_sun_direction is not None:
        if isinstance(base_sun_direction, torch.Tensor):
            d = base_sun_direction.cpu().float()
        else:
            d = torch.tensor(base_sun_direction, dtype=torch.float32)
        rotated = rot_y @ d
    else:
        # Construct direction from angles (elevation from horizon, azimuth around Y)
        if elevation_rad is None:
            elevation_rad = np.pi / 4  # default 45 degrees
        cos_e = np.cos(elevation_rad)
        sin_e = np.sin(elevation_rad)
        d = torch.tensor([cos_e * np.sin(azimuth_rad), sin_e, cos_e * np.cos(azimuth_rad)],
                         dtype=torch.float32)
        rotated = d

    rotated = rotated / (torch.norm(rotated) + 1e-8)
    return rotated.cuda()


def render_shadowed_sun(gaussians, viewpoint_cam, pipeline, background,
                        normal_vectors, multiplier, emb_idx,
                        sun_direction, sun_elevation,
                        shadow_method, shadow_map_resolution, shadow_bias,
                        ray_march_steps, voxel_resolution):
    """
    Render a shadowed image using the sun model (directional or PBR).

    Matches the rendering logic used during training (train.py and
    training_report in train_utils.py) exactly.

    Returns:
        rendering_shadowed: Clamped rendered image [3, H, W]
        rendering_unshadowed: Clamped unshadowed rendered image [3, H, W]
    """
    # --- Unshadowed ---
    if gaussians.full_pbr:
        rgb_unshadowed, _, sun_dir_out, components = gaussians.compute_directional_pbr(
            emb_idx, normal_vectors, sun_direction, viewpoint_cam.camera_center,
            sun_elevation=sun_elevation
        )
    else:
        rgb_unshadowed, _, sun_dir_out, components = gaussians.compute_directional_rgb(
            emb_idx, normal_vectors, sun_direction, sun_elevation=sun_elevation
        )
    render_pkg_unshadowed = render(viewpoint_cam, gaussians, pipeline, background,
                                   override_color=rgb_unshadowed)
    rendering_unshadowed = torch.clamp(render_pkg_unshadowed["render"], 0.0, 1.0)

    # --- Shadows ---
    # Use the normalized sun direction returned by the model (matches training)
    effective_shadow_method = shadow_method
    if getattr(pipeline, "use_gaussians", False) and effective_shadow_method == "shadow_map":
        effective_shadow_method = "ray_march"

    shadow_mask, _, _ = compute_shadows_for_gaussians(
        gaussians,
        sun_dir_out,
        pipeline,
        method=effective_shadow_method,
        shadow_map_resolution=shadow_map_resolution,
        shadow_bias=shadow_bias,
        ray_march_steps=ray_march_steps,
        voxel_resolution=voxel_resolution,
        device="cuda"
    )
    shadow_mask = shadow_mask.unsqueeze(-1)  # [N, 1]

    # --- Shadowed ---
    if gaussians.full_pbr:
        rgb_shadowed, _, _, _ = gaussians.compute_directional_pbr(
            emb_idx, normal_vectors, sun_direction, viewpoint_cam.camera_center,
            sun_elevation=sun_elevation, shadow_mask=shadow_mask
        )
    else:
        # Reuse components from the unshadowed call (same lighting params)
        direct_light = components['direct']
        ambient_light = components['ambient']
        residual_light = components['residual']

        # Shadowed intensity: direct * shadow + ambient + residual
        # (matches train.py line ~301)
        intensity_hdr = direct_light * shadow_mask + ambient_light + residual_light
        intensity_hdr = torch.clamp_min(intensity_hdr, 0.00001)
        intensity = intensity_hdr ** (1 / 2.2)

        albedo = gaussians.get_albedo
        rgb_shadowed = torch.clamp(intensity * albedo, 0.0)

    render_pkg = render(viewpoint_cam, gaussians, pipeline, background,
                        override_color=rgb_shadowed)
    rendering_shadowed = torch.clamp(render_pkg["render"], 0.0, 1.0)

    return rendering_shadowed, rendering_unshadowed


def render_set(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
               opt: OptimizationParams, test_config: str, no_sweep: bool = False):

    dataset.eval = True

    # ---- Create GaussianModel ----
    gaussians = GaussianModel(
        dataset.sh_degree, dataset.with_mlp, dataset.mlp_W, dataset.mlp_D, dataset.N_a,
        use_sun=True, n_images=1700,
        use_residual_sh=dataset.use_residual_sh,
        full_pbr=dataset.full_pbr, use_ao=dataset.use_ao,
        sky_sh_degree=dataset.sky_sh_degree,
        use_color_bias=dataset.use_color_bias,
        optimize_casts_shadow=dataset.optimize_casts_shadow
    )

    scene = Scene(dataset, gaussians, load_iteration=iteration)

    # Update n_images and verify sun data
    n_images = len(scene.getTrainCameras())
    missing_sun = [cam.image_name for cam in scene.getTrainCameras() if cam.sun_direction is None]
    if missing_sun:
        print(f"Warning: Sun data missing for {len(missing_sun)} training images. "
              f"First few: {missing_sun[:5]}")

    gaussians.n_images = n_images
    if gaussians.sun_model is not None:
        gaussians.sun_model.eval()
        print(f"SunModel loaded for {n_images} images")
    else:
        raise RuntimeError("Sun model was not loaded. Check checkpoint files.")

    # Shadow parameters — use same defaults as training_report() in train_utils.py
    # (training test-view rendering uses resolution=512, bias=0.1 regardless of
    #  dataset args, to keep test evaluation consistent)
    shadow_method = "shadow_map"
    shadow_map_resolution = 512
    shadow_bias = 0.1
    ray_march_steps = 64
    voxel_resolution = 128

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ---- Output directories ----
    method_tag = "sun_pbr" if dataset.full_pbr else "sun"
    gt_path = os.path.join(dataset.model_path, f"test_with_gt_env_map_{method_tag}", "gt_image")
    makedirs(gt_path, exist_ok=True)
    render_path = os.path.join(dataset.model_path, f"test_with_gt_env_map_{method_tag}",
                               f"render_gt_env_map", f"ours_{scene.loaded_iter}")
    makedirs(render_path, exist_ok=True)

    # ---- Load appearance LUT ----
    lut_path = os.path.join(dataset.model_path, "appearance_lut.json")
    if os.path.exists(lut_path):
        with open(lut_path) as f:
            appearance_lut = json.load(f)
    else:
        # Build a simple LUT from training cameras order
        appearance_lut = {cam.image_name: i for i, cam in enumerate(scene.getTrainCameras())}
        print("Warning: appearance_lut.json not found, built from training camera order.")

    # ---- Read test config ----
    sys.path.append(test_config)
    config = importlib.import_module("test_config").config

    # ---- Select test cameras ----
    tmp_cameras = scene.getTestCameras()
    test_cameras = [c for c in tmp_cameras if c.image_name in config.keys()]

    if not test_cameras:
        print("No test cameras match the config keys. Config keys:", list(config.keys()))
        print("Available test cams:", [c.image_name for c in tmp_cameras[:10]])
        return

    # ---- Metric accumulators ----
    ssims_unshadowed, psnrs_unshadowed, mse_unshadowed, mae_unshadowed = [], [], [], []
    ssims_shadowed, psnrs_shadowed, mse_shadowed, mae_shadowed = [], [], [], []
    img_names, used_angles = [], []

    for viewpoint_cam in tqdm(test_cameras, desc="Test images"):
        print(viewpoint_cam.image_name)

        image_config = config[viewpoint_cam.image_name]
        mask_path = image_config["mask_path"]
        sun_angle_range = image_config["sun_angles"]

        # ---- Appearance index ----
        # For test cameras, use the first training image's appearance (matches
        # training_report() in train_utils.py which does the same)
        if viewpoint_cam.image_name in appearance_lut:
            emb_idx = appearance_lut[viewpoint_cam.image_name]
        else:
            emb_idx = list(appearance_lut.values())[0]

        # ---- Ground truth ----
        gt_image = viewpoint_cam.original_image.cuda()
        torchvision.utils.save_image(gt_image, os.path.join(gt_path, viewpoint_cam.image_name))

        # ---- Evaluation mask ----
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (gt_image.shape[2], gt_image.shape[1]))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = torch.from_numpy(mask // 255).cuda()

        # ---- Normals ----
        quaternions = gaussians.get_rotation
        scales = gaussians.get_scaling
        normal_vectors, multiplier = compute_normal_world_space(
            quaternions, scales, viewpoint_cam.world_view_transform, gaussians.get_xyz
        )

        # ---- Sun direction ----
        # Use camera's own sun direction (matches training_report behavior)
        base_sun_dir = viewpoint_cam.sun_direction
        sun_elevation = viewpoint_cam.sun_elevation  # Pass as-is (can be None, like training)

        if base_sun_dir is None:
            print(f"  Warning: No sun direction for {viewpoint_cam.image_name}, using default.")
            base_sun_dir = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

        if no_sweep:
            # ---- Direct render (matches training test-view render exactly) ----
            best_angle = torch.tensor(0.0)
            best_sun_dir = base_sun_dir

            rendering_shadowed, rendering_unshadowed = render_shadowed_sun(
                gaussians, viewpoint_cam, pipeline, background,
                normal_vectors, multiplier, emb_idx,
                best_sun_dir, sun_elevation,
                shadow_method, shadow_map_resolution, shadow_bias,
                ray_march_steps, voxel_resolution
            )
        else:
            # ---- Sun direction search (SOLNERF protocol) ----
            # Sweep sun direction around Y axis over angle range from test config
            best_psnr = 0
            best_angle = None

            n = 51
            angle_list = torch.linspace(sun_angle_range[0], sun_angle_range[1], n)

            for angle in tqdm(angle_list, desc="  Angle sweep", leave=False):
                # Rotate base sun direction around Y axis by the candidate angle
                sun_dir = sun_direction_from_azimuth(float(angle), base_sun_direction=base_sun_dir)

                rendering_shadowed, _ = render_shadowed_sun(
                    gaussians, viewpoint_cam, pipeline, background,
                    normal_vectors, multiplier, emb_idx,
                    sun_dir, sun_elevation,
                    shadow_method, shadow_map_resolution, shadow_bias,
                    ray_march_steps, voxel_resolution
                )

                current_psnr = mse2psnr(img2mse(rendering_shadowed, gt_image, mask=mask))
                if current_psnr > best_psnr:
                    best_angle = angle
                    best_psnr = current_psnr

            # ---- Render best angle ----
            best_sun_dir = sun_direction_from_azimuth(float(best_angle), base_sun_direction=base_sun_dir)

            rendering_shadowed, rendering_unshadowed = render_shadowed_sun(
                gaussians, viewpoint_cam, pipeline, background,
                normal_vectors, multiplier, emb_idx,
                best_sun_dir, sun_elevation,
                shadow_method, shadow_map_resolution, shadow_bias,
                ray_march_steps, voxel_resolution
            )

        # ---- Save rendered images ----
        torchvision.utils.save_image(rendering_unshadowed * mask,
                                     os.path.join(render_path, "unshadowed_" + viewpoint_cam.image_name))
        torchvision.utils.save_image(rendering_shadowed * mask,
                                     os.path.join(render_path, "shadowed_" + viewpoint_cam.image_name))

        # Also render and save albedo
        rgb_precomp_alb = gaussians.get_albedo
        render_pkg_alb = render(viewpoint_cam, gaussians, pipeline, background,
                                override_color=rgb_precomp_alb)
        albedo_img = render_pkg_alb["render"]
        torchvision.utils.save_image(albedo_img * mask,
                                     os.path.join(render_path, "albedo_" + viewpoint_cam.image_name))

        used_angles.append(best_angle)
        img_names.append(viewpoint_cam.image_name)

        # ---- Compute metrics ----
        psnrs_unshadowed.append(mse2psnr(img2mse(rendering_unshadowed, gt_image, mask=mask)))
        mae_unshadowed.append(img2mae(rendering_unshadowed, gt_image, mask=mask))
        mse_unshadowed.append(img2mse(rendering_unshadowed, gt_image, mask=mask))

        psnrs_shadowed.append(mse2psnr(img2mse(rendering_shadowed, gt_image, mask=mask)))
        mae_shadowed.append(img2mae(rendering_shadowed, gt_image, mask=mask))
        mse_shadowed.append(img2mse(rendering_shadowed, gt_image, mask=mask))

        unshadowed_np = rendering_unshadowed.cpu().detach().numpy().transpose(1, 2, 0)
        shadowed_np = rendering_shadowed.cpu().detach().numpy().transpose(1, 2, 0)
        gt_image_np = gt_image.cpu().detach().numpy().transpose(1, 2, 0)

        _, full = ssim_skimage(unshadowed_np, gt_image_np, win_size=5,
                               channel_axis=2, full=True, data_range=1.0)
        mssim = (torch.tensor(full).cuda() * mask.unsqueeze(-1)).sum() / (3 * mask.sum())
        ssims_unshadowed.append(mssim)

        _, full = ssim_skimage(shadowed_np, gt_image_np, win_size=5,
                               channel_axis=2, full=True, data_range=1.0)
        mssim = (torch.tensor(full).cuda() * mask.unsqueeze(-1)).sum() / (3 * mask.sum())
        ssims_shadowed.append(mssim)

    # ---- Save metrics ----
    with open(os.path.join(render_path, "metrics.txt"), 'w') as f:
        print("  PSNR unshadowed: {:>12.7f}".format(torch.tensor(psnrs_unshadowed).mean(), ".5"), file=f)
        print("  MSE unshadowed: {:>12.7f}".format(torch.tensor(mse_unshadowed).mean(), ".5"), file=f)
        print("  MAE unshadowed: {:>12.7f}".format(torch.tensor(mae_unshadowed).mean(), ".5"), file=f)
        print("  SSIM skimage unshadowed: {:>12.7f}".format(torch.tensor(ssims_unshadowed).mean(), ".5"), file=f)

        print("  PSNR shadowed: {:>12.7f}".format(torch.tensor(psnrs_shadowed).mean(), ".5"), file=f)
        print("  MSE shadowed: {:>12.7f}".format(torch.tensor(mse_shadowed).mean(), ".5"), file=f)
        print("  MAE shadowed: {:>12.7f}".format(torch.tensor(mae_shadowed).mean(), ".5"), file=f)
        print("  SSIM skimage shadowed: {:>12.7f}".format(torch.tensor(ssims_shadowed).mean(), ".5"), file=f)

        print(f"  best psnrs, image order: {psnrs_shadowed}. optimized for psnr shadowed", file=f)
        print(f"  best angles: {used_angles}", file=f)
        print(f"  image names: {img_names}", file=f)
        print(f"  method: {'full_pbr' if dataset.full_pbr else 'sun_directional'}", file=f)
        print(f"  shadow_method: {shadow_method}", file=f)

    # Also save per-image metrics as JSON for easier downstream analysis
    per_image_metrics = {}
    for i, name in enumerate(img_names):
        per_image_metrics[name] = {
            "psnr_shadowed": float(psnrs_shadowed[i]),
            "psnr_unshadowed": float(psnrs_unshadowed[i]),
            "mse_shadowed": float(mse_shadowed[i]),
            "mse_unshadowed": float(mse_unshadowed[i]),
            "mae_shadowed": float(mae_shadowed[i]),
            "mae_unshadowed": float(mae_unshadowed[i]),
            "ssim_shadowed": float(ssims_shadowed[i]),
            "ssim_unshadowed": float(ssims_unshadowed[i]),
            "best_angle": float(used_angles[i]),
        }
    with open(os.path.join(render_path, "per_image_metrics.json"), 'w') as f:
        json.dump(per_image_metrics, f, indent=2)

    print("\n=== Results ===")
    print("PSNR shadowed:  {:>12.7f}".format(torch.tensor(psnrs_shadowed).mean(), ".5"))
    print("MSE shadowed:   {:>12.7f}".format(torch.tensor(mse_shadowed).mean(), ".5"))
    print("MAE shadowed:   {:>12.7f}".format(torch.tensor(mae_shadowed).mean(), ".5"))
    print("SSIM shadowed:  {:>12.7f}".format(torch.tensor(ssims_shadowed).mean(), ".5"))
    print()
    print("PSNR unshadowed: {:>12.7f}".format(torch.tensor(psnrs_unshadowed).mean(), ".5"))
    print("MSE unshadowed:  {:>12.7f}".format(torch.tensor(mse_unshadowed).mean(), ".5"))
    print("MAE unshadowed:  {:>12.7f}".format(torch.tensor(mae_unshadowed).mean(), ".5"))
    print("SSIM unshadowed: {:>12.7f}".format(torch.tensor(ssims_unshadowed).mean(), ".5"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Test sun-based models with GT env map protocol")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--test_config", default="", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_sweep", action="store_true",
                        help="Skip angle sweep — render with the camera's own sun direction "
                             "(reproduces the training test-view render exactly)")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    with torch.no_grad():
        render_set(model.extract(args), args.iteration, pipeline.extract(args),
                   opt.extract(args), args.test_config,
                   no_sweep=getattr(args, 'no_sweep', False))
