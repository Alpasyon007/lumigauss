"""
Optimise per-image sun directions for eval/test views while keeping
everything else in the scene frozen, then evaluate with the optimised
sun directions to produce best-case metrics.

This gives an upper bound on evaluation quality: the sun direction for
each test view is gradient-optimised to minimise the photometric loss
against the ground-truth image. All gaussian parameters, the sun model,
and shadow computation remain frozen.

The only learnable parameter per test view is a 3D delta added to the
camera's initial sun direction, identical to the sun_cal mechanism used
during training but applied independently to each test view.

Usage:
    python optimise_sun_eval.py -m output/lk2_sun_v3.1.10 --iteration 40000 \
        --test_config example_test_configs/lk2 --opt_iters 200 --lr 0.05
"""

import cv2
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import sys
import json
import importlib
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.normal_utils import compute_normal_world_space
from utils.shadow_utils import compute_shadows_for_gaussians
from utils.loss_utils import l1_loss, ssim, img2mse, img2mae, mse2psnr
from skimage.metrics import structural_similarity as ssim_skimage


TINY_NUMBER = 1e-6


def rebase_config_path(path, config_dir):
    """Rebase a possibly-absolute path from test_config.py to config_dir."""
    if os.path.isfile(path):
        return path
    marker = 'eval_files' + os.sep
    alt_marker = 'eval_files/'
    for m in (marker, alt_marker):
        idx = path.find(m)
        if idx != -1:
            relative = path[idx + len(m):]
            rebased = os.path.join(config_dir, relative)
            if os.path.isfile(rebased):
                return rebased
    basename_path = os.path.join(config_dir, os.path.basename(path))
    if os.path.isfile(basename_path):
        return basename_path
    return path


def render_with_sun_direction(gaussians, viewpoint_cam, pipeline, background,
                              normal_vectors, multiplier, emb_idx,
                              sun_direction, sun_elevation,
                              shadow_method, shadow_map_resolution, shadow_bias,
                              ray_march_steps, voxel_resolution,
                              shadow_scale_modifier=1.5, shadow_dilation_kernel=5,
                              shadow_alpha_threshold=0.01):
    """
    Render unshadowed + shadowed images for a given sun direction.

    Returns:
        rendering_shadowed [3,H,W], rendering_unshadowed [3,H,W], components dict
    """
    # Unshadowed
    if gaussians.full_pbr:
        rgb_unshadowed, _, sun_dir_out, components = gaussians.compute_directional_pbr(
            emb_idx, normal_vectors, sun_direction, viewpoint_cam.camera_center,
            sun_elevation=sun_elevation
        )
    else:
        rgb_unshadowed, _, sun_dir_out, components = gaussians.compute_directional_rgb(
            emb_idx, normal_vectors, sun_direction, sun_elevation=sun_elevation,
            normal_multiplier=multiplier
        )
    render_pkg_unshadowed = render(viewpoint_cam, gaussians, pipeline, background,
                                   override_color=rgb_unshadowed)
    rendering_unshadowed = torch.clamp(render_pkg_unshadowed["render"], 0.0, 1.0)

    # Shadows
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
        device="cuda",
        normal_vectors=normal_vectors,
        shadow_scale_modifier=shadow_scale_modifier,
        shadow_dilation_kernel=shadow_dilation_kernel,
        alpha_threshold=shadow_alpha_threshold,
        normal_multiplier=multiplier,
    )
    shadow_mask = shadow_mask.unsqueeze(-1)  # [N, 1]

    # Shadowed
    if gaussians.full_pbr:
        rgb_shadowed, _, _, _ = gaussians.compute_directional_pbr(
            emb_idx, normal_vectors, sun_direction, viewpoint_cam.camera_center,
            sun_elevation=sun_elevation, shadow_mask=shadow_mask
        )
    else:
        direct_light = components['direct']
        ambient_light = components['ambient']
        residual_light = components['residual']
        intensity_hdr = direct_light * shadow_mask + ambient_light + residual_light
        intensity_hdr = torch.clamp_min(intensity_hdr, 0.00001)
        intensity = intensity_hdr ** (1 / 2.2)
        albedo = gaussians.get_albedo
        rgb_shadowed = torch.clamp(intensity * albedo, 0.0)

    render_pkg = render(viewpoint_cam, gaussians, pipeline, background,
                        override_color=rgb_shadowed)
    rendering_shadowed = torch.clamp(render_pkg["render"], 0.0, 1.0)

    return rendering_shadowed, rendering_unshadowed, components


def optimise_sun_for_view(gaussians, viewpoint_cam, pipeline, background,
                          normal_vectors, multiplier, emb_idx,
                          sun_elevation, eval_mask,
                          opt_iters, lr, lambda_dssim,
                          shadow_method, shadow_map_resolution, shadow_bias,
                          ray_march_steps, voxel_resolution,
                          verbose=False):
    """
    Optimise the sun direction delta for a single test view.

    Everything is frozen except a learnable 3D delta_sun_dir which is added
    to the camera's initial sun direction (same mechanism as sun_cal in
    training).

    Args:
        eval_mask: [H,W] binary tensor (1 = evaluate this pixel)

    Returns:
        best_sun_dir: Optimised sun direction [3] tensor
        best_delta: Best delta_sun_dir [3] tensor
        best_loss: Best loss value achieved
    """
    gt_image = viewpoint_cam.original_image.cuda()
    base_sun_dir = viewpoint_cam.get_adjusted_sun_direction()
    if base_sun_dir is None:
        raise ValueError(f"No sun direction for {viewpoint_cam.image_name}")

    # Learnable delta (same as Camera.delta_sun_dir)
    delta_sun_dir = torch.nn.Parameter(torch.zeros(3, device="cuda"))
    optimizer = torch.optim.Adam([delta_sun_dir], lr=lr)

    # Use the eval_mask expanded for the loss (matching the [3,H,W] image)
    # eval_mask is [H,W], we need [1,H,W] for broadcasting
    mask_3d = eval_mask.unsqueeze(0).float()  # [1,H,W]

    best_loss = float("inf")
    best_delta = torch.zeros(3, device="cuda")

    for i in range(opt_iters):
        optimizer.zero_grad()

        # Compute adjusted sun direction (differentiable)
        adjusted_sun_dir = F.normalize(base_sun_dir + delta_sun_dir, dim=0)

        # Unshadowed lighting (differentiable through sun direction)
        if gaussians.full_pbr:
            rgb_unshadowed, _, sun_dir_out, components = gaussians.compute_directional_pbr(
                emb_idx, normal_vectors, adjusted_sun_dir, viewpoint_cam.camera_center,
                sun_elevation=sun_elevation
            )
        else:
            rgb_unshadowed, _, sun_dir_out, components = gaussians.compute_directional_rgb(
                emb_idx, normal_vectors, adjusted_sun_dir, sun_elevation=sun_elevation,
                normal_multiplier=multiplier
            )

        # Shadow computation (non-differentiable geometry operation)
        effective_shadow_method = shadow_method
        if getattr(pipeline, "use_gaussians", False) and effective_shadow_method == "shadow_map":
            effective_shadow_method = "ray_march"

        with torch.no_grad():
            shadow_mask_raw, _, _ = compute_shadows_for_gaussians(
                gaussians,
                sun_dir_out.detach(),
                pipeline,
                method=effective_shadow_method,
                shadow_map_resolution=shadow_map_resolution,
                shadow_bias=shadow_bias,
                ray_march_steps=ray_march_steps,
                voxel_resolution=voxel_resolution,
                device="cuda",
                normal_vectors=normal_vectors,
                shadow_scale_modifier=1.5,
                shadow_dilation_kernel=5,
                alpha_threshold=0.01,
                normal_multiplier=multiplier,
            )
        shadow_mask = shadow_mask_raw.unsqueeze(-1)  # [N, 1]

        # Shadowed rendering (gradients flow through lighting, not shadows)
        if gaussians.full_pbr:
            rgb_shadowed, _, _, _ = gaussians.compute_directional_pbr(
                emb_idx, normal_vectors, adjusted_sun_dir, viewpoint_cam.camera_center,
                sun_elevation=sun_elevation, shadow_mask=shadow_mask
            )
        else:
            direct_light = components['direct']
            ambient_light = components['ambient']
            residual_light = components['residual']
            intensity_hdr = direct_light * shadow_mask + ambient_light + residual_light
            intensity_hdr = torch.clamp_min(intensity_hdr, 0.00001)
            intensity = intensity_hdr ** (1 / 2.2)
            albedo = gaussians.get_albedo
            rgb_shadowed = torch.clamp(intensity * albedo, 0.0)

        render_pkg = render(viewpoint_cam, gaussians, pipeline, background,
                            override_color=rgb_shadowed)
        image_shadowed = render_pkg["render"]

        # Masked photometric loss
        loss_l1 = (torch.abs(image_shadowed - gt_image) * mask_3d).sum() / (mask_3d.sum() * 3 + TINY_NUMBER)
        loss_ssim_val = 1.0 - ssim(image_shadowed, gt_image, mask=viewpoint_cam.mask)
        loss = (1.0 - lambda_dssim) * loss_l1 + lambda_dssim * loss_ssim_val

        # Small regularization to keep delta small (prefer directions close to initial)
        reg = 0.01 * delta_sun_dir.norm() ** 2
        total_loss = loss + reg

        total_loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta_sun_dir.detach().clone()

        if verbose and (i % 50 == 0 or i == opt_iters - 1):
            adjusted = F.normalize(base_sun_dir + delta_sun_dir.detach(), dim=0)
            print(f"    iter {i:4d}/{opt_iters}  loss={loss.item():.5f}  "
                  f"delta_norm={delta_sun_dir.detach().norm().item():.4f}  "
                  f"sun_dir=[{adjusted[0]:.3f}, {adjusted[1]:.3f}, {adjusted[2]:.3f}]")

    best_sun_dir = F.normalize(base_sun_dir + best_delta, dim=0)
    return best_sun_dir, best_delta, best_loss


def run(dataset, opt, pipe, iteration, test_config_path,
        opt_iters, lr, lambda_dssim, save_images, quiet):
    """Main entry point."""

    if not dataset.use_sun:
        print("ERROR: This script is for the sun-based path only (--use_sun).")
        sys.exit(1)

    # ---- Load trained scene ----
    gaussians = GaussianModel(
        dataset.sh_degree, dataset.with_mlp, dataset.mlp_W, dataset.mlp_D, dataset.N_a,
        use_sun=True, n_images=1700,
        use_residual_sh=dataset.use_residual_sh,
        full_pbr=dataset.full_pbr,
        scene_lighting_sh=dataset.scene_lighting_sh,
        sky_sh_degree=dataset.sky_sh_degree
    )
    scene = Scene(dataset, gaussians, load_iteration=iteration)

    n_images = len(scene.getTrainCameras())
    gaussians.n_images = n_images
    if gaussians.sun_model is None:
        raise RuntimeError("Sun model was not loaded. Check checkpoint files.")
    print(f"Loaded SunModel for {n_images} images at iteration {scene.loaded_iter}")

    # ---- Freeze EVERYTHING ----
    # Freeze gaussian params
    for attr in ['_xyz', '_features_dc_positive', '_features_rest_positive',
                 '_features_dc_negative', '_features_rest_negative',
                 '_albedo', '_roughness', '_metallic', '_scaling', '_rotation',
                 '_opacity', '_casts_shadow']:
        param = getattr(gaussians, attr, None)
        if param is not None and isinstance(param, torch.nn.Parameter):
            param.requires_grad_(False)

    # Freeze sun model
    gaussians.sun_model.eval()
    for p in gaussians.sun_model.parameters():
        p.requires_grad_(False)

    # Shadow parameters (consistent with test_gt_env_map_sun.py)
    shadow_method = "shadow_map"
    shadow_map_resolution = 512
    shadow_bias = 0.1
    ray_march_steps = 64
    voxel_resolution = 128

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ---- Load appearance LUT ----
    lut_path = os.path.join(dataset.model_path, "appearance_lut.json")
    if os.path.exists(lut_path):
        with open(lut_path) as f:
            appearance_lut = json.load(f)
    else:
        appearance_lut = {cam.image_name: i for i, cam in enumerate(scene.getTrainCameras())}
        print("Warning: appearance_lut.json not found, built from training camera order.")

    # ---- Read test config ----
    sys.path.append(test_config_path)
    config = importlib.import_module("test_config").config

    # ---- Select test cameras ----
    all_test_cameras = scene.getTestCameras()
    test_cameras = [c for c in all_test_cameras if c.image_name in config.keys()]

    if not test_cameras:
        print("No test cameras match the config keys.")
        print("Config keys:", list(config.keys()))
        print("Available test cams:", [c.image_name for c in all_test_cameras[:10]])
        return

    print(f"\nOptimising sun directions for {len(test_cameras)} test views")
    print(f"  opt_iters={opt_iters}, lr={lr}, lambda_dssim={lambda_dssim}")
    method_tag = "sun_pbr" if dataset.full_pbr else "sun"
    print(f"  method: {method_tag}")
    print()

    # ---- Output directories ----
    out_dir = os.path.join(dataset.model_path,
                           f"optimised_sun_eval_{method_tag}",
                           f"iteration_{scene.loaded_iter}")
    gt_path = os.path.join(out_dir, "gt_image")
    render_path = os.path.join(out_dir, "renders")
    makedirs(gt_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)

    # ---- Metric accumulators ----
    ssims_unshadowed, psnrs_unshadowed, mse_unshadowed, mae_unshadowed = [], [], [], []
    ssims_shadowed, psnrs_shadowed, mse_shadowed, mae_shadowed = [], [], [], []
    img_names = []
    optimised_sun_dirs = {}
    per_image_metrics = {}

    for viewpoint_cam in tqdm(test_cameras, desc="Optimising + evaluating"):
        image_config = config[viewpoint_cam.image_name]
        mask_path = rebase_config_path(image_config["mask_path"], test_config_path)

        # Appearance index
        if viewpoint_cam.image_name in appearance_lut:
            emb_idx = appearance_lut[viewpoint_cam.image_name]
        else:
            emb_idx = list(appearance_lut.values())[0]

        # Ground truth
        gt_image = viewpoint_cam.original_image.cuda()

        # Evaluation mask
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_np is None:
            print(f"  Warning: Could not read mask at {mask_path}, using full image")
            mask_np = np.ones((gt_image.shape[1], gt_image.shape[2]), dtype=np.uint8) * 255
        mask_np = cv2.resize(mask_np, (gt_image.shape[2], gt_image.shape[1]))
        kernel = np.ones((5, 5), np.uint8)
        mask_np = cv2.erode(mask_np, kernel, iterations=1)
        eval_mask = torch.from_numpy(mask_np // 255).cuda()

        # Normals
        quaternions = gaussians.get_rotation
        scales = gaussians.get_scaling
        normal_vectors, multiplier = compute_normal_world_space(
            quaternions, scales, viewpoint_cam.world_view_transform, gaussians.get_xyz
        )

        sun_elevation = viewpoint_cam.sun_elevation

        # ============================================================
        # Phase 1: Optimise sun direction
        # ============================================================
        print(f"\n  [{viewpoint_cam.image_name}] Optimising sun direction...")
        best_sun_dir, best_delta, best_loss = optimise_sun_for_view(
            gaussians, viewpoint_cam, pipeline, background,
            normal_vectors, multiplier, emb_idx,
            sun_elevation, eval_mask,
            opt_iters, lr, lambda_dssim,
            shadow_method, shadow_map_resolution, shadow_bias,
            ray_march_steps, voxel_resolution,
            verbose=not quiet
        )
        print(f"  [{viewpoint_cam.image_name}] Best loss: {best_loss:.5f}  "
              f"delta_norm: {best_delta.norm().item():.4f}  "
              f"sun_dir: [{best_sun_dir[0]:.3f}, {best_sun_dir[1]:.3f}, {best_sun_dir[2]:.3f}]")

        optimised_sun_dirs[viewpoint_cam.image_name] = {
            "optimised_sun_dir": best_sun_dir.cpu().tolist(),
            "delta_sun_dir": best_delta.cpu().tolist(),
            "delta_norm": best_delta.norm().item(),
            "opt_loss": best_loss,
        }

        # ============================================================
        # Phase 2: Evaluate with optimised sun direction
        # ============================================================
        with torch.no_grad():
            rendering_shadowed, rendering_unshadowed, _ = render_with_sun_direction(
                gaussians, viewpoint_cam, pipeline, background,
                normal_vectors, multiplier, emb_idx,
                best_sun_dir, sun_elevation,
                shadow_method, shadow_map_resolution, shadow_bias,
                ray_march_steps, voxel_resolution
            )

        # Save images
        torchvision.utils.save_image(gt_image,
                                     os.path.join(gt_path, viewpoint_cam.image_name))
        torchvision.utils.save_image(rendering_unshadowed * eval_mask,
                                     os.path.join(render_path, "unshadowed_" + viewpoint_cam.image_name))
        torchvision.utils.save_image(rendering_shadowed * eval_mask,
                                     os.path.join(render_path, "shadowed_" + viewpoint_cam.image_name))

        # Albedo
        rgb_precomp_alb = gaussians.get_albedo
        render_pkg_alb = render(viewpoint_cam, gaussians, pipeline, background,
                                override_color=rgb_precomp_alb)
        albedo_img = render_pkg_alb["render"]
        torchvision.utils.save_image(albedo_img * eval_mask,
                                     os.path.join(render_path, "albedo_" + viewpoint_cam.image_name))

        if save_images:
            # Save comparison grid: GT | shadowed | unshadowed | albedo
            grid = torch.cat([
                gt_image * eval_mask,
                rendering_shadowed * eval_mask,
                rendering_unshadowed * eval_mask,
                torch.clamp(albedo_img, 0, 1) * eval_mask,
            ], dim=2)
            torchvision.utils.save_image(grid, os.path.join(render_path,
                                         f"comparison_{viewpoint_cam.image_name}"))

        img_names.append(viewpoint_cam.image_name)

        # ---- Compute metrics (matching test_gt_env_map_sun.py) ----
        psnrs_unshadowed.append(mse2psnr(img2mse(rendering_unshadowed, gt_image, mask=eval_mask)))
        mae_unshadowed.append(img2mae(rendering_unshadowed, gt_image, mask=eval_mask))
        mse_unshadowed.append(img2mse(rendering_unshadowed, gt_image, mask=eval_mask))

        psnrs_shadowed.append(mse2psnr(img2mse(rendering_shadowed, gt_image, mask=eval_mask)))
        mae_shadowed.append(img2mae(rendering_shadowed, gt_image, mask=eval_mask))
        mse_shadowed.append(img2mse(rendering_shadowed, gt_image, mask=eval_mask))

        # SSIM (skimage, masked — matching test_gt_env_map_sun.py)
        unshadowed_np = rendering_unshadowed.cpu().numpy().transpose(1, 2, 0)
        shadowed_np = rendering_shadowed.cpu().numpy().transpose(1, 2, 0)
        gt_np = gt_image.cpu().numpy().transpose(1, 2, 0)

        _, full = ssim_skimage(unshadowed_np, gt_np, win_size=5,
                               channel_axis=2, full=True, data_range=1.0)
        mssim = (torch.tensor(full).cuda() * eval_mask.unsqueeze(-1)).sum() / (3 * eval_mask.sum())
        ssims_unshadowed.append(mssim)

        _, full = ssim_skimage(shadowed_np, gt_np, win_size=5,
                               channel_axis=2, full=True, data_range=1.0)
        mssim = (torch.tensor(full).cuda() * eval_mask.unsqueeze(-1)).sum() / (3 * eval_mask.sum())
        ssims_shadowed.append(mssim)

        per_image_metrics[viewpoint_cam.image_name] = {
            "psnr_shadowed": float(psnrs_shadowed[-1]),
            "psnr_unshadowed": float(psnrs_unshadowed[-1]),
            "mse_shadowed": float(mse_shadowed[-1]),
            "mse_unshadowed": float(mse_unshadowed[-1]),
            "mae_shadowed": float(mae_shadowed[-1]),
            "mae_unshadowed": float(mae_unshadowed[-1]),
            "ssim_shadowed": float(ssims_shadowed[-1]),
            "ssim_unshadowed": float(ssims_unshadowed[-1]),
            "optimised_sun_dir": best_sun_dir.cpu().tolist(),
            "delta_norm": best_delta.norm().item(),
        }

        print(f"  [{viewpoint_cam.image_name}] PSNR shadowed: {float(psnrs_shadowed[-1]):.2f}  "
              f"SSIM: {float(ssims_shadowed[-1]):.4f}")

    # ======================================================================
    # Summary
    # ======================================================================
    mean_psnr_shad = torch.tensor(psnrs_shadowed).mean()
    mean_mse_shad = torch.tensor(mse_shadowed).mean()
    mean_mae_shad = torch.tensor(mae_shadowed).mean()
    mean_ssim_shad = torch.tensor(ssims_shadowed).mean()

    mean_psnr_unshad = torch.tensor(psnrs_unshadowed).mean()
    mean_mse_unshad = torch.tensor(mse_unshadowed).mean()
    mean_mae_unshad = torch.tensor(mae_unshadowed).mean()
    mean_ssim_unshad = torch.tensor(ssims_unshadowed).mean()

    print(f"\n{'='*70}")
    print(f"Optimised Sun Directions — Test Set ({len(test_cameras)} views)")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'Shadowed':>12} {'Unshadowed':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    print(f"  {'PSNR (dB)':<20} {mean_psnr_shad:>12.4f} {mean_psnr_unshad:>12.4f}")
    print(f"  {'MSE':<20} {mean_mse_shad:>12.7f} {mean_mse_unshad:>12.7f}")
    print(f"  {'MAE':<20} {mean_mae_shad:>12.7f} {mean_mae_unshad:>12.7f}")
    print(f"  {'SSIM':<20} {mean_ssim_shad:>12.4f} {mean_ssim_unshad:>12.4f}")
    print(f"{'='*70}")

    # ---- Save metrics (matching test_gt_env_map_sun.py format) ----
    with open(os.path.join(render_path, "metrics.txt"), 'w') as f:
        f.write(f"  PSNR unshadowed: {mean_psnr_unshad:>12.7f}\n")
        f.write(f"  MSE unshadowed: {mean_mse_unshad:>12.7f}\n")
        f.write(f"  MAE unshadowed: {mean_mae_unshad:>12.7f}\n")
        f.write(f"  SSIM skimage unshadowed: {mean_ssim_unshad:>12.7f}\n")
        f.write(f"\n")
        f.write(f"  PSNR shadowed: {mean_psnr_shad:>12.7f}\n")
        f.write(f"  MSE shadowed: {mean_mse_shad:>12.7f}\n")
        f.write(f"  MAE shadowed: {mean_mae_shad:>12.7f}\n")
        f.write(f"  SSIM skimage shadowed: {mean_ssim_shad:>12.7f}\n")
        f.write(f"\n")
        f.write(f"  method: optimised_sun_direction ({'full_pbr' if dataset.full_pbr else 'sun_directional'})\n")
        f.write(f"  shadow_method: {shadow_method}\n")
        f.write(f"  opt_iters: {opt_iters}\n")
        f.write(f"  lr: {lr}\n")
        f.write(f"  image names: {img_names}\n")

    # Per-image metrics JSON
    with open(os.path.join(render_path, "per_image_metrics.json"), 'w') as f:
        json.dump(per_image_metrics, f, indent=2)

    # Optimised sun directions JSON
    with open(os.path.join(out_dir, "optimised_sun_directions.json"), 'w') as f:
        json.dump(optimised_sun_dirs, f, indent=2)

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Optimise sun directions for eval views and compute metrics")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int,
                        help="Checkpoint iteration to load (-1 = latest)")
    parser.add_argument("--test_config", default="", type=str,
                        help="Path to eval config directory containing test_config.py")
    parser.add_argument("--opt_iters", default=200, type=int,
                        help="Number of optimisation iterations per test view")
    parser.add_argument("--lr", default=0.05, type=float,
                        help="Learning rate for sun direction optimisation")
    parser.add_argument("--save_images", action="store_true",
                        help="Save comparison grid images")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Optimising sun directions for: " + args.model_path)

    safe_state(args.quiet)

    dataset = model.extract(args)
    # Force use_sun since this script is sun-only
    dataset.use_sun = True

    run(dataset,
        opt.extract(args),
        pipeline.extract(args),
        args.iteration,
        args.test_config,
        args.opt_iters,
        args.lr,
        opt.extract(args).lambda_dssim,
        getattr(args, 'save_images', False),
        args.quiet)
