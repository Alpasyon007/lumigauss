"""
Optimise per-image environment SH for test views of an already-trained scene.

Loads a trained Gaussian model (non-sun path, with MLP or env_params), freezes
all Gaussian parameters and the MLP, then creates a fresh embedding for each
test view and optimises it so that the resulting env SH minimises the
photometric loss against the ground-truth image.

The optimised embeddings are saved alongside the model so that downstream
evaluation/rendering scripts can load them.

Usage (inside Docker):
    python optimise_env.py -m output/st_mlp --iteration 40000 \
        --opt_iters 300 --lr 0.01
"""

import os
import sys
import json
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from utils.normal_utils import compute_normal_world_space

LPIPS_AVAILABLE = False
try:
    from lpipsPyTorch import lpips as compute_lpips
    LPIPS_AVAILABLE = True
except ImportError:
    pass


@torch.no_grad()
def evaluate_view(gaussians, view, pipe, background, env_sh, multiplier, normal_vectors):
    """Render unshadowed and shadowed images for a view with the given env SH."""
    # Unshadowed
    rgb_unshadowed, _ = gaussians.compute_gaussian_rgb(
        env_sh, shadowed=False, normal_vectors=normal_vectors
    )
    pkg_unshadowed = render(view, gaussians, pipe, background, override_color=rgb_unshadowed)
    img_unshadowed = torch.clamp(pkg_unshadowed["render"], 0.0, 1.0)

    # Shadowed
    rgb_shadowed, _ = gaussians.compute_gaussian_rgb(env_sh, multiplier=multiplier)
    pkg_shadowed = render(view, gaussians, pipe, background, override_color=rgb_shadowed)
    img_shadowed = torch.clamp(pkg_shadowed["render"], 0.0, 1.0)

    # Albedo
    rgb_albedo = gaussians.get_albedo
    pkg_albedo = render(view, gaussians, pipe, background, override_color=rgb_albedo)
    img_albedo = torch.clamp(pkg_albedo["render"], 0.0, 1.0)

    return img_albedo, img_unshadowed, img_shadowed


def optimise_env_for_view(gaussians, view, pipe, background, opt_iters, lr,
                           lambda_dssim=0.2, verbose=False):
    """
    Optimise a fresh appearance embedding for a single test view.

    Returns the optimised env SH tensor and metrics dict.
    """
    gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
    mask = view.mask

    # Normals + multiplier for this view
    quaternions = gaussians.get_rotation
    scales = gaussians.get_scaling
    normal_vectors, multiplier = compute_normal_world_space(
        quaternions, scales, view.world_view_transform, gaussians.get_xyz
    )

    if gaussians.with_mlp:
        # Optimise a fresh embedding vector that feeds through the frozen MLP
        emb_dim = gaussians.N_a
        test_emb = torch.nn.Parameter(torch.randn(1, emb_dim, device="cuda") * 0.01)
        optimiser = torch.optim.Adam([test_emb], lr=lr)

        def get_env_sh():
            return gaussians.mlp(test_emb)
    else:
        # Optimise a fresh env SH parameter directly (same shape as env_params entries)
        # Initialise from the first training env param as a starting point
        first_key = next(iter(gaussians.env_params.keys()))
        init_sh = gaussians.env_params[first_key].detach().clone()
        test_sh = torch.nn.Parameter(init_sh)
        optimiser = torch.optim.Adam([test_sh], lr=lr)

        def get_env_sh():
            return test_sh

    best_loss = float("inf")
    best_env_sh = None

    for i in range(opt_iters):
        optimiser.zero_grad()
        env_sh = get_env_sh()

        # Shadowed render (primary loss target — GT has real shadows)
        rgb_shadowed, _ = gaussians.compute_gaussian_rgb(env_sh, multiplier=multiplier)
        pkg = render(view, gaussians, pipe, background, override_color=rgb_shadowed)
        image_shadowed = pkg["render"]

        loss_l1 = l1_loss(image_shadowed, gt_image, mask=mask)
        loss_ssim_val = 1.0 - ssim(image_shadowed, gt_image, mask=mask)
        loss = (1.0 - lambda_dssim) * loss_l1 + lambda_dssim * loss_ssim_val

        loss.backward()
        optimiser.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_env_sh = get_env_sh().detach().clone()

        if verbose and (i % 50 == 0 or i == opt_iters - 1):
            print(f"    iter {i:4d}/{opt_iters}  loss={loss.item():.5f}  l1={loss_l1.item():.5f}")

    # Final evaluation with best env SH
    with torch.no_grad():
        img_albedo, img_unshadowed, img_shadowed = evaluate_view(
            gaussians, view, pipe, background, best_env_sh, multiplier, normal_vectors
        )

    metrics = {}
    metrics["l1_albedo"] = l1_loss(img_albedo, gt_image).mean().item()
    metrics["l1_unshadowed"] = l1_loss(img_unshadowed, gt_image).mean().item()
    metrics["l1_shadowed"] = l1_loss(img_shadowed, gt_image).mean().item()
    metrics["psnr_albedo"] = psnr(img_albedo, gt_image).mean().item()
    metrics["psnr_unshadowed"] = psnr(img_unshadowed, gt_image).mean().item()
    metrics["psnr_shadowed"] = psnr(img_shadowed, gt_image).mean().item()
    metrics["ssim_albedo"] = ssim(img_albedo, gt_image).item()
    metrics["ssim_unshadowed"] = ssim(img_unshadowed, gt_image).item()
    metrics["ssim_shadowed"] = ssim(img_shadowed, gt_image).item()

    if LPIPS_AVAILABLE:
        metrics["lpips_albedo"] = compute_lpips(img_albedo.unsqueeze(0), gt_image.unsqueeze(0), net_type="alex").item()
        metrics["lpips_unshadowed"] = compute_lpips(img_unshadowed.unsqueeze(0), gt_image.unsqueeze(0), net_type="alex").item()
        metrics["lpips_shadowed"] = compute_lpips(img_shadowed.unsqueeze(0), gt_image.unsqueeze(0), net_type="alex").item()

    return best_env_sh, metrics, img_albedo, img_unshadowed, img_shadowed


def run(dataset, opt, pipe, iteration, opt_iters, lr, lambda_dssim, save_images, quiet):
    """Main entry point."""
    # ---- Load trained scene (non-sun path) ----
    if dataset.use_sun:
        print("ERROR: This script is for the non-sun (MLP/env_params) path only.")
        print("       The loaded model has use_sun=True. Exiting.")
        sys.exit(1)

    gaussians = GaussianModel(dataset.sh_degree, dataset.with_mlp,
                               dataset.mlp_W, dataset.mlp_D, dataset.N_a)
    scene = Scene(dataset, gaussians, load_iteration=iteration)

    if gaussians.with_mlp:
        gaussians.mlp.eval()
        gaussians.embedding.eval()
        print(f"Loaded MLP model at iteration {scene.loaded_iter}")
    else:
        print(f"Loaded env_params model at iteration {scene.loaded_iter}")

    # Freeze all gaussian parameters
    for attr in ['_xyz', '_features_dc_positive', '_features_rest_positive',
                 '_features_dc_negative', '_features_rest_negative',
                 '_albedo', '_roughness', '_metallic', '_scaling', '_rotation',
                 '_opacity', '_casts_shadow']:
        param = getattr(gaussians, attr, None)
        if param is not None and isinstance(param, torch.nn.Parameter):
            param.requires_grad_(False)
    if gaussians.with_mlp:
        for p in gaussians.mlp.parameters():
            p.requires_grad_(False)
        for p in gaussians.embedding.parameters():
            p.requires_grad_(False)
    else:
        for p in gaussians.env_params.parameters():
            p.requires_grad_(False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_cameras = scene.getTestCameras()
    if not test_cameras:
        print("No test cameras found. Nothing to optimise.")
        return

    print(f"\nOptimising environment for {len(test_cameras)} test views")
    print(f"  opt_iters={opt_iters}, lr={lr}, lambda_dssim={lambda_dssim}")
    print(f"  Model type: {'MLP + embedding' if gaussians.with_mlp else 'env_params'}")
    print()

    # Output directories
    out_dir = os.path.join(dataset.model_path, "optimised_env",
                           f"iteration_{scene.loaded_iter}")
    os.makedirs(out_dir, exist_ok=True)
    if save_images:
        img_dir = os.path.join(out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

    all_metrics = {}
    all_env_sh = {}

    for view in tqdm(test_cameras, desc="Optimising test views"):
        env_sh, metrics, img_albedo, img_unshadowed, img_shadowed = optimise_env_for_view(
            gaussians, view, pipe, background, opt_iters, lr,
            lambda_dssim=lambda_dssim, verbose=not quiet
        )

        all_metrics[view.image_name] = metrics
        all_env_sh[view.image_name] = env_sh.cpu()

        if save_images:
            gt = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            grid = torch.cat([gt, img_albedo, img_unshadowed, img_shadowed], dim=2)
            torchvision.utils.save_image(grid, os.path.join(img_dir, f"{view.image_name}_comparison.png"))

    # ---- Save results ----
    torch.save(all_env_sh, os.path.join(out_dir, "test_env_sh.pth"))

    # Compute and print mean metrics
    metric_keys = list(next(iter(all_metrics.values())).keys())
    means = {}
    for k in metric_keys:
        vals = [m[k] for m in all_metrics.values()]
        means[k] = float(np.mean(vals))

    print(f"\n{'='*70}")
    print(f"Optimised Environment — Test Set Means ({len(test_cameras)} views)")
    print(f"{'='*70}")
    print(f"  {'Metric':<12} {'Albedo':>10} {'Unshadowed':>12} {'Shadowed':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
    print(f"  {'PSNR (dB)':<12} {means['psnr_albedo']:>10.2f} {means['psnr_unshadowed']:>12.2f} {means['psnr_shadowed']:>12.2f}")
    print(f"  {'SSIM':<12} {means['ssim_albedo']:>10.4f} {means['ssim_unshadowed']:>12.4f} {means['ssim_shadowed']:>12.4f}")
    if LPIPS_AVAILABLE:
        print(f"  {'LPIPS':<12} {means['lpips_albedo']:>10.4f} {means['lpips_unshadowed']:>12.4f} {means['lpips_shadowed']:>12.4f}")
    print(f"  {'L1':<12} {means['l1_albedo']:>10.5f} {means['l1_unshadowed']:>12.5f} {means['l1_shadowed']:>12.5f}")
    print(f"{'='*70}")

    # Save metrics JSON
    output_json = {
        "iteration": scene.loaded_iter,
        "opt_iters": opt_iters,
        "lr": lr,
        "lambda_dssim": lambda_dssim,
        "num_views": len(test_cameras),
        "mean_metrics": means,
        "per_view_metrics": all_metrics,
    }
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Optimise environment SH for test views")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt_group = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Model iteration to load (-1 = latest)")
    parser.add_argument("--opt_iters", default=300, type=int,
                        help="Number of optimisation iterations per test view")
    parser.add_argument("--lr", default=0.01, type=float,
                        help="Learning rate for embedding/env optimisation")
    parser.add_argument("--save_images", action="store_true",
                        help="Save comparison images (GT | albedo | unshadowed | shadowed)")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print(f"Optimising environment for model: {args.model_path}")
    safe_state(args.quiet)

    opt = opt_group.extract(args)
    run(model.extract(args),
        opt,
        pipeline.extract(args),
        args.iteration,
        args.opt_iters,
        args.lr,
        opt.lambda_dssim,
        args.save_images,
        args.quiet)
