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

import os
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.dataset_readers import load_sky_masks
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import json
import os
from utils.normal_utils import compute_normal_world_space
from utils.loss_utils import compute_sh_gauss_losses, compute_sh_env_loss
from utils.train_utils import (
    prepare_output_and_logger, training_report, update_lambdas,
    get_training_phase, create_loss_components, LossComponents, TrainingPhase,
    generate_final_metrics_report, export_metrics_to_csv
)
from utils.shadow_utils import compute_shadows_for_gaussians
from utils.adaptive_dens_utils import AdaptiveDensGrid



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    assert opt.warmup <= opt.start_shadowed

    first_iter = 0
    tb_writer, metrics_logger = prepare_output_and_logger(dataset)

    # Create GaussianModel - sun data is loaded automatically in Scene via dataset_readers
    # and stored in cameras. n_images is needed for use_sun mode.
    if dataset.use_sun:
        # We need to know the number of images upfront for the SunModel
        # This will be set properly after Scene is created
        gaussians = GaussianModel(dataset.sh_degree, dataset.with_mlp, dataset.mlp_W, dataset.mlp_D, dataset.N_a,
                                   use_sun=dataset.use_sun, n_images=1700, use_residual_sh=dataset.use_residual_sh,
                                   full_pbr=dataset.full_pbr)  # Temporary, will be reset
    else:
        gaussians = GaussianModel(dataset.sh_degree, dataset.with_mlp, dataset.mlp_W, dataset.mlp_D, dataset.N_a)

    scene = Scene(dataset, gaussians)

    # Now that Scene is created, update n_images with actual count and reinitialize sun_model
    if dataset.use_sun:
        n_images = len(scene.getTrainCameras())
        # Check that all cameras have sun_direction
        missing_sun = [cam.image_name for cam in scene.getTrainCameras() if cam.sun_direction is None]
        if missing_sun:
            raise ValueError(f"Sun data missing for {len(missing_sun)} images. "
                           f"Place sun_data.json in dataset folder or provide --sun_json_path. "
                           f"First few missing: {missing_sun[:5]}")

        # Reinitialize with correct n_images
        gaussians.n_images = n_images
        gaussians.setup_sun_model()
        print(f"Initialized SunModel for {n_images} training images")

    # Load sky masks if path is provided (for marking sky gaussians as non-shadow-casting)
    sky_masks = {}
    if dataset.sky_mask_path:
        image_names = [cam.image_name for cam in scene.getTrainCameras()]
        sky_masks = load_sky_masks(dataset.sky_mask_path, image_names)
        if sky_masks:
            print(f"Loaded {len(sky_masks)} sky masks for sky mask loss training")

    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        checkpoint_dir = os.path.dirname(checkpoint)
        gaussians.restore(model_params, opt)
        if gaussians.use_sun:
            gaussians.sun_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "chkpnt_sun" + str(first_iter) + ".pth")))
        elif gaussians.with_mlp:
            gaussians.mlp.load_state_dict(torch.load(os.path.join(checkpoint_dir, "chkpnt_mlp" + str(first_iter) + ".pth")))
            gaussians.embedding.load_state_dict(torch.load(os.path.join(checkpoint_dir, "chkpnt_embedding" + str(first_iter) + ".pth")))
        else:
            gaussians.env_params.load_state_dict(torch.load(os.path.join(checkpoint_dir, "chkpnt_env" + str(first_iter) + ".pth")))
        gaussians.optimizer_env.load_state_dict(torch.load(os.path.join(checkpoint_dir, "chkpnt_optimizer_env" + str(first_iter) + ".pth")))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    #prepare look up table for appearance (lightning)
    viewpoint_stack = scene.getTrainCameras().copy()
    appearance_lut = {}
    for i, s in enumerate(viewpoint_stack):
        appearance_lut[s.image_name] = i
    with open(os.path.join(scene.model_path, "appearance_lut.json"), "w") as outfile:
        json.dump(appearance_lut, outfile)


    viewpoint_stack = None

    # ---- Adaptive grid-based densification setup ----
    adaptive_grid = None
    if opt.use_adaptive_dens:
        adaptive_grid = AdaptiveDensGrid(
            gaussians, scene.cameras_extent,
            grid_resolution=opt.adaptive_dens_grid_res,
            fill_empty=getattr(opt, "adaptive_dens_fill_empty", True),
            zero_depth_max_pixels=getattr(opt, "adaptive_dens_zero_depth_max_pixels", 4096),
            zero_depth_samples=getattr(opt, "adaptive_dens_zero_depth_samples", 1),
            surface_samples=getattr(opt, "adaptive_dens_surface_samples", 1),
            surface_jitter=getattr(opt, "adaptive_dens_surface_jitter", 0.0),
            ema_decay=getattr(opt, "adaptive_dens_ema_decay", 0.8),
            use_highfreq=getattr(opt, "adaptive_dens_use_highfreq", True),
            highfreq_boost=getattr(opt, "adaptive_dens_hf_boost", 0.75),
            highfreq_quantile=getattr(opt, "adaptive_dens_hf_quantile", 0.6),
            hole_score_quantile=getattr(opt, "adaptive_dens_hole_score_quantile", 0.5),
        )
        print(f"[Adaptive Densification] Enabled  –  grid {opt.adaptive_dens_grid_res}³, "
              f"interval={opt.adaptive_dens_interval}, "
              f"iters [{opt.adaptive_dens_from_iter}, {opt.adaptive_dens_until_iter}]")

    print("\n[Saving Gaussians before trainig")
    scene.save(0)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image = viewpoint_cam.original_image
        mask = viewpoint_cam.mask

        # compute env sh for given camera; like NerfOSR - add small random noise for env map
        emb_idx = appearance_lut[viewpoint_cam.image_name]

        # For use_sun mode, we don't use SH environment representation
        if not gaussians.use_sun:
            sh_env = gaussians.compute_env_sh(emb_idx)
            sh_random_noise = torch.randn_like(sh_env)*0.025
        else:
            sh_env = None
            sh_random_noise = None

        #get normals in world space
        quaternions = gaussians.get_rotation
        scales = gaussians.get_scaling
        normal_vectors, multiplier = compute_normal_world_space(quaternions, scales, viewpoint_cam.world_view_transform, gaussians.get_xyz)

        # Update loss lambdas depending on shadowed/unshadowed mode
        lambdas = update_lambdas(iteration, opt, use_sun=gaussians.use_sun)
        shadowed_image_loss_lambda = lambdas["shadowed_image_loss_lambda"]
        unshadowed_image_loss_lambda = lambdas["unshadowed_image_loss_lambda"]
        consistency_loss_lambda = lambdas["consistency_loss_lambda"]
        sh_gauss_lambda = lambdas["sh_gauss_lambda"]
        shadow_loss_lambda = lambdas["shadow_loss_lambda"]
        env_loss_lambda = lambdas["env_loss_lambda"]
        lambda_normal = lambdas["lambda_normal"]
        lambda_dist = lambdas["lambda_dist"]

        if getattr(pipe, "use_gaussians", False):
            lambda_normal = 0.0
            lambda_dist = 0.0

        # photometric loss for unshadowed
        # NOTE: For use_sun mode, the unshadowed image should NOT be the primary loss target
        # during warmup, because GT images have real shadows. If we fit unshadowed to GT,
        # the model bakes shadow patterns into albedo. Instead, unshadowed serves as a
        # small regularizer to encourage clean, shadow-free albedo.
        if unshadowed_image_loss_lambda >0:
            if gaussians.use_sun:
                # Use explicit directional lighting with sun color prior
                # Get sun direction and elevation from camera
                sun_dir = viewpoint_cam.sun_direction
                sun_elev = viewpoint_cam.sun_elevation
                if sun_dir is None:
                    raise ValueError(f"Sun direction missing for camera {viewpoint_cam.image_name}")
                # Shadows are not applied here - just unshadowed lighting
                if gaussians.full_pbr:
                    rgb_precomp_unshadowed, _, sun_dir, _ = gaussians.compute_directional_pbr(
                        emb_idx, normal_vectors, sun_dir, viewpoint_cam.camera_center, sun_elevation=sun_elev
                    )
                else:
                    rgb_precomp_unshadowed, _, sun_dir, _ = gaussians.compute_directional_rgb(emb_idx, normal_vectors, sun_dir, sun_elevation=sun_elev)
            else:
                rgb_precomp_unshadowed, _ = gaussians.compute_gaussian_rgb(sh_env+sh_random_noise, shadowed=False, normal_vectors=normal_vectors)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=rgb_precomp_unshadowed)
            image_unshadowed = render_pkg["render"]
            viewspace_point_tensor_unshadowed, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            Ll1_unshadowed = l1_loss(image_unshadowed, gt_image, mask=mask)
            unshadowed_image_loss = (1.0 - opt.lambda_dssim) * Ll1_unshadowed + opt.lambda_dssim * (1.0 - ssim(image_unshadowed, gt_image, mask=mask))
            unshadowed_image_loss *= unshadowed_image_loss_lambda
        else:
            Ll1_unshadowed = torch.tensor(0.0, device="cuda", dtype=torch.float32)
            unshadowed_image_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)

        # photometric loss for shadowed
        if shadowed_image_loss_lambda >0:
            if gaussians.use_sun:
                # Get sun direction and elevation from camera
                sun_dir = viewpoint_cam.sun_direction
                sun_elev = viewpoint_cam.sun_elevation
                if sun_dir is None:
                    raise ValueError(f"Sun direction missing for camera {viewpoint_cam.image_name}")
                # Compute shadow mask using selected method
                shadow_method = dataset.shadow_method
                if getattr(pipe, "use_gaussians", False) and shadow_method == "shadow_map":
                    shadow_method = "ray_march"
                shadow_mask, _, _ = compute_shadows_for_gaussians(
                    gaussians,
                    sun_dir,
                    pipe,
                    method=shadow_method,
                    shadow_map_resolution=dataset.shadow_map_resolution,
                    shadow_bias=dataset.shadow_bias,
                    ray_march_steps=dataset.ray_march_steps,
                    voxel_resolution=dataset.voxel_resolution,
                    device="cuda"
                )
                shadow_mask = shadow_mask.unsqueeze(-1)  # [N, 1]
                if gaussians.full_pbr:
                    rgb_precomp_shadowed, _, sun_dir, _ = gaussians.compute_directional_pbr(
                        emb_idx, normal_vectors, sun_dir, viewpoint_cam.camera_center,
                        sun_elevation=sun_elev, shadow_mask=shadow_mask
                    )
                else:
                    # Use explicit directional lighting with sun color prior
                    _, _, sun_dir, components = gaussians.compute_directional_rgb(emb_idx, normal_vectors, sun_dir, sun_elevation=sun_elev)

                    # Apply shadow to direct lighting only (ambient and residual remain unaffected)
                    direct_light = components['direct']
                    ambient_light = components['ambient']
                    residual_light = components['residual']

                    # Shadowed intensity = direct * shadow + ambient + residual
                    intensity_hdr_shadowed = direct_light * shadow_mask + ambient_light + residual_light
                    intensity_hdr_shadowed = torch.clamp_min(intensity_hdr_shadowed, 0.00001)
                    intensity_shadowed = intensity_hdr_shadowed ** (1 / 2.2)  # gamma correction

                    albedo = gaussians.get_albedo
                    rgb_precomp_shadowed = torch.clamp(intensity_shadowed * albedo, 0.0)
            else:
                rgb_precomp_shadowed, _  = gaussians.compute_gaussian_rgb(sh_env+sh_random_noise, multiplier=multiplier)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=rgb_precomp_shadowed)
            image_shadowed = render_pkg["render"]
            viewspace_point_tensor_shadowed, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            Ll1_shadowed = l1_loss(image_shadowed, gt_image, mask=mask)
            shadowed_image_loss = (1.0 - opt.lambda_dssim) * Ll1_shadowed + opt.lambda_dssim * (1.0 - ssim(image_shadowed, gt_image, mask=mask))
            shadowed_image_loss *= shadowed_image_loss_lambda
        else:
            Ll1_shadowed = torch.tensor(0.0, device="cuda", dtype=torch.float32)
            shadowed_image_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)

        # physical losses for SH_gauss (not applicable for use_sun mode)
        if not gaussians.use_sun and (sh_gauss_lambda >0 or consistency_loss_lambda >0 or shadow_loss_lambda >0):
            shs_gauss = gaussians.get_features(multiplier).transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
            shs_gauss_stacked = torch.cat([shs_gauss], dim=0)
            normal_vectors_stacked = torch.cat([normal_vectors], dim=0).detach()
            sh_gauss_loss_raw, consistency_loss_raw, shadow_loss_raw = compute_sh_gauss_losses(shs_gauss_stacked, normal_vectors_stacked)

            sh_gauss_loss = sh_gauss_lambda*sh_gauss_loss_raw # eq. 12
            consistency_loss = consistency_loss_lambda*consistency_loss_raw #  eq.14
            shadow_loss = shadow_loss_lambda*shadow_loss_raw  # eq. 15
        else:
            sh_gauss_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)
            consistency_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)
            shadow_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)

        #Environment map loss for SH_env, eq. 13 (not applicable for use_sun mode)
        if not gaussians.use_sun and env_loss_lambda >0:
            sh_env_loss = env_loss_lambda*compute_sh_env_loss(sh_env)
        else:
            sh_env_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)

        # Sun model regularization: encourage direct light to dominate over ambient/sky
        sun_reg_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        if gaussians.use_sun:
            # Get current lighting parameters
            sun_int = gaussians.sun_model.get_sun_intensity(emb_idx)  # [3]
            ambient = gaussians.sun_model.get_ambient(emb_idx)  # [3]

            # 1. Regularize ambient to stay lower than sun intensity
            # Ambient should be ~10-30% of sun intensity for realistic outdoor scenes
            ambient_ratio = ambient.mean() / (sun_int.mean() + 1e-6)
            if ambient_ratio > 0.3:
                # Penalize if ambient is more than 30% of sun
                sun_reg_loss = sun_reg_loss + 0.1 * (ambient_ratio - 0.3) ** 2

            # 2. Regularize sun_color_correction to stay near 1.0
            color_corr = gaussians.sun_model.sun_color_correction[emb_idx]
            sun_reg_loss = sun_reg_loss + 0.01 * ((color_corr - 1.0) ** 2).mean()

            # 3. Regularize global sky SH to stay bounded
            if gaussians.sun_model.use_residual_sh:
                sky_sh = gaussians.sun_model.sky_sh  # Global sky SH [3, n_coeffs]
                # L2 regularization - sky should be subtle
                sun_reg_loss = sun_reg_loss + 0.01 * (sky_sh ** 2).mean()
                # Sky SH DC term should be lower than ambient
                sky_dc = sky_sh[:, 0].abs().mean() * 0.282095  # DC contribution
                if sky_dc > ambient.mean() * 0.5:
                    sun_reg_loss = sun_reg_loss + 0.05 * (sky_dc - ambient.mean() * 0.5) ** 2

        # Sky mask loss: train _casts_shadow to match sky mask via rendering
        sky_mask_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        if sky_masks and gaussians.use_sun and viewpoint_cam.image_name in sky_masks:
            sky_mask = sky_masks[viewpoint_cam.image_name]
            sky_mask_loss_raw, _ = gaussians.compute_sky_mask_loss(
                viewpoint_cam, sky_mask, render, pipe, background
            )
            sky_mask_loss = opt.sky_mask_loss_weight * sky_mask_loss_raw

        # 2DGS original regularization
        if lambda_normal>0 or lambda_dist>0:
            rend_dist = render_pkg["rend_dist"]*mask
            rend_normal  = render_pkg['rend_normal']*mask
            surf_normal = render_pkg['surf_normal']*mask
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()
        else:
            normal_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)
            dist_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)

        total_loss = unshadowed_image_loss + shadowed_image_loss + dist_loss + normal_loss + shadow_loss + sh_gauss_loss + sh_env_loss + consistency_loss + sun_reg_loss + sky_mask_loss

        # Only backward if we have a loss with gradients
        if torch.is_tensor(total_loss) and total_loss.requires_grad:
            total_loss.backward()
        else:
            # Skip backward if no gradients (happens during certain training phases)
            pass

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():

            if iteration % 10 == 0:
                # Create structured loss components
                loss_components = create_loss_components(
                    unshadowed_image_loss=unshadowed_image_loss,
                    shadowed_image_loss=shadowed_image_loss,
                    l1_unshadowed=Ll1_unshadowed,
                    l1_shadowed=Ll1_shadowed,
                    normal_loss=normal_loss,
                    dist_loss=dist_loss,
                    sh_gauss_loss=sh_gauss_loss,
                    sh_env_loss=sh_env_loss,
                    consistency_loss=consistency_loss,
                    shadow_loss=shadow_loss,
                    sun_reg_loss=sun_reg_loss,
                    sky_mask_loss=sky_mask_loss,
                    total_loss=total_loss
                )

                # Get training phase info
                phase = get_training_phase(iteration, opt)

                # Log to metrics logger with structured data
                if metrics_logger is not None:
                    metrics_logger.log_training_losses(iteration, loss_components, phase, lambdas)

                # Console output - more compact and informative
                phase_name = phase.get_phase_name()
                def _val(x):
                    return x.item() if torch.is_tensor(x) else float(x)
                loss_dict = {
                    "Phase": phase_name,
                    "Total": f"{_val(total_loss):.5f}",
                    "Unshadowed": f"{_val(unshadowed_image_loss):.5f}",
                    "Shadowed": f"{_val(shadowed_image_loss):.5f}",
                    "Normal": f"{_val(normal_loss):.5f}",
                    "Dist": f"{_val(dist_loss):.5f}",
                    "SHgauss": f"{_val(sh_gauss_loss):.5f}",
                    "SHenv": f"{_val(sh_env_loss):.5f}",
                    "Consist": f"{_val(consistency_loss):.5f}",
                    "Shadow": f"{_val(shadow_loss):.5f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                # Add sun-specific losses if in use_sun mode
                if gaussians.use_sun:
                    loss_dict["SunReg"] = f"{_val(sun_reg_loss):.5f}"
                    if sky_masks:
                        loss_dict["SkyMask"] = f"{_val(sky_mask_loss):.5f}"

                print(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            psnr_metric = training_report(tb_writer, iteration, Ll1_unshadowed, Ll1_shadowed, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),
                                          appearance_lut=appearance_lut, source_path=dataset.source_path, sky_masks=sky_masks,
                                          metrics_logger=metrics_logger)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter and (iteration<opt.warmup or iteration>opt.start_shadowed):
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # Accumulate gradient stats from whichever renders were active
                gradient_stats = None
                if unshadowed_image_loss_lambda > 0:
                    gradient_stats = viewspace_point_tensor_unshadowed.grad
                if shadowed_image_loss_lambda > 0:
                    if gradient_stats is not None:
                        gradient_stats = gradient_stats + viewspace_point_tensor_shadowed.grad
                    else:
                        gradient_stats = viewspace_point_tensor_shadowed.grad
                if gradient_stats is not None:
                    gaussians.add_densification_stats(gradient_stats, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # ---- Adaptive grid-based densification ----
            if (adaptive_grid is not None
                    and iteration >= opt.adaptive_dens_from_iter
                    and iteration <= opt.adaptive_dens_until_iter):
                # Accumulate per-pixel loss into the 3D voxel grid every iteration
                # Pick whichever rendered image + loss is active
                active_image = None
                if shadowed_image_loss_lambda > 0:
                    active_image = image_shadowed
                elif unshadowed_image_loss_lambda > 0:
                    active_image = image_unshadowed
                if active_image is not None:
                    per_pixel_loss = torch.abs(active_image - gt_image).mean(dim=0)  # [H, W]
                    surf_depth = render_pkg.get("surf_depth", None)
                    if surf_depth is not None:
                        dens_mask = mask
                        # Ignore sky for adaptive densification (sky is naturally high-error and dynamic).
                        if sky_masks and viewpoint_cam.image_name in sky_masks:
                            import torch.nn.functional as F
                            sky_non_sky_mask = sky_masks[viewpoint_cam.image_name]  # [H, W], 1=non-sky

                            # Resize sky mask to current view resolution if needed.
                            target_h, target_w = per_pixel_loss.shape
                            if sky_non_sky_mask.shape[0] != target_h or sky_non_sky_mask.shape[1] != target_w:
                                sky_non_sky_mask = F.interpolate(
                                    sky_non_sky_mask.unsqueeze(0).unsqueeze(0),
                                    size=(target_h, target_w),
                                    mode='nearest'
                                ).squeeze(0).squeeze(0)

                            if torch.is_tensor(dens_mask):
                                if dens_mask.dim() == 3:
                                    dens_mask = dens_mask * sky_non_sky_mask.unsqueeze(0)
                                else:
                                    dens_mask = dens_mask * sky_non_sky_mask
                            else:
                                dens_mask = sky_non_sky_mask

                        dens_debug = adaptive_grid.accumulate(
                            per_pixel_loss, surf_depth.squeeze(0),
                            viewpoint_cam, mask=dens_mask, reference_image=gt_image
                        )

                        if tb_writer and dens_debug is not None:
                            vis_interval = max(int(getattr(opt, "adaptive_dens_vis_interval", 100)), 1)
                            if iteration % vis_interval == 0:
                                err_map = dens_debug["error_map"]
                                score_map = dens_debug["score_map"]
                                hf_map = dens_debug["highfreq_map"]
                                selected_mask = dens_debug["selected_mask"].float()

                                err_map = err_map / (err_map.max().clamp_min(1e-6))
                                score_map = score_map / (score_map.max().clamp_min(1e-6))
                                hf_map = hf_map / (hf_map.max().clamp_min(1e-6))

                                selected_rgb = selected_mask.unsqueeze(0).expand(3, -1, -1)
                                overlay = torch.clamp(gt_image, 0.0, 1.0).clone()
                                overlay[0] = torch.clamp(overlay[0] + 0.8 * selected_mask, 0.0, 1.0)
                                overlay[1] = torch.clamp(overlay[1] * (1.0 - 0.6 * selected_mask), 0.0, 1.0)
                                overlay[2] = torch.clamp(overlay[2] * (1.0 - 0.6 * selected_mask), 0.0, 1.0)

                                tag = f"adaptive_dens/{viewpoint_cam.image_name}"
                                tb_writer.add_images(tag + "/01_error_map", err_map.unsqueeze(0).unsqueeze(0), global_step=iteration)
                                tb_writer.add_images(tag + "/02_highfreq_map", hf_map.unsqueeze(0).unsqueeze(0), global_step=iteration)
                                tb_writer.add_images(tag + "/03_score_map", score_map.unsqueeze(0).unsqueeze(0), global_step=iteration)
                                tb_writer.add_images(tag + "/04_selected_pixels", selected_rgb.unsqueeze(0), global_step=iteration)
                                tb_writer.add_images(tag + "/05_densify_overlay", overlay.unsqueeze(0), global_step=iteration)

                # Trigger adaptive densification at the configured interval
                if (iteration > opt.adaptive_dens_from_iter
                        and iteration % opt.adaptive_dens_interval == 0):
                    adaptive_grid.trigger_densification(
                        gaussians,
                        loss_thresh_quantile=opt.adaptive_dens_loss_thresh,
                        count_thresh=opt.adaptive_dens_count_thresh,
                        max_new_gaussians=opt.adaptive_dens_max_gaussians,
                    )

            # Sky mask loss is now applied during training loop (see sky_mask_loss computation above)
            # No need for periodic update_sky_gaussians calls

            # update lrs
            if iteration>opt.warmup and iteration<=opt.start_shadowed:
                if gaussians.use_sun:
                    # For use_sun mode: keep sun model learning rates active during this phase
                    # since we're still training with shadowed rendering
                    for param_group in gaussians.optimizer_env.param_groups:
                        if param_group["name"] == "sun_intensity":
                            param_group['lr'] = opt.env_lr * 1.0  # Keep learning
                        elif param_group["name"] == "sun_color_correction":
                            param_group['lr'] = opt.env_lr * 0.3
                        elif param_group["name"] == "ambient_color":
                            param_group['lr'] = opt.env_lr * 1.0
                        elif param_group["name"] == "sky_sh":
                            param_group['lr'] = opt.env_lr * 0.5
                    # Don't freeze rotation for sun mode - normals matter for N·L
                else:
                    for param_group in gaussians.optimizer_env.param_groups:
                        param_group['lr'] = 0.0
                    for param_group in gaussians.optimizer.param_groups:
                        if "rotation" in param_group["name"]:
                            param_group['lr'] = 0.0

            if iteration>opt.start_shadowed:
                for param_group in gaussians.optimizer_env.param_groups:
                    if param_group["name"] == "mlp":
                        param_group['lr'] = opt.mlp_lr/ opt.mlp_lr_final_ratio
                    if param_group["name"] == "embedding":
                        param_group['lr'] = opt.embedding_lr/ opt.embedding_lr_final_ratio
                    if param_group["name"] == "env_params_lr":
                        param_group['lr'] = opt.env_lr
                    # Sun model learning rates
                    if param_group["name"] == "sun_intensity":
                        param_group['lr'] = opt.env_lr * 2.0
                    if param_group["name"] == "sun_color_correction":
                        param_group['lr'] = opt.env_lr * 0.5
                    if param_group["name"] == "ambient_color":
                        param_group['lr'] = opt.env_lr * 2.0
                    if param_group["name"] == "sky_sh":
                        param_group['lr'] = opt.env_lr

                for param_group in gaussians.optimizer.param_groups:
                    if param_group["name"] == "rotation":
                        param_group['lr'] = opt.rotation_lr

            #optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer_env.step()
                gaussians.optimizer_env.zero_grad(set_to_none = True)


            # save checkpoints
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                scene.save(iteration)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if gaussians.use_sun:
                    torch.save(gaussians.sun_model.state_dict(), scene.model_path + "/chkpnt_sun" + str(iteration) + ".pth")
                    torch.save(gaussians.optimizer_env.state_dict(), scene.model_path + "/chkpnt_optimizer_env" + str(iteration) + ".pth")
                elif gaussians.with_mlp:
                    torch.save(gaussians.mlp.state_dict(), scene.model_path + "/chkpnt_mlp" + str(iteration) + ".pth")
                    torch.save(gaussians.embedding.state_dict(), scene.model_path + "/chkpnt_embedding" + str(iteration) + ".pth")
                    torch.save(gaussians.optimizer_env.state_dict(), scene.model_path + "/chkpnt_optimizer_env" + str(iteration) + ".pth")
                else:
                    torch.save(gaussians.env_params.state_dict(), scene.model_path + "/chkpnt_env" + str(iteration) + ".pth")
                    torch.save(gaussians.optimizer_env.state_dict(), scene.model_path + "/chkpnt_optimizer_env" + str(iteration) + ".pth")

    # Generate final metrics report
    if metrics_logger is not None:
        generate_final_metrics_report(metrics_logger)
        export_metrics_to_csv(metrics_logger)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    iter_list = [1, 1000, 5000, 7500, *list(range(10000, 1000001, 2500))] #[30000]
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=iter_list)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=iter_list)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=iter_list)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(False) #args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")