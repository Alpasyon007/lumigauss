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
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import json
import os
from utils.normal_utils import compute_normal_world_space
from utils.loss_utils import compute_sh_gauss_losses, compute_sh_env_loss
from utils.train_utils import prepare_output_and_logger, training_report, update_lambdas
from utils.sun_utils import load_sun_data



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    assert opt.warmup <= opt.start_shadowed

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Load sun data if use_sun is enabled
    sun_data = None
    if dataset.use_sun:
        if not dataset.sun_json_path:
            raise ValueError("--sun_json_path must be provided when --use_sun is enabled")
        print(f"Loading sun position data from: {dataset.sun_json_path}")
        sun_data = load_sun_data(dataset.sun_json_path)
        print(f"Loaded sun data for {len(sun_data)} images")

    # For sun model, we need image names upfront, so we create Scene first temporarily
    # to get image names, then create GaussianModel with sun data
    if dataset.use_sun:
        # Create a temporary scene to get image names
        from scene.dataset_readers import sceneLoadTypeCallbacks
        import glob
        split_file_pattern = os.path.join(dataset.source_path, "*split.csv")
        split_files = glob.glob(split_file_pattern)
        if split_files:
            eval_file = split_files[0]
        else:
            eval_file = None

        if os.path.exists(os.path.join(dataset.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](dataset.source_path, dataset.images, dataset.eval, eval_file)
        elif os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](dataset.source_path, dataset.white_background, dataset.eval, eval_file)
        else:
            raise ValueError("Could not recognize scene type!")

        # Get image names from train cameras
        image_names = [cam.image_name for cam in scene_info.train_cameras]
        print(f"Found {len(image_names)} training images for sun model")

        gaussians = GaussianModel(dataset.sh_degree, dataset.with_mlp, dataset.mlp_W, dataset.mlp_D, dataset.N_a,
                                   use_sun=dataset.use_sun, sun_data=sun_data, image_names=image_names)
    else:
        gaussians = GaussianModel(dataset.sh_degree, dataset.with_mlp, dataset.mlp_W, dataset.mlp_D, dataset.N_a)

    scene = Scene(dataset, gaussians)
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
        lambdas = update_lambdas(iteration, opt)
        shadowed_image_loss_lambda = lambdas["shadowed_image_loss_lambda"]
        unshadowed_image_loss_lambda = lambdas["unshadowed_image_loss_lambda"]
        consistency_loss_lambda = lambdas["consistency_loss_lambda"]
        sh_gauss_lambda = lambdas["sh_gauss_lambda"]
        shadow_loss_lambda = lambdas["shadow_loss_lambda"]
        env_loss_lambda = lambdas["env_loss_lambda"]
        lambda_normal = lambdas["lambda_normal"]
        lambda_dist = lambdas["lambda_dist"]

        # photometric loss for unshadowed
        if unshadowed_image_loss_lambda >0:
            if gaussians.use_sun:
                # Use explicit directional lighting (no SH)
                # Shadows are not applied here - just unshadowed lighting
                rgb_precomp_unshadowed, _, sun_dir, _ = gaussians.compute_directional_rgb(emb_idx, normal_vectors)
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
                # Use explicit directional lighting
                # Shadow will be applied externally via multiplier on the intensity
                rgb_precomp_unshadowed_for_shadow, intensity_unshadowed, sun_dir, components = gaussians.compute_directional_rgb(emb_idx, normal_vectors)

                # Apply shadow externally: modulate the direct lighting component by the shadow mask
                # The multiplier represents visibility (1=lit, 0=shadowed)
                if multiplier is not None:
                    # Convert multiplier to proper shape
                    if len(multiplier.shape) == 1:
                        shadow_mask = multiplier.unsqueeze(-1).float()  # [N, 1]
                    else:
                        shadow_mask = multiplier.float()

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
                    rgb_precomp_shadowed = rgb_precomp_unshadowed_for_shadow
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

        # Sun model regularization: encourage residual SH to stay small
        sun_reg_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        if gaussians.use_sun and gaussians.sun_model.use_residual_sh:
            residual = gaussians.sun_model.residual_sh[emb_idx]
            # L2 regularization on residual SH coefficients
            sun_reg_loss = 0.01 * (residual ** 2).mean()

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

        total_loss = unshadowed_image_loss + shadowed_image_loss + dist_loss + normal_loss + shadow_loss + sh_gauss_loss + sh_env_loss + consistency_loss + sun_reg_loss

        # Only backward if we have a loss with gradients
        # For use_sun mode during warmup->shadowed transition, SH losses are skipped and image losses may be 0
        if total_loss.requires_grad:
            total_loss.backward()
        else:
            # Skip backward if no gradients (happens during certain training phases with use_sun)
            pass

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():

            if iteration % 10 == 0:
                loss_dict = {
                    "UnshadowedImLoss": f"{unshadowed_image_loss:.{5}f}",
                    "ShadowedImLoss": f"{shadowed_image_loss:.{5}f}",
                    "DistortLoss": f"{dist_loss:.{5}f}",
                    "NormalLoss": f"{normal_loss:.{5}f}",
                    "SHgaussLoss": f"{sh_gauss_loss:.{5}f}",
                    "SHenvLoss": f"{sh_env_loss:.{5}f}",
                    "ConsistencyLoss": f"{consistency_loss:.{5}f}",
                    "ShadowLoss": f"{shadow_loss:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                print(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            psnr_metric = training_report(tb_writer, iteration, Ll1_unshadowed, Ll1_shadowed, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),
                                          appearance_lut=appearance_lut, source_path=dataset.source_path)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter and (iteration<opt.warmup or iteration>opt.start_shadowed):
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if iteration > opt.start_shadowed:
                    gradient_stats = viewspace_point_tensor_unshadowed.grad + viewspace_point_tensor_shadowed.grad
                else:
                    gradient_stats = viewspace_point_tensor_unshadowed.grad
                gaussians.add_densification_stats(gradient_stats, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # update lrs
            if iteration>opt.warmup and iteration<=opt.start_shadowed:
                for param_group in gaussians.optimizer_env.param_groups:
                    # For sun model, keep a small learning rate during SH_gauss tuning
                    # This helps maintain good sun priors while tuning Gaussian SH
                    if gaussians.use_sun and param_group["name"] in ["sun_intensity", "sky_zenith", "sky_horizon", "residual_sh"]:
                        param_group['lr'] = opt.env_lr * 0.1  # Reduced but not zero
                    else:
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
                    if param_group["name"] == "sky_zenith":
                        param_group['lr'] = opt.env_lr * 2.0
                    if param_group["name"] == "sky_horizon":
                        param_group['lr'] = opt.env_lr * 2.0
                    if param_group["name"] == "residual_sh":
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



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    iter_list = [1000, 5000, 7500, *list(range(10000, 1000001, 2500))] #[30000]
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