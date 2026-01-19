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
import pandas as pd
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json
import torch
from utils.sh_vis_utils import shReconstructDiffuseMap
from utils.normal_utils import compute_normal_world_space
from utils.shadow_utils import compute_shadows_for_gaussians
import imageio.v2 as im
import numpy as np

def render_set(model_path, imgs_subset, iteration, views, train_cameras, gaussians, pipeline, background,
               appearance_lut, appearance_list = None, only_from_appearence_list = False, args=None):

    # Shadow method parameters with defaults
    shadow_method = getattr(args, 'shadow_method', 'shadow_map') if args else 'shadow_map'
    shadow_map_resolution = getattr(args, 'shadow_map_resolution', 1024) if args else 1024
    shadow_bias = getattr(args, 'shadow_bias', 0.1) if args else 0.1
    ray_march_steps = getattr(args, 'ray_march_steps', 64) if args else 64
    voxel_resolution = getattr(args, 'voxel_resolution', 128) if args else 128

    if only_from_appearence_list:
        assert appearance_list is not None, 'only_from_appearences_list requires appearance_list'

    render_path = os.path.join(model_path, imgs_subset, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    if appearance_list:
        appearance_df = pd.read_csv(appearance_list, delimiter=';', index_col=0)

    for render_idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if only_from_appearence_list and view.image_name not in appearance_df.index.tolist():
            continue
        quaternions = gaussians.get_rotation
        scales = gaussians.get_scaling
        normal_vectors, multiplier = compute_normal_world_space(quaternions, scales, view.world_view_transform, gaussians.get_xyz)

        rgb_precomp = gaussians.get_albedo
        render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp)
        albedo = render_pkg["render"]
        normals = render_pkg["rend_normal"]
        normals_view = render_pkg["view_normal"]
        combined_rendering = torch.cat((view.original_image, torch.clamp(albedo, 0.0 ,1.0)), 2)
        torchvision.utils.save_image(combined_rendering, os.path.join(render_path, view.image_name + "_albedo.png"))
        torchvision.utils.save_image(normals*0.5+0.5, os.path.join(render_path, view.image_name + "_normals_world.png"))

        #if other convention needed:
        # torchvision.utils.save_image(normals_view*0.5+0.5, os.path.join(render_path, view.image_name + "_normals_view.png"))
        # normals_qual = normals_view.clone()
        # #normals_qual *=-1
        # normals_qual[0]*=-1
        # torchvision.utils.save_image(normals_qual*0.5+0.5, os.path.join(render_path, view.image_name + "_normals_view_qualitative.png"))

        # Recreate imgs is possible only for train cameras
        if "train" in imgs_subset:

            # get sh env map or use directional lighting
            appearance_idx = appearance_lut[view.image_name]

            if gaussians.use_sun:
                # Get sun direction from camera
                sun_dir = view.sun_direction
                if sun_dir is None:
                    print(f"Warning: No sun direction for {view.image_name}, skipping")
                    continue

                # Directional sun lighting mode - no diffuse map, use explicit lighting
                # unshadowed version
                rgb_precomp_unshadowed, intensity, sun_dir, components = gaussians.compute_directional_rgb(appearance_idx, normal_vectors, sun_dir)
                render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp_unshadowed)
                rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                combined_rendering = torch.cat((view.original_image, rendering), 2)
                torchvision.utils.save_image(combined_rendering, os.path.join(render_path, view.image_name + "_recreate_appearace_unshadowed.png"))

                # shadowed version - compute shadows using selected method
                shadow_mask, shadow_depth_map, _ = compute_shadows_for_gaussians(
                    gaussians,
                    sun_dir,
                    pipeline,
                    method=shadow_method,
                    shadow_map_resolution=shadow_map_resolution,
                    shadow_bias=shadow_bias,
                    ray_march_steps=ray_march_steps,
                    voxel_resolution=voxel_resolution,
                    device="cuda"
                )
                shadow_mask = shadow_mask.unsqueeze(-1)  # [N, 1]

                # Apply shadow to direct lighting only
                direct_light = components['direct']
                ambient_light = components['ambient']
                residual_light = components['residual']

                intensity_hdr_shadowed = direct_light * shadow_mask + ambient_light + residual_light
                intensity_hdr_shadowed = torch.clamp_min(intensity_hdr_shadowed, 0.00001)
                intensity_shadowed = intensity_hdr_shadowed ** (1 / 2.2)

                albedo = gaussians.get_albedo
                rgb_precomp_shadowed = torch.clamp(intensity_shadowed * albedo, 0.0)

                render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp_shadowed)
                rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                combined_rendering = torch.cat((view.original_image, rendering), 2)
                torchvision.utils.save_image(combined_rendering, os.path.join(render_path, view.image_name + "_recreate_appearace_shadowed.png"))

                # Save shadow map visualization
                shadow_rgb = shadow_mask.expand(-1, 3)
                render_pkg_shadow = render(view, gaussians, pipeline, background, override_color=shadow_rgb)
                shadow_vis = torch.clamp(render_pkg_shadow["render"], 0.0, 1.0)
                torchvision.utils.save_image(shadow_vis, os.path.join(render_path, view.image_name + "_shadow_map.png"))
            else:
                env_sh = gaussians.compute_env_sh(appearance_idx)
                diffuse_map=shReconstructDiffuseMap(env_sh.T.cpu().detach().numpy())
                im.imwrite(os.path.join(render_path, view.image_name +'_diffuse_map.exr'), diffuse_map)

                #unshadowed version
                rgb_precomp_unshadowed, _ =gaussians.compute_gaussian_rgb(env_sh, shadowed=False, normal_vectors=normal_vectors)
                render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp_unshadowed)
                rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                combined_rendering = torch.cat((view.original_image, rendering), 2)
                torchvision.utils.save_image(combined_rendering, os.path.join(render_path, view.image_name + "_recreate_appearace_unshadowed.png"))

                #shadowed version
                rgb_precomp_shadowed, _ = gaussians.compute_gaussian_rgb(env_sh, multiplier=multiplier)
                render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp_shadowed)
                rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                combined_rendering = torch.cat((view.original_image, rendering), 2)
                torchvision.utils.save_image(combined_rendering, os.path.join(render_path, view.image_name + "_recreate_appearace_shadowed.png"))

        if appearance_list:

            appearance_img_names = appearance_df.loc[view.image_name].appearance_imgs.split(',')
            for app_name in appearance_img_names:

                #get appearance idx and image
                if app_name in appearance_lut:
                    appearance_idx = appearance_lut[app_name]
                else:
                    print(f"Key '{app_name}' not found in appearance_lut")
                    continue
                appearance_idx = appearance_lut[app_name]
                app_image = [c.original_image for c in train_cameras if c.image_name == app_name][0]

                # Get sun direction from source appearance camera
                app_camera = [c for c in train_cameras if c.image_name == app_name][0]
                sun_dir = app_camera.sun_direction
                if sun_dir is None:
                    print(f"Warning: No sun direction for {app_name}, skipping")
                    continue

                n_rows = view.original_image.shape[1]
                n_cols = int(app_image.shape[2]*albedo.shape[1]/app_image.shape[1])
                app_image = torch.nn.functional.interpolate(app_image.unsqueeze(0), (n_rows, n_cols))
                app_image = torch.clamp(app_image, min=0.0, max = 1.0)

                if gaussians.use_sun:
                    # Directional sun lighting mode
                    # unshadowed version
                    rgb_precomp_unshadowed, intensity, sun_dir, components = gaussians.compute_directional_rgb(appearance_idx, normal_vectors, sun_dir)
                    render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp_unshadowed)
                    rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    combined_rendering = torch.cat((view.original_image, app_image.squeeze(), rendering), 2)
                    torchvision.utils.save_image(combined_rendering, os.path.join(render_path, '{}_to_{}'.format(view.image_name, app_name) + "_unshadowed.png"))

                    # shadowed version - compute shadows using selected method
                    shadow_mask, _, _ = compute_shadows_for_gaussians(
                        gaussians,
                        sun_dir,
                        pipeline,
                        method=shadow_method,
                        shadow_map_resolution=shadow_map_resolution,
                        shadow_bias=shadow_bias,
                        ray_march_steps=ray_march_steps,
                        voxel_resolution=voxel_resolution,
                        device="cuda"
                    )
                    shadow_mask = shadow_mask.unsqueeze(-1)  # [N, 1]

                    direct_light = components['direct']
                    ambient_light = components['ambient']
                    residual_light = components['residual']

                    intensity_hdr_shadowed = direct_light * shadow_mask + ambient_light + residual_light
                    intensity_hdr_shadowed = torch.clamp_min(intensity_hdr_shadowed, 0.00001)
                    intensity_shadowed = intensity_hdr_shadowed ** (1 / 2.2)

                    albedo = gaussians.get_albedo
                    rgb_precomp_shadowed = torch.clamp(intensity_shadowed * albedo, 0.0)

                    render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp_shadowed)
                    rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    combined_rendering = torch.cat((view.original_image, app_image.squeeze(), rendering), 2)
                    torchvision.utils.save_image(combined_rendering, os.path.join(render_path, '{}_to_{}'.format(view.image_name, app_name) + "_shadowed.png"))
                else:
                    #get new sh env, save it
                    env_sh = gaussians.compute_env_sh(appearance_idx)
                    diffuse_map=np.clip(shReconstructDiffuseMap(env_sh.T.cpu().detach().numpy()), 0, None)**(1/ 2.2)
                    im.imwrite(os.path.join(render_path, '{}_to_{}'.format(view.image_name, app_name) + '_diffuse_map.exr'), diffuse_map)

                    # unshadowed version
                    rgb_precomp_unshadowed, _ =gaussians.compute_gaussian_rgb(env_sh, shadowed=False, normal_vectors=normal_vectors)
                    render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp_unshadowed)
                    rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    combined_rendering = torch.cat((view.original_image, app_image.squeeze(), rendering), 2)
                    torchvision.utils.save_image(combined_rendering, os.path.join(render_path, '{}_to_{}'.format(view.image_name, app_name) + "_unshadowed.png"))

                    # shadowed version
                    rgb_precomp_shadowed, _ = gaussians.compute_gaussian_rgb(env_sh, multiplier=multiplier)
                    render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp_shadowed)
                    rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    combined_rendering = torch.cat((view.original_image, app_image.squeeze(), rendering), 2)
                    torchvision.utils.save_image(combined_rendering, os.path.join(render_path, '{}_to_{}'.format(view.image_name, app_name) + "_shadowed.png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, appearance_list = None, only_from_appearance_list = False, args=None):
    with torch.no_grad():

        # Create GaussianModel - sun data is loaded via Scene/cameras
        if dataset.use_sun:
            # Temporary n_images - will be updated after Scene creation
            gaussians = GaussianModel(dataset.sh_degree, dataset.with_mlp, dataset.mlp_W, dataset.mlp_D, dataset.N_a,
                                       use_sun=dataset.use_sun, n_images=1700)
        else:
            gaussians = GaussianModel(dataset.sh_degree, dataset.with_mlp, dataset.mlp_W, dataset.mlp_D, dataset.N_a)

        scene = Scene(dataset, gaussians, load_iteration=iteration)

        # Update n_images and initialize sun_model with correct count
        if dataset.use_sun:
            n_images = len(scene.getTrainCameras())
            # Verify cameras have sun data
            missing_sun = [cam.image_name for cam in scene.getTrainCameras() if cam.sun_direction is None]
            if missing_sun:
                raise ValueError(f"Sun data missing for {len(missing_sun)} images. "
                               f"First few missing: {missing_sun[:5]}")
            gaussians.n_images = n_images
            gaussians.setup_sun_model()
            gaussians.sun_model.eval()
            print(f"Initialized SunModel for {n_images} images")
        elif gaussians.with_mlp:
            gaussians.mlp.eval()
            gaussians.embedding.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        with open(os.path.join(dataset.model_path, "appearance_lut.json")) as handle:
            appearance_lut = json.loads(handle.read())

        if not skip_train:
             render_set(dataset.model_path, "render_train", scene.loaded_iter, scene.getTrainCameras(), scene.getTrainCameras(), gaussians,
                        pipeline, background, appearance_lut, appearance_list=appearance_list, only_from_appearence_list=only_from_appearance_list, args=args)

        if not skip_test:
             render_set(dataset.model_path, "render_test", scene.loaded_iter, scene.getTestCameras(), scene.getTrainCameras(), gaussians,
                        pipeline, background, appearance_lut, appearance_list=appearance_list, only_from_appearence_list=only_from_appearance_list, args=args)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--appearance_list", type=str, default = '')
    parser.add_argument("--only_from_appearance_list", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                args.appearance_list, args.only_from_appearance_list, args=args)