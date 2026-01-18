
import os
import numpy as np
import torch
from gaussian_renderer import render
import uuid
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import os
from utils.sh_vis_utils import shReconstructDiffuseMap
from utils.normal_utils import compute_normal_world_space
from scene import Scene

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def update_lambdas(iteration, opt):
    """
    Returns the loss lambda values based on the current iteration and shadowed/unshadowed mode.
    Returns:
        dict: A dictionary containing adjusted lambda values.
    """

    lambda_normal = opt.lambda_normal if iteration > opt.start_regularization else 0.0
    lambda_dist = opt.lambda_dist if iteration > opt.start_regularization else 0.0

    if iteration < opt.warmup:
        shadowed_image_loss_lambda = 0.0
        unshadowed_image_loss_lambda = 1.0
        consistency_loss_lambda = 0.0
        sh_gauss_lambda = 0.0
        shadow_loss_lambda = 0.0
        env_loss_lambda = opt.env_loss_lambda

    # For small number of iterations, tune only SH_gauss
    elif opt.warmup <= iteration <= opt.start_shadowed:
        shadowed_image_loss_lambda = 0.0
        unshadowed_image_loss_lambda = 0.0
        consistency_loss_lambda = opt.consistency_loss_lambda_init
        sh_gauss_lambda = opt.gauss_loss_lambda
        shadow_loss_lambda = 0.0
        env_loss_lambda = 0.0

        #Turn off 2DGS regularization while tuning SH_gauss
        lambda_normal = 0.0
        lambda_dist = 0.0

    elif iteration > opt.start_shadowed:
        shadowed_image_loss_lambda = 1.0
        unshadowed_image_loss_lambda = 0.001
        consistency_loss_lambda = opt.consistency_loss_lambda_init / opt.consistency_loss_lambda_final_ratio
        sh_gauss_lambda = opt.gauss_loss_lambda
        shadow_loss_lambda = opt.shadow_loss_lambda
        env_loss_lambda = opt.env_loss_lambda
    else:
        raise ValueError("Iteration doesn't fit into any defined conditions - verify logic.")



    return {
        "shadowed_image_loss_lambda": shadowed_image_loss_lambda,
        "unshadowed_image_loss_lambda": unshadowed_image_loss_lambda,
        "consistency_loss_lambda": consistency_loss_lambda,
        "sh_gauss_lambda": sh_gauss_lambda,
        "shadow_loss_lambda": shadow_loss_lambda,
        "env_loss_lambda": env_loss_lambda,
        "lambda_normal": lambda_normal,
        "lambda_dist": lambda_dist
    }


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path, flush_secs=5)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1_unshadowed, Ll1_shadowed, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, appearance_lut=None, source_path=None):
    if tb_writer:
        tb_writer.add_scalar('Ll1_unshadowed', Ll1_unshadowed, iteration)
        tb_writer.add_scalar('Ll1_shadowed', Ll1_shadowed, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    psnr_test = 10000000
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        test_cameras = scene.getTestCameras()
        validation_configs = ({'name': 'test', 'cameras' : test_cameras[:5] if len(test_cameras) > 5 else test_cameras},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    #render albedo
                    rgb_precomp = scene.gaussians.get_albedo
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                    image_albedo = torch.clamp(render_pkg["render"], 0.0, 1.0)

                    #get normals in world space
                    quaternions = scene.gaussians.get_rotation
                    scales = scene.gaussians.get_scaling
                    normal_vectors, multiplier = compute_normal_world_space(
                        quaternions, scales, viewpoint.world_view_transform, scene.gaussians.get_xyz)

                    # Initialize variables
                    image_shadowed = None
                    image_unshadowed = None
                    env_sh_learned = None

                    if scene.gaussians.use_sun:
                        # Directional sun lighting mode - no SH environment
                        if config["name"] == "train":
                            emb_idx = appearance_lut[viewpoint.image_name]
                        else:
                            # For test, use first training image's lighting
                            emb_idx = list(appearance_lut.values())[0] if appearance_lut else 0

                        # render shadowed with directional lighting
                        rgb_precomp, _ = scene.gaussians.compute_directional_rgb(emb_idx, normal_vectors, multiplier=multiplier, shadowed=True)
                        render_pkg_shadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                        image_shadowed = torch.clamp(render_pkg_shadowed["render"], 0.0, 1.0)

                        # render unshadowed with directional lighting
                        rgb_precomp, _ = scene.gaussians.compute_directional_rgb(emb_idx, normal_vectors, multiplier=None, shadowed=False)
                        render_pkg_unshadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                        image_unshadowed = torch.clamp(render_pkg_unshadowed["render"], 0.0, 1.0)

                        # No env_sh_learned visualization for directional mode
                        env_sh_learned = None

                    elif config["name"]=="train":
                        # get env sh from this view's appearance
                        emb_idx = appearance_lut[viewpoint.image_name]
                        env_sh = scene.gaussians.compute_env_sh(emb_idx)
                        # vis env map
                        env_sh_learned = np.clip(shReconstructDiffuseMap(env_sh.T.cpu().detach().numpy(), width=300), 0, None)
                        env_sh_learned = torch.clamp(torch.tensor(env_sh_learned**(1/ 2.2)).permute(2,0,1), 0.0, 1.0)

                        # render shadowed
                        rgb_precomp,_ = scene.gaussians.compute_gaussian_rgb(env_sh, multiplier=multiplier)
                        render_pkg_shadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                        image_shadowed = torch.clamp(render_pkg_shadowed["render"], 0.0, 1.0)

                        #render unshadowed
                        rgb_precomp,_ = scene.gaussians.compute_gaussian_rgb(env_sh, shadowed=False, normal_vectors=normal_vectors)
                        render_pkg_unshadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                        image_unshadowed = torch.clamp(render_pkg_unshadowed["render"], 0.0, 1.0)
                    else:
                        # For test images, use first training image's environment
                        if appearance_lut:
                            first_emb_idx = list(appearance_lut.values())[0]
                            env_sh = scene.gaussians.compute_env_sh(first_emb_idx)
                        else:
                            # Fallback to hardcoded environment if no appearance_lut
                            env_sh = torch.tensor(np.array(
                                    [[2.5, 2.389, 2.562],
                                    [0.545, 0.436, 0.373],
                                    [1.46, 1.724, 2.118],
                                    [0.771, 0.623, 0.53],
                                    [0.407, 0.355, 0.313],
                                    [0.667, 0.516, 0.42],
                                    [0.38, 0.314, 0.399],
                                    [0.817, 0.637, 0.517],
                                    [0.193, 0.151, 0.148]]),
                                    dtype=torch.float32, device=scene.gaussians._albedo.device).T

                        env_sh_learned = np.clip(shReconstructDiffuseMap(env_sh.T.cpu().detach().numpy(), width=300), 0, None)
                        env_sh_learned = torch.clamp(torch.tensor(env_sh_learned**(1/ 2.2)).permute(2,0,1), 0.0, 1.0)

                        # render shadowed
                        rgb_precomp,_ = scene.gaussians.compute_gaussian_rgb(env_sh, multiplier=multiplier)
                        render_pkg_shadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                        image_shadowed = torch.clamp(render_pkg_shadowed["render"], 0.0, 1.0)

                        #render unshadowed
                        rgb_precomp,_ = scene.gaussians.compute_gaussian_rgb(env_sh, shadowed=False, normal_vectors=normal_vectors)
                        render_pkg_unshadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                        image_unshadowed = torch.clamp(render_pkg_unshadowed["render"], 0.0, 1.0)

                    if config["name"]=="train" and not scene.gaussians.use_sun:
                        # Appearance transfer - visualize render with lightning from other training image
                        # (Not available for use_sun mode)

                        # BIG TODO remove hardcoded idx! For now, please do it yourself if needed
                        if "trevi" in source_path:
                            emb_idx = appearance_lut["00851798_4967549624.jpg"]
                        elif "lk2" in source_path:
                            emb_idx = appearance_lut["C3_DSC_8_3.png"]
                        elif "lwp" in source_path:
                            emb_idx = appearance_lut["23-04_10_00_DSC_1687.jpg"]
                        elif "st" in source_path:
                            emb_idx = appearance_lut["12-04_18_00_DSC_0483.jpg"]
                        else:
                            raise("Other datasets than trevi, lk2, lwp, st not implemented. Moreover, viewpoints are hardcoded. Please, for now, fix it by yourself.")

                        env_sh = scene.gaussians.compute_env_sh(emb_idx)
                        env_sh_transfer = shReconstructDiffuseMap(env_sh.T.cpu().detach().numpy(), width=300)
                        env_sh_transfer = (torch.clamp(torch.tensor(env_sh_transfer** (1 / 2.2)).permute(2,0,1), 0.0, 1.0))

                        #render shadowed
                        rgb_precomp,_ = scene.gaussians.compute_gaussian_rgb(env_sh, multiplier=multiplier)
                        render_pkg_shadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                        image_shadowed_transfer = torch.clamp(render_pkg_shadowed["render"], 0.0, 1.0)

                        #render unshadowed
                        rgb_precomp,_ = scene.gaussians.compute_gaussian_rgb(env_sh, shadowed=False, normal_vectors=normal_vectors)
                        render_pkg_unshadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                        image_unshadowed_transfer = torch.clamp(render_pkg_unshadowed["render"], 0.0, 1.0)


                        # Relightning - visualize render with lightning from external env map

                        env_sh=torch.tensor(np.array(
                                [[2.5, 2.389, 2.562],
                                [0.545, 0.436, 0.373],
                                [1.46, 1.724, 2.118],
                                [0.771, 0.623, 0.53],
                                [0.407, 0.355, 0.313],
                                [0.667, 0.516, 0.42],
                                [0.38, 0.314, 0.399],
                                [0.817, 0.637, 0.517],
                                [0.193, 0.151, 0.148]]),
                                dtype=torch.float32, device=scene.gaussians._albedo.device).T

                        env_sh_external = shReconstructDiffuseMap(env_sh.T.cpu().detach().numpy(), width=300)
                        env_sh_external = (torch.clamp(torch.tensor(env_sh_external** (1 / 2.2)).permute(2,0,1), 0.0, 1.0))

                        #render shadowed
                        rgb_precomp,_ = scene.gaussians.compute_gaussian_rgb(env_sh, multiplier=multiplier)
                        render_pkg_shadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                        image_shadowed_external = torch.clamp(render_pkg_shadowed["render"], 0.0, 1.0)

                        #render unshadowed
                        rgb_precomp,_ = scene.gaussians.compute_gaussian_rgb(env_sh, shadowed=False, normal_vectors=normal_vectors)
                        render_pkg_unshadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                        image_unshadowed_external = torch.clamp(render_pkg_unshadowed["render"], 0.0, 1.0)


                    if tb_writer:
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')

                        # Use view tag for proper grouping in TensorBoard
                        view_tag = "{}_view_{}".format(config['name'], viewpoint.image_name)

                        tb_writer.add_images(view_tag + "/01_ground_truth", gt_image[None], global_step=iteration)
                        tb_writer.add_images(view_tag + "/02_albedo", image_albedo[None], global_step=iteration)
                        tb_writer.add_images(view_tag + "/03_recreate_unshadowed", image_unshadowed[None], global_step=iteration)
                        tb_writer.add_images(view_tag + "/04_recreate_shadowed", image_shadowed[None], global_step=iteration)
                        if env_sh_learned is not None:
                            tb_writer.add_images(view_tag + "/05_env_recreate", env_sh_learned[None], global_step=iteration)
                        tb_writer.add_images(view_tag + "/06_depth", depth[None], global_step=iteration)

                        if config["name"]=="train" and not scene.gaussians.use_sun:
                            tb_writer.add_images(view_tag + "/07_transfer_unshadowed", image_unshadowed_transfer[None], global_step=iteration)
                            tb_writer.add_images(view_tag + "/08_transfer_shadowed", image_shadowed_transfer[None], global_step=iteration)
                            tb_writer.add_images(view_tag + "/09_env_transfer", env_sh_transfer[None], global_step=iteration)
                            tb_writer.add_images(view_tag + "/10_relight_unshadowed", image_unshadowed_external[None], global_step=iteration)
                            tb_writer.add_images(view_tag + "/11_relight_shadowed", image_shadowed_external[None], global_step=iteration)
                            tb_writer.add_images(view_tag + "/12_env_relight", env_sh_external[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            normals_precomp = (normal_vectors*0.5 + 0.5)
                            render_pkg = render(viewpoint, scene.gaussians,*renderArgs, override_color=normals_precomp)
                            rend_normal = render_pkg["render"]
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(view_tag + "/13_rend_normal", rend_normal[None], global_step=iteration)
                            tb_writer.add_images(view_tag + "/14_surf_normal", surf_normal[None], global_step=iteration)
                            tb_writer.add_images(view_tag + "/15_rend_alpha", rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(view_tag + "/16_rend_dist", rend_dist[None], global_step=iteration)
                        except:
                            pass

                    l1_test += l1_loss(image_albedo, gt_image).mean().double()
                    psnr_test += psnr(image_albedo, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()
    return psnr_test