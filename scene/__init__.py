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
import glob
import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        # Store progressive resolution config for use in training loop
        self.progressive_resolution = getattr(args, 'progressive_resolution', False)
        self.progressive_switch_iter = getattr(args, 'progressive_switch_iter', 15000)
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.source_path = args.source_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.sun_data = None  # Will be populated from scene_info

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        split_file_pattern = os.path.join(args.source_path, "*split.csv")
        split_files = glob.glob(split_file_pattern)

        if split_files:
            args.eval_file = split_files[0]
            print(f"SPLIT FILE USED: {args.eval_file}")
        else:
            args.eval_file = None
            print("WARNING: No split file found. Please check if one exists in the directory or modify scene/init.py.")

        # Get sun_json_path if available
        sun_json_path = getattr(args, 'sun_json_path', None)
        use_sun = getattr(args, 'use_sun', False)
        eval_config_path = getattr(args, 'eval_config_path', None)

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.eval_file, sun_json_path=sun_json_path, use_sun=use_sun, eval_config_path=eval_config_path)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.eval_file)
        else:
            assert False, "Could not recognize scene type!"

        # Store sun data from scene_info (single source of truth)
        self.sun_data = scene_info.sun_data

        if shuffle:
            random.Random(567464).shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.Random(774452).shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras", len(scene_info.train_cameras))
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras", len(scene_info.test_cameras))
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            if self.gaussians.use_sun:
                # Setup sun_model with correct n_images before loading checkpoint
                n_images = len(scene_info.train_cameras)
                self.gaussians.n_images = n_images

                # Auto-detect checkpoint mode and sky SH degree from saved keys
                sun_ckpt = torch.load(self.model_path + "/chkpnt_sun" + str(self.loaded_iter) + ".pth", weights_only=True)
                ckpt_scene_sh = "intensity_sh" in sun_ckpt
                ckpt_sky_degree = self.gaussians.sky_sh_degree
                if "sky_sh" in sun_ckpt:
                    n_coeffs = sun_ckpt["sky_sh"].shape[1]
                    ckpt_sky_degree = int(n_coeffs ** 0.5) - 1

                if ckpt_scene_sh != self.gaussians.scene_lighting_sh:
                    mode = "scene-global SH" if ckpt_scene_sh else "per-image"
                    print(f"Note: checkpoint uses {mode} mode, reconfiguring SunModel to match")
                    self.gaussians.scene_lighting_sh = ckpt_scene_sh
                if ckpt_sky_degree != self.gaussians.sky_sh_degree:
                    print(f"Note: checkpoint sky_sh degree={ckpt_sky_degree}, reconfiguring (was {self.gaussians.sky_sh_degree})")
                    self.gaussians.sky_sh_degree = ckpt_sky_degree

                self.gaussians.setup_sun_model()
                self.gaussians.sun_model.load_state_dict(sun_ckpt)
            elif self.gaussians.with_mlp:
                self.gaussians.mlp.load_state_dict(torch.load(self.model_path + "/chkpnt_mlp" + str(self.loaded_iter) + ".pth", weights_only=True))
                self.gaussians.embedding.load_state_dict(torch.load(self.model_path + "/chkpnt_embedding" + str(self.loaded_iter) + ".pth", weights_only=True))
            else:
                self.gaussians.env_params.load_state_dict(torch.load(self.model_path + "/chkpnt_env" + str(self.loaded_iter) + ".pth", weights_only=True))

        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        # Camera calibration refinement state
        self.cam_cal_enabled = False
        self.optimizer_cam_cal = None

        # Sun direction calibration state
        self.sun_cal_enabled = False
        self.optimizer_sun_cal = None

    def setup_cam_cal(self, opt):
        """Enable learnable camera pose refinement for all training cameras.

        Creates a dedicated Adam optimizer for the rotation and translation
        deltas on every training camera.

        Args:
            opt: OptimizationParams with cam_cal_rot_lr and cam_cal_trans_lr.
        """
        rot_params = []
        trans_params = []
        for cam in self.getTrainCameras():
            cam.enable_cam_cal()
            rot_params.append(cam.delta_rot)
            trans_params.append(cam.delta_trans)

        self.optimizer_cam_cal = torch.optim.Adam([
            {"params": rot_params, "lr": opt.cam_cal_rot_lr, "name": "cam_cal_rot"},
            {"params": trans_params, "lr": opt.cam_cal_trans_lr, "name": "cam_cal_trans"},
        ], lr=0.0, eps=1e-15)
        self.cam_cal_enabled = True
        print(f"[Camera Calibration] Enabled for {len(rot_params)} training cameras "
              f"(rot_lr={opt.cam_cal_rot_lr}, trans_lr={opt.cam_cal_trans_lr})")

    def setup_sun_cal(self, opt):
        """Enable learnable sun direction refinement for all training cameras.

        Creates a dedicated Adam optimizer for the sun direction delta on every
        training camera that has a sun_direction.

        Args:
            opt: OptimizationParams with sun_cal_lr.
        """
        sun_params = []
        n_enabled = 0
        for cam in self.getTrainCameras():
            if cam.sun_direction is not None:
                cam.enable_sun_cal()
                sun_params.append(cam.delta_sun_dir)
                n_enabled += 1

        if not sun_params:
            print("[Sun Calibration] WARNING: No cameras have sun_direction – nothing to optimise.")
            return

        self.optimizer_sun_cal = torch.optim.Adam([
            {"params": sun_params, "lr": opt.sun_cal_lr, "name": "sun_cal"},
        ], lr=0.0, eps=1e-15)
        self.sun_cal_enabled = True
        print(f"[Sun Calibration] Enabled for {n_enabled} training cameras "
              f"(lr={opt.sun_cal_lr})")

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.gaussians.use_sun:
            torch.save(self.gaussians.sun_model.state_dict(), self.model_path + "/chkpnt_sun" + str(iteration) + ".pth")
        elif self.gaussians.with_mlp:
            torch.save(self.gaussians.mlp.state_dict(), self.model_path + "/chkpnt_mlp" + str(iteration) + ".pth")
            torch.save(self.gaussians.embedding.state_dict(), self.model_path + "/chkpnt_embedding" + str(iteration) + ".pth")
        else:
            torch.save(self.gaussians.env_params.state_dict(), self.model_path + "/chkpnt_env" + str(iteration) + ".pth")

        # Save camera calibration deltas
        if self.cam_cal_enabled:
            cam_cal_state = {}
            for cam in self.getTrainCameras():
                cam_cal_state[cam.image_name] = {
                    "delta_rot": cam.delta_rot.detach().cpu(),
                    "delta_trans": cam.delta_trans.detach().cpu(),
                }
            torch.save(cam_cal_state, os.path.join(self.model_path, f"chkpnt_cam_cal{iteration}.pth"))

        # Save sun direction calibration deltas
        if self.sun_cal_enabled:
            sun_cal_state = {}
            for cam in self.getTrainCameras():
                if cam._sun_cal_enabled:
                    sun_cal_state[cam.image_name] = {
                        "delta_sun_dir": cam.delta_sun_dir.detach().cpu(),
                    }
            torch.save(sun_cal_state, os.path.join(self.model_path, f"chkpnt_sun_cal{iteration}.pth"))

    def load_cam_cal(self, iteration):
        """Load camera calibration deltas from a checkpoint."""
        path = os.path.join(self.model_path, f"chkpnt_cam_cal{iteration}.pth")
        if not os.path.exists(path):
            print(f"[Camera Calibration] No checkpoint found at {path}")
            return
        cam_cal_state = torch.load(path, weights_only=True)
        for cam in self.getTrainCameras():
            if cam.image_name in cam_cal_state:
                entry = cam_cal_state[cam.image_name]
                cam.delta_rot.data.copy_(entry["delta_rot"].cuda())
                cam.delta_trans.data.copy_(entry["delta_trans"].cuda())
                cam.apply_cam_cal()
        print(f"[Camera Calibration] Loaded deltas for {len(cam_cal_state)} cameras from iter {iteration}")

    def load_sun_cal(self, iteration):
        """Load sun direction calibration deltas from a checkpoint."""
        path = os.path.join(self.model_path, f"chkpnt_sun_cal{iteration}.pth")
        if not os.path.exists(path):
            print(f"[Sun Calibration] No checkpoint found at {path}")
            return
        sun_cal_state = torch.load(path, weights_only=True)
        for cam in self.getTrainCameras():
            if cam.image_name in sun_cal_state:
                entry = sun_cal_state[cam.image_name]
                cam.delta_sun_dir.data.copy_(entry["delta_sun_dir"].cuda())
        print(f"[Sun Calibration] Loaded deltas for {len(sun_cal_state)} cameras from iter {iteration}")


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getSunData(self):
        """Get sun data loaded from the dataset. Returns None if no sun data available."""
        return self.sun_data