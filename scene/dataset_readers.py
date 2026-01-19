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
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from pathlib import Path

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array = None
    sun_direction: np.array = None  # Sun direction vector [3] for this image

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    sun_data: dict = None  # Dictionary mapping image names to sun data


def load_sun_data_from_path(path: str) -> Optional[dict]:
    """
    Load sun data from a JSON file in the dataset directory.

    Looks for files named: sun_data.json, sun_positions.json, or sun.json

    Args:
        path: Path to the dataset directory

    Returns:
        Dictionary with sun data, or None if not found
    """
    sun_file_names = ["sun_data.json", "sun_positions.json", "sun.json"]

    for filename in sun_file_names:
        sun_path = os.path.join(path, filename)
        if os.path.exists(sun_path):
            print(f"Loading sun data from: {sun_path}")
            with open(sun_path, 'r') as f:
                sun_data = json.load(f)
            print(f"Loaded sun data for {len(sun_data)} images")
            return sun_data

    return None


def load_sun_data(source_path: str = None, explicit_path: str = None) -> Optional[dict]:
    """
    Single entry point for loading sun data.

    This is the canonical way to load sun data. It first tries to load from the
    source_path directory (auto-discovery), then falls back to explicit_path.

    Args:
        source_path: Path to the dataset directory (will auto-discover sun_data.json, etc.)
        explicit_path: Explicit path to a sun data JSON file

    Returns:
        Dictionary with sun data, or None if not found

    Example:
        # Auto-load from dataset folder
        sun_data = load_sun_data(source_path="/path/to/dataset")

        # Load from explicit path
        sun_data = load_sun_data(explicit_path="/path/to/sun.json")

        # Try auto-load first, fall back to explicit
        sun_data = load_sun_data(source_path="/path/to/dataset",
                                  explicit_path="/path/to/fallback.json")
    """
    sun_data = None

    # First try auto-discovery from source path
    if source_path is not None:
        sun_data = load_sun_data_from_path(source_path)

    # Fall back to explicit path if not found
    if sun_data is None and explicit_path is not None:
        if os.path.exists(explicit_path):
            print(f"Loading sun data from explicit path: {explicit_path}")
            with open(explicit_path, 'r') as f:
                sun_data = json.load(f)
            print(f"Loaded sun data for {len(sun_data)} images")
        else:
            print(f"Warning: Sun data file not found: {explicit_path}")

    return sun_data


def flip_sun_direction(direction: np.ndarray) -> np.ndarray:
	"""
	Apply coordinate system transformations to sun direction vector.

	Args:
		direction: Sun direction vector as numpy array [3]

	Returns:
		Transformed sun direction vector
	"""
	arr = np.array(direction, dtype=np.float32)
	arr[0] = -arr[0]  # Flip x direction
	return arr


def get_sun_direction_for_image(sun_data: dict, image_name: str) -> Optional[np.ndarray]:
	"""
	Get the sun direction vector for a specific image.

	Args:
		sun_data: Dictionary with sun position data for all images
		image_name: Name of the image to get sun direction for

	Returns:
		Sun direction vector as numpy array [3], or None if not found
		Note: X component is flipped to match coordinate system
	"""
	# Try exact match first
	if image_name in sun_data:
		direction = sun_data[image_name].get("sun_direction_vector")
		if direction is not None:
			return flip_sun_direction(direction)

	# Try with different extensions
	base_name = image_name.rsplit('.', 1)[0] if '.' in image_name else image_name
	extensions = [".JPG", ".jpg", ".png", ".PNG", ".jpeg", ".JPEG"]

	for ext in extensions:
		key = base_name + ext
		if key in sun_data:
			direction = sun_data[key].get("sun_direction_vector")
			if direction is not None:
				return flip_sun_direction(direction)

	# Try to find a match by removing extension from keys
	for key in sun_data:
		key_base = key.rsplit('.', 1)[0] if '.' in key else key
		if key_base == base_name:
			direction = sun_data[key].get("sun_direction_vector")
			if direction is not None:
				return flip_sun_direction(direction)

	return None


def load_sky_masks(sky_mask_path: str, image_names: list = None) -> dict:
    """
    Load sky masks from a folder.

    Sky masks are expected to be images where black (0) = sky, white (255) = not sky.
    The mask filename should match the image filename (with potentially different extension).

    Args:
        sky_mask_path: Path to folder containing sky mask images
        image_names: Optional list of image names to load masks for

    Returns:
        Dictionary mapping image_name to sky mask tensor [H, W] where 0=sky, 1=not sky
    """
    import torch

    sky_masks = {}

    if not sky_mask_path or not os.path.exists(sky_mask_path):
        print(f"Sky mask path not found or not specified: {sky_mask_path}")
        return sky_masks

    print(f"Loading sky masks from: {sky_mask_path}")

    # Get list of mask files
    mask_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    mask_files = {}
    for f in os.listdir(sky_mask_path):
        if any(f.endswith(ext) for ext in mask_extensions):
            # Store by base name (without extension)
            base_name = f.rsplit('.', 1)[0]
            mask_files[base_name] = os.path.join(sky_mask_path, f)

    # Load masks for each image
    loaded_count = 0
    for image_name in (image_names or mask_files.keys()):
        base_name = image_name.rsplit('.', 1)[0] if '.' in image_name else image_name

        if base_name in mask_files:
            mask_path = mask_files[base_name]
            try:
                # Load mask as grayscale
                mask_img = Image.open(mask_path).convert('L')
                mask_np = np.array(mask_img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
                # Black (0) = sky, White (1) = not sky -> already correct!
                # Convert to tensor
                mask_tensor = torch.from_numpy(mask_np).cuda()
                sky_masks[image_name] = mask_tensor
                loaded_count += 1
            except Exception as e:
                print(f"Warning: Failed to load sky mask {mask_path}: {e}")

    print(f"Loaded {loaded_count} sky masks")
    return sky_masks


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, path, reading_dir, sun_data=None):
    images_folder=os.path.join(path, reading_dir)
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)

        precomputed_mask=True
        if precomputed_mask:
            mask_path = Path(images_folder.replace(reading_dir, "masks"))
            base_image_name = Path(image_name).stem
            mask_files = list(mask_path.glob(f"{base_image_name}.*"))
            if mask_files:
                mask = Image.open(mask_files[0]).convert('L')
            else:
                raise FileNotFoundError(f"No mask found for {base_image_name} in {mask_path}")
        else:
            mask = Image.fromarray(np.uint8(np.ones_like(image)*255)).convert('L')

        # Get sun direction for this image if available
        sun_direction = None
        if sun_data is not None:
            sun_direction = get_sun_direction_for_image(sun_data, image_name)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              mask=mask, sun_direction=sun_direction)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, eval_file, llffhold=20, sun_json_path=None):
    # load cameras
    print(path)
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # Load sun data - try auto-discovery first, then explicit path
    sun_data = load_sun_data(source_path=path, explicit_path=sun_json_path)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                            path=path, reading_dir=reading_dir, sun_data=sun_data)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    train_cam_infos = cam_infos

    if eval:
        if eval_file:
            import pandas as pd
            df = pd.read_csv(eval_file, sep=';', header=0, index_col=0)
            train_imgs = df[df['split']=='train'].index.tolist()
            test_imgs = df[df['split']=='test'].index.tolist()

            train_cam_infos = [c for c in cam_infos if c.image_name in train_imgs]
            test_cam_infos = [c for c in cam_infos if c.image_name in test_imgs]
        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           sun_data=sun_data)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    raise NotImplementedError("To be tested and added")
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
}