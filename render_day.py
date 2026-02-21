#
# Render test views with sun moving through the day
# Creates a video/image sequence showing lighting changes throughout the day
# Includes 3D visualization of point cloud with sun position
#

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
import numpy as np
from utils.normal_utils import compute_normal_world_space
from utils.shadow_utils import compute_shadows_for_gaussians, SunShadowCamera, render_shadow_map
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont


def compute_sun_direction_from_angles(azimuth_deg: float, elevation_deg: float, north_offset_deg: float = 0,
                                      flip_x: bool = True, flip_y: bool = False, flip_z: bool = True) -> torch.Tensor:
    """
    Compute sun direction vector from azimuth and elevation angles.

    Args:
        azimuth_deg: Sun azimuth in degrees (0 = North, 90 = East, 180 = South, 270 = West)
        elevation_deg: Sun elevation in degrees (0 = horizon, 90 = zenith)
        north_offset_deg: North offset for coordinate system alignment (default from lk2 dataset)
        flip_x: Negate X component
        flip_y: Negate Y component
        flip_z: Negate Z component

    Returns:
        Sun direction vector [3] in Blender world coordinates
    """
    # Convert to radians
    azimuth_rad = np.radians(azimuth_deg + north_offset_deg)
    elevation_rad = np.radians(elevation_deg)

    # Compute direction (spherical to cartesian)
    # In Blender coordinate system: X=right, Y=forward, Z=up
    cos_elev = np.cos(elevation_rad)
    x = cos_elev * np.sin(azimuth_rad)
    y = -cos_elev * np.cos(azimuth_rad)
    z = np.sin(elevation_rad)

    if flip_x:
        x = -x
    if flip_y:
        y = -y
    if flip_z:
        z = -z

    direction = torch.tensor([x, y, z], dtype=torch.float32, device="cuda")
    return direction / (torch.norm(direction) + 1e-8)


def generate_day_sun_positions(latitude: float = 49.23, longitude: float = 7.0,
                                start_hour: float = 6.0, end_hour: float = 20.0,
                                num_frames: int = 60, day_of_year: int = 200) -> list:
    """
    Generate sun positions throughout the day using simplified solar position calculation.

    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        start_hour: Start time (e.g., 6.0 for 6 AM)
        end_hour: End time (e.g., 20.0 for 8 PM)
        num_frames: Number of frames to generate
        day_of_year: Day of year (1-365, affects sun path)

    Returns:
        List of (azimuth_deg, elevation_deg) tuples
    """
    positions = []

    # Simplified solar position calculation
    lat_rad = np.radians(latitude)

    # Solar declination (simplified)
    declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    dec_rad = np.radians(declination)

    hours = np.linspace(start_hour, end_hour, num_frames)

    for hour in hours:
        # Hour angle (15 degrees per hour from solar noon at 12:00)
        hour_angle = 15 * (hour - 12)
        ha_rad = np.radians(hour_angle)

        # Solar elevation
        sin_elev = np.sin(lat_rad) * np.sin(dec_rad) + \
                   np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
        elevation = np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))

        # Solar azimuth
        cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * sin_elev) / \
                 (np.cos(lat_rad) * np.cos(np.radians(elevation)) + 1e-8)
        azimuth = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))

        # Adjust azimuth based on hour
        if hour > 12:
            azimuth = 360 - azimuth

        positions.append((azimuth, elevation, hour))

    return positions


def create_3d_sun_visualization(
    gaussian_positions: torch.Tensor,
    sun_direction: torch.Tensor,
    camera_position: torch.Tensor,
    camera_forward: torch.Tensor = None,
    hour: float = 12.0,
    elevation: float = 60.0,
    azimuth: float = 180.0,
    all_sun_positions: list = None,
    north_offset: float = -134.0,
    max_points: int = 3000,
    figsize: tuple = (6, 6),
    flip_x: bool = True, flip_y: bool = False, flip_z: bool = True,
) -> np.ndarray:
    """
    Create a 3D visualization of the point cloud with sun position and sun path arc.
    Returns a numpy array image.

    Args:
        all_sun_positions: List of (azimuth, elevation, hour) tuples for the full day
        north_offset: North offset for coordinate system alignment
    """
    def swap_yz(arr):
        """Remap so -Y is up: (x, y, z) -> (x, z, -y)"""
        if arr.ndim == 1:
            return np.array([arr[0], arr[2], -arr[1]])
        else:
            return np.stack([arr[:, 0], arr[:, 2], -arr[:, 1]], axis=1)

    def angles_to_direction(az_deg, el_deg, n_offset, fx=True, fy=False, fz=True):
        """Convert azimuth/elevation to direction vector with configurable axis flips"""
        az_rad = np.radians(az_deg + n_offset)
        el_rad = np.radians(el_deg)
        cos_el = np.cos(el_rad)
        x = cos_el * np.sin(az_rad)
        y = -cos_el * np.cos(az_rad)
        z = np.sin(el_rad)
        if fx:
            x = -x
        if fy:
            y = -y
        if fz:
            z = -z
        d = np.array([x, y, z])
        return d / (np.linalg.norm(d) + 1e-8)

    # Convert to numpy
    if torch.is_tensor(gaussian_positions):
        positions = gaussian_positions.detach().cpu().numpy()
    else:
        positions = np.array(gaussian_positions)

    if torch.is_tensor(sun_direction):
        sun_dir = sun_direction.detach().cpu().numpy()
    else:
        sun_dir = np.array(sun_direction)

    if torch.is_tensor(camera_position):
        cam_pos = camera_position.detach().cpu().numpy()
    else:
        cam_pos = np.array(camera_position)

    if camera_forward is not None and torch.is_tensor(camera_forward):
        cam_forward = camera_forward.detach().cpu().numpy()
    else:
        cam_forward = np.array(camera_forward) if camera_forward is not None else None

    # Subsample points
    if len(positions) > max_points:
        indices = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[indices]

    # Swap Y and Z for visualization
    positions = swap_yz(positions)
    sun_dir = swap_yz(sun_dir)
    cam_pos = swap_yz(cam_pos)
    if cam_forward is not None:
        cam_forward = swap_yz(cam_forward)

    # Compute scene center and extent
    scene_center = positions.mean(axis=0)
    scene_extent = np.linalg.norm(positions - scene_center, axis=1).max()
    arrow_length = scene_extent * 0.6
    sun_arc_radius = arrow_length * 1.2

    # Normalize sun direction
    sun_dir_norm = sun_dir / (np.linalg.norm(sun_dir) + 1e-8)
    sun_pos = scene_center + sun_dir_norm * sun_arc_radius

    # Precompute sun path points (shared across views)
    sun_path_data = None
    if all_sun_positions is not None:
        sun_path_points = []
        sun_path_colors = []
        for i, (az, el, h) in enumerate(all_sun_positions):
            if el >= 0:
                dir_vec = angles_to_direction(az, el, north_offset, flip_x, flip_y, flip_z)
                dir_vec = swap_yz(dir_vec)
                pos = scene_center + dir_vec * sun_arc_radius
                sun_path_points.append(pos)
                t = i / max(len(all_sun_positions) - 1, 1)
                if t < 0.5:
                    c = (1.0, 0.5 + t, 0.0, 0.6)
                else:
                    c = (1.0, 1.0 - (t - 0.5), 0.0, 0.6)
                sun_path_colors.append(c)
        if len(sun_path_points) > 1:
            sun_path_data = (np.array(sun_path_points), sun_path_colors)

    # Multiple view angles: (elevation, azimuth, label)
    view_angles = [
        (25, -60, "3/4 View"),
        (90, -90, "Top Down"),
        (0, 0, "Front"),
        (0, -90, "Side"),
    ]

    # Precompute shared data
    z_norm = (positions[:, 2] - positions[:, 2].min()) / (positions[:, 2].max() - positions[:, 2].min() + 1e-8)
    pt_colors = plt.cm.coolwarm(z_norm)
    max_range = scene_extent * 0.9
    mid_x, mid_y, mid_z = scene_center[0], scene_center[1], scene_center[2]
    time_str = f"{int(hour):02d}:{int((hour % 1) * 60):02d}"

    cam_forward_norm = None
    if cam_forward is not None:
        cam_forward_norm = cam_forward / (np.linalg.norm(cam_forward) + 1e-8)

    view_images = []
    per_view_size = (figsize[0] // 2 + 1, figsize[1] // 2 + 1)

    for view_elev, view_azim, view_label in view_angles:
        fig = plt.figure(figsize=per_view_size, facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')

        # Plot points
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c=pt_colors, s=0.5, alpha=0.4)

        # Sun path arc
        if sun_path_data is not None:
            spp, spc = sun_path_data
            ax.plot(spp[:, 0], spp[:, 1], spp[:, 2],
                    color='orange', linewidth=2, alpha=0.7, linestyle='-')
            for i, (pt, col) in enumerate(zip(spp, spc)):
                if i % 5 == 0:
                    ax.scatter([pt[0]], [pt[1]], [pt[2]], c=[col[:3]], s=20, alpha=0.5)

        # Sun arrow + symbol
        ax.quiver(scene_center[0], scene_center[1], scene_center[2],
                  sun_dir_norm[0] * arrow_length,
                  sun_dir_norm[1] * arrow_length,
                  sun_dir_norm[2] * arrow_length,
                  color='orange', linewidth=3, arrow_length_ratio=0.15)
        ax.scatter([sun_pos[0]], [sun_pos[1]], [sun_pos[2]],
                   c='yellow', s=200, marker='o', edgecolors='orange', linewidths=2)

        # Camera
        ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
                   c='red', s=100, marker='^')
        if cam_forward_norm is not None:
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                      cam_forward_norm[0] * arrow_length * 0.4,
                      cam_forward_norm[1] * arrow_length * 0.4,
                      cam_forward_norm[2] * arrow_length * 0.4,
                      color='red', linewidth=2, arrow_length_ratio=0.2)

        ax.view_init(elev=view_elev, azim=view_azim)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X', color='white', fontsize=6)
        ax.set_ylabel('Z', color='white', fontsize=6)
        ax.set_zlabel('-Y (up)', color='white', fontsize=6)
        ax.tick_params(colors='white', labelsize=5)

        ax.set_title(f"{view_label}\n{time_str}  El:{elevation:.0f}° Az:{azimuth:.0f}°",
                     color='white', fontsize=8, pad=5)

        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')

        plt.tight_layout()
        fig.canvas.draw()
        view_buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        view_images.append(view_buf)

    # Tile into 2x2 grid
    # Resize all to same size
    target_h = min(v.shape[0] for v in view_images)
    target_w = min(v.shape[1] for v in view_images)
    resized = []
    for v in view_images:
        if v.shape[0] != target_h or v.shape[1] != target_w:
            resized.append(np.array(Image.fromarray(v).resize((target_w, target_h), Image.LANCZOS)))
        else:
            resized.append(v)

    top_row = np.concatenate([resized[0], resized[1]], axis=1)
    bot_row = np.concatenate([resized[2], resized[3]], axis=1)
    buf = np.concatenate([top_row, bot_row], axis=0)

    return buf


def combine_render_and_visualization(render_panel: np.ndarray, vis_3d: np.ndarray, time_str: str) -> np.ndarray:
    """
    Combine the rendered image and 3D visualization side by side.
    """
    render_h, render_w = render_panel.shape[:2]
    vis_h, vis_w = vis_3d.shape[:2]

    # Resize visualization to match render height
    scale = render_h / vis_h
    new_vis_w = int(vis_w * scale)
    vis_3d_resized = np.array(Image.fromarray(vis_3d).resize((new_vis_w, render_h), Image.LANCZOS))

    # Combine side by side
    combined = np.concatenate([render_panel, vis_3d_resized], axis=1)

    # Add time label at top
    combined_pil = Image.fromarray(combined)
    draw = ImageDraw.Draw(combined_pil)

    # Try to use a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()

    # Draw time text with shadow for visibility
    text = f"Time: {time_str}"
    draw.text((12, 12), text, fill=(0, 0, 0), font=font)
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)

    return np.array(combined_pil)


def compute_shadow_mask_fixed(
    gaussian_positions: torch.Tensor,
    sun_camera,
    shadow_depth_map: torch.Tensor,
    shadow_alpha_map: torch.Tensor,
    bias: float = 0.02,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Fixed shadow mask computation with zeros padding and tighter bias.
    """
    N = gaussian_positions.shape[0]

    ones = torch.ones(N, 1, device=device, dtype=gaussian_positions.dtype)
    positions_homo = torch.cat([gaussian_positions, ones], dim=1)

    clip_coords = positions_homo @ sun_camera.full_proj_transform
    ndc_coords = clip_coords[:, :3] / (clip_coords[:, 3:4] + 1e-8)

    uv = (ndc_coords[:, :2] + 1.0) * 0.5
    gaussian_depth = ndc_coords[:, 2]

    grid = (uv * 2.0 - 1.0).view(1, 1, N, 2)

    # Use 'zeros' padding instead of 'border' to avoid edge depth leaking
    sampled_depth = torch.nn.functional.grid_sample(
        shadow_depth_map.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    ).view(N)

    sampled_alpha = torch.nn.functional.grid_sample(
        shadow_alpha_map.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    ).view(N)

    in_shadow = (gaussian_depth > sampled_depth + bias) & (sampled_alpha > 0.1)
    shadow_mask = (~in_shadow).float()

    # Out-of-bounds → lit (unchanged logic but now zeros padding handles edges better)
    out_of_bounds = (uv[:, 0] < 0) | (uv[:, 0] > 1) | (uv[:, 1] < 0) | (uv[:, 1] > 1)
    shadow_mask[out_of_bounds] = 1.0

    return shadow_mask


def compute_shadows_fixed(
    gaussians,
    sun_direction: torch.Tensor,
    pipe,
    shadow_map_resolution: int = 512,
    shadow_bias: float = 0.02,
    hard_threshold: bool = True,
    ortho_scale: float = 2.0,
    device: str = "cuda"
) -> tuple:
    """
    Fixed shadow map computation with:
    - Hard thresholded casts_shadow
    - Reduced bias
    - Larger ortho frustum
    - Zeros padding in grid_sample
    """
    positions = gaussians.get_xyz
    scene_center = positions.mean(dim=0)
    scene_extent = (positions - scene_center).norm(dim=1).max().item()

    # Create sun camera with larger frustum
    sun_camera = SunShadowCamera(
        sun_direction=sun_direction,
        scene_center=scene_center,
        scene_extent=scene_extent,
        resolution=shadow_map_resolution,
        device=device
    )
    # Override ortho_size for larger frustum coverage
    sun_camera.ortho_size = scene_extent * ortho_scale

    # Hard-threshold casts_shadow before rendering shadow map
    original_casts_shadow = None
    if hard_threshold:
        original_casts_shadow = gaussians._casts_shadow.data.clone()
        hard_mask = (gaussians.get_casts_shadow > 0.5).float()
        gaussians._casts_shadow.data = hard_mask

    shadow_depth_map, shadow_alpha_map = render_shadow_map(
        sun_camera, gaussians, pipe, device
    )

    # Restore original casts_shadow
    if original_casts_shadow is not None:
        gaussians._casts_shadow.data = original_casts_shadow

    shadow_mask = compute_shadow_mask_fixed(
        positions,
        sun_camera,
        shadow_depth_map,
        shadow_alpha_map,
        bias=shadow_bias,
        device=device
    )

    return shadow_mask, shadow_depth_map, sun_camera


def render_day_sequence(model_path, iteration, views, train_cameras, gaussians, pipeline, background,
                        appearance_lut, args, output_suffix="", selected_view_idx=0):
    """
    Render a sequence of frames with sun moving through the day.
    """
    # Shadow method parameters
    shadow_method = getattr(args, 'shadow_method', 'shadow_map')
    shadow_map_resolution = getattr(args, 'shadow_map_resolution', 1024)
    shadow_bias = getattr(args, 'shadow_bias', 0.1)
    ray_march_steps = getattr(args, 'ray_march_steps', 64)
    voxel_resolution = getattr(args, 'voxel_resolution', 128)

    # Shadow fix parameters
    use_shadow_fixes = getattr(args, 'shadow_fixes', False)
    fix_bias = getattr(args, 'shadow_fix_bias', 0.02)
    fix_hard_threshold = getattr(args, 'shadow_fix_hard_threshold', True)
    fix_ortho_scale = getattr(args, 'shadow_fix_ortho_scale', 2.0)
    fix_flip_normals = getattr(args, 'shadow_fix_flip_normals', False)
    show_debug_panels = getattr(args, 'shadow_debug_panels', False)

    # Axis flip parameters
    flip_x = getattr(args, 'flip_x', True)
    flip_y = getattr(args, 'flip_y', False)
    flip_z = getattr(args, 'flip_z', True)

    print(f"Sun direction axis flips: X={flip_x} Y={flip_y} Z={flip_z}")

    # Day rendering parameters
    num_frames = getattr(args, 'num_frames', 60)
    start_hour = getattr(args, 'start_hour', 6.0)
    end_hour = getattr(args, 'end_hour', 20.0)
    latitude = getattr(args, 'latitude', 49.23)  # Default: lk2 dataset location
    day_of_year = getattr(args, 'day_of_year', 200)  # Mid-July
    north_offset = getattr(args, 'north_offset', -134.0)  # lk2 dataset north offset

    # Visualization options
    show_3d_vis = getattr(args, 'show_3d', True)

    render_path = os.path.join(model_path, f"render_day{output_suffix}", f"ours_{iteration}")
    makedirs(render_path, exist_ok=True)

    # Generate sun positions for the day
    sun_positions = generate_day_sun_positions(
        latitude=latitude,
        start_hour=start_hour,
        end_hour=end_hour,
        num_frames=num_frames,
        day_of_year=day_of_year
    )

    # Select which views to render
    if selected_view_idx >= 0:
        views_to_render = [views[min(selected_view_idx, len(views)-1)]]
    else:
        views_to_render = views

    # Use first train camera's appearance as reference
    reference_cam = train_cameras[0]
    appearance_idx = appearance_lut.get(reference_cam.image_name, 0)

    # Get gaussian positions for 3D visualization (subsample for speed)
    gaussian_positions = gaussians.get_xyz.detach()

    for view_idx, view in enumerate(views_to_render):
        view_render_path = os.path.join(render_path, view.image_name)
        makedirs(view_render_path, exist_ok=True)

        # Get camera position for visualization
        W2C = view.world_view_transform.T.cpu().numpy()
        C2W = np.linalg.inv(W2C)
        camera_position = torch.tensor(C2W[:3, 3], dtype=torch.float32)
        camera_forward = torch.tensor(-C2W[:3, 2], dtype=torch.float32)  # -Z is forward

        # Compute normals for this view
        quaternions = gaussians.get_rotation
        scales = gaussians.get_scaling
        normal_vectors, multiplier = compute_normal_world_space(
            quaternions, scales, view.world_view_transform, gaussians.get_xyz
        )

        frames_combined = []
        frames_shadowed_direct = []

        print(f"\nRendering view: {view.image_name}")

        for frame_idx, (azimuth, elevation, hour) in enumerate(tqdm(sun_positions, desc="Sun positions")):
            # Skip if sun is below horizon
            if elevation < 0:
                continue

            # Compute sun direction for this time
            sun_dir = compute_sun_direction_from_angles(azimuth, elevation, north_offset,
                                                         flip_x=flip_x, flip_y=flip_y, flip_z=flip_z)

            # Optionally flip normals that face away from the sun (before lighting)
            frame_normals = normal_vectors
            if use_shadow_fixes and fix_flip_normals:
                sun_dir_norm = sun_dir / (torch.norm(sun_dir) + 1e-8)
                n_dot_l_check = torch.sum(normal_vectors * sun_dir_norm.unsqueeze(0), dim=-1)  # [N]
                flip_mask = (n_dot_l_check < 0).unsqueeze(-1)  # [N, 1]
                frame_normals = torch.where(flip_mask, -normal_vectors, normal_vectors)

            # Compute lighting
            if gaussians.full_pbr:
                rgb_precomp_unshadowed, intensity, sun_dir_out, components = gaussians.compute_directional_pbr(
                    appearance_idx, frame_normals, sun_dir, view.camera_center, sun_elevation=elevation
                )
            else:
                rgb_precomp_unshadowed, intensity, sun_dir_out, components = gaussians.compute_directional_rgb(
                    appearance_idx, frame_normals, sun_dir, sun_elevation=elevation
                )

            # Compute shadows
            if use_shadow_fixes and shadow_method == "shadow_map":
                # Use fixed shadow computation with all patches
                shadow_mask, _, _ = compute_shadows_fixed(
                    gaussians,
                    sun_dir,
                    pipeline,
                    shadow_map_resolution=shadow_map_resolution,
                    shadow_bias=fix_bias,
                    hard_threshold=fix_hard_threshold,
                    ortho_scale=fix_ortho_scale,
                    device="cuda"
                )
            else:
                effective_shadow_method = shadow_method
                if getattr(pipeline, "use_gaussians", False) and effective_shadow_method == "shadow_map":
                    effective_shadow_method = "ray_march"
                shadow_mask, _, _ = compute_shadows_for_gaussians(
                    gaussians,
                    sun_dir,
                    pipeline,
                    method=effective_shadow_method,
                    shadow_map_resolution=shadow_map_resolution,
                    shadow_bias=shadow_bias,
                    ray_march_steps=ray_march_steps,
                    voxel_resolution=voxel_resolution,
                    device="cuda"
                )
            shadow_mask = shadow_mask.unsqueeze(-1)  # [N, 1]

            if gaussians.full_pbr:
                rgb_precomp_shadowed, intensity_shadowed, _, _ = gaussians.compute_directional_pbr(
                    appearance_idx, frame_normals, sun_dir, view.camera_center,
                    sun_elevation=elevation, shadow_mask=shadow_mask
                )
                if 'direct_pbr' in components:
                    direct_light = components['direct_pbr']
                else:
                    direct_light = components['direct']
            else:
                # Apply shadow to direct lighting only
                direct_light = components['direct']
                ambient_light = components['ambient']
                residual_light = components['residual']

                intensity_hdr_shadowed = direct_light * shadow_mask + ambient_light + residual_light
                intensity_hdr_shadowed = torch.clamp_min(intensity_hdr_shadowed, 0.00001)
                intensity_shadowed = intensity_hdr_shadowed ** (1 / 2.2)

                albedo = gaussians.get_albedo
                rgb_precomp_shadowed = torch.clamp(intensity_shadowed * albedo, 0.0)

            # Render direct sun only (with shadow applied), matching train visualizations
            casts_shadow_flag = gaussians.get_casts_shadow.unsqueeze(-1)  # [N, 1]
            is_sky = casts_shadow_flag < 0.5
            sun_intensity = gaussians.sun_model.get_sun_intensity(appearance_idx, sun_elevation=elevation).unsqueeze(0)  # [1, 3]
            direct_for_vis = torch.where(is_sky, sun_intensity.expand(direct_light.shape[0], -1), direct_light)
            direct_shadowed = direct_for_vis * shadow_mask
            direct_shadowed_gamma = torch.clamp_min(direct_shadowed, 0.00001) ** (1 / 2.2)
            direct_shadowed_rgb = torch.clamp(direct_shadowed_gamma * gaussians.get_albedo, 0.0)
            render_pkg_direct = render(view, gaussians, pipeline, background, override_color=direct_shadowed_rgb)
            rendering_direct = torch.clamp(render_pkg_direct["render"], 0.0, 1.0)

            # Render shadowed
            render_pkg = render(view, gaussians, pipeline, background, override_color=rgb_precomp_shadowed)
            rendering_shadowed = torch.clamp(render_pkg["render"], 0.0, 1.0)

            # Time string
            time_str = f"{int(hour):02d}:{int((hour % 1) * 60):02d}"

            # Convert render to numpy
            frame_shadowed_np = (rendering_shadowed.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame_direct_np = (rendering_direct.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame_shadowed_direct_np = np.concatenate([frame_shadowed_np, frame_direct_np], axis=1)

            # Debug panels: shadow mask, N·L, normals
            if show_debug_panels:
                sun_dir_norm = sun_dir / (torch.norm(sun_dir) + 1e-8)
                normals_norm = frame_normals / (torch.norm(frame_normals, dim=-1, keepdim=True) + 1e-8)
                n_dot_l_debug = torch.clamp(torch.sum(normals_norm * sun_dir_norm.unsqueeze(0), dim=-1, keepdim=True), 0.0, 1.0)  # [N, 1]

                # Render shadow mask as grayscale
                shadow_vis = shadow_mask.expand(-1, 3)  # [N, 3]
                render_pkg_shadow = render(view, gaussians, pipeline, background, override_color=shadow_vis)
                shadow_panel = torch.clamp(render_pkg_shadow["render"], 0.0, 1.0)
                shadow_panel_np = (shadow_panel.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Render N·L as grayscale
                ndotl_vis = n_dot_l_debug.expand(-1, 3)  # [N, 3]
                render_pkg_ndotl = render(view, gaussians, pipeline, background, override_color=ndotl_vis)
                ndotl_panel = torch.clamp(render_pkg_ndotl["render"], 0.0, 1.0)
                ndotl_panel_np = (ndotl_panel.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Render normals (remap [-1,1] to [0,1])
                normal_vis = (normals_norm * 0.5 + 0.5)  # [N, 3]
                render_pkg_normal = render(view, gaussians, pipeline, background, override_color=normal_vis)
                normal_panel = torch.clamp(render_pkg_normal["render"], 0.0, 1.0)
                normal_panel_np = (normal_panel.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Build 2x2 debug grid: [shadowed, direct] / [shadow_mask, ndotl] + normals
                h, w = frame_shadowed_np.shape[:2]
                # Resize debug panels to match render size
                debug_bottom = np.concatenate([shadow_panel_np, ndotl_panel_np], axis=1)
                normal_row = np.concatenate([normal_panel_np, np.zeros_like(normal_panel_np)], axis=1)

                # Stack: top=shadowed+direct, middle=shadow_mask+ndotl, bottom=normals+blank
                frame_shadowed_direct_np = np.concatenate([
                    frame_shadowed_direct_np,
                    debug_bottom,
                    normal_row
                ], axis=0)

                # Add labels
                debug_img = Image.fromarray(frame_shadowed_direct_np)
                draw_debug = ImageDraw.Draw(debug_img)
                try:
                    label_font = ImageFont.truetype("arial.ttf", 20)
                except:
                    label_font = ImageFont.load_default()
                labels = [
                    (5, 5, "Shadowed"), (w + 5, 5, "Direct Sun Only"),
                    (5, h + 5, "Shadow Mask"), (w + 5, h + 5, "N·L"),
                    (5, 2*h + 5, "Normals")
                ]
                for lx, ly, ltxt in labels:
                    draw_debug.text((lx+1, ly+1), ltxt, fill=(0, 0, 0), font=label_font)
                    draw_debug.text((lx, ly), ltxt, fill=(255, 255, 255), font=label_font)

                if use_shadow_fixes:
                    info_text = f"bias={fix_bias} hard_th={fix_hard_threshold} ortho={fix_ortho_scale}x flip_n={fix_flip_normals}"
                    draw_debug.text((5, frame_shadowed_direct_np.shape[0] - 25), info_text, fill=(255, 255, 0), font=label_font)

                frame_shadowed_direct_np = np.array(debug_img)

            # Create 3D visualization if enabled
            if show_3d_vis:
                vis_3d = create_3d_sun_visualization(
                    gaussian_positions,
                    sun_dir,
                    camera_position,
                    camera_forward,
                    hour=hour,
                    elevation=elevation,
                    azimuth=azimuth,
                    all_sun_positions=sun_positions,
                    north_offset=north_offset,
                    max_points=3000,
                    figsize=(5, 5),
                    flip_x=flip_x, flip_y=flip_y, flip_z=flip_z,
                )

                # Combine render and visualization
                combined_frame = combine_render_and_visualization(frame_shadowed_direct_np, vis_3d, time_str)
                frames_combined.append(combined_frame)

                # Save combined frame
                Image.fromarray(combined_frame).save(
                    os.path.join(view_render_path, f"frame_{frame_idx:04d}_combined_{time_str.replace(':', '')}.png")
                )

            # Save side-by-side actual render and direct-sun-only render
            Image.fromarray(frame_shadowed_direct_np).save(
                os.path.join(view_render_path, f"frame_{frame_idx:04d}_shadowed_direct_{time_str.replace(':', '')}.png")
            )

            # Save individual frames as well
            torchvision.utils.save_image(
                rendering_shadowed,
                os.path.join(view_render_path, f"frame_{frame_idx:04d}_shadowed_{time_str.replace(':', '')}.png")
            )
            torchvision.utils.save_image(
                rendering_direct,
                os.path.join(view_render_path, f"frame_{frame_idx:04d}_direct_sun_only_{time_str.replace(':', '')}.png")
            )

            frames_shadowed_direct.append(frame_shadowed_direct_np)

        # Save videos
        if frames_shadowed_direct:
            video_path_shadowed = os.path.join(render_path, f"{view.image_name}_day_shadowed_direct.mp4")
            imageio.mimsave(video_path_shadowed, frames_shadowed_direct, fps=args.fps)
            print(f"Saved shadowed+direct video to: {video_path_shadowed}")

        if frames_combined:
            video_path_combined = os.path.join(render_path, f"{view.image_name}_day_combined.mp4")
            imageio.mimsave(video_path_combined, frames_combined, fps=args.fps)
            print(f"Saved combined video to: {video_path_combined}")


def render_day(dataset: ModelParams, iteration: int, pipeline: PipelineParams, args):
    """Main function to render day sequence."""
    with torch.no_grad():
        # Create GaussianModel
        if dataset.use_sun:
            gaussians = GaussianModel(
                dataset.sh_degree, dataset.with_mlp, dataset.mlp_W, dataset.mlp_D, dataset.N_a,
                use_sun=dataset.use_sun, n_images=1700, use_residual_sh=dataset.use_residual_sh,
                full_pbr=dataset.full_pbr
            )
        else:
            raise ValueError("render_day.py requires --use_sun flag")

        scene = Scene(dataset, gaussians, load_iteration=iteration)

        # sun_model is now set up and loaded by Scene
        if gaussians.sun_model is not None:
            gaussians.sun_model.eval()
            print(f"SunModel loaded for {gaussians.n_images} images")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        with open(os.path.join(dataset.model_path, "appearance_lut.json")) as handle:
            appearance_lut = json.loads(handle.read())

        # Determine which views to use
        if args.use_train:
            views = scene.getTrainCameras()
        else:
            views = scene.getTestCameras()

        if not views:
            print("No views available. Trying train cameras...")
            views = scene.getTrainCameras()

        render_day_sequence(
            dataset.model_path,
            scene.loaded_iter,
            views,
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            background,
            appearance_lut,
            args,
            output_suffix="_test" if not args.use_train else "_train",
            selected_view_idx=args.view_idx
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Render scene with sun moving through the day")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    # Standard args
    parser.add_argument("--iteration", default=-1, type=int, help="Model iteration to load")
    parser.add_argument("--quiet", action="store_true")

    # Day rendering args
    parser.add_argument("--num_frames", default=60, type=int, help="Number of frames in day sequence")
    parser.add_argument("--start_hour", default=6.0, type=float, help="Start time (e.g., 6.0 for 6 AM)")
    parser.add_argument("--end_hour", default=20.0, type=float, help="End time (e.g., 20.0 for 8 PM)")
    parser.add_argument("--latitude", default=49.23, type=float, help="Location latitude")
    parser.add_argument("--day_of_year", default=200, type=int, help="Day of year (1-365)")
    parser.add_argument("--north_offset", default=-134.0, type=float, help="North offset in degrees")
    parser.add_argument("--fps", default=15, type=int, help="Video frames per second")
    parser.add_argument("--view_idx", default=0, type=int, help="Index of view to render (-1 for all)")
    parser.add_argument("--use_train", action="store_true", help="Use train cameras instead of test")
    parser.add_argument("--no_3d", action="store_true", help="Disable 3D visualization")

    # Shadow fix args (render_day only, for testing)
    parser.add_argument("--shadow_fixes", action="store_true", help="Enable all shadow fixes")
    parser.add_argument("--shadow_fix_bias", default=0.02, type=float, help="Shadow bias (lower = tighter, try 0.01-0.05)")
    parser.add_argument("--shadow_fix_hard_threshold", action="store_true", default=True, help="Hard-threshold casts_shadow at 0.5")
    parser.add_argument("--no_shadow_fix_hard_threshold", dest="shadow_fix_hard_threshold", action="store_false")
    parser.add_argument("--shadow_fix_ortho_scale", default=2.0, type=float, help="Ortho frustum = extent * scale (default 1.5, fix 2.0)")
    parser.add_argument("--shadow_fix_flip_normals", action="store_true", help="Flip normals facing away from sun")
    parser.add_argument("--shadow_debug_panels", action="store_true", help="Render debug panels (shadow mask, N·L, normals)")

    # Axis flip args for finding correct sun direction convention
    parser.add_argument("--flip_x", action="store_true", default=True, help="Negate X of sun direction (default: on)")
    parser.add_argument("--no_flip_x", dest="flip_x", action="store_false", help="Don't negate X")
    parser.add_argument("--flip_y", action="store_true", default=False, help="Negate Y of sun direction")
    parser.add_argument("--no_flip_y", dest="flip_y", action="store_false")
    parser.add_argument("--flip_z", action="store_true", default=True, help="Negate Z of sun direction (default: on)")
    parser.add_argument("--no_flip_z", dest="flip_z", action="store_false", help="Don't negate Z")

    args = get_combined_args(parser)

    # Convert no_3d flag to show_3d
    args.show_3d = not getattr(args, 'no_3d', False)

    print(f"Rendering day sequence for: {args.model_path}")

    safe_state(args.quiet)
    render_day(model.extract(args), args.iteration, pipeline.extract(args), args)
