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
from utils.shadow_utils import compute_shadows_for_gaussians
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont


def compute_sun_direction_from_angles(azimuth_deg: float, elevation_deg: float, north_offset_deg: float = 0) -> torch.Tensor:
    """
    Compute sun direction vector from azimuth and elevation angles.

    Args:
        azimuth_deg: Sun azimuth in degrees (0 = North, 90 = East, 180 = South, 270 = West)
        elevation_deg: Sun elevation in degrees (0 = horizon, 90 = zenith)
        north_offset_deg: North offset for coordinate system alignment (default from lk2 dataset)

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
) -> np.ndarray:
    """
    Create a 3D visualization of the point cloud with sun position and sun path arc.
    Returns a numpy array image.

    Args:
        all_sun_positions: List of (azimuth, elevation, hour) tuples for the full day
        north_offset: North offset for coordinate system alignment
    """
    def swap_yz(arr):
        """Swap Y and Z for Y-up to Z-up conversion, flip X and Y"""
        if arr.ndim == 1:
            return np.array([arr[0], arr[2], arr[1]])
        else:
            return np.stack([arr[:, 0], arr[:, 2], arr[:, 1]], axis=1)

    def angles_to_direction(az_deg, el_deg, n_offset):
        """Convert azimuth/elevation to direction vector"""
        az_rad = np.radians(az_deg + n_offset)
        el_rad = np.radians(el_deg)
        cos_el = np.cos(el_rad)
        x = cos_el * np.sin(az_rad)
        y = -cos_el * np.cos(az_rad)
        z = np.sin(el_rad)
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

    # Create figure
    fig = plt.figure(figsize=figsize, facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Plot points with gradient color based on height
    z_norm = (positions[:, 2] - positions[:, 2].min()) / (positions[:, 2].max() - positions[:, 2].min() + 1e-8)
    colors = plt.cm.coolwarm(z_norm)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=colors, s=0.5, alpha=0.4)

    # Draw the sun path arc if all_sun_positions is provided
    if all_sun_positions is not None:
        sun_path_points = []
        sun_path_colors = []
        for i, (az, el, h) in enumerate(all_sun_positions):
            if el >= 0:  # Only show sun positions above horizon
                dir_vec = angles_to_direction(az, el, north_offset)
                dir_vec = swap_yz(dir_vec)
                pos = scene_center + dir_vec * sun_arc_radius
                sun_path_points.append(pos)
                # Color gradient from orange (morning) to yellow (noon) to red (evening)
                t = i / max(len(all_sun_positions) - 1, 1)
                if t < 0.5:
                    # Morning: orange to yellow
                    c = (1.0, 0.5 + t, 0.0, 0.6)
                else:
                    # Evening: yellow to red
                    c = (1.0, 1.0 - (t - 0.5), 0.0, 0.6)
                sun_path_colors.append(c)

        if len(sun_path_points) > 1:
            sun_path_points = np.array(sun_path_points)
            # Draw the sun arc as a line
            ax.plot(sun_path_points[:, 0], sun_path_points[:, 1], sun_path_points[:, 2],
                    color='orange', linewidth=2, alpha=0.7, linestyle='-')
            # Draw small dots along the path
            for i, (pt, col) in enumerate(zip(sun_path_points, sun_path_colors)):
                # Draw every 5th point as a small marker
                if i % 5 == 0:
                    ax.scatter([pt[0]], [pt[1]], [pt[2]], c=[col[:3]], s=20, alpha=0.5)

    # Plot sun direction arrow (thick orange)
    ax.quiver(scene_center[0], scene_center[1], scene_center[2],
              sun_dir_norm[0] * arrow_length,
              sun_dir_norm[1] * arrow_length,
              sun_dir_norm[2] * arrow_length,
              color='orange', linewidth=3, arrow_length_ratio=0.15)

    # Sun symbol (bright yellow sphere) - current position
    ax.scatter([sun_pos[0]], [sun_pos[1]], [sun_pos[2]],
               c='yellow', s=200, marker='o', edgecolors='orange', linewidths=2)

    # Camera position (red triangle)
    ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
               c='red', s=100, marker='^')

    # Camera forward direction
    if cam_forward is not None:
        cam_forward_norm = cam_forward / (np.linalg.norm(cam_forward) + 1e-8)
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                  cam_forward_norm[0] * arrow_length * 0.4,
                  cam_forward_norm[1] * arrow_length * 0.4,
                  cam_forward_norm[2] * arrow_length * 0.4,
                  color='red', linewidth=2, arrow_length_ratio=0.2)

    # Set view angle - nice 3/4 view
    ax.view_init(elev=25, azim=-60)

    # Set limits
    max_range = scene_extent * 0.9
    mid_x, mid_y, mid_z = scene_center[0], scene_center[1], scene_center[2]
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Style
    ax.set_xlabel('X', color='white', fontsize=8)
    ax.set_ylabel('Z', color='white', fontsize=8)
    ax.set_zlabel('Y (up)', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=6)

    # Add time and sun info as title
    time_str = f"{int(hour):02d}:{int((hour % 1) * 60):02d}"
    ax.set_title(f"Time: {time_str}\nElev: {elevation:.1f}° Az: {azimuth:.1f}°",
                 color='white', fontsize=10, pad=10)

    # Remove grid for cleaner look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    buf = buf[:, :, :3]  # RGB only
    plt.close(fig)

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
            sun_dir = compute_sun_direction_from_angles(azimuth, elevation, north_offset)

            # Compute lighting
            if gaussians.full_pbr:
                rgb_precomp_unshadowed, intensity, sun_dir_out, components = gaussians.compute_directional_pbr(
                    appearance_idx, normal_vectors, sun_dir, view.camera_center, sun_elevation=elevation
                )
            else:
                rgb_precomp_unshadowed, intensity, sun_dir_out, components = gaussians.compute_directional_rgb(
                    appearance_idx, normal_vectors, sun_dir, sun_elevation=elevation
                )

            # Compute shadows
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
                    appearance_idx, normal_vectors, sun_dir, view.camera_center,
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
                    figsize=(5, 5)
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

    args = get_combined_args(parser)

    # Convert no_3d flag to show_3d
    args.show_3d = not getattr(args, 'no_3d', False)

    print(f"Rendering day sequence for: {args.model_path}")

    safe_state(args.quiet)
    render_day(model.extract(args), args.iteration, pipeline.extract(args), args)
