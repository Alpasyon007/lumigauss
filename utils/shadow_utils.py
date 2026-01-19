"""
Shadow mapping utilities for Gaussian Splatting with directional sun lighting.

This module provides functions to compute shadows from a directional light source
using shadow mapping technique adapted for Gaussian splatting.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional


def create_sun_camera(
    sun_direction: torch.Tensor,
    scene_center: torch.Tensor,
    scene_extent: float,
    resolution: int = 1024,
    device: str = "cuda"
) -> dict:
    """
    Create an orthographic camera looking along the sun direction for shadow mapping.

    Args:
        sun_direction: Normalized direction vector pointing TOWARD the sun [3]
        scene_center: Center of the scene bounding box [3]
        scene_extent: Approximate extent/radius of the scene
        resolution: Shadow map resolution (width and height)
        device: Device to create tensors on

    Returns:
        Dictionary with camera parameters for rendering shadow map
    """
    sun_dir = sun_direction / (torch.norm(sun_direction) + 1e-8)
    sun_dir = sun_dir.float().to(device)
    scene_center = scene_center.float().to(device)

    # Position the camera far along the sun direction, looking back at scene
    # Camera is placed at scene_center + sun_dir * distance
    distance = scene_extent * 3.0  # Place camera far enough to see entire scene
    cam_pos = scene_center + sun_dir * distance

    # Camera looks toward scene center (opposite of sun direction)
    forward = -sun_dir  # Looking toward scene (opposite of sun direction)

    # Create orthonormal basis for camera
    # Choose an arbitrary up vector (avoid parallel to forward)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    if abs(torch.dot(forward, world_up)) > 0.99:
        world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)

    right = torch.cross(forward, world_up)
    right = right / (torch.norm(right) + 1e-8)
    up = torch.cross(right, forward)
    up = up / (torch.norm(up) + 1e-8)

    # Build rotation matrix (world to camera)
    # Camera convention: -Z is forward, X is right, Y is up
    R = torch.stack([right, up, -forward], dim=0)  # [3, 3]

    # Translation: camera position in world, need to convert to camera space
    T = -R @ cam_pos  # [3]

    # Build world-to-view matrix (4x4)
    world_view = torch.eye(4, device=device, dtype=torch.float32)
    world_view[:3, :3] = R
    world_view[:3, 3] = T
    world_view = world_view.T  # Transpose for column-major convention

    # Build orthographic projection matrix
    # Scene extent determines the frustum size
    ortho_size = scene_extent * 1.5  # Make frustum slightly larger than scene
    znear = 0.1
    zfar = distance * 2 + scene_extent * 2

    proj = create_orthographic_projection(
        left=-ortho_size, right=ortho_size,
        bottom=-ortho_size, top=ortho_size,
        znear=znear, zfar=zfar,
        device=device
    )
    proj = proj.T  # Transpose for column-major convention

    # Full projection = world_view @ projection
    full_proj = world_view @ proj

    return {
        'world_view_transform': world_view,
        'projection_matrix': proj,
        'full_proj_transform': full_proj,
        'camera_center': cam_pos,
        'resolution': resolution,
        'znear': znear,
        'zfar': zfar,
        'ortho_size': ortho_size,
    }


def create_orthographic_projection(
    left: float, right: float,
    bottom: float, top: float,
    znear: float, zfar: float,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Create an orthographic projection matrix.

    Args:
        left, right: Left and right clipping planes
        bottom, top: Bottom and top clipping planes
        znear, zfar: Near and far clipping planes
        device: Device to create tensor on

    Returns:
        4x4 orthographic projection matrix
    """
    P = torch.zeros(4, 4, device=device, dtype=torch.float32)

    P[0, 0] = 2.0 / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    P[2, 2] = 1.0 / (zfar - znear)  # Different convention for depth

    P[0, 3] = -(right + left) / (right - left)
    P[1, 3] = -(top + bottom) / (top - bottom)
    P[2, 3] = -znear / (zfar - znear)
    P[3, 3] = 1.0

    return P


class SunShadowCamera:
    """
    A minimal camera class for shadow map rendering from sun direction.
    Compatible with the Gaussian rasterizer.
    """
    def __init__(
        self,
        sun_direction: torch.Tensor,
        scene_center: torch.Tensor,
        scene_extent: float,
        resolution: int = 1024,
        device: str = "cuda"
    ):
        self.device = device
        self.image_width = resolution
        self.image_height = resolution

        # Create camera matrices
        cam_params = create_sun_camera(
            sun_direction, scene_center, scene_extent, resolution, device
        )

        self.world_view_transform = cam_params['world_view_transform']
        self.projection_matrix = cam_params['projection_matrix']
        self.full_proj_transform = cam_params['full_proj_transform']
        self.camera_center = cam_params['camera_center']
        self.znear = cam_params['znear']
        self.zfar = cam_params['zfar']
        self.ortho_size = cam_params['ortho_size']

        # For orthographic, FoV is not really used but rasterizer needs it
        # Use a value that approximates the orthographic frustum
        self.FoVx = 2.0 * math.atan(cam_params['ortho_size'] / (cam_params['zfar'] / 2))
        self.FoVy = self.FoVx


def render_shadow_map(
    sun_camera: SunShadowCamera,
    gaussians,
    pipe,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Render a depth/shadow map from the sun's viewpoint.

    Args:
        sun_camera: SunShadowCamera instance
        gaussians: GaussianModel instance
        pipe: Pipeline parameters
        device: Device

    Returns:
        Shadow depth map [1, H, W]
    """
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    bg_color = torch.zeros(3, device=device)

    # For orthographic projection, tanfov should be large (approximating parallel rays)
    # But we use the computed FoV for compatibility
    tanfovx = math.tan(sun_camera.FoVx * 0.5)
    tanfovy = math.tan(sun_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=sun_camera.image_height,
        image_width=sun_camera.image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=sun_camera.world_view_transform,
        projmatrix=sun_camera.full_proj_transform,
        sh_degree=0,  # Don't need SH for shadow map
        campos=sun_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation

    # Filter out non-shadow-casting gaussians (sky gaussians)
    # by setting their opacity to 0 for shadow map rendering
    casts_shadow = gaussians.get_casts_shadow  # [N]
    if casts_shadow.shape[0] == opacity.shape[0]:
        # Multiply opacity by casts_shadow flag (0 for sky, 1 for solid)
        opacity = opacity * casts_shadow.unsqueeze(-1)

    screenspace_points = torch.zeros_like(means3D, requires_grad=False, device=device)

    # Render with white color to get depth
    colors_precomp = torch.ones_like(means3D)

    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )

    # Extract depth from allmap
    # allmap[0] is expected depth, allmap[5] is median depth
    render_depth = allmap[0:1]  # Expected depth
    render_depth = torch.nan_to_num(render_depth, 0, 0)

    # Normalize by alpha to get proper depth
    render_alpha = torch.nan_to_num(allmap[1:2], 0, 0)
    shadow_depth = render_depth / render_alpha.clamp_min(1e-6)

    return shadow_depth, render_alpha


def compute_shadow_mask(
    gaussian_positions: torch.Tensor,
    sun_camera: SunShadowCamera,
    shadow_depth_map: torch.Tensor,
    shadow_alpha_map: torch.Tensor,
    bias: float = 0.05,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute shadow mask for each Gaussian by comparing its depth to the shadow map.

    Args:
        gaussian_positions: World positions of Gaussians [N, 3]
        sun_camera: SunShadowCamera used to render shadow map
        shadow_depth_map: Rendered depth from sun view [1, H, W]
        shadow_alpha_map: Rendered alpha from sun view [1, H, W]
        bias: Depth bias to avoid self-shadowing artifacts
        device: Device

    Returns:
        Shadow mask [N] where 1=lit, 0=shadowed
    """
    N = gaussian_positions.shape[0]

    # Transform Gaussian positions to sun camera clip space
    ones = torch.ones(N, 1, device=device, dtype=gaussian_positions.dtype)
    positions_homo = torch.cat([gaussian_positions, ones], dim=1)  # [N, 4]

    # Apply full projection transform
    # full_proj_transform is [4, 4] in column-major, so we do pos @ transform
    clip_coords = positions_homo @ sun_camera.full_proj_transform  # [N, 4]

    # Perspective divide (for orthographic w should be 1, but do it anyway)
    ndc_coords = clip_coords[:, :3] / (clip_coords[:, 3:4] + 1e-8)  # [N, 3]

    # NDC to texture coordinates [0, 1]
    # NDC is in [-1, 1], convert to [0, 1] for sampling
    uv = (ndc_coords[:, :2] + 1.0) * 0.5  # [N, 2]

    # Depth in sun view (Z coordinate after projection)
    gaussian_depth = ndc_coords[:, 2]  # [N]

    # Sample shadow map at UV coordinates
    # Need to convert to grid_sample format: [1, 1, N, 2] with values in [-1, 1]
    grid = (uv * 2.0 - 1.0).view(1, 1, N, 2)  # [1, 1, N, 2]

    # Sample depth map
    sampled_depth = torch.nn.functional.grid_sample(
        shadow_depth_map.unsqueeze(0),  # [1, 1, H, W]
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    ).view(N)  # [N]

    # Sample alpha map (to check if there's geometry at this location)
    sampled_alpha = torch.nn.functional.grid_sample(
        shadow_alpha_map.unsqueeze(0),  # [1, 1, H, W]
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    ).view(N)  # [N]

    # Compare depths: if gaussian_depth > sampled_depth + bias, it's in shadow
    # Also check if the sampled location has valid geometry (alpha > threshold)
    in_shadow = (gaussian_depth > sampled_depth + bias) & (sampled_alpha > 0.1)

    # Shadow mask: 1 = lit, 0 = shadowed
    shadow_mask = (~in_shadow).float()

    # Points outside the shadow map frustum should be lit (or could be marked specially)
    out_of_bounds = (uv[:, 0] < 0) | (uv[:, 0] > 1) | (uv[:, 1] < 0) | (uv[:, 1] > 1)
    shadow_mask[out_of_bounds] = 1.0

    return shadow_mask


# =============================================================================
# SHADOW METHOD 1: No shadows (self-shadowing only via N·L)
# =============================================================================

def compute_shadows_none(
    gaussians,
    sun_direction: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    No shadow computation - all points are fully lit.
    Self-shadowing is handled by the N·L term in the lighting equation.

    Args:
        gaussians: GaussianModel instance
        sun_direction: Direction toward the sun [3] (unused)
        device: Device

    Returns:
        shadow_mask: [N] tensor of all ones (all lit)
    """
    N = gaussians.get_xyz.shape[0]
    return torch.ones(N, device=device)


# =============================================================================
# SHADOW METHOD 2: Shadow Mapping (already implemented above)
# =============================================================================

def compute_shadows_shadow_map(
    gaussians,
    sun_direction: torch.Tensor,
    pipe,
    shadow_map_resolution: int = 512,
    shadow_bias: float = 0.1,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, SunShadowCamera]:
    """
    Shadow mapping - render depth from sun's viewpoint and compare.

    This is the standard shadow mapping technique adapted for Gaussian splatting.

    Args:
        gaussians: GaussianModel instance
        sun_direction: Direction toward the sun [3]
        pipe: Pipeline parameters
        shadow_map_resolution: Resolution of shadow map
        shadow_bias: Depth comparison bias
        device: Device

    Returns:
        Tuple of (shadow_mask, shadow_depth_map, sun_camera)
    """
    positions = gaussians.get_xyz
    scene_center = positions.mean(dim=0)
    scene_extent = (positions - scene_center).norm(dim=1).max().item()

    sun_camera = SunShadowCamera(
        sun_direction=sun_direction,
        scene_center=scene_center,
        scene_extent=scene_extent,
        resolution=shadow_map_resolution,
        device=device
    )

    shadow_depth_map, shadow_alpha_map = render_shadow_map(
        sun_camera, gaussians, pipe, device
    )

    shadow_mask = compute_shadow_mask(
        positions,
        sun_camera,
        shadow_depth_map,
        shadow_alpha_map,
        bias=shadow_bias,
        device=device
    )

    return shadow_mask, shadow_depth_map, sun_camera


# =============================================================================
# SHADOW METHOD 3: Ray Marching through Gaussians
# =============================================================================

def compute_shadows_ray_march(
    gaussians,
    sun_direction: torch.Tensor,
    num_steps: int = 64,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Ray marching through Gaussians - trace rays toward sun, accumulate opacity.

    For each Gaussian, march a ray toward the sun and accumulate the opacity
    of Gaussians along the ray. If accumulated opacity exceeds threshold,
    the point is in shadow.

    This method produces soft shadows naturally based on Gaussian opacity.

    Args:
        gaussians: GaussianModel instance
        sun_direction: Direction toward the sun [3]
        num_steps: Number of ray marching steps
        device: Device

    Returns:
        shadow_mask: [N] tensor where 1=lit, 0=shadowed
    """
    positions = gaussians.get_xyz  # [N, 3]
    opacities = gaussians.get_opacity.squeeze()  # [N]
    scales = gaussians.get_scaling  # [N, 3]

    N = positions.shape[0]

    # Normalize sun direction
    sun_dir = sun_direction / (torch.norm(sun_direction) + 1e-8)
    sun_dir = sun_dir.to(device).float()

    # Compute scene extent for ray length
    scene_center = positions.mean(dim=0)
    scene_extent = (positions - scene_center).norm(dim=1).max().item()
    ray_length = scene_extent * 2.0
    step_size = ray_length / num_steps

    # Compute average scale for each Gaussian (approximate radius)
    gaussian_radii = scales.mean(dim=1)  # [N]
    avg_radius = gaussian_radii.mean().item()

    # Build a spatial grid for acceleration
    # Grid cell size based on average Gaussian radius
    cell_size = max(avg_radius * 6.0, scene_extent / 50.0)  # At least 50 cells across

    # Compute grid bounds
    pos_min = positions.min(dim=0)[0] - cell_size
    pos_max = positions.max(dim=0)[0] + cell_size
    grid_size = ((pos_max - pos_min) / cell_size).long() + 1
    grid_size = torch.clamp(grid_size, 1, 100)  # Limit grid size

    # Assign Gaussians to grid cells
    grid_coords = ((positions - pos_min) / cell_size).long()
    grid_coords = torch.clamp(grid_coords, 0, grid_size.unsqueeze(0) - 1)

    # Create a dictionary mapping grid cells to Gaussian indices
    # Use a flat index for the grid
    flat_indices = (grid_coords[:, 0] * grid_size[1] * grid_size[2] +
                   grid_coords[:, 1] * grid_size[2] +
                   grid_coords[:, 2])

    # Build cell to Gaussian mapping (on CPU for dict operations)
    cell_to_gaussians = {}
    flat_indices_cpu = flat_indices.cpu().numpy()
    for i, cell_idx in enumerate(flat_indices_cpu):
        if cell_idx not in cell_to_gaussians:
            cell_to_gaussians[cell_idx] = []
        cell_to_gaussians[cell_idx].append(i)

    shadow_mask = torch.ones(N, device=device)

    # Process each Gaussian
    batch_size = 256  # Smaller batches

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_positions = positions[batch_start:batch_end]  # [B, 3]
        B = batch_positions.shape[0]

        # Accumulated opacity along ray for each point in batch
        accumulated_opacity = torch.zeros(B, device=device)

        # March along the ray
        for step in range(1, num_steps + 1):
            t = step * step_size
            sample_pos = batch_positions + sun_dir.unsqueeze(0) * t  # [B, 3]

            # For each sample, find which grid cell it's in and check nearby Gaussians
            sample_grid = ((sample_pos - pos_min) / cell_size).long()

            # Process each sample in batch
            for b in range(B):
                # Skip if outside grid
                sg = sample_grid[b]
                if (sg < 0).any() or (sg >= grid_size).any():
                    continue

                # Get nearby cells (3x3x3 neighborhood)
                step_opacity = 0.0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            cx = sg[0].item() + dx
                            cy = sg[1].item() + dy
                            cz = sg[2].item() + dz

                            if cx < 0 or cy < 0 or cz < 0:
                                continue
                            if cx >= grid_size[0] or cy >= grid_size[1] or cz >= grid_size[2]:
                                continue

                            cell_idx = cx * grid_size[1].item() * grid_size[2].item() + cy * grid_size[2].item() + cz

                            if cell_idx not in cell_to_gaussians:
                                continue

                            # Check Gaussians in this cell
                            for g_idx in cell_to_gaussians[cell_idx]:
                                # Skip self
                                if g_idx == batch_start + b:
                                    continue

                                # Compute distance
                                dist = torch.norm(sample_pos[b] - positions[g_idx])
                                eff_radius = gaussian_radii[g_idx] * 3.0

                                # Gaussian weight
                                weight = torch.exp(-0.5 * (dist / (eff_radius + 1e-6)) ** 2)
                                contrib = weight * opacities[g_idx]

                                if contrib.item() > step_opacity:
                                    step_opacity = contrib.item()

                accumulated_opacity[b] += step_opacity * 0.1

        # Convert accumulated opacity to shadow mask
        shadow_factor = torch.exp(-accumulated_opacity * 2.0)
        shadow_mask[batch_start:batch_end] = shadow_factor

    return shadow_mask


# =============================================================================
# SHADOW METHOD 4: Voxel-based Shadows
# =============================================================================

def compute_shadows_voxel(
    gaussians,
    sun_direction: torch.Tensor,
    voxel_resolution: int = 128,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Voxel-based shadow computation - voxelize scene and trace through voxels.

    This method voxelizes the Gaussian scene into a 3D grid, then traces rays
    from each Gaussian toward the sun through the voxel grid to determine shadowing.

    Args:
        gaussians: GaussianModel instance
        sun_direction: Direction toward the sun [3]
        voxel_resolution: Resolution of voxel grid (NxNxN)
        device: Device

    Returns:
        shadow_mask: [N] tensor where 1=lit, 0=shadowed
    """
    positions = gaussians.get_xyz  # [N, 3]
    opacities = gaussians.get_opacity.squeeze()  # [N]
    scales = gaussians.get_scaling  # [N, 3]

    N = positions.shape[0]
    res = voxel_resolution

    # Normalize sun direction
    sun_dir = sun_direction / (torch.norm(sun_direction) + 1e-8)
    sun_dir = sun_dir.to(device).float()

    # Compute scene bounds with padding
    pos_min = positions.min(dim=0)[0]
    pos_max = positions.max(dim=0)[0]
    scene_extent = (pos_max - pos_min).max().item()
    padding = scene_extent * 0.1

    voxel_min = pos_min - padding
    voxel_max = pos_max + padding
    voxel_size = (voxel_max - voxel_min) / res

    # Create voxel grid for opacity
    voxel_grid = torch.zeros(res, res, res, device=device)

    # Splat Gaussians into voxel grid
    # Convert positions to voxel indices
    voxel_coords = ((positions - voxel_min) / voxel_size).long()  # [N, 3]
    voxel_coords = torch.clamp(voxel_coords, 0, res - 1)

    # Simple splatting: add opacity to voxel (could be improved with proper Gaussian splatting)
    for i in range(N):
        x, y, z = voxel_coords[i]
        # Splat to neighboring voxels based on scale
        radius = max(1, int((scales[i].mean() / voxel_size.mean()).item()))
        radius = min(radius, 3)  # Limit radius for performance

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < res and 0 <= ny < res and 0 <= nz < res:
                        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                        weight = math.exp(-0.5 * (dist / max(radius, 1)) ** 2)
                        voxel_grid[nx, ny, nz] += opacities[i].item() * weight

    # Clamp voxel values
    voxel_grid = torch.clamp(voxel_grid, 0, 1)

    # Ray march through voxel grid for each Gaussian
    shadow_mask = torch.ones(N, device=device)

    # Number of steps based on grid diagonal
    num_steps = int(res * 1.73)  # sqrt(3) * res
    step_size = voxel_size.mean()

    for i in range(N):
        pos = positions[i]
        accumulated_opacity = 0.0

        # March toward sun
        for step in range(1, num_steps + 1):
            sample_pos = pos + sun_dir * step * step_size

            # Convert to voxel coordinates
            voxel_coord = ((sample_pos - voxel_min) / voxel_size).long()

            # Check bounds
            if (voxel_coord < 0).any() or (voxel_coord >= res).any():
                break  # Exited grid

            # Sample voxel
            vx, vy, vz = voxel_coord
            accumulated_opacity += voxel_grid[vx, vy, vz].item() * 0.5

            # Early termination if fully shadowed
            if accumulated_opacity > 3.0:
                break

        # Convert to shadow factor
        shadow_mask[i] = math.exp(-accumulated_opacity)

    return shadow_mask


# =============================================================================
# Unified Shadow Computation Interface
# =============================================================================

def compute_shadows_for_gaussians(
    gaussians,
    sun_direction: torch.Tensor,
    pipe,
    method: str = "shadow_map",
    shadow_map_resolution: int = 512,
    shadow_bias: float = 0.1,
    ray_march_steps: int = 64,
    voxel_resolution: int = 128,
    device: str = "cuda"
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[SunShadowCamera]]:
    """
    Unified interface for computing shadows using different methods.

    Args:
        gaussians: GaussianModel instance
        sun_direction: Direction toward the sun [3]
        pipe: Pipeline parameters (for shadow_map method)
        method: Shadow computation method:
            - 'none': No shadows (all points lit, only self-shadowing via N·L)
            - 'shadow_map': Shadow mapping (render depth from sun view)
            - 'ray_march': Ray marching through Gaussians
            - 'voxel': Voxel-based ray tracing
        shadow_map_resolution: Resolution for shadow mapping
        shadow_bias: Depth comparison bias for shadow mapping
        ray_march_steps: Number of steps for ray marching
        voxel_resolution: Resolution for voxel grid
        device: Device

    Returns:
        Tuple of:
        - shadow_mask: [N] tensor where 1=lit, 0=shadowed
        - shadow_depth_map: [1, H, W] depth map (only for shadow_map method)
        - sun_camera: SunShadowCamera (only for shadow_map method)
    """
    if method == "none":
        shadow_mask = compute_shadows_none(gaussians, sun_direction, device)
        return shadow_mask, None, None

    elif method == "shadow_map":
        return compute_shadows_shadow_map(
            gaussians, sun_direction, pipe,
            shadow_map_resolution, shadow_bias, device
        )

    elif method == "ray_march":
        shadow_mask = compute_shadows_ray_march(
            gaussians, sun_direction, ray_march_steps, device
        )
        return shadow_mask, None, None

    elif method == "voxel":
        shadow_mask = compute_shadows_voxel(
            gaussians, sun_direction, voxel_resolution, device
        )
        return shadow_mask, None, None

    else:
        raise ValueError(f"Unknown shadow method: {method}. "
                        f"Choose from: 'none', 'shadow_map', 'ray_march', 'voxel'")


def visualize_sun_and_camera(
    gaussian_positions: torch.Tensor,
    sun_direction: torch.Tensor,
    camera_position: torch.Tensor,
    camera_forward: torch.Tensor = None,
    shadow_mask: torch.Tensor = None,
    max_points: int = 5000,
    figsize: Tuple[int, int] = (12, 12),
    title: str = "Scene with Sun and Camera",
    save_path: str = None,
    return_image: bool = True
) -> Optional[np.ndarray]:
    """
    Visualize the point cloud with sun direction and camera position in 3D.
    Shows a 2x2 grid with Front, Top, Left, and Right views.

    Note: Scene uses Y-up coordinate system. We swap Y and Z for matplotlib
    which uses Z-up, so the visualization appears correct.

    Args:
        gaussian_positions: Positions of Gaussians [N, 3] in Y-up coordinate system
        sun_direction: Direction vector pointing TOWARD the sun [3]
        camera_position: Camera position in world space [3]
        camera_forward: Camera forward/look direction [3] (optional)
        shadow_mask: Optional shadow mask [N] for coloring points (1=lit, 0=shadow)
        max_points: Maximum number of points to display (for performance)
        figsize: Figure size tuple
        title: Plot title
        save_path: If provided, save figure to this path
        return_image: If True, return the figure as a numpy array

    Returns:
        If return_image is True, returns RGB image as numpy array [H, W, 3]
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Helper function to swap Y and Z for Y-up to Z-up conversion
    def swap_yz(arr):
        """Swap Y and Z coordinates: [X, Y, Z] -> [X, Z, Y] for Y-up to Z-up"""
        if arr.ndim == 1:
            return np.array([arr[0], arr[2], arr[1]])
        else:
            return np.stack([arr[:, 0], arr[:, 2], arr[:, 1]], axis=1)

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

    if shadow_mask is not None and torch.is_tensor(shadow_mask):
        shadows = shadow_mask.detach().cpu().numpy()
    else:
        shadows = shadow_mask

    # Subsample points if too many
    if len(positions) > max_points:
        indices = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[indices]
        if shadows is not None:
            shadows = shadows[indices]

    # Swap Y and Z for visualization (Y-up scene -> Z-up matplotlib)
    positions = swap_yz(positions)
    sun_dir = swap_yz(sun_dir)
    cam_pos = swap_yz(cam_pos)
    if cam_forward is not None:
        cam_forward = swap_yz(cam_forward)

    # Compute scene center and extent for arrow scaling
    scene_center = positions.mean(axis=0)
    scene_extent = np.linalg.norm(positions - scene_center, axis=1).max()
    arrow_length = scene_extent * 0.5

    # Normalize sun direction
    sun_dir_norm = sun_dir / (np.linalg.norm(sun_dir) + 1e-8)
    sun_pos = scene_center + sun_dir_norm * arrow_length * 1.1

    # Prepare colors
    if shadows is not None:
        colors = np.zeros((len(positions), 3))
        colors[:, 0] = shadows  # R
        colors[:, 1] = shadows * 0.8  # G
        colors[:, 2] = 1 - shadows  # B (shadowed points are blue)
    else:
        colors = 'gray'

    # Camera forward normalized
    if cam_forward is not None:
        cam_forward_norm = cam_forward / (np.linalg.norm(cam_forward) + 1e-8)
    else:
        cam_forward_norm = None

    # Create 2x2 grid figure
    fig = plt.figure(figsize=figsize)

    # View configurations for Z-up matplotlib (after Y/Z swap):
    # (elev, azim, title)
    # Front: looking along -Y (which was -Z in original scene)
    # Top: looking down -Z (which was -Y/up in original scene)
    # Left/Right: looking along ±X
    views = [
        (20, -60, 'Front'),      # Front perspective view
        (90, 0, 'Top'),          # Top-down view (looking down Z, which is original Y/up)
        (0, 0, 'Right'),         # Right side view (looking along +Y)
        (0, -90, 'Back'),        # Back view (looking along -X)
    ]

    max_range = scene_extent * 0.8
    mid_x, mid_y, mid_z = scene_center[0], scene_center[1], scene_center[2]

    for idx, (elev, azim, view_name) in enumerate(views):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

        # Plot points
        if shadows is not None:
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                       c=colors, s=1, alpha=0.5)
        else:
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                       c='gray', s=1, alpha=0.5)

        # Plot sun direction arrow
        ax.quiver(scene_center[0], scene_center[1], scene_center[2],
                  sun_dir_norm[0] * arrow_length,
                  sun_dir_norm[1] * arrow_length,
                  sun_dir_norm[2] * arrow_length,
                  color='orange', linewidth=2, arrow_length_ratio=0.15)

        # Sun symbol
        ax.scatter([sun_pos[0]], [sun_pos[1]], [sun_pos[2]],
                   c='yellow', s=100, marker='o', edgecolors='orange', linewidths=2)

        # Camera position
        ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
                   c='red', s=60, marker='^')

        # Camera forward direction
        if cam_forward_norm is not None:
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                      cam_forward_norm[0] * arrow_length * 0.5,
                      cam_forward_norm[1] * arrow_length * 0.5,
                      cam_forward_norm[2] * arrow_length * 0.5,
                      color='red', linewidth=2, arrow_length_ratio=0.2)

        # Scene center
        ax.scatter([scene_center[0]], [scene_center[1]], [scene_center[2]],
                   c='green', s=30, marker='x')

        # Set view angle
        ax.view_init(elev=elev, azim=azim)

        # Set limits
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Labels (note: Z in plot is Y in original scene)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Z', fontsize=8)  # Original Z
        ax.set_zlabel('Y (up)', fontsize=8)  # Original Y (up)
        ax.set_title(view_name, fontsize=10)
        ax.tick_params(labelsize=6)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Convert to image array if requested
    if return_image:
        fig.canvas.draw()
        # Get the RGBA buffer from the figure (use buffer_rgba instead of deprecated tostring_rgb)
        buf = np.asarray(fig.canvas.buffer_rgba())
        # Convert RGBA to RGB
        buf = buf[:, :, :3]
        plt.close(fig)
        return buf
    else:
        plt.close(fig)
        return None


def create_sun_camera_visualization_tensor(
    gaussian_positions: torch.Tensor,
    sun_direction: torch.Tensor,
    camera_position: torch.Tensor,
    camera_forward: torch.Tensor = None,
    shadow_mask: torch.Tensor = None,
    max_points: int = 5000,
) -> torch.Tensor:
    """
    Create a visualization tensor suitable for TensorBoard.

    Returns:
        Tensor of shape [3, H, W] suitable for tb_writer.add_images()
    """
    img = visualize_sun_and_camera(
        gaussian_positions=gaussian_positions,
        sun_direction=sun_direction,
        camera_position=camera_position,
        camera_forward=camera_forward,
        shadow_mask=shadow_mask,
        max_points=max_points,
        return_image=True
    )

    # Convert to tensor [H, W, 3] -> [3, H, W] and normalize to [0, 1]
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # [3, H, W]

    return img_tensor
