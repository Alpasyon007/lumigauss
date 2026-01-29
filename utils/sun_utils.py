"""
Sun position utilities for LumiGauss.

This module provides:
1. SunModel class for explicit directional sun lighting
2. Helper functions for sun direction SH coefficients
3. Directional lighting with ambient sky illumination

The sun model follows the equation:
    Li(ωi) = Isun · δ(ωi − ωsun) + Lsky(ωi)

where ωsun is the sun direction from camera metadata, Isun is the sun intensity,
and Lsky(ωi) represents ambient sky illumination.

NOTE: Sun data is loaded in scene/dataset_readers.py and stored in Camera objects.
The sun direction is passed to SunModel.forward() at runtime from the camera.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


def get_sun_direction(sun_data: Dict, image_name: str) -> Optional[torch.Tensor]:
    """
    Get the sun direction vector for a specific image.

    NOTE: This function is kept for backward compatibility but sun directions
    should generally be accessed from Camera.sun_direction directly.

    Args:
        sun_data: Dictionary with sun position data for all images.
        image_name: Name of the image to get sun direction for.

    Returns:
        Sun direction vector as a torch tensor [3], or None if not found.
    """
    # Try exact match first
    if image_name in sun_data:
        direction = sun_data[image_name]["sun_direction_vector"]
        return torch.tensor(direction, dtype=torch.float32)

    # Try with .JPG extension
    if image_name + ".JPG" in sun_data:
        direction = sun_data[image_name + ".JPG"]["sun_direction_vector"]
        return torch.tensor(direction, dtype=torch.float32)

    # Try with .jpg extension
    if image_name + ".jpg" in sun_data:
        direction = sun_data[image_name + ".jpg"]["sun_direction_vector"]
        return torch.tensor(direction, dtype=torch.float32)

    # Try to find a match by removing extension from image_name
    base_name = image_name.rsplit('.', 1)[0] if '.' in image_name else image_name
    for key in sun_data:
        key_base = key.rsplit('.', 1)[0] if '.' in key else key
        if key_base == base_name:
            direction = sun_data[key]["sun_direction_vector"]
            return torch.tensor(direction, dtype=torch.float32)

    print(f"Warning: Sun direction not found for image: {image_name}")
    return None


def direction_to_sh_coefficients(direction: torch.Tensor, intensity: float = 1.0,
                                  ambient: float = 0.1, sh_degree: int = 2) -> torch.Tensor:
    """
    Convert a directional light source to spherical harmonic coefficients.

    This creates an environment map with a directional sun light and ambient sky illumination.
    The SH representation approximates:
        Li(ωi) = Isun · δ(ωi − ωsun) + Lsky(ωi)

    For a directional light, we use the SH basis functions evaluated at the light direction.

    Args:
        direction: Normalized sun direction vector [3] (pointing towards the sun)
        intensity: Sun intensity (Isun)
        ambient: Ambient sky illumination level (Lsky)
        sh_degree: SH degree (0, 1, or 2)

    Returns:
        SH coefficients [3, (sh_degree+1)^2] for RGB channels
    """
    device = direction.device

    # Ensure direction is normalized
    direction = direction / (torch.norm(direction) + 1e-8)

    x, y, z = direction[0], direction[1], direction[2]

    # SH basis function constants
    c1 = 0.282095   # Y_0^0 = 1/(2*sqrt(pi))
    c2 = 0.488603   # Y_1^m = sqrt(3/(4*pi))
    c3 = 1.092548   # Y_2^{-2,2} = sqrt(15/(4*pi))
    c4 = 0.315392   # Y_2^0 = sqrt(5/(16*pi))
    c5 = 0.546274   # Y_2^{-1,1} = sqrt(15/(4*pi)) / 2

    # Number of SH coefficients based on degree
    n_coeffs = (sh_degree + 1) ** 2

    # Initialize SH coefficients for grayscale (will be replicated for RGB)
    sh = torch.zeros(n_coeffs, device=device)

    # L=0: DC term (ambient + directional contribution)
    # The DC coefficient represents the average irradiance
    sh[0] = (ambient + intensity) / c1

    if sh_degree >= 1:
        # L=1: First-order terms (directional component)
        # These encode the dominant light direction
        sh[1] = intensity * y / c2  # Y_1^{-1}
        sh[2] = intensity * z / c2  # Y_1^0
        sh[3] = intensity * x / c2  # Y_1^1

    if sh_degree >= 2:
        # L=2: Second-order terms (additional angular detail)
        sh[4] = intensity * x * y / c3           # Y_2^{-2}
        sh[5] = intensity * y * z / c3           # Y_2^{-1}
        sh[6] = intensity * (3*z*z - 1) / (3*c4) # Y_2^0
        sh[7] = intensity * x * z / c3           # Y_2^1
        sh[8] = intensity * (x*x - y*y) / (2*c5) # Y_2^2

    # Replicate for RGB channels (sun is assumed white by default)
    # Shape: [3, n_coeffs]
    sh_rgb = sh.unsqueeze(0).repeat(3, 1)

    return sh_rgb


def create_sun_sh_with_color(direction: torch.Tensor,
                              sun_intensity: torch.Tensor,
                              sky_color: torch.Tensor,
                              sh_degree: int = 2) -> torch.Tensor:
    """
    Create SH coefficients for sun lighting with separate sun intensity and sky color.

    This is a more advanced version that allows for:
    - Learnable sun intensity per channel
    - Learnable ambient sky color

    Args:
        direction: Normalized sun direction vector [3]
        sun_intensity: Sun intensity per RGB channel [3] (learnable)
        sky_color: Ambient sky color [3] (learnable)
        sh_degree: SH degree (0, 1, or 2)

    Returns:
        SH coefficients [3, (sh_degree+1)^2]
    """
    device = direction.device

    # Ensure direction is normalized
    direction = direction / (torch.norm(direction) + 1e-8)

    x, y, z = direction[0], direction[1], direction[2]

    # SH basis function constants
    c1 = 0.282095
    c2 = 0.488603
    c3 = 1.092548
    c4 = 0.315392
    c5 = 0.546274

    n_coeffs = (sh_degree + 1) ** 2

    # Initialize SH coefficients for each RGB channel
    sh_rgb = torch.zeros(3, n_coeffs, device=device)

    for ch in range(3):
        intensity = sun_intensity[ch]
        ambient = sky_color[ch]

        # L=0: DC term
        sh_rgb[ch, 0] = (ambient + intensity) / c1

        if sh_degree >= 1:
            # L=1: Directional terms
            sh_rgb[ch, 1] = intensity * y / c2
            sh_rgb[ch, 2] = intensity * z / c2
            sh_rgb[ch, 3] = intensity * x / c2

        if sh_degree >= 2:
            # L=2: Second-order terms
            sh_rgb[ch, 4] = intensity * x * y / c3
            sh_rgb[ch, 5] = intensity * y * z / c3
            sh_rgb[ch, 6] = intensity * (3*z*z - 1) / (3*c4)
            sh_rgb[ch, 7] = intensity * x * z / c3
            sh_rgb[ch, 8] = intensity * (x*x - y*y) / (2*c5)

    return sh_rgb


def compute_sun_color_from_elevation(elevation_deg: float, device: str = "cuda") -> torch.Tensor:
    """
    Compute physically-based sun color from elevation angle using atmospheric scattering.

    Based on the Nishita sky model - sunlight color changes with elevation due to
    Rayleigh and Mie scattering through the atmosphere. Lower sun = more atmosphere
    traversed = redder light.

    This is a simplified polynomial approximation fitted to the Nishita model.

    Args:
        elevation_deg: Sun elevation angle in degrees (0 = horizon, 90 = zenith)
        device: Device for output tensor

    Returns:
        Sun color multiplier [3] for RGB channels (normalized, multiply with intensity)
    """
    # Clamp elevation to valid range
    elevation = max(0.0, min(90.0, elevation_deg))

    # Convert to radians for calculations
    elev_rad = elevation * np.pi / 180.0

    # Approximate optical depth based on elevation
    # At horizon (0°), light travels through ~38x more atmosphere than at zenith
    # Air mass approximation: AM ≈ 1/sin(elevation) for elevation > 0
    if elevation < 5.0:
        # Very low sun - extreme reddening
        air_mass = 10.0 + (5.0 - elevation) * 2.0  # Cap at reasonable value
    else:
        air_mass = 1.0 / np.sin(elev_rad)

    # Rayleigh scattering coefficients (wavelength dependent: ~1/λ^4)
    # Blue scatters more than red, so blue is attenuated more at high air mass
    # Approximate for RGB: R=650nm, G=550nm, B=450nm
    beta_r = 0.0596  # Rayleigh scattering coefficient at sea level

    # Wavelength-dependent extinction (normalized to red channel)
    lambda_r, lambda_g, lambda_b = 650.0, 550.0, 450.0
    ext_r = (lambda_r / 650.0) ** (-4)  # = 1.0
    ext_g = (lambda_g / 650.0) ** (-4)  # ≈ 1.95
    ext_b = (lambda_b / 650.0) ** (-4)  # ≈ 4.36

    # Transmittance through atmosphere: T = exp(-beta * air_mass * ext)
    # Scale factor to get reasonable color shifts
    scale = 0.15

    t_r = np.exp(-beta_r * air_mass * ext_r * scale)
    t_g = np.exp(-beta_r * air_mass * ext_g * scale)
    t_b = np.exp(-beta_r * air_mass * ext_b * scale)

    # Normalize so that at zenith (90°) we get approximately white light
    # At zenith, air_mass ≈ 1, so normalize by zenith values
    t_r_zenith = np.exp(-beta_r * 1.0 * ext_r * scale)
    t_g_zenith = np.exp(-beta_r * 1.0 * ext_g * scale)
    t_b_zenith = np.exp(-beta_r * 1.0 * ext_b * scale)

    sun_color = torch.tensor([
        t_r / t_r_zenith,
        t_g / t_g_zenith,
        t_b / t_b_zenith
    ], dtype=torch.float32, device=device)

    # Clamp to reasonable range
    sun_color = torch.clamp(sun_color, min=0.1, max=1.5)

    return sun_color


class SunModel(torch.nn.Module):
    """
    Explicit directional sun model with sun color prior and global sky SH.

    This model combines:
    1. Explicit directional sun light with physically-based color prior
    2. Global (shared) first-order SH for sky lighting

    The lighting equation is:
        L = sun_color(elevation) * sun_intensity * max(0, N·L) + ambient + SH_sky(N)

    Note: Shadowing is handled separately using geometry-based shadow computation.

    Key features:
    - Sun color prior: Sun color is derived from elevation angle using atmospheric scattering
    - Global sky SH: Single shared SH environment for all images (enables relighting)
    - Per-image parameters: sun_intensity (scalar multiplier), ambient_color

    The sun direction is obtained from Camera objects at forward time.
    """

    def __init__(self, n_images: int, device: str = "cuda",
                 use_residual_sh: bool = True, sh_degree: int = 1):
        """
        Initialize the directional sun model with sun color prior.

        Args:
            n_images: Number of images in the dataset.
            device: Device to store tensors on.
            use_residual_sh: Whether to use global sky SH (kept for API compatibility).
            sh_degree: Degree of spherical harmonics (default 1 = 4 coefficients for global sky).
        """
        super().__init__()

        self.n_images = n_images
        self.device = device
        self.use_residual_sh = use_residual_sh
        self.sh_degree = sh_degree
        self.n_sh_coeffs = (sh_degree + 1) ** 2  # 4 for degree 1

        # Learnable sun intensity multiplier per image (scalar, color from elevation prior)
        # Shape: [n_images, 1]
        self.sun_intensity_multiplier = nn.Parameter(
            torch.ones(self.n_images, 1, device=device) * 3.0
        )

        # Learnable color correction for sun (small adjustment to physical prior)
        # Initialized to [1, 1, 1] = no correction, learned as multiplier
        # Shape: [n_images, 3]
        self.sun_color_correction = nn.Parameter(
            torch.ones(self.n_images, 3, device=device)
        )

        # Learnable ambient color per image (sky/indirect illumination)
        # Shape: [n_images, 3] for RGB
        self.ambient_color = nn.Parameter(
            torch.ones(self.n_images, 3, device=device) * 0.3
        )

        # GLOBAL sky SH coefficients (shared across all images)
        # This enables relighting with novel environments
        # Shape: [3, n_sh_coeffs] - initialized with slight sky gradient (blue at top)
        if use_residual_sh:
            # Initialize with slight upward bias for sky
            init_sh = torch.zeros(3, self.n_sh_coeffs, device=device)
            # DC term - base sky color
            init_sh[0, 0] = 0.2  # R
            init_sh[1, 0] = 0.25  # G
            init_sh[2, 0] = 0.4  # B (more blue)
            if self.n_sh_coeffs >= 4:
                # L1 z-component - gradient from ground to sky
                init_sh[2, 2] = 0.1  # Blue increases upward

            self.sky_sh = nn.Parameter(init_sh)
            print(f"SunModel: Using GLOBAL sky SH with {self.n_sh_coeffs} coefficients (degree {sh_degree})")

        # Keep legacy attribute for compatibility
        self.sun_intensity = self.sun_intensity_multiplier  # Alias

    def evaluate_sh(self, sh_coeffs: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spherical harmonics at given normal directions.

        Args:
            sh_coeffs: SH coefficients [3, n_coeffs] for RGB channels
            normals: Normal vectors [N, 3]

        Returns:
            Evaluated SH values [N, 3]
        """
        # Normalize normals
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]

        # SH basis functions (real spherical harmonics)
        N = normals.shape[0]
        result = torch.zeros(N, 3, device=normals.device)

        for ch in range(3):
            coeffs = sh_coeffs[ch]  # [n_coeffs]

            # L=0 (DC)
            val = coeffs[0] * 0.282095  # Y_0^0

            if self.sh_degree >= 1 and self.n_sh_coeffs >= 4:
                # L=1
                val = val + coeffs[1] * 0.488603 * y  # Y_1^{-1}
                val = val + coeffs[2] * 0.488603 * z  # Y_1^0
                val = val + coeffs[3] * 0.488603 * x  # Y_1^1

            if self.sh_degree >= 2 and self.n_sh_coeffs >= 9:
                # L=2
                val = val + coeffs[4] * 1.092548 * x * y  # Y_2^{-2}
                val = val + coeffs[5] * 1.092548 * y * z  # Y_2^{-1}
                val = val + coeffs[6] * 0.315392 * (3 * z * z - 1)  # Y_2^0
                val = val + coeffs[7] * 1.092548 * x * z  # Y_2^1
                val = val + coeffs[8] * 0.546274 * (x * x - y * y)  # Y_2^2

            result[:, ch] = val

        return result

    def forward(self, image_idx: int, normal_vectors: torch.Tensor,
                sun_direction: torch.Tensor = None,
                sun_elevation: float = None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute direct illumination using sun color prior and global sky SH.

        Implements: intensity = sun_color(elev) * intensity * max(0, N·L) + ambient + SH_sky(N)

        Args:
            image_idx: Index of the image in the dataset.
            normal_vectors: Surface normals [N, 3] (should be normalized).
            sun_direction: Sun direction vector [3] from camera. Required.
            sun_elevation: Sun elevation angle in degrees. If None, uses default (45°).

        Returns:
            Tuple of:
            - intensity: HDR intensity values [N, 3] (unshadowed)
            - sun_direction: Sun direction vector [3] for shadow computation
            - lighting_components: Dict with 'direct', 'ambient', 'sky_sh', 'sun_color' for debugging
        """
        if sun_direction is None:
            raise ValueError("sun_direction must be provided from camera")

        # Ensure sun_direction is on correct device
        if not isinstance(sun_direction, torch.Tensor):
            sun_direction = torch.tensor(sun_direction, dtype=torch.float32, device=self.device)
        elif sun_direction.device != self.device:
            sun_direction = sun_direction.to(self.device)

        # Get sun color from elevation prior
        if sun_elevation is None:
            sun_elevation = 45.0  # Default to mid-day sun
        sun_color_prior = compute_sun_color_from_elevation(sun_elevation, self.device)

        # Apply learnable color correction (small adjustment to physical model)
        color_correction = torch.clamp(self.sun_color_correction[image_idx], min=0.5, max=2.0)
        sun_color = sun_color_prior * color_correction  # [3]

        # Get intensity multiplier and ambient
        intensity_mult = torch.clamp(self.sun_intensity_multiplier[image_idx], min=0.01)  # [1]
        sun_int = sun_color * intensity_mult  # [3]
        ambient = torch.clamp(self.ambient_color[image_idx], min=0.01)  # [3]

        # Normalize sun direction and normals
        sun_dir_norm = sun_direction / (torch.norm(sun_direction) + 1e-8)
        normals_norm = normal_vectors / (torch.norm(normal_vectors, dim=-1, keepdim=True) + 1e-8)

        # Lambert's cosine law: N·L
        n_dot_l = torch.sum(normals_norm * sun_dir_norm.unsqueeze(0), dim=-1, keepdim=True)  # [N, 1]
        n_dot_l = torch.clamp(n_dot_l, min=0.0)  # [N, 1]

        # Direct sun illumination: sun_color * intensity * N·L
        direct_light = n_dot_l * sun_int.unsqueeze(0)  # [N, 3]

        # Ambient term
        ambient_light = ambient.unsqueeze(0).expand(normal_vectors.shape[0], -1)  # [N, 3]

        # Global sky SH contribution (same for all images)
        if self.use_residual_sh:
            sky_light = self.evaluate_sh(self.sky_sh, normals_norm)  # [N, 3]
            sky_light = torch.clamp(sky_light, min=0.0)  # Sky light should be non-negative
        else:
            sky_light = torch.zeros_like(direct_light)

        # Total illumination
        intensity = direct_light + ambient_light + sky_light  # [N, 3]
        intensity = torch.clamp(intensity, min=0.0)  # Ensure non-negative

        lighting_components = {
            'direct': direct_light,
            'ambient': ambient_light,
            'residual': sky_light,  # Keep 'residual' key for compatibility
            'sky_sh': sky_light,
            'n_dot_l': n_dot_l,
            'sun_color': sun_int,
            'sun_color_prior': sun_color_prior,
        }

        return intensity, sun_dir_norm, lighting_components

    def get_sun_intensity(self, image_idx: int, sun_elevation: float = None) -> torch.Tensor:
        """Get the sun intensity (with color prior) for a specific image."""
        if sun_elevation is None:
            sun_elevation = 45.0
        sun_color_prior = compute_sun_color_from_elevation(sun_elevation, self.device)
        color_correction = torch.clamp(self.sun_color_correction[image_idx], min=0.5, max=2.0)
        intensity_mult = torch.clamp(self.sun_intensity_multiplier[image_idx], min=0.01)
        return sun_color_prior * color_correction * intensity_mult

    def get_ambient(self, image_idx: int) -> torch.Tensor:
        """Get the ambient color for a specific image."""
        return torch.clamp(self.ambient_color[image_idx], min=0.01)

    def get_residual_sh(self, image_idx: int = None) -> torch.Tensor:
        """Get the global sky SH coefficients (same for all images)."""
        if self.use_residual_sh:
            return self.sky_sh
        else:
            return None

    def get_sky_sh(self) -> torch.Tensor:
        """Get the global sky SH coefficients."""
        if self.use_residual_sh:
            return self.sky_sh
        return None


def compute_sun_sh_simple(direction: torch.Tensor, intensity: float = 3.0,
                          ambient: float = 0.3) -> torch.Tensor:
    """
    Simple function to compute SH coefficients from sun direction.

    This is a convenience function for when you don't need the full SunModel class.

    Args:
        direction: Sun direction vector [3]
        intensity: Sun intensity scalar
        ambient: Ambient light level

    Returns:
        SH coefficients [3, 9] for degree-2 SH
    """
    return direction_to_sh_coefficients(direction, intensity, ambient, sh_degree=2)

