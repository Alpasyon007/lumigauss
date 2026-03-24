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
    Explicit directional sun model for outdoor scenes.

    Supports two parameterisation modes controlled by ``scene_sh``:

    **Per-image mode** (``scene_sh=False``, default / original):
        Each training image has its own learnable ``sun_intensity_multiplier``,
        ``sun_color_correction`` and ``ambient_color``.

    **Scene-global SH mode** (``scene_sh=True``, ``--scene_lighting_sh``):
        Those three quantities are represented as low-order spherical-harmonic
        functions of the sun direction, making them scene-specific rather than
        view-specific.  This reduces parameters to O(1), enforces physical
        consistency across similar sun positions, and enables evaluation at
        novel sun directions.

    In both modes the lighting equation is:
        L = sun_color(θ) · I · max(0, N·L) + A + SH_sky(N)

    Shadowing is handled separately using geometry-based shadow computation.
    """

    def __init__(self, n_images: int, device: str = "cuda",
                 use_residual_sh: bool = True, sh_degree: int = 1,
                 scene_sh: bool = False, param_sh_degree: int = 2):
        """
        Args:
            n_images: Number of images in the dataset.
            device: Device to store tensors on.
            use_residual_sh: Whether to use global sky SH for indirect light.
            sh_degree: Degree of sky SH (default 1 → 4 coefficients).
            scene_sh: If True, use scene-global SH parameterisation for
                      intensity / colour / ambient.  If False (default),
                      use the original per-image parameters.
            param_sh_degree: (scene_sh only) Degree of the lighting-parameter
                             SH functions (default 2 → 9 coefficients).
        """
        super().__init__()

        self.n_images = n_images
        self.device = device
        self.use_residual_sh = use_residual_sh
        self.sh_degree = sh_degree
        self.n_sh_coeffs = (sh_degree + 1) ** 2
        self.scene_sh = scene_sh

        if scene_sh:
            # ---------- Scene-global SH mode ----------
            self.param_sh_degree = param_sh_degree
            self.n_param_sh_coeffs = (param_sh_degree + 1) ** 2

            # Sun intensity multiplier  I(ω_sun) → scalar
            init_intensity = torch.zeros(1, self.n_param_sh_coeffs, device=device)
            init_intensity[0, 0] = 4.0 / 0.282095
            self.intensity_sh = nn.Parameter(init_intensity)

            # Sun colour correction  C(ω_sun) → RGB
            init_color = torch.zeros(3, self.n_param_sh_coeffs, device=device)
            init_color[:, 0] = 1.0 / 0.282095
            self.color_correction_sh = nn.Parameter(init_color)

            # Ambient colour  A(ω_sun) → RGB
            init_ambient = torch.zeros(3, self.n_param_sh_coeffs, device=device)
            init_ambient[:, 0] = 0.15 / 0.282095
            self.ambient_sh = nn.Parameter(init_ambient)

            print(f"SunModel: Scene-global lighting SH with {self.n_param_sh_coeffs} coefficients "
                  f"(degree {param_sh_degree}) for intensity / colour / ambient")
        else:
            # ---------- Original per-image mode ----------
            self.param_sh_degree = 0
            self.n_param_sh_coeffs = 0

            self.sun_intensity_multiplier = nn.Parameter(
                torch.ones(self.n_images, 1, device=device) * 4.0
            )
            self.sun_color_correction = nn.Parameter(
                torch.ones(self.n_images, 3, device=device)
            )
            self.ambient_color = nn.Parameter(
                torch.ones(self.n_images, 3, device=device) * 0.15
            )
            # Legacy alias
            self.sun_intensity = self.sun_intensity_multiplier

        # ----- Global sky SH (normal-dependent indirect illumination) -----
        if use_residual_sh:
            init_sh = torch.zeros(3, self.n_sh_coeffs, device=device)
            init_sh[0, 0] = 0.05
            init_sh[1, 0] = 0.07
            init_sh[2, 0] = 0.1
            if self.n_sh_coeffs >= 4:
                init_sh[2, 2] = 0.02
            self.sky_sh = nn.Parameter(init_sh)
            print(f"SunModel: Using GLOBAL sky SH with {self.n_sh_coeffs} coefficients (degree {sh_degree})")

    # ------------------------------------------------------------------
    #  SH evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_sh_at_direction(sh_coeffs: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Evaluate SH coefficients at a single normalised direction.  [C,K]→[C]"""
        x, y, z = direction[0], direction[1], direction[2]
        K = sh_coeffs.shape[1]

        result = sh_coeffs[:, 0] * 0.282095
        if K >= 4:
            result = result + sh_coeffs[:, 1] * 0.488603 * y
            result = result + sh_coeffs[:, 2] * 0.488603 * z
            result = result + sh_coeffs[:, 3] * 0.488603 * x
        if K >= 9:
            result = result + sh_coeffs[:, 4] * 1.092548 * x * y
            result = result + sh_coeffs[:, 5] * 1.092548 * y * z
            result = result + sh_coeffs[:, 6] * 0.315392 * (3 * z * z - 1)
            result = result + sh_coeffs[:, 7] * 1.092548 * x * z
            result = result + sh_coeffs[:, 8] * 0.546274 * (x * x - y * y)
        return result

    def evaluate_sh(self, sh_coeffs: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        """Evaluate SH coefficients at a batch of normal directions.  [3,K],[N,3]→[N,3]"""
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]

        N = normals.shape[0]
        result = torch.zeros(N, 3, device=normals.device)

        for ch in range(3):
            coeffs = sh_coeffs[ch]
            val = coeffs[0] * 0.282095
            if self.sh_degree >= 1 and self.n_sh_coeffs >= 4:
                val = val + coeffs[1] * 0.488603 * y
                val = val + coeffs[2] * 0.488603 * z
                val = val + coeffs[3] * 0.488603 * x
            if self.sh_degree >= 2 and self.n_sh_coeffs >= 9:
                val = val + coeffs[4] * 1.092548 * x * y
                val = val + coeffs[5] * 1.092548 * y * z
                val = val + coeffs[6] * 0.315392 * (3 * z * z - 1)
                val = val + coeffs[7] * 1.092548 * x * z
                val = val + coeffs[8] * 0.546274 * (x * x - y * y)
            result[:, ch] = val

        return result

    # ------------------------------------------------------------------
    #  Internal:  compute intensity / colour / ambient
    # ------------------------------------------------------------------

    def _get_lighting_params(self, image_idx: int,
                             sun_direction: torch.Tensor,
                             sun_elevation: float = None):
        """
        Returns (sun_int [3], ambient [3], sun_color_prior [3]).

        Dispatches to scene-global SH or per-image parameters depending on
        ``self.scene_sh``.
        """
        sun_dir_norm = sun_direction / (torch.norm(sun_direction) + 1e-8)

        if sun_elevation is None:
            sun_elevation = 45.0
        sun_color_prior = compute_sun_color_from_elevation(sun_elevation, self.device)

        if self.scene_sh:
            intensity_mult = self._evaluate_sh_at_direction(self.intensity_sh, sun_dir_norm)  # [1]
            intensity_mult = torch.clamp(intensity_mult, min=0.01)
            color_correction = self._evaluate_sh_at_direction(self.color_correction_sh, sun_dir_norm)  # [3]
            color_correction = torch.clamp(color_correction, min=0.5, max=2.0)
            ambient = self._evaluate_sh_at_direction(self.ambient_sh, sun_dir_norm)  # [3]
            ambient = torch.clamp(ambient, min=0.01)
        else:
            intensity_mult = torch.clamp(self.sun_intensity_multiplier[image_idx], min=0.01)  # [1]
            color_correction = torch.clamp(self.sun_color_correction[image_idx], min=0.5, max=2.0)  # [3]
            ambient = torch.clamp(self.ambient_color[image_idx], min=0.01)  # [3]

        sun_int = sun_color_prior * color_correction * intensity_mult  # [3]
        return sun_int, ambient, sun_color_prior

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, image_idx: int, normal_vectors: torch.Tensor,
                sun_direction: torch.Tensor = None,
                sun_elevation: float = None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute direct illumination.

        Args:
            image_idx: Image index (used only in per-image mode).
            normal_vectors: Surface normals [N, 3].
            sun_direction: Sun direction vector [3] from camera.  Required.
            sun_elevation: Sun elevation in degrees.  None → 45°.

        Returns:
            (intensity [N,3], sun_direction [3], lighting_components dict)
        """
        if sun_direction is None:
            raise ValueError("sun_direction must be provided from camera")

        if not isinstance(sun_direction, torch.Tensor):
            sun_direction = torch.tensor(sun_direction, dtype=torch.float32, device=self.device)
        elif sun_direction.device != self.device:
            sun_direction = sun_direction.to(self.device)

        sun_dir_norm = sun_direction / (torch.norm(sun_direction) + 1e-8)
        normals_norm = normal_vectors / (torch.norm(normal_vectors, dim=-1, keepdim=True) + 1e-8)

        sun_int, ambient, sun_color_prior = self._get_lighting_params(
            image_idx, sun_direction, sun_elevation
        )

        # Lambert N·L
        n_dot_l = torch.sum(normals_norm * sun_dir_norm.unsqueeze(0), dim=-1, keepdim=True)
        n_dot_l = torch.clamp(n_dot_l, min=0.0)

        direct_light = n_dot_l * sun_int.unsqueeze(0)
        ambient_light = ambient.unsqueeze(0).expand(normal_vectors.shape[0], -1)

        if self.use_residual_sh:
            sky_light = self.evaluate_sh(self.sky_sh, normals_norm)
            sky_light = torch.clamp(sky_light, min=0.0)
        else:
            sky_light = torch.zeros_like(direct_light)

        intensity = torch.clamp(direct_light + ambient_light + sky_light, min=0.0)

        lighting_components = {
            'direct': direct_light,
            'ambient': ambient_light,
            'residual': sky_light,
            'sky_sh': sky_light,
            'n_dot_l': n_dot_l,
            'sun_color': sun_int,
            'sun_color_prior': sun_color_prior,
        }

        return intensity, sun_dir_norm, lighting_components

    # ------------------------------------------------------------------
    #  Public accessors  (accept both image_idx and sun_direction)
    # ------------------------------------------------------------------

    def get_sun_intensity(self, image_idx_or_dir, sun_elevation: float = None) -> torch.Tensor:
        """
        Get effective sun colour × intensity.

        In per-image mode ``image_idx_or_dir`` is an int index.
        In scene-SH mode it is a sun direction tensor [3].
        A tensor is also accepted in per-image mode (it will be treated as
        a direction and a dummy index of 0 is used internally).
        """
        if isinstance(image_idx_or_dir, torch.Tensor):
            sun_dir = image_idx_or_dir
            if sun_dir.device != self.device:
                sun_dir = sun_dir.to(self.device)
            image_idx = 0
        elif isinstance(image_idx_or_dir, (list, tuple)):
            sun_dir = torch.tensor(image_idx_or_dir, dtype=torch.float32, device=self.device)
            image_idx = 0
        else:
            image_idx = image_idx_or_dir
            # Need a direction for the colour prior; use zenith as default
            sun_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        sun_int, _, _ = self._get_lighting_params(image_idx, sun_dir, sun_elevation)
        return sun_int

    def get_ambient(self, image_idx_or_dir, sun_elevation: float = None) -> torch.Tensor:
        """Get ambient colour.  Same argument convention as get_sun_intensity."""
        if isinstance(image_idx_or_dir, torch.Tensor):
            sun_dir = image_idx_or_dir
            if sun_dir.device != self.device:
                sun_dir = sun_dir.to(self.device)
            image_idx = 0
        elif isinstance(image_idx_or_dir, (list, tuple)):
            sun_dir = torch.tensor(image_idx_or_dir, dtype=torch.float32, device=self.device)
            image_idx = 0
        else:
            image_idx = image_idx_or_dir
            sun_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        _, ambient, _ = self._get_lighting_params(image_idx, sun_dir, sun_elevation)
        return ambient

    def get_residual_sh(self, image_idx: int = None) -> torch.Tensor:
        """Get the global sky SH coefficients (same for all images)."""
        if self.use_residual_sh:
            return self.sky_sh
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

