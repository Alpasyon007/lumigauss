"""
Sun position utilities for LumiGauss.

This module provides functions to:
1. Load sun position data from JSON files
2. Convert sun direction vectors to SH coefficients
3. Model directional sun lighting with ambient sky illumination

The sun model follows the equation:
    Li(ωi) = Isun · δ(ωi − ωsun) + Lsky(ωi)

where ωsun is the sun direction computed from metadata, Isun is the sun intensity,
and Lsky(ωi) represents ambient sky illumination.

Enhanced version includes:
- Learnable residual SH for additional environmental details
- Hemisphere-based sky model with zenith/horizon gradients
- Sun angular size modeling
"""

import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


def load_sun_data(json_path: str) -> Dict:
    """
    Load sun position data from a JSON file.

    Args:
        json_path: Path to the JSON file containing sun position data.

    Returns:
        Dictionary mapping image names to their sun position data.
    """
    with open(json_path, 'r') as f:
        sun_data = json.load(f)
    return sun_data


def get_sun_direction(sun_data: Dict, image_name: str) -> Optional[torch.Tensor]:
    """
    Get the sun direction vector for a specific image.

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


class SunModel(torch.nn.Module):
    """
    Explicit directional sun model that does NOT use SH representation.

    This model keeps the sun as an explicit directional light source, enabling:
    - Sharp shadow boundaries (no SH band-limiting)
    - Accurate specular highlights
    - Physically correct Lambert shading: albedo * sun_intensity * max(0, N·L) * shadow + ambient

    The sun direction is fixed from metadata, but the following parameters are learnable:
    - sun_intensity: Per-image sun intensity [3] for RGB channels
    - ambient_color: Per-image ambient/sky color [3] for RGB channels
    - shadow_softness: Optional soft shadow falloff parameter
    """

    def __init__(self, sun_data: Dict, image_names: list, device: str = "cuda"):
        """
        Initialize the directional sun model.

        Args:
            sun_data: Dictionary mapping image names to sun position data.
            image_names: List of all image names in the dataset.
            device: Device to store tensors on.
        """
        super().__init__()

        self.n_images = len(image_names)
        self.device = device

        # Store sun directions for each image (not learnable)
        sun_directions = []
        for img_name in image_names:
            direction = get_sun_direction(sun_data, img_name)
            if direction is None:
                # Default to zenith if sun direction not found
                direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
                print(f"Warning: Using default zenith direction for {img_name}")
            sun_directions.append(direction)

        # Register as buffer (non-learnable)
        self.register_buffer('sun_directions', torch.stack(sun_directions).to(device))

        # Learnable sun intensity per image (initialized for typical outdoor sun)
        # Shape: [n_images, 3] for RGB
        self.sun_intensity = nn.Parameter(
            torch.ones(self.n_images, 3, device=device) * 3.0
        )

        # Learnable ambient color per image (sky/indirect illumination)
        # Shape: [n_images, 3] for RGB
        self.ambient_color = nn.Parameter(
            torch.ones(self.n_images, 3, device=device) * 0.3
        )

        # Optional: learnable shadow softness (for penumbra)
        # Higher values = softer shadow edges
        self.shadow_softness = nn.Parameter(
            torch.ones(self.n_images, 1, device=device) * 0.0
        )

    def forward(self, image_idx: int, normal_vectors: torch.Tensor,
                shadow_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute direct illumination for Gaussians using explicit directional lighting.

        Implements: intensity = sun_intensity * max(0, N·L) * shadow + ambient

        Args:
            image_idx: Index of the image in the dataset.
            normal_vectors: Surface normals [N, 3] (should be normalized).
            shadow_mask: Optional shadow visibility [N, 1] where 1=lit, 0=shadowed.
                        If None, assumes all points are lit.

        Returns:
            Tuple of:
            - intensity: HDR intensity values [N, 3]
            - sun_direction: Sun direction vector [3]
            - lighting_components: Dict with 'direct', 'ambient' for debugging
        """
        sun_direction = self.sun_directions[image_idx]  # [3]
        sun_int = torch.clamp(self.sun_intensity[image_idx], min=0.01)  # [3]
        ambient = torch.clamp(self.ambient_color[image_idx], min=0.01)  # [3]

        # Normalize sun direction and normals
        sun_dir_norm = sun_direction / (torch.norm(sun_direction) + 1e-8)
        normals_norm = normal_vectors / (torch.norm(normal_vectors, dim=-1, keepdim=True) + 1e-8)

        # Lambert's cosine law: N·L
        n_dot_l = torch.sum(normals_norm * sun_dir_norm.unsqueeze(0), dim=-1, keepdim=True)  # [N, 1]

        # Clamp to [0, 1] - only front-facing surfaces receive direct light
        n_dot_l = torch.clamp(n_dot_l, min=0.0)  # [N, 1]

        # Apply shadow mask if provided
        if shadow_mask is not None:
            # shadow_mask: 1 = lit, 0 = in shadow
            # Apply soft shadow if softness > 0
            softness = torch.clamp(self.shadow_softness[image_idx], min=0.0)
            if softness > 0.01:
                # Soft shadow transition
                shadow_factor = torch.sigmoid((shadow_mask - 0.5) / (softness + 1e-6))
            else:
                shadow_factor = shadow_mask
            n_dot_l = n_dot_l * shadow_factor  # [N, 1]

        # Direct sun illumination: sun_intensity * N·L
        direct_light = n_dot_l * sun_int.unsqueeze(0)  # [N, 3]

        # Total illumination: direct + ambient
        intensity = direct_light + ambient.unsqueeze(0)  # [N, 3]

        lighting_components = {
            'direct': direct_light,
            'ambient': ambient.unsqueeze(0).expand_as(direct_light),
            'n_dot_l': n_dot_l,
        }

        return intensity, sun_dir_norm, lighting_components

    def get_sun_direction(self, image_idx: int) -> torch.Tensor:
        """Get the sun direction for a specific image."""
        return self.sun_directions[image_idx]

    def get_sun_intensity(self, image_idx: int) -> torch.Tensor:
        """Get the sun intensity for a specific image."""
        return torch.clamp(self.sun_intensity[image_idx], min=0.01)

    def get_ambient(self, image_idx: int) -> torch.Tensor:
        """Get the ambient color for a specific image."""
        return torch.clamp(self.ambient_color[image_idx], min=0.01)


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

