
import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np
import torch
from gaussian_renderer import render
import uuid
from utils.image_utils import psnr, mse
from utils.loss_utils import ssim, img2mse, img2mae, mse2psnr
from argparse import ArgumentParser, Namespace
from skimage.metrics import structural_similarity as ssim_skimage
import os
from utils.sh_vis_utils import shReconstructDiffuseMap, getCoefficientsFromImage
from utils.sh_rotate_utils import Rotation
from utils.normal_utils import compute_normal_world_space
from utils.shadow_utils import compute_shadows_for_gaussians, create_sun_camera_visualization_tensor
from scene import Scene
import cv2 as _cv2
import matplotlib.pyplot as _plt
import importlib
import sys

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Try to import LPIPS for perceptual metrics
try:
    from lpipsPyTorch import lpips as compute_lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


# =============================================================================
# Metrics Data Classes
# =============================================================================

@dataclass
class ImageMetrics:
    """Metrics for a single image comparison."""
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0
    l1: float = 0.0
    l2: float = 0.0
    mse: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class LossComponents:
    """Individual loss components during training."""
    unshadowed_image: float = 0.0
    shadowed_image: float = 0.0
    l1_unshadowed: float = 0.0
    l1_shadowed: float = 0.0
    ssim_unshadowed: float = 0.0
    ssim_shadowed: float = 0.0
    normal: float = 0.0
    dist: float = 0.0
    sh_gauss: float = 0.0
    sh_env: float = 0.0
    consistency: float = 0.0
    shadow: float = 0.0
    sun_reg: float = 0.0
    sky_mask: float = 0.0
    depth_est: float = 0.0
    ao_reg: float = 0.0
    manhattan: float = 0.0
    casts_shadow_reg: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class TrainingPhase:
    """Training phase information."""
    name: str = "unknown"
    iteration: int = 0
    warmup_end: int = 0
    sh_tuning_end: int = 0

    @property
    def is_warmup(self) -> bool:
        return self.iteration < self.warmup_end

    @property
    def is_sh_tuning(self) -> bool:
        return self.warmup_end <= self.iteration <= self.sh_tuning_end

    @property
    def is_shadowed(self) -> bool:
        return self.iteration > self.sh_tuning_end

    def get_phase_name(self) -> str:
        if self.is_warmup:
            return "warmup"
        elif self.is_sh_tuning:
            return "sh_tuning"
        else:
            return "shadowed"


@dataclass
class EvaluationResult:
    """Complete evaluation result for a viewpoint."""
    viewpoint_name: str
    config_name: str  # 'train' or 'test'
    iteration: int

    # Metrics for different render types
    albedo_metrics: ImageMetrics = field(default_factory=ImageMetrics)
    shadowed_metrics: ImageMetrics = field(default_factory=ImageMetrics)
    unshadowed_metrics: ImageMetrics = field(default_factory=ImageMetrics)

    # Additional info
    num_gaussians: int = 0
    render_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint_name": self.viewpoint_name,
            "config_name": self.config_name,
            "iteration": self.iteration,
            "albedo_metrics": self.albedo_metrics.to_dict(),
            "shadowed_metrics": self.shadowed_metrics.to_dict(),
            "unshadowed_metrics": self.unshadowed_metrics.to_dict(),
            "num_gaussians": self.num_gaussians,
            "render_time_ms": self.render_time_ms
        }


@dataclass
class EvaluationSummary:
    """Summary statistics across multiple viewpoints."""
    config_name: str
    iteration: int
    num_viewpoints: int = 0

    # Aggregated metrics (mean)
    mean_psnr_albedo: float = 0.0
    mean_psnr_shadowed: float = 0.0
    mean_psnr_unshadowed: float = 0.0
    mean_ssim_albedo: float = 0.0
    mean_ssim_shadowed: float = 0.0
    mean_ssim_unshadowed: float = 0.0
    mean_lpips_albedo: float = 0.0
    mean_lpips_shadowed: float = 0.0
    mean_lpips_unshadowed: float = 0.0
    mean_l1_albedo: float = 0.0
    mean_l1_shadowed: float = 0.0
    mean_l1_unshadowed: float = 0.0

    # Standard deviations
    std_psnr_albedo: float = 0.0
    std_psnr_shadowed: float = 0.0
    std_psnr_unshadowed: float = 0.0

    # Best/worst
    best_psnr_viewpoint: str = ""
    worst_psnr_viewpoint: str = ""
    best_psnr: float = 0.0
    worst_psnr: float = float('inf')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Metrics Calculator
# =============================================================================

class MetricsCalculator:
    """Utility class for computing image quality metrics."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._lpips_net = None

    @property
    def lpips_net(self):
        """Lazy load LPIPS network."""
        if self._lpips_net is None and LPIPS_AVAILABLE:
            from lpipsPyTorch.modules.lpips import LPIPS
            self._lpips_net = LPIPS('alex', '0.1').to(self.device)
            self._lpips_net.eval()
        return self._lpips_net

    def compute_all_metrics(self, pred: torch.Tensor, gt: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> ImageMetrics:
        """Compute all image metrics between prediction and ground truth."""
        metrics = ImageMetrics()

        # Ensure proper shape [C, H, W] or [1, C, H, W]
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)

        # PSNR
        metrics.psnr = psnr(pred, gt, mask).mean().item()

        # MSE
        metrics.mse = mse(pred, gt, mask).mean().item()

        # SSIM
        metrics.ssim = ssim(pred.squeeze(0), gt.squeeze(0), mask=mask).item()

        # L1
        if mask is not None:
            metrics.l1 = (torch.abs(pred - gt) * mask).sum().item() / (mask.sum().item() * pred.shape[1] + 1e-6)
        else:
            metrics.l1 = torch.abs(pred - gt).mean().item()

        # L2
        if mask is not None:
            metrics.l2 = ((pred - gt) ** 2 * mask).sum().item() / (mask.sum().item() * pred.shape[1] + 1e-6)
        else:
            metrics.l2 = ((pred - gt) ** 2).mean().item()

        # LPIPS (perceptual)
        if LPIPS_AVAILABLE and self.lpips_net is not None:
            try:
                # LPIPS expects [N, C, H, W] in range [-1, 1]
                pred_lpips = pred * 2 - 1
                gt_lpips = gt * 2 - 1
                with torch.no_grad():
                    metrics.lpips = self.lpips_net(pred_lpips, gt_lpips).mean().item()
            except Exception as e:
                metrics.lpips = 0.0

        return metrics

    def compute_quick_metrics(self, pred: torch.Tensor, gt: torch.Tensor,
                              mask: Optional[torch.Tensor] = None) -> ImageMetrics:
        """Compute only PSNR and L1 (faster for training logging)."""
        metrics = ImageMetrics()

        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)

        metrics.psnr = psnr(pred, gt, mask).mean().item()

        if mask is not None:
            metrics.l1 = (torch.abs(pred - gt) * mask).sum().item() / (mask.sum().item() * pred.shape[1] + 1e-6)
        else:
            metrics.l1 = torch.abs(pred - gt).mean().item()

        return metrics


# =============================================================================
# Metrics Logger
# =============================================================================

class MetricsLogger:
    """Centralized logging for training and evaluation metrics."""

    def __init__(self, tb_writer: Optional[SummaryWriter], model_path: str):
        self.tb_writer = tb_writer
        self.model_path = model_path
        self.metrics_calculator = MetricsCalculator()

        # Storage for aggregating metrics
        self.training_losses: List[LossComponents] = []
        self.evaluation_results: Dict[int, List[EvaluationResult]] = defaultdict(list)
        self.evaluation_summaries: Dict[int, Dict[str, EvaluationSummary]] = {}

        # Timing
        self.iteration_times: List[float] = []

        # Create metrics output directory
        self.metrics_dir = os.path.join(model_path, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

    def log_training_losses(self, iteration: int, losses: LossComponents,
                           phase: TrainingPhase, lambdas: Dict[str, float]):
        """Log training losses to TensorBoard with proper organization."""
        if self.tb_writer is None:
            return

        phase_name = phase.get_phase_name()

        # Group 1: Total and Combined Losses
        self.tb_writer.add_scalar('Loss/total', losses.total, iteration)
        self.tb_writer.add_scalar('Loss/image_combined',
                                  losses.unshadowed_image + losses.shadowed_image, iteration)

        # Group 2: Image Reconstruction Losses
        self.tb_writer.add_scalar('Loss_Image/unshadowed', losses.unshadowed_image, iteration)
        self.tb_writer.add_scalar('Loss_Image/shadowed', losses.shadowed_image, iteration)
        self.tb_writer.add_scalar('Loss_Image/l1_unshadowed', losses.l1_unshadowed, iteration)
        self.tb_writer.add_scalar('Loss_Image/l1_shadowed', losses.l1_shadowed, iteration)

        # Group 3: Regularization Losses
        self.tb_writer.add_scalar('Loss_Regularization/normal', losses.normal, iteration)
        self.tb_writer.add_scalar('Loss_Regularization/dist', losses.dist, iteration)
        self.tb_writer.add_scalar('Loss_Regularization/sun_reg', losses.sun_reg, iteration)
        self.tb_writer.add_scalar('Loss_Regularization/sky_mask', losses.sky_mask, iteration)
        self.tb_writer.add_scalar('Loss_Regularization/depth_est', losses.depth_est, iteration)
        self.tb_writer.add_scalar('Loss_Regularization/ao_reg', losses.ao_reg, iteration)
        self.tb_writer.add_scalar('Loss_Regularization/manhattan', losses.manhattan, iteration)
        self.tb_writer.add_scalar('Loss_Regularization/casts_shadow_reg', losses.casts_shadow_reg, iteration)

        # Group 4: SH/Lighting Losses
        self.tb_writer.add_scalar('Loss_Lighting/sh_gauss', losses.sh_gauss, iteration)
        self.tb_writer.add_scalar('Loss_Lighting/sh_env', losses.sh_env, iteration)
        self.tb_writer.add_scalar('Loss_Lighting/consistency', losses.consistency, iteration)
        self.tb_writer.add_scalar('Loss_Lighting/shadow', losses.shadow, iteration)

        # Group 5: Lambda Values (loss weights)
        for name, value in lambdas.items():
            self.tb_writer.add_scalar(f'Lambda/{name}', value, iteration)

        # Group 6: Training Phase
        phase_idx = {"warmup": 0, "sh_tuning": 1, "shadowed": 2}.get(phase_name, -1)
        self.tb_writer.add_scalar('Training/phase', phase_idx, iteration)

    def log_iteration_time(self, iteration: int, elapsed_ms: float):
        """Log iteration timing."""
        self.iteration_times.append(elapsed_ms)
        if self.tb_writer:
            self.tb_writer.add_scalar('Performance/iter_time_ms', elapsed_ms, iteration)
            # Running average
            if len(self.iteration_times) >= 100:
                avg_time = sum(self.iteration_times[-100:]) / 100
                self.tb_writer.add_scalar('Performance/iter_time_avg100', avg_time, iteration)

    def log_gaussian_stats(self, iteration: int, gaussians):
        """Log Gaussian model statistics."""
        if self.tb_writer is None:
            return

        num_points = gaussians.get_xyz.shape[0]
        self.tb_writer.add_scalar('Gaussians/total_count', num_points, iteration)

        # Scale statistics
        scales = gaussians.get_scaling
        self.tb_writer.add_scalar('Gaussians/scale_mean', scales.mean().item(), iteration)
        self.tb_writer.add_scalar('Gaussians/scale_max', scales.max().item(), iteration)
        self.tb_writer.add_scalar('Gaussians/scale_min', scales.min().item(), iteration)

        # Opacity statistics
        opacity = gaussians.get_opacity
        self.tb_writer.add_scalar('Gaussians/opacity_mean', opacity.mean().item(), iteration)
        self.tb_writer.add_histogram('Gaussians/opacity_hist', opacity, iteration)

        # PBR material statistics (available regardless of training mode)
        if hasattr(gaussians, 'get_roughness') and hasattr(gaussians, 'get_metallic'):
            roughness = gaussians.get_roughness
            metallic = gaussians.get_metallic
            self.tb_writer.add_scalar('Gaussians/roughness_mean', roughness.mean().item(), iteration)
            self.tb_writer.add_scalar('Gaussians/roughness_min', roughness.min().item(), iteration)
            self.tb_writer.add_scalar('Gaussians/roughness_max', roughness.max().item(), iteration)
            self.tb_writer.add_histogram('Gaussians/roughness_hist', roughness, iteration)
            self.tb_writer.add_scalar('Gaussians/metallic_mean', metallic.mean().item(), iteration)
            self.tb_writer.add_scalar('Gaussians/metallic_min', metallic.min().item(), iteration)
            self.tb_writer.add_scalar('Gaussians/metallic_max', metallic.max().item(), iteration)
            self.tb_writer.add_histogram('Gaussians/metallic_hist', metallic, iteration)

        # For sun mode: casts_shadow statistics
        if gaussians.use_sun:
            casts_shadow = gaussians.get_casts_shadow
            sky_ratio = (casts_shadow < 0.5).float().mean().item()
            self.tb_writer.add_scalar('Gaussians/sky_ratio', sky_ratio, iteration)
            self.tb_writer.add_histogram('Gaussians/casts_shadow_hist', casts_shadow, iteration)

        # AO statistics
        if gaussians.use_ao:
            ao = gaussians.get_ao
            self.tb_writer.add_scalar('Gaussians/ao_mean', ao.mean().item(), iteration)
            self.tb_writer.add_scalar('Gaussians/ao_min', ao.min().item(), iteration)
            self.tb_writer.add_scalar('Gaussians/ao_max', ao.max().item(), iteration)
            self.tb_writer.add_histogram('Gaussians/ao_hist', ao, iteration)

    def log_sun_model_params(self, iteration: int, sun_model, emb_idx: int = 0):
        """Log sun model parameters for use_sun mode."""
        if self.tb_writer is None or sun_model is None:
            return

        # Sun intensity
        sun_intensity = sun_model.get_sun_intensity(emb_idx)
        self.tb_writer.add_scalar('SunModel/intensity_r', sun_intensity[0].item(), iteration)
        self.tb_writer.add_scalar('SunModel/intensity_g', sun_intensity[1].item(), iteration)
        self.tb_writer.add_scalar('SunModel/intensity_b', sun_intensity[2].item(), iteration)
        self.tb_writer.add_scalar('SunModel/intensity_mean', sun_intensity.mean().item(), iteration)

        # Ambient color
        ambient = sun_model.get_ambient(emb_idx)
        self.tb_writer.add_scalar('SunModel/ambient_r', ambient[0].item(), iteration)
        self.tb_writer.add_scalar('SunModel/ambient_g', ambient[1].item(), iteration)
        self.tb_writer.add_scalar('SunModel/ambient_b', ambient[2].item(), iteration)

        # Color correction
        if hasattr(sun_model, 'sun_color_correction'):
            color_corr = sun_model.sun_color_correction[emb_idx]
            self.tb_writer.add_scalar('SunModel/color_correction_r', color_corr[0].item(), iteration)
            self.tb_writer.add_scalar('SunModel/color_correction_g', color_corr[1].item(), iteration)
            self.tb_writer.add_scalar('SunModel/color_correction_b', color_corr[2].item(), iteration)

    def log_evaluation_result(self, result: EvaluationResult):
        """Log a single evaluation result (means only, no per-image TB scalars)."""
        self.evaluation_results[result.iteration].append(result)

    def compute_and_log_summary(self, iteration: int, config_name: str) -> EvaluationSummary:
        """Compute and log summary statistics for an evaluation config."""
        results = [r for r in self.evaluation_results[iteration] if r.config_name == config_name]

        if not results:
            return EvaluationSummary(config_name=config_name, iteration=iteration)

        summary = EvaluationSummary(
            config_name=config_name,
            iteration=iteration,
            num_viewpoints=len(results)
        )

        # Collect metrics
        psnr_albedo = [r.albedo_metrics.psnr for r in results]
        psnr_shadowed = [r.shadowed_metrics.psnr for r in results]
        psnr_unshadowed = [r.unshadowed_metrics.psnr for r in results]
        ssim_albedo = [r.albedo_metrics.ssim for r in results]
        ssim_shadowed = [r.shadowed_metrics.ssim for r in results]
        ssim_unshadowed = [r.unshadowed_metrics.ssim for r in results]
        lpips_albedo = [r.albedo_metrics.lpips for r in results]
        lpips_shadowed = [r.shadowed_metrics.lpips for r in results]
        lpips_unshadowed = [r.unshadowed_metrics.lpips for r in results]
        l1_albedo = [r.albedo_metrics.l1 for r in results]
        l1_shadowed = [r.shadowed_metrics.l1 for r in results]
        l1_unshadowed = [r.unshadowed_metrics.l1 for r in results]

        # Compute means
        summary.mean_psnr_albedo = np.mean(psnr_albedo)
        summary.mean_psnr_shadowed = np.mean(psnr_shadowed)
        summary.mean_psnr_unshadowed = np.mean(psnr_unshadowed)
        summary.mean_ssim_albedo = np.mean(ssim_albedo)
        summary.mean_ssim_shadowed = np.mean(ssim_shadowed)
        summary.mean_ssim_unshadowed = np.mean(ssim_unshadowed)
        summary.mean_lpips_albedo = np.mean(lpips_albedo)
        summary.mean_lpips_shadowed = np.mean(lpips_shadowed)
        summary.mean_lpips_unshadowed = np.mean(lpips_unshadowed)
        summary.mean_l1_albedo = np.mean(l1_albedo)
        summary.mean_l1_shadowed = np.mean(l1_shadowed)
        summary.mean_l1_unshadowed = np.mean(l1_unshadowed)

        # Compute std
        summary.std_psnr_albedo = np.std(psnr_albedo)
        summary.std_psnr_shadowed = np.std(psnr_shadowed)
        summary.std_psnr_unshadowed = np.std(psnr_unshadowed)

        # Best/worst
        best_idx = np.argmax(psnr_albedo)
        worst_idx = np.argmin(psnr_albedo)
        summary.best_psnr_viewpoint = results[best_idx].viewpoint_name
        summary.worst_psnr_viewpoint = results[worst_idx].viewpoint_name
        summary.best_psnr = psnr_albedo[best_idx]
        summary.worst_psnr = psnr_albedo[worst_idx]

        # Store summary
        if iteration not in self.evaluation_summaries:
            self.evaluation_summaries[iteration] = {}
        self.evaluation_summaries[iteration][config_name] = summary

        # Log to TensorBoard
        if self.tb_writer:
            prefix = f"Eval_{config_name}_Summary"

            # PSNR
            self.tb_writer.add_scalar(f'{prefix}/psnr_albedo_mean', summary.mean_psnr_albedo, iteration)
            self.tb_writer.add_scalar(f'{prefix}/psnr_shadowed_mean', summary.mean_psnr_shadowed, iteration)
            self.tb_writer.add_scalar(f'{prefix}/psnr_unshadowed_mean', summary.mean_psnr_unshadowed, iteration)

            # SSIM
            self.tb_writer.add_scalar(f'{prefix}/ssim_albedo_mean', summary.mean_ssim_albedo, iteration)
            self.tb_writer.add_scalar(f'{prefix}/ssim_shadowed_mean', summary.mean_ssim_shadowed, iteration)
            self.tb_writer.add_scalar(f'{prefix}/ssim_unshadowed_mean', summary.mean_ssim_unshadowed, iteration)

            # LPIPS
            self.tb_writer.add_scalar(f'{prefix}/lpips_albedo_mean', summary.mean_lpips_albedo, iteration)
            self.tb_writer.add_scalar(f'{prefix}/lpips_shadowed_mean', summary.mean_lpips_shadowed, iteration)
            self.tb_writer.add_scalar(f'{prefix}/lpips_unshadowed_mean', summary.mean_lpips_unshadowed, iteration)

            # L1
            self.tb_writer.add_scalar(f'{prefix}/l1_albedo_mean', summary.mean_l1_albedo, iteration)
            self.tb_writer.add_scalar(f'{prefix}/l1_shadowed_mean', summary.mean_l1_shadowed, iteration)
            self.tb_writer.add_scalar(f'{prefix}/l1_unshadowed_mean', summary.mean_l1_unshadowed, iteration)

        return summary

    def save_evaluation_json(self, iteration: int):
        """Save evaluation summary (means only) to JSON file."""
        summaries = self.evaluation_summaries.get(iteration, {})

        output = {
            "iteration": iteration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summaries": {k: v.to_dict() for k, v in summaries.items()}
        }

        filepath = os.path.join(self.metrics_dir, f"evaluation_{iteration:06d}.json")
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

    def print_evaluation_summary(self, iteration: int, config_name: str):
        """Print formatted evaluation summary (mean metrics only)."""
        summary = self.evaluation_summaries.get(iteration, {}).get(config_name)
        if summary is None:
            return

        print(f"\n{'='*70}")
        print(f"[ITER {iteration}] Evaluation Summary - {config_name.upper()} ({summary.num_viewpoints} viewpoints)")
        print(f"{'='*70}")
        print(f"  {'Metric':<12} {'Albedo':>10} {'Unshadowed':>12} {'Shadowed':>12}")
        print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
        print(f"  {'PSNR (dB)':<12} {summary.mean_psnr_albedo:>10.2f} {summary.mean_psnr_unshadowed:>12.2f} {summary.mean_psnr_shadowed:>12.2f}")
        print(f"  {'SSIM':<12} {summary.mean_ssim_albedo:>10.4f} {summary.mean_ssim_unshadowed:>12.4f} {summary.mean_ssim_shadowed:>12.4f}")
        if LPIPS_AVAILABLE:
            print(f"  {'LPIPS':<12} {summary.mean_lpips_albedo:>10.4f} {summary.mean_lpips_unshadowed:>12.4f} {summary.mean_lpips_shadowed:>12.4f}")
        print(f"  {'L1':<12} {summary.mean_l1_albedo:>10.5f} {summary.mean_l1_unshadowed:>12.5f} {summary.mean_l1_shadowed:>12.5f}")
        print(f"{'='*70}\n")


def create_loss_components(
    unshadowed_image_loss: torch.Tensor,
    shadowed_image_loss: torch.Tensor,
    l1_unshadowed: torch.Tensor,
    l1_shadowed: torch.Tensor,
    normal_loss: torch.Tensor,
    dist_loss: torch.Tensor,
    sh_gauss_loss: torch.Tensor,
    sh_env_loss: torch.Tensor,
    consistency_loss: torch.Tensor,
    shadow_loss: torch.Tensor,
    sun_reg_loss: torch.Tensor = None,
    sky_mask_loss: torch.Tensor = None,
    depth_est_loss: torch.Tensor = None,
    ao_reg_loss: torch.Tensor = None,
    manhattan_loss: torch.Tensor = None,
    casts_shadow_reg_loss: torch.Tensor = None,
    total_loss: torch.Tensor = None
) -> LossComponents:
    """Helper to create LossComponents from tensor values."""
    return LossComponents(
        unshadowed_image=unshadowed_image_loss.item() if torch.is_tensor(unshadowed_image_loss) else unshadowed_image_loss,
        shadowed_image=shadowed_image_loss.item() if torch.is_tensor(shadowed_image_loss) else shadowed_image_loss,
        l1_unshadowed=l1_unshadowed.item() if torch.is_tensor(l1_unshadowed) else l1_unshadowed,
        l1_shadowed=l1_shadowed.item() if torch.is_tensor(l1_shadowed) else l1_shadowed,
        normal=normal_loss.item() if torch.is_tensor(normal_loss) else normal_loss,
        dist=dist_loss.item() if torch.is_tensor(dist_loss) else dist_loss,
        sh_gauss=sh_gauss_loss.item() if torch.is_tensor(sh_gauss_loss) else sh_gauss_loss,
        sh_env=sh_env_loss.item() if torch.is_tensor(sh_env_loss) else sh_env_loss,
        consistency=consistency_loss.item() if torch.is_tensor(consistency_loss) else consistency_loss,
        shadow=shadow_loss.item() if torch.is_tensor(shadow_loss) else shadow_loss,
        sun_reg=sun_reg_loss.item() if sun_reg_loss is not None and torch.is_tensor(sun_reg_loss) else (sun_reg_loss or 0.0),
        sky_mask=sky_mask_loss.item() if sky_mask_loss is not None and torch.is_tensor(sky_mask_loss) else (sky_mask_loss or 0.0),
        depth_est=depth_est_loss.item() if depth_est_loss is not None and torch.is_tensor(depth_est_loss) else (depth_est_loss or 0.0),
        ao_reg=ao_reg_loss.item() if ao_reg_loss is not None and torch.is_tensor(ao_reg_loss) else (ao_reg_loss or 0.0),
        manhattan=manhattan_loss.item() if manhattan_loss is not None and torch.is_tensor(manhattan_loss) else (manhattan_loss or 0.0),
        casts_shadow_reg=casts_shadow_reg_loss.item() if casts_shadow_reg_loss is not None and torch.is_tensor(casts_shadow_reg_loss) else (casts_shadow_reg_loss or 0.0),
        total=total_loss.item() if total_loss is not None and torch.is_tensor(total_loss) else (total_loss or 0.0)
    )

def update_lambdas(iteration, opt, use_sun=False) -> Dict[str, float]:
    """
    Returns the loss lambda values based on the current iteration and shadowed/unshadowed mode.

    For use_sun mode, the training schedule is fundamentally different:
    - Shadowed rendering is used from the START (warmup included)
    - This prevents shadows from being baked into albedo
    - The unshadowed loss is only a small regularizer, never the primary loss
    - The SH_gauss tuning phase is skipped (not applicable for explicit sun model)

    Returns:
        dict: A dictionary containing adjusted lambda values.
    """

    lambda_normal = opt.lambda_normal if iteration > opt.start_regularization else 0.0
    lambda_dist = opt.lambda_dist if iteration > opt.start_regularization else 0.0

    if use_sun:
        # ===== USE_SUN MODE =====
        # Key insight: We MUST use shadowed rendering from the start.
        # If we train unshadowed against GT (which has real shadows), the model
        # will bake shadow patterns into albedo since there's no shadow mask to
        # explain the dark regions.
        #
        # Training phases for use_sun:
        # 1. Early warmup (0 → warmup): Shadowed + unshadowed, build geometry
        # 2. Post-warmup (warmup → start_shadowed): Shadowed only, refine lighting
        # 3. Full training (start_shadowed+): Shadowed + small unshadowed regularizer

        if iteration < opt.warmup:
            # Early phase: primarily shadowed, with unshadowed as regularizer
            # Shadowed fits GT (which has shadows), unshadowed encourages clean albedo
            shadowed_image_loss_lambda = 1.0
            unshadowed_image_loss_lambda = 0.01  # Small regularizer to prevent albedo collapse
            consistency_loss_lambda = 0.0
            sh_gauss_lambda = 0.0
            shadow_loss_lambda = 0.0
            env_loss_lambda = 0.0

        elif opt.warmup <= iteration <= opt.start_shadowed:
            # Refinement: shadowed only, let shadow mask and lighting settle
            # SH_gauss tuning is not applicable for sun model
            shadowed_image_loss_lambda = 1.0
            unshadowed_image_loss_lambda = 0.0
            consistency_loss_lambda = 0.0
            sh_gauss_lambda = 0.0
            shadow_loss_lambda = 0.0
            env_loss_lambda = 0.0

            # Keep 2DGS regularization on for sun mode (normals matter for N·L)
            # lambda_normal and lambda_dist stay as set above

        elif iteration > opt.start_shadowed:
            # Full training: shadowed primary only (disable unshadowed GT supervision)
            shadowed_image_loss_lambda = 1.0
            unshadowed_image_loss_lambda = 0.0 #Old was 0.01
            consistency_loss_lambda = 0.0
            sh_gauss_lambda = 0.0
            shadow_loss_lambda = 0.0
            env_loss_lambda = 0.0
        else:
            raise ValueError("Iteration doesn't fit into any defined conditions - verify logic.")

    else:
        # ===== ORIGINAL SH MODE =====
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


def get_training_phase(iteration: int, opt) -> TrainingPhase:
    """Get the current training phase information."""
    return TrainingPhase(
        name="training",
        iteration=iteration,
        warmup_end=opt.warmup,
        sh_tuning_end=opt.start_shadowed
    )


def prepare_output_and_logger(args) -> tuple:
    """
    Prepare output directory and logging infrastructure.

    Returns:
        tuple: (tb_writer, metrics_logger)
    """
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

    # Create metrics logger
    metrics_logger = MetricsLogger(tb_writer, args.model_path)

    return tb_writer, metrics_logger


@torch.no_grad()
def _process_environment_map_image(img_path, scale_high, threshold):
    """Process an environment map image into SH coefficients (from test_gt_env_map.py)."""
    img = _plt.imread(img_path)
    img = torch.from_numpy(img).float() / 255
    img[img > threshold] *= scale_high
    coeffs = getCoefficientsFromImage(img.numpy(), 2)
    return coeffs


def _sun_direction_from_azimuth(azimuth_rad, base_sun_direction):
    """Rotate a sun direction around the vertical (Y) axis (from test_gt_env_map_sun.py)."""
    cos_a = np.cos(azimuth_rad)
    sin_a = np.sin(azimuth_rad)
    rot_y = torch.tensor([
        [cos_a,  0, sin_a],
        [0,      1, 0    ],
        [-sin_a, 0, cos_a]
    ], dtype=torch.float32)
    if isinstance(base_sun_direction, torch.Tensor):
        d = base_sun_direction.cpu().float()
    else:
        d = torch.tensor(base_sun_direction, dtype=torch.float32)
    rotated = rot_y @ d
    rotated = rotated / (torch.norm(rotated) + 1e-8)
    return rotated.cuda()


def _render_shadowed_sun(gaussians, viewpoint_cam, pipeline, background,
                         normal_vectors, multiplier, emb_idx,
                         sun_direction, sun_elevation,
                         shadow_method="shadow_map", shadow_map_resolution=512,
                         shadow_bias=0.1, ray_march_steps=64, voxel_resolution=128):
    """Render a shadowed image using the sun model (from test_gt_env_map_sun.py)."""
    if gaussians.full_pbr:
        rgb_unshadowed, _, sun_dir_out, components = gaussians.compute_directional_pbr(
            emb_idx, normal_vectors, sun_direction, viewpoint_cam.camera_center,
            sun_elevation=sun_elevation
        )
    else:
        rgb_unshadowed, _, sun_dir_out, components = gaussians.compute_directional_rgb(
            emb_idx, normal_vectors, sun_direction, sun_elevation=sun_elevation
        )

    effective_shadow_method = shadow_method
    if getattr(pipeline, "use_gaussians", False) and effective_shadow_method == "shadow_map":
        effective_shadow_method = "ray_march"

    shadow_mask, _, _ = compute_shadows_for_gaussians(
        gaussians, sun_dir_out, pipeline,
        method=effective_shadow_method,
        shadow_map_resolution=shadow_map_resolution,
        shadow_bias=shadow_bias,
        ray_march_steps=ray_march_steps,
        voxel_resolution=voxel_resolution,
        device="cuda"
    )
    shadow_mask = shadow_mask.unsqueeze(-1)  # [N, 1]

    if gaussians.full_pbr:
        rgb_shadowed, _, _, _ = gaussians.compute_directional_pbr(
            emb_idx, normal_vectors, sun_direction, viewpoint_cam.camera_center,
            sun_elevation=sun_elevation, shadow_mask=shadow_mask
        )
    else:
        direct_light = components['direct']
        ambient_light = components['ambient']
        residual_light = components['residual']
        intensity_hdr = direct_light * shadow_mask + ambient_light + residual_light
        intensity_hdr = torch.clamp_min(intensity_hdr, 0.00001)
        intensity = intensity_hdr ** (1 / 2.2)
        albedo = gaussians.get_albedo
        rgb_shadowed = torch.clamp(intensity * albedo, 0.0)

    render_pkg = render(viewpoint_cam, gaussians, pipeline, background,
                        override_color=rgb_shadowed)
    rendering_shadowed = torch.clamp(render_pkg["render"], 0.0, 1.0)
    return rendering_shadowed


def training_report(tb_writer, iteration, Ll1_unshadowed, Ll1_shadowed, l1_loss, elapsed,
                    testing_iterations, scene: Scene, renderFunc, renderArgs,
                    appearance_lut=None, source_path=None, sky_masks=None,
                    metrics_logger: Optional[MetricsLogger] = None,
                    test_config_path: str = None):
    """
    Generate training report with detailed metrics logging.

    Args:
        tb_writer: TensorBoard writer
        iteration: Current training iteration
        Ll1_unshadowed: L1 loss for unshadowed render
        Ll1_shadowed: L1 loss for shadowed render
        l1_loss: L1 loss function
        elapsed: Elapsed time for iteration (ms)
        testing_iterations: List of iterations to run evaluation
        scene: Scene object
        renderFunc: Render function
        renderArgs: Arguments for render function
        appearance_lut: Appearance lookup table
        source_path: Source data path
        sky_masks: Sky masks dictionary
        metrics_logger: MetricsLogger instance for detailed logging
        test_config_path: Path to test config directory (containing test_config.py).
                          When provided, eval_masked uses the full test_gt_env_map /
                          test_gt_env_map_sun protocol (GT env map + angle sweep).

    Returns:
        float: PSNR value from evaluation (or large number if not evaluating)
    """
    # Use metrics_logger if available, otherwise fall back to basic logging
    if metrics_logger is not None:
        metrics_logger.log_iteration_time(iteration, elapsed)
        metrics_logger.log_gaussian_stats(iteration, scene.gaussians)

        # Log sun model parameters if in use_sun mode
        if scene.gaussians.use_sun and appearance_lut:
            first_emb_idx = list(appearance_lut.values())[0]
            metrics_logger.log_sun_model_params(iteration, scene.gaussians.sun_model, first_emb_idx)
    elif tb_writer:
        # Fallback to basic logging
        tb_writer.add_scalar('Loss_Image/l1_unshadowed', Ll1_unshadowed, iteration)
        tb_writer.add_scalar('Loss_Image/l1_shadowed', Ll1_shadowed, iteration)
        tb_writer.add_scalar('Performance/iter_time_ms', elapsed, iteration)
        tb_writer.add_scalar('Gaussians/total_count', scene.gaussians.get_xyz.shape[0], iteration)

    psnr_test = 10000000

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        test_cameras = scene.getTestCameras()

        # Hardcoded test image names per dataset — these match the test_config.py
        # files used for evaluation, so TensorBoard shows the exact same views.
        TEST_IMAGES = {
            "st": [
                "01-09_14_00_IMG_0706.JPG",
                "24-08_11_30_IMG_9690.JPG",
                "24-08_16_30_IMG_0061.JPG",
                "25-08_19_30_IMG_0306.JPG",
                "31-08_07_30_IMG_0501.JPG",
            ],
            "lk2": [
                "01-08_07_30_IMG_6710.JPG",
                "08-08_16_00_IMG_7850.JPG",
                "28-07_10_00_DSC_0055.jpg",
                "29-07_12_00_IMG_5424.JPG",
                "29-07_20_30_IMG_5607.JPG",
            ],
            "lwp": [
                "01-09_14_00_IMG_0821.JPG",
                "24-08_11_30_IMG_9765.JPG",
                "24-08_16_30_IMG_0216.JPG",
                "25-08_19_30_IMG_0406.JPG",
                "31-08_07_30_IMG_0631.JPG",
            ],
        }

        # Find which dataset we're working with
        target_names = None
        if source_path:
            for key, names in TEST_IMAGES.items():
                if key in source_path.lower():
                    target_names = set(names)
                    break

        if target_names:
            # Filter to only the hardcoded test images, preserving config order
            test_cameras_filtered = [c for c in test_cameras if c.image_name in target_names]
            if not test_cameras_filtered:
                print(f"Warning: none of the hardcoded test images found in test set. "
                      f"Available: {[c.image_name for c in test_cameras[:10]]}")
                test_cameras_filtered = sorted(test_cameras, key=lambda c: c.image_name)[:5]
        else:
            # Unknown dataset — fall back to sorted first 5
            test_cameras_filtered = sorted(test_cameras, key=lambda c: c.image_name)[:5]

        validation_configs = (
            {'name': 'test', 'cameras': test_cameras_filtered},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                # Masked evaluation metric accumulators (matches test script exactly)
                eval_psnrs_shadowed = []
                eval_mse_shadowed = []
                eval_mae_shadowed = []
                eval_ssim_shadowed = []

                # Load test config dict once per validation config
                _test_config_dict = {}
                if config['name'] == 'test' and test_config_path:
                    try:
                        _tc_path = test_config_path
                        if _tc_path not in sys.path:
                            sys.path.insert(0, _tc_path)
                        # Force reimport in case module was cached from a previous iteration
                        if 'test_config' in sys.modules:
                            del sys.modules['test_config']
                        _test_config_dict = importlib.import_module("test_config").config
                        print(f"[ITER {iteration}] Loaded test config from {_tc_path} "
                              f"({len(_test_config_dict)} images)")
                    except Exception as e:
                        print(f"[ITER {iteration}] Warning: failed to load test config: {e}")
                        _test_config_dict = {}

                for idx, viewpoint in enumerate(config['cameras']):
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    # Render albedo
                    rgb_precomp = scene.gaussians.get_albedo
                    render_start = time.time()
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp)
                    render_time_ms = (time.time() - render_start) * 1000
                    image_albedo = torch.clamp(render_pkg["render"], 0.0, 1.0)

                    # Get normals in world space
                    quaternions = scene.gaussians.get_rotation
                    scales = scene.gaussians.get_scaling
                    normal_vectors, multiplier = compute_normal_world_space(
                        quaternions, scales, viewpoint.world_view_transform, scene.gaussians.get_xyz)

                    # Initialize variables
                    image_shadowed = None
                    image_unshadowed = None
                    env_sh_learned = None
                    residual_env_map = None
                    shadow_map_vis = None
                    direct_light_vis = None
                    casts_shadow_vis = None

                    if scene.gaussians.use_sun:
                        # Directional sun lighting mode - no SH environment
                        if config["name"] == "train":
                            emb_idx = appearance_lut[viewpoint.image_name]
                        else:
                            # For test, use first training image's lighting
                            emb_idx = list(appearance_lut.values())[0] if appearance_lut else 0

                        # Get unshadowed lighting + components (with sun color prior)
                        sun_elev = viewpoint.sun_elevation
                        if scene.gaussians.full_pbr:
                            rgb_precomp_unshadowed, intensity, sun_dir, components = scene.gaussians.compute_directional_pbr(
                                emb_idx, normal_vectors, viewpoint.sun_direction, viewpoint.camera_center, sun_elevation=sun_elev
                            )
                        else:
                            rgb_precomp_unshadowed, intensity, sun_dir, components = scene.gaussians.compute_directional_rgb(emb_idx, normal_vectors, viewpoint.sun_direction, sun_elevation=sun_elev)

                        # render unshadowed with directional lighting
                        render_pkg_unshadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp_unshadowed)
                        image_unshadowed = torch.clamp(render_pkg_unshadowed["render"], 0.0, 1.0)

                        # Compute shadow mask for TensorBoard visualization
                        viz_shadow_method = "shadow_map"
                        if getattr(renderArgs[0], "use_gaussians", False):
                            viz_shadow_method = "ray_march"
                        shadow_mask, shadow_depth_map, sun_camera = compute_shadows_for_gaussians(
                            scene.gaussians,
                            sun_dir,
                            renderArgs[0],  # pipe
                            method=viz_shadow_method,
                            shadow_map_resolution=512,
                            shadow_bias=0.1,
                            device="cuda"
                        )
                        shadow_mask = shadow_mask.unsqueeze(-1)  # [N, 1]
                        albedo = scene.gaussians.get_albedo

                        if scene.gaussians.full_pbr:
                            rgb_precomp_shadowed, intensity_shadowed, _, components_shadowed = scene.gaussians.compute_directional_pbr(
                                emb_idx, normal_vectors, viewpoint.sun_direction, viewpoint.camera_center,
                                sun_elevation=sun_elev, shadow_mask=shadow_mask
                            )
                            direct_light = components_shadowed['direct_pbr'] if 'direct_pbr' in components_shadowed else components_shadowed['direct']
                        else:
                            direct_light = components['direct']
                            ambient_light = components['ambient']
                            residual_light = components['residual']

                            intensity_hdr_shadowed = direct_light * shadow_mask + ambient_light + residual_light
                            intensity_hdr_shadowed = torch.clamp_min(intensity_hdr_shadowed, 0.00001)
                            intensity_shadowed = intensity_hdr_shadowed ** (1 / 2.2)

                            rgb_precomp_shadowed = torch.clamp(intensity_shadowed * albedo, 0.0)

                        # Render shadow map for visualization with black background
                        shadow_rgb = shadow_mask.expand(-1, 3)  # [N, 3]
                        black_bg = torch.zeros(3, device="cuda")
                        render_pkg_shadow = renderFunc(viewpoint, scene.gaussians, renderArgs[0], black_bg, override_color=shadow_rgb)
                        shadow_map_vis = torch.clamp(render_pkg_shadow["render"], 0.0, 1.0)

                        # Render direct light only (with shadow applied)
                        # direct_light already contains sun_color * I * max(N·L, 0) for all gaussians
                        # (casts_shadow only controls shadow map opacity, not lighting mode)
                        direct_shadowed = direct_light * shadow_mask  # [N, 3]
                        direct_shadowed_gamma = torch.clamp_min(direct_shadowed, 0.00001) ** (1 / 2.2)
                        direct_shadowed_rgb = torch.clamp(direct_shadowed_gamma * albedo, 0.0)
                        render_pkg_direct = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=direct_shadowed_rgb)
                        direct_light_vis = torch.clamp(render_pkg_direct["render"], 0.0, 1.0)

                        # Render casts_shadow visualization as grayscale mask
                        # Black = sky (non-shadow-casting, value 0), White = non-sky (shadow-casting, value 1)
                        # Use black background for clarity
                        casts_shadow = scene.gaussians.get_casts_shadow  # [N]
                        # Create grayscale RGB: white for shadow-casting (1), black for non-shadow-casting (0)
                        casts_shadow_rgb = casts_shadow.unsqueeze(-1).expand(-1, 3)  # [N, 3]
                        render_pkg_casts_shadow = renderFunc(viewpoint, scene.gaussians, renderArgs[0], black_bg, override_color=casts_shadow_rgb)
                        casts_shadow_vis = torch.clamp(render_pkg_casts_shadow["render"], 0.0, 1.0)

                        # Create sky mask comparison visualization
                        sky_mask_vis = None
                        sky_mask_comparison = None
                        if sky_masks is not None and viewpoint.image_name in sky_masks:
                            sky_mask = sky_masks[viewpoint.image_name]  # [H, W] where 0=sky, 1=not sky
                            # Resize sky mask to match rendered image size if needed
                            H_render, W_render = casts_shadow_vis.shape[1], casts_shadow_vis.shape[2]
                            H_mask, W_mask = sky_mask.shape
                            if H_mask != H_render or W_mask != W_render:
                                import torch.nn.functional as F
                                sky_mask_resized = F.interpolate(
                                    sky_mask.unsqueeze(0).unsqueeze(0),
                                    size=(H_render, W_render),
                                    mode='nearest'
                                ).squeeze()
                            else:
                                sky_mask_resized = sky_mask

                            # Create RGB visualization of sky mask as grayscale (white=not sky, black=sky)
                            # This matches the casts_shadow_vis format
                            sky_mask_vis = sky_mask_resized.unsqueeze(0).expand(3, -1, -1)  # [3, H, W]

                            # Create comparison: where rendered casts_shadow differs from sky mask
                            # casts_shadow_vis is grayscale: high value = shadow casting (not sky)
                            # sky_mask: 1 = not sky (should cast shadow), 0 = sky
                            # If they match: green. If they differ: red or blue
                            rendered_is_shadow_caster = casts_shadow_vis[0] > 0.5  # any channel works, it's grayscale
                            mask_says_not_sky = sky_mask_resized > 0.5

                            sky_mask_comparison = torch.zeros(3, H_render, W_render, device=sky_mask.device)
                            # Green where both agree it should cast shadow (not sky)
                            sky_mask_comparison[1] = (rendered_is_shadow_caster & mask_says_not_sky).float()
                            # Red where rendered says casts shadow but mask says sky (false positive)
                            sky_mask_comparison[0] = (rendered_is_shadow_caster & ~mask_says_not_sky).float()
                            # Blue where rendered says no shadow but mask says not sky (sky gaussians over non-sky)
                            sky_mask_comparison[2] = (~rendered_is_shadow_caster & mask_says_not_sky).float()

                        render_pkg_shadowed = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=rgb_precomp_shadowed)
                        image_shadowed = torch.clamp(render_pkg_shadowed["render"], 0.0, 1.0)

                        # Visualize global sky SH environment map
                        if scene.gaussians.sun_model.use_residual_sh:
                            # sky_sh shape: [3, n_sh_coeffs] -> need [n_sh_coeffs, 3] for shReconstructDiffuseMap
                            sky_sh_coeffs = scene.gaussians.sun_model.sky_sh  # [3, 4] for degree 1
                            sky_sh_for_vis = sky_sh_coeffs.T.cpu().detach().numpy()  # [4, 3]
                            # Pad to 9 coeffs if needed for visualization (shReconstructDiffuseMap expects degree 2)
                            if sky_sh_for_vis.shape[0] < 9:
                                sky_sh_padded = np.zeros((9, 3))
                                sky_sh_padded[:sky_sh_for_vis.shape[0]] = sky_sh_for_vis
                                sky_sh_for_vis = sky_sh_padded
                            residual_env_map = np.clip(shReconstructDiffuseMap(sky_sh_for_vis, width=300), 0, None)
                            # Apply gamma correction and convert to tensor
                            residual_env_map = torch.clamp(torch.tensor(residual_env_map ** (1 / 2.2)).permute(2, 0, 1), 0.0, 1.0)

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

                        # Visualizations for use_sun mode
                        if scene.gaussians.use_sun:
                            if residual_env_map is not None:
                                tb_writer.add_images(view_tag + "/07_residual_env_map", residual_env_map[None], global_step=iteration)
                            if shadow_map_vis is not None:
                                tb_writer.add_images(view_tag + "/08_shadow_map", shadow_map_vis[None], global_step=iteration)
                            if direct_light_vis is not None:
                                tb_writer.add_images(view_tag + "/09_direct_sun_only", direct_light_vis[None], global_step=iteration)
                            if casts_shadow_vis is not None:
                                tb_writer.add_images(view_tag + "/10_casts_shadow_mask", casts_shadow_vis[None], global_step=iteration)
                            if sky_mask_vis is not None:
                                tb_writer.add_images(view_tag + "/11_sky_mask_gt", sky_mask_vis[None], global_step=iteration)
                            if sky_mask_comparison is not None:
                                tb_writer.add_images(view_tag + "/12_sky_mask_comparison", sky_mask_comparison[None], global_step=iteration)

                            # 3D visualization of sun direction and camera
                            try:
                                # Get camera forward direction from view matrix
                                view_matrix = viewpoint.world_view_transform
                                cam_forward = -view_matrix[:3, 2]  # Forward is -Z in camera space

                                # Get adjusted (optimised) sun direction if sun_cal is active
                                adjusted_sun = viewpoint.get_adjusted_sun_direction()
                                # original_sun = viewpoint.sun_direction (already used as sun_dir above)
                                # If sun_cal changed the direction, show both; otherwise just one
                                original_sun = viewpoint.sun_direction

                                sun_cam_vis = create_sun_camera_visualization_tensor(
                                    gaussian_positions=scene.gaussians.get_xyz,
                                    sun_direction=adjusted_sun if adjusted_sun is not None else sun_dir,
                                    camera_position=viewpoint.camera_center,
                                    camera_forward=cam_forward,
                                    shadow_mask=shadow_mask.squeeze() if shadow_mask is not None else None,
                                    original_sun_direction=original_sun,
                                    max_points=3000
                                )
                                tb_writer.add_images(view_tag + "/10_sun_camera_3d", sun_cam_vis[None], global_step=iteration)
                            except Exception as e:
                                print(f"Warning: Could not create sun/camera visualization: {e}")

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

                            # Material map renders (similar pipeline to normal map render)
                            if hasattr(scene.gaussians, 'get_roughness') and hasattr(scene.gaussians, 'get_metallic'):
                                roughness_precomp = scene.gaussians.get_roughness.expand(-1, 3)
                                render_pkg_roughness = render(viewpoint, scene.gaussians, *renderArgs, override_color=roughness_precomp)
                                roughness_map = torch.clamp(render_pkg_roughness["render"], 0.0, 1.0)
                                tb_writer.add_images(view_tag + "/17_roughness_map", roughness_map[None], global_step=iteration)

                                metallic_precomp = scene.gaussians.get_metallic.expand(-1, 3)
                                render_pkg_metallic = render(viewpoint, scene.gaussians, *renderArgs, override_color=metallic_precomp)
                                metallic_map = torch.clamp(render_pkg_metallic["render"], 0.0, 1.0)
                                tb_writer.add_images(view_tag + "/18_metallic_map", metallic_map[None], global_step=iteration)
                        except:
                            pass

                    # Compute detailed metrics using MetricsLogger if available
                    if metrics_logger is not None:
                        # Compute comprehensive metrics for albedo
                        albedo_metrics = metrics_logger.metrics_calculator.compute_all_metrics(
                            image_albedo, gt_image
                        )

                        # Compute metrics for shadowed and unshadowed renders
                        shadowed_metrics = ImageMetrics()
                        unshadowed_metrics = ImageMetrics()

                        if image_shadowed is not None:
                            shadowed_metrics = metrics_logger.metrics_calculator.compute_all_metrics(
                                image_shadowed, gt_image
                            )

                        if image_unshadowed is not None:
                            unshadowed_metrics = metrics_logger.metrics_calculator.compute_all_metrics(
                                image_unshadowed, gt_image
                            )

                        # Create and log evaluation result
                        eval_result = EvaluationResult(
                            viewpoint_name=viewpoint.image_name,
                            config_name=config['name'],
                            iteration=iteration,
                            albedo_metrics=albedo_metrics,
                            shadowed_metrics=shadowed_metrics,
                            unshadowed_metrics=unshadowed_metrics,
                            num_gaussians=scene.gaussians.get_xyz.shape[0],
                            render_time_ms=render_time_ms
                        )
                        metrics_logger.log_evaluation_result(eval_result)

                        # Accumulate for backward compatibility
                        l1_test += albedo_metrics.l1
                        psnr_test += albedo_metrics.psnr
                    else:
                        # Fallback to basic metrics computation
                        l1_test += l1_loss(image_albedo, gt_image).mean().double()
                        psnr_test += psnr(image_albedo, gt_image).mean().double()

                    # ---- Masked evaluation metrics (test_gt_env_map protocol) ----
                    if config['name'] == 'test' and test_config_path and viewpoint.image_name in _test_config_dict:
                        image_config = _test_config_dict[viewpoint.image_name]
                        mask_path = image_config["mask_path"]
                        sun_angle_range = image_config["sun_angles"]

                        # Load and erode mask from test config (same as test_gt_env_map.py)
                        eval_mask = _cv2.imread(mask_path, _cv2.IMREAD_GRAYSCALE)
                        eval_mask = _cv2.resize(eval_mask, (gt_image.shape[2], gt_image.shape[1]))
                        _kernel = np.ones((5, 5), np.uint8)
                        eval_mask = _cv2.erode(eval_mask, _kernel, iterations=1)
                        eval_mask = torch.from_numpy(eval_mask // 255).to(gt_image.device)

                        if scene.gaussians.use_sun:
                            # --- Sun method: sweep sun azimuth (test_gt_env_map_sun.py) ---
                            if appearance_lut and viewpoint.image_name in appearance_lut:
                                _emb_idx = appearance_lut[viewpoint.image_name]
                            elif appearance_lut:
                                _emb_idx = list(appearance_lut.values())[0]
                            else:
                                _emb_idx = 0

                            base_sun_dir = viewpoint.sun_direction
                            _sun_elev = viewpoint.sun_elevation
                            if base_sun_dir is None:
                                base_sun_dir = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

                            best_psnr_sweep = 0
                            n_sweep = 51
                            angle_list = torch.linspace(sun_angle_range[0], sun_angle_range[1], n_sweep)
                            best_angle = angle_list[0]

                            for _angle in angle_list:
                                _sun_dir = _sun_direction_from_azimuth(float(_angle), base_sun_direction=base_sun_dir)
                                _rendering = _render_shadowed_sun(
                                    scene.gaussians, viewpoint, renderArgs[0], renderArgs[1],
                                    normal_vectors, multiplier, _emb_idx,
                                    _sun_dir, _sun_elev)
                                _cur_psnr = float(mse2psnr(img2mse(_rendering, gt_image, mask=eval_mask)))
                                if _cur_psnr > best_psnr_sweep:
                                    best_psnr_sweep = _cur_psnr
                                    best_angle = _angle

                            # Render best angle for final metrics
                            best_sun_dir = _sun_direction_from_azimuth(float(best_angle), base_sun_direction=base_sun_dir)
                            eval_shadowed = _render_shadowed_sun(
                                scene.gaussians, viewpoint, renderArgs[0], renderArgs[1],
                                normal_vectors, multiplier, _emb_idx,
                                best_sun_dir, _sun_elev)
                        else:
                            # --- Non-sun method: GT env map + angle sweep (test_gt_env_map.py) ---
                            envmap_img_path = image_config["env_map_path"]
                            init_rot_x = image_config["initial_env_map_rotation"]["x"]
                            init_rot_y = image_config["initial_env_map_rotation"]["y"]
                            init_rot_z = image_config["initial_env_map_rotation"]["z"]
                            threshold = image_config["env_map_scaling"]["threshold"]
                            scale = image_config["env_map_scaling"]["scale"]

                            env_sh_gt = _process_environment_map_image(envmap_img_path, scale, threshold)

                            best_psnr_sweep = 0
                            n_sweep = 51
                            angle_list = torch.linspace(sun_angle_range[0], sun_angle_range[1], n_sweep)
                            sun_angles_list = [torch.tensor([a, 0, 0]) for a in angle_list]
                            best_angle = sun_angles_list[0]

                            rotation = Rotation()
                            init_rot = np.float32(np.dot(rotation.rot_y(init_rot_y),
                                                         np.dot(rotation.rot_x(init_rot_x), rotation.rot_z(init_rot_z))))

                            for _angle in sun_angles_list:
                                env_sh = np.matmul(init_rot, env_sh_gt)
                                rotation2 = Rotation()
                                rot2 = np.float32(np.dot(rotation2.rot_y(_angle[0]),
                                                         np.dot(rotation2.rot_x(_angle[1]), rotation2.rot_z(_angle[2]))))
                                env_sh = np.matmul(rot2, env_sh)
                                env_sh_torch = torch.tensor(env_sh.T, dtype=torch.float32).cuda()

                                rgb_precomp_sweep, _ = scene.gaussians.compute_gaussian_rgb(env_sh_torch, multiplier=multiplier)
                                render_pkg_sweep = render(viewpoint, scene.gaussians, renderArgs[0], renderArgs[1],
                                                          override_color=rgb_precomp_sweep)
                                _rendering = torch.clamp(render_pkg_sweep["render"], 0.0, 1.0)
                                _cur_psnr = float(mse2psnr(img2mse(_rendering, gt_image, mask=eval_mask)))
                                if _cur_psnr > best_psnr_sweep:
                                    best_psnr_sweep = _cur_psnr
                                    best_angle = _angle

                            # Render best angle for final metrics
                            env_sh = np.matmul(init_rot, env_sh_gt)
                            rotation3 = Rotation()
                            rot3 = np.float32(np.dot(rotation3.rot_y(best_angle[0]),
                                                     np.dot(rotation3.rot_x(best_angle[1]), rotation3.rot_z(best_angle[2]))))
                            env_sh = np.matmul(rot3, env_sh)
                            env_sh_torch = torch.tensor(env_sh.T, dtype=torch.float32).cuda()

                            rgb_precomp_best, _ = scene.gaussians.compute_gaussian_rgb(env_sh_torch, multiplier=multiplier)
                            render_pkg_best = render(viewpoint, scene.gaussians, renderArgs[0], renderArgs[1],
                                                     override_color=rgb_precomp_best)
                            eval_shadowed = torch.clamp(render_pkg_best["render"], 0.0, 1.0)

                        # Compute metrics on the best-angle rendering
                        _mse_val = img2mse(eval_shadowed, gt_image, mask=eval_mask)
                        eval_mse_shadowed.append(float(_mse_val))
                        eval_psnrs_shadowed.append(float(mse2psnr(_mse_val)))
                        eval_mae_shadowed.append(float(img2mae(eval_shadowed, gt_image, mask=eval_mask)))

                        _shad_np = eval_shadowed.cpu().detach().numpy().transpose(1, 2, 0)
                        _gt_np = gt_image.cpu().detach().numpy().transpose(1, 2, 0)
                        _, _full = ssim_skimage(_shad_np, _gt_np, win_size=5,
                                                channel_axis=2, full=True, data_range=1.0)
                        _mssim = (torch.tensor(_full).to(gt_image.device) * eval_mask.unsqueeze(-1)).sum() / (3 * eval_mask.sum())
                        eval_ssim_shadowed.append(float(_mssim))

                    elif config['name'] == 'test' and image_shadowed is not None:
                        # Fallback: no test_config provided, use old behavior
                        eval_mask = viewpoint.mask.squeeze(0)
                        mask_np = (eval_mask.cpu().numpy() * 255).astype(np.uint8)
                        _kernel = np.ones((5, 5), np.uint8)
                        mask_np = _cv2.erode(mask_np, _kernel, iterations=1)
                        eval_mask = torch.from_numpy(mask_np // 255).to(gt_image.device)

                        _mse_val = img2mse(image_shadowed, gt_image, mask=eval_mask)
                        eval_mse_shadowed.append(float(_mse_val))
                        eval_psnrs_shadowed.append(float(mse2psnr(_mse_val)))
                        eval_mae_shadowed.append(float(img2mae(image_shadowed, gt_image, mask=eval_mask)))

                        _shad_np = image_shadowed.cpu().detach().numpy().transpose(1, 2, 0)
                        _gt_np = gt_image.cpu().detach().numpy().transpose(1, 2, 0)
                        _, _full = ssim_skimage(_shad_np, _gt_np, win_size=5,
                                                channel_axis=2, full=True, data_range=1.0)
                        _mssim = (torch.tensor(_full).to(gt_image.device) * eval_mask.unsqueeze(-1)).sum() / (3 * eval_mask.sum())
                        eval_ssim_shadowed.append(float(_mssim))

                # Compute averages
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                # Log masked evaluation scalars to TensorBoard
                if tb_writer and config['name'] == 'test' and eval_psnrs_shadowed:
                    mean_psnr = np.mean(eval_psnrs_shadowed)
                    mean_mse = np.mean(eval_mse_shadowed)
                    mean_mae = np.mean(eval_mae_shadowed)
                    mean_ssim = np.mean(eval_ssim_shadowed)
                    tb_writer.add_scalar('eval_masked/PSNR_shadowed', mean_psnr, iteration)
                    tb_writer.add_scalar('eval_masked/MSE_shadowed', mean_mse, iteration)
                    tb_writer.add_scalar('eval_masked/MAE_shadowed', mean_mae, iteration)
                    tb_writer.add_scalar('eval_masked/SSIM_shadowed', mean_ssim, iteration)
                    print(f"\n[ITER {iteration}] Eval masked (test): "
                          f"PSNR={mean_psnr:.4f}  MSE={mean_mse:.6f}  "
                          f"MAE={mean_mae:.6f}  SSIM={mean_ssim:.4f}")

                # Log summary using MetricsLogger if available
                if metrics_logger is not None:
                    summary = metrics_logger.compute_and_log_summary(iteration, config['name'])
                    metrics_logger.print_evaluation_summary(iteration, config['name'])
                else:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # Save evaluation results to JSON
        if metrics_logger is not None:
            metrics_logger.save_evaluation_json(iteration)

        torch.cuda.empty_cache()
    return psnr_test


# =============================================================================
# Utility Functions for Final Metrics Export
# =============================================================================

def generate_final_metrics_report(metrics_logger: MetricsLogger, output_path: str = None):
    """
    Generate a comprehensive final metrics report.

    Args:
        metrics_logger: The MetricsLogger instance with accumulated data
        output_path: Optional path to save the report (defaults to metrics_dir)
    """
    if output_path is None:
        output_path = os.path.join(metrics_logger.metrics_dir, "final_report.json")

    # Collect all summaries
    all_summaries = {}
    for iteration, configs in metrics_logger.evaluation_summaries.items():
        all_summaries[iteration] = {config: summary.to_dict() for config, summary in configs.items()}

    # Find best iteration by PSNR
    best_iter_train = None
    best_psnr_train = 0
    best_iter_test = None
    best_psnr_test = 0

    for iteration, configs in metrics_logger.evaluation_summaries.items():
        if 'train' in configs and configs['train'].mean_psnr_albedo > best_psnr_train:
            best_psnr_train = configs['train'].mean_psnr_albedo
            best_iter_train = iteration
        if 'test' in configs and configs['test'].mean_psnr_albedo > best_psnr_test:
            best_psnr_test = configs['test'].mean_psnr_albedo
            best_iter_test = iteration

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluations": len(metrics_logger.evaluation_summaries),
        "best_train_iteration": best_iter_train,
        "best_train_psnr": best_psnr_train,
        "best_test_iteration": best_iter_test,
        "best_test_psnr": best_psnr_test,
        "iteration_summaries": all_summaries,
        "performance": {
            "mean_iter_time_ms": np.mean(metrics_logger.iteration_times) if metrics_logger.iteration_times else 0,
            "total_iterations": len(metrics_logger.iteration_times)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL METRICS REPORT")
    print(f"{'='*60}")
    print(f"Report saved to: {output_path}")
    print(f"Total evaluations: {report['total_evaluations']}")
    if best_iter_train:
        print(f"Best train iteration: {best_iter_train} (PSNR: {best_psnr_train:.2f} dB)")
    if best_iter_test:
        print(f"Best test iteration: {best_iter_test} (PSNR: {best_psnr_test:.2f} dB)")
    print(f"{'='*60}\n")

    return report


def export_metrics_to_csv(metrics_logger: MetricsLogger, output_path: str = None):
    """
    Export evaluation metrics to CSV format for easy analysis.

    Args:
        metrics_logger: The MetricsLogger instance with accumulated data
        output_path: Optional path for the CSV file
    """
    if output_path is None:
        output_path = os.path.join(metrics_logger.metrics_dir, "metrics_summary.csv")

    rows = []
    headers = [
        "iteration", "config", "num_viewpoints",
        "psnr_albedo_mean", "psnr_albedo_std",
        "psnr_shadowed_mean", "psnr_unshadowed_mean",
        "ssim_albedo_mean", "ssim_shadowed_mean", "ssim_unshadowed_mean",
        "lpips_albedo_mean", "lpips_shadowed_mean",
        "l1_albedo_mean", "l1_shadowed_mean", "l1_unshadowed_mean",
        "best_viewpoint", "worst_viewpoint"
    ]

    for iteration in sorted(metrics_logger.evaluation_summaries.keys()):
        for config_name, summary in metrics_logger.evaluation_summaries[iteration].items():
            row = [
                iteration, config_name, summary.num_viewpoints,
                summary.mean_psnr_albedo, summary.std_psnr_albedo,
                summary.mean_psnr_shadowed, summary.mean_psnr_unshadowed,
                summary.mean_ssim_albedo, summary.mean_ssim_shadowed, summary.mean_ssim_unshadowed,
                summary.mean_lpips_albedo, summary.mean_lpips_shadowed,
                summary.mean_l1_albedo, summary.mean_l1_shadowed, summary.mean_l1_unshadowed,
                summary.best_psnr_viewpoint, summary.worst_psnr_viewpoint
            ]
            rows.append(row)

    # Write CSV
    with open(output_path, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in rows:
            f.write(','.join(str(v) for v in row) + '\n')

    print(f"Metrics exported to CSV: {output_path}")
    return output_path


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    'ImageMetrics',
    'LossComponents',
    'TrainingPhase',
    'EvaluationResult',
    'EvaluationSummary',
    # Classes
    'MetricsCalculator',
    'MetricsLogger',
    # Functions
    'create_loss_components',
    'update_lambdas',
    'get_training_phase',
    'prepare_output_and_logger',
    'training_report',
    'generate_final_metrics_report',
    'export_metrics_to_csv',
    # Constants
    'TENSORBOARD_FOUND',
    'LPIPS_AVAILABLE',
]