"""Monocular depth estimation for surface regularisation and densification.

Pre-computes pseudo-depth maps for every training image using a pretrained
DPT model, then provides:
  1. A scale-and-shift invariant loss (Pearson correlation) that penalises
     Gaussians whose rendered depth deviates from the monocular estimate.
  2. Depth-guided densification that spawns new Gaussians where rendered
     alpha is low but the monocular depth indicates a surface exists.

Guarded by ``--use_depth_est``.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
#  Pre-computation
# ---------------------------------------------------------------------------

def precompute_depth_maps(cameras, model_name="Intel/dpt-hybrid-midas", device="cuda"):
    """Run a pretrained monocular depth estimator on every training image.

    The model is loaded once, inference is run for each camera, and then the
    model is deleted to free GPU memory.  Two versions of each depth map are
    stored:
      * **normalised** (``[0, 1]``) – for the Pearson-correlation loss.
      * **raw** (metric-scale disparity-inverted) – for depth-guided
        densification, where we align the mono estimate to the rendered depth
        via per-image least-squares.

    Both are stored on **CPU** to save device memory during training.

    Args:
        cameras: list of Camera objects (each with ``.original_image`` [3,H,W]
                 and ``.image_name``).
        model_name: HuggingFace model identifier for ``DPTForDepthEstimation``.
        device: torch device used for inference.

    Returns:
        ``dict[image_name -> dict]`` with keys ``'norm'`` (normalised [H,W])
        and ``'raw'`` (un-normalised [H,W]) – both on CPU.
    """
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    print(f"[DepthEst] Loading model  {model_name}  ...")
    processor = DPTImageProcessor.from_pretrained(model_name)
    model = DPTForDepthEstimation.from_pretrained(model_name, use_safetensors=True).to(device).eval()

    depth_maps = {}
    print(f"[DepthEst] Estimating depth for {len(cameras)} training images ...")
    for cam in tqdm(cameras, desc="Depth estimation"):
        img = cam.original_image                   # [3, H, W], cuda, [0,1]
        H, W = img.shape[1], img.shape[2]

        # DPT processor expects uint8 numpy HWC or PIL
        img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        inputs = processor(images=img_np, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.predicted_depth          # [1, h_model, w_model]

        # Resize to the camera's native resolution
        pred = F.interpolate(
            pred.unsqueeze(0), size=(H, W),
            mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)                     # [H, W]

        # MiDaS-family models output *inverse depth* (disparity): larger = closer.
        # Convert to regular depth (larger = farther) to match surf_depth.
        pred = pred.clamp_min(1e-3)
        depth_raw = 1.0 / pred

        # Per-image normalisation to [0, 1] for the Pearson loss
        d_min, d_max = depth_raw.min(), depth_raw.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_raw - d_min) / (d_max - d_min)
        else:
            depth_norm = torch.zeros_like(depth_raw)

        depth_maps[cam.image_name] = {
            "norm": depth_norm.cpu(),
            "raw": depth_raw.cpu(),
        }

    # Free model memory
    del model, processor
    torch.cuda.empty_cache()
    print(f"[DepthEst] Done – {len(depth_maps)} depth maps cached on CPU.")
    return depth_maps


# ---------------------------------------------------------------------------
#  Loss function
# ---------------------------------------------------------------------------

def scale_invariant_depth_loss(rendered_depth, estimated_depth_entry,
                               mask=None, render_alpha=None,
                               alpha_threshold=0.5):
    """Pearson-correlation depth loss (scale-and-shift invariant).

    Because the monocular depth estimate is only correct up to an unknown
    global scale and shift, we measure how well the *ranking* of depths
    agrees between ``rendered_depth`` and ``estimated_depth`` using the
    Pearson correlation coefficient.

    Loss = 1 − corr(r, e)   ∈ [0, 2].  0 = perfect agreement.

    Args:
        rendered_depth:      ``[1, H, W]`` rendered surface depth from 2DGS.
        estimated_depth_entry: either a ``[H, W]`` tensor (legacy) or a dict
                               with key ``'norm'`` → ``[H, W]``.
        mask:                optional ``[1, H, W]`` or ``[H, W]`` training /
                             sky mask (1 = valid pixel).
        render_alpha:        optional ``[1, H, W]`` rendered alpha; pixels
                             below *alpha_threshold* are excluded.
        alpha_threshold:     minimum alpha for a pixel to be considered valid.

    Returns:
        Scalar loss tensor (differentiable w.r.t. ``rendered_depth``).
    """
    # Support both legacy plain-tensor and new dict formats.
    if isinstance(estimated_depth_entry, dict):
        estimated_depth = estimated_depth_entry["norm"]
    else:
        estimated_depth = estimated_depth_entry

    r = rendered_depth.squeeze()                   # [H, W]
    e = estimated_depth.squeeze().to(r.device)      # [H, W]

    # Handle resolution mismatch (different camera vs. pre-computed size)
    if r.shape != e.shape:
        e = F.interpolate(
            e.unsqueeze(0).unsqueeze(0), size=r.shape,
            mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)

    # Build validity mask
    valid = (r > 1e-6) & (e > 1e-6)

    if mask is not None:
        m = mask.squeeze().float()
        if m.shape != r.shape:
            m = F.interpolate(
                m.unsqueeze(0).unsqueeze(0), size=r.shape, mode="nearest"
            ).squeeze(0).squeeze(0)
        valid = valid & (m > 0.5)

    if render_alpha is not None:
        a = render_alpha.squeeze()
        valid = valid & (a > alpha_threshold)

    r_v = r[valid]
    e_v = e[valid]

    if r_v.numel() < 10:
        return torch.tensor(0.0, device=r.device)

    # Pearson correlation
    r_c = r_v - r_v.mean()
    e_c = e_v - e_v.mean()
    corr = (r_c * e_c).sum() / (r_c.norm() * e_c.norm() + 1e-8)

    return 1.0 - corr


# ---------------------------------------------------------------------------
#  Depth-guided densification
# ---------------------------------------------------------------------------

def _align_mono_to_rendered(mono_raw, rendered_depth, render_alpha,
                            mask=None, alpha_thresh=0.5):
    """Least-squares scale+shift alignment:  aligned = scale * mono + shift.

    Only uses pixels where both the rendered depth and render alpha are valid
    so the alignment is driven by existing well-represented Gaussians.

    Returns ``(aligned_depth, scale, shift)`` all on the same device as
    ``rendered_depth``.  If alignment fails (too few pixels) returns
    ``(None, None, None)``.
    """
    device = rendered_depth.device
    r = rendered_depth.squeeze().detach()             # [H, W]
    m = mono_raw.squeeze().to(device)                  # [H, W]

    if r.shape != m.shape:
        m = F.interpolate(
            m.unsqueeze(0).unsqueeze(0), size=r.shape,
            mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)

    valid = (r > 1e-6) & (m > 1e-6)
    if render_alpha is not None:
        valid = valid & (render_alpha.squeeze() > alpha_thresh)
    if mask is not None:
        mk = mask.squeeze().float()
        if mk.shape != r.shape:
            mk = F.interpolate(mk.unsqueeze(0).unsqueeze(0), size=r.shape,
                               mode="nearest").squeeze(0).squeeze(0)
        valid = valid & (mk > 0.5)

    if valid.sum() < 50:
        return None, None, None

    r_v = r[valid]
    m_v = m[valid]

    # Solve  r ≈ scale * m + shift  via normal equations
    A = torch.stack([m_v, torch.ones_like(m_v)], dim=-1)   # [N, 2]
    b = r_v                                                  # [N]
    AtA = A.T @ A
    Atb = A.T @ b
    try:
        params = torch.linalg.solve(AtA, Atb)  # [scale, shift]
    except Exception:
        return None, None, None

    scale, shift = params[0].item(), params[1].item()
    aligned = m * scale + shift
    return aligned, scale, shift


def depth_guided_densify(gaussians, viewpoint_cam, render_pkg,
                         depth_est_entry, sky_mask=None,
                         alpha_threshold=0.3, max_new=1024,
                         per_pixel_loss=None, loss_quantile=0.5):
    """Spawn new Gaussians where rendered alpha is low but the monocular depth
    estimate predicts a surface.

    The monocular estimate is first aligned (scale + shift) to the rendered
    depth using pixels that *do* have good Gaussians, then sparse pixels are
    unprojected to world space and fed to ``gaussians.densification_postfix``.

    Args:
        gaussians:       ``GaussianModel`` instance.
        viewpoint_cam:   Camera used for the current render.
        render_pkg:      dict from ``render()`` including ``surf_depth``,
                         ``rend_alpha``.
        depth_est_entry: dict with ``'raw'`` → ``[H, W]`` un-normalised mono depth.
        sky_mask:        optional ``[H, W]`` (1 = non-sky) to exclude sky from
                         densification.
        alpha_threshold: pixels with ``rend_alpha`` below this are considered
                         "sparse" / empty.
        max_new:         maximum number of Gaussians to add per call.
        per_pixel_loss:  optional ``[H, W]`` loss map; when provided, only
                         pixels above *loss_quantile* within the sparse set
                         are selected.
        loss_quantile:   quantile threshold within sparse high-loss pixels.

    Returns:
        int – number of Gaussians added (0 if nothing was done).
    """
    device = gaussians.get_xyz.device
    mono_raw = depth_est_entry["raw"]
    surf_depth = render_pkg["surf_depth"]               # [1, H, W]
    rend_alpha = render_pkg["rend_alpha"]                # [1, H, W]
    H, W = surf_depth.shape[1], surf_depth.shape[2]

    # ---- 1. Align mono depth to rendered depth scale ----
    align_mask = None
    if sky_mask is not None:
        align_mask = sky_mask
    aligned_depth, scale, shift = _align_mono_to_rendered(
        mono_raw, surf_depth, rend_alpha, mask=align_mask
    )
    if aligned_depth is None:
        return 0

    aligned_depth = aligned_depth.clamp_min(1e-4)

    # ---- 2. Find sparse pixels: low alpha + valid mono depth + not sky ----
    sparse = rend_alpha.squeeze() < alpha_threshold
    valid_mono = aligned_depth > 1e-4

    combined = sparse & valid_mono
    if sky_mask is not None:
        sky_m = sky_mask.squeeze().to(device)
        if sky_m.shape[0] != H or sky_m.shape[1] != W:
            sky_m = F.interpolate(
                sky_m.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest"
            ).squeeze(0).squeeze(0)
        combined = combined & (sky_m > 0.5)

    # Optionally prioritise by loss
    if per_pixel_loss is not None and combined.any():
        losses_at_sparse = per_pixel_loss.squeeze()[combined]
        if losses_at_sparse.numel() > 10:
            thresh = torch.quantile(losses_at_sparse, loss_quantile)
            loss_filter = per_pixel_loss.squeeze() >= thresh
            combined = combined & loss_filter

    if combined.sum() == 0:
        return 0

    ys, xs = torch.where(combined)

    # Sub-sample if too many
    K = ys.shape[0]
    if K > max_new:
        perm = torch.randperm(K, device=device)[:max_new]
        ys = ys[perm]
        xs = xs[perm]

    # ---- 3. Unproject to 3D world coordinates ----
    depths = aligned_depth[ys, xs]

    tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

    ndc_x = (2.0 * xs.float() / W - 1.0)
    ndc_y = (2.0 * ys.float() / H - 1.0)

    cam_x = ndc_x * tanfovx * depths
    cam_y = ndc_y * tanfovy * depths
    cam_z = depths

    pts_cam = torch.stack([cam_x, cam_y, cam_z, torch.ones_like(cam_z)], dim=-1)

    w2c = viewpoint_cam.world_view_transform.T
    c2w = torch.inverse(w2c).to(device)
    pts_world = (c2w @ pts_cam.T).T[:, :3]

    n_new = pts_world.shape[0]
    if n_new == 0:
        return 0

    # ---- 4. Initialise Gaussian attributes (same scheme as adaptive_densify) ----
    sh_dim = (gaussians.max_sh_degree + 1) ** 2

    # Scale: small surfels – use median existing scale or a fraction of scene extent
    median_scale_lin = gaussians.get_scaling.median(dim=0).values
    init_scale_lin = median_scale_lin * 0.5
    new_scaling = gaussians.scaling_inverse_activation(
        init_scale_lin.unsqueeze(0).expand(n_new, -1).clone()
    )

    new_rotation = torch.randn(n_new, 4, device=device)
    new_rotation = F.normalize(new_rotation, dim=-1)

    new_opacity = gaussians.inverse_opacity_activation(
        0.15 * torch.ones(n_new, 1, device=device)
    )

    new_features_dc_pos = torch.zeros(n_new, 1, 3, device=device) + 0.02
    new_features_rest_pos = torch.zeros(n_new, sh_dim - 1, 3, device=device) + 0.01
    new_features_dc_neg = torch.zeros(n_new, 1, 3, device=device) + 0.02
    new_features_rest_neg = torch.zeros(n_new, sh_dim - 1, 3, device=device) + 0.01

    mean_albedo = gaussians._albedo.detach().mean(dim=0, keepdim=True)
    new_albedo = mean_albedo.expand(n_new, -1).clone()
    mean_roughness = gaussians._roughness.detach().mean(dim=0, keepdim=True)
    mean_metallic = gaussians._metallic.detach().mean(dim=0, keepdim=True)
    new_roughness = mean_roughness.expand(n_new, -1).clone()
    new_metallic = mean_metallic.expand(n_new, -1).clone()

    new_casts_shadow = torch.ones(n_new, device=device)

    gaussians.densification_postfix(
        pts_world, new_features_dc_pos, new_features_rest_pos,
        new_features_dc_neg, new_features_rest_neg,
        new_albedo, new_opacity, new_scaling, new_rotation,
        new_casts_shadow, new_roughness=new_roughness, new_metallic=new_metallic
    )

    print(f"[DepthEst Densify] Added {n_new} Gaussians at sparse surfaces "
          f"(total now: {gaussians.get_xyz.shape[0]})")
    return n_new
