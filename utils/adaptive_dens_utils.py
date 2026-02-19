"""
Adaptive grid-based densification utilities.

Splits the scene bounding box into a 3D voxel grid, accumulates per-pixel
photometric loss into the grid via depth-unprojection, and identifies sparse
high-loss cells where additional Gaussians should be spawned.
"""

import torch
import torch.nn.functional as F


class AdaptiveDensGrid:
    """Maintains a voxel grid over the scene for loss-driven densification.

    Workflow (called from train.py):
        1. ``__init__``  – builds the grid AABB from the initial Gaussian positions.
        2. ``accumulate`` – every training iteration, unprojects the per-pixel loss
           map using the depth buffer and camera intrinsics, then splatts it into
           the voxel grid.
        3. ``trigger_densification`` – every *N* iterations, counts Gaussians per
           cell, finds cells that are simultaneously high-loss AND sparse, and
           asks ``GaussianModel.adaptive_densify_from_loss_grid`` to spawn new
           Gaussians there.  Then resets the accumulator.
    """

    def __init__(
        self,
        gaussians,
        cameras_extent,
        grid_resolution=32,
        padding_factor=0.1,
        fill_empty=True,
        zero_depth_max_pixels=4096,
        zero_depth_samples=1,
        surface_samples=1,
        surface_jitter=0.0,
        ema_decay=0.8,
        use_highfreq=True,
        highfreq_boost=0.75,
        highfreq_quantile=0.6,
        hole_score_quantile=0.5,
        camera_center=None,
        max_densify_distance=-1.0,
    ):
        """
        Args:
            gaussians: GaussianModel – used to read current positions for AABB.
            cameras_extent: float – scene radius from Scene (nerf_normalization).
            grid_resolution: int – number of voxels per axis.
            padding_factor: float – fraction of AABB size added as padding.
        """
        self.grid_res = grid_resolution
        device = gaussians.get_xyz.device

        self.fill_empty = bool(fill_empty)
        self.zero_depth_max_pixels = int(zero_depth_max_pixels)
        self.zero_depth_samples = int(zero_depth_samples)
        self.surface_samples = max(int(surface_samples), 1)
        self.surface_jitter = float(surface_jitter)
        self.ema_decay = float(ema_decay)
        self.use_highfreq = bool(use_highfreq)
        self.highfreq_boost = float(highfreq_boost)
        self.highfreq_quantile = float(highfreq_quantile)
        self.hole_score_quantile = float(hole_score_quantile)
        self.camera_center = camera_center.to(device) if torch.is_tensor(camera_center) else None
        self.max_densify_distance = float(max_densify_distance)

        # Build AABB from current Gaussian positions with some padding
        xyz = gaussians.get_xyz.detach()
        pts_min = xyz.min(dim=0).values
        pts_max = xyz.max(dim=0).values
        extent = pts_max - pts_min
        pad = extent * padding_factor
        self.grid_min = (pts_min - pad).to(device)
        self.grid_max = (pts_max + pad).to(device)

        # Accumulators
        self.loss_grid = torch.zeros(grid_resolution, grid_resolution, grid_resolution,
                                     device=device, dtype=torch.float32)
        self.hit_count = torch.zeros_like(self.loss_grid)  # how many pixels mapped here
        self.loss_ema = torch.zeros_like(self.loss_grid)

    # ------------------------------------------------------------------
    # Accumulate
    # ------------------------------------------------------------------
    def accumulate(self, per_pixel_loss, depth_map, viewpoint_cam, mask=None, reference_image=None):
        """Unproject pixels into 3D using *depth_map* and scatter their loss
        into the voxel grid.

        Args:
            per_pixel_loss: [H, W] tensor – per-pixel L1 (or similar) loss.
            depth_map: [H, W] tensor – rendered surface depth.
            viewpoint_cam: Camera object with projection / view matrices.
            mask: optional [1 or 3, H, W] mask (0 = ignore pixel).
        """
        device = self.loss_grid.device
        H, W = per_pixel_loss.shape

        # Flatten mask
        if mask is not None:
            if mask.dim() == 3:
                m = mask[0]  # [H, W]
            else:
                m = mask
            in_mask = (m > 0.5)
        else:
            in_mask = torch.ones_like(depth_map, dtype=torch.bool)

        valid_surface = in_mask & (depth_map > 0)
        valid_holes = in_mask & (depth_map <= 0)

        if (valid_surface.sum() == 0) and (not (self.fill_empty and valid_holes.any())):
            return None

        # Build densification score map (photometric error + optional high-frequency boost)
        score_map = per_pixel_loss.clone()
        highfreq_map = torch.zeros_like(per_pixel_loss)
        if self.use_highfreq and reference_image is not None:
            highfreq_map = self._compute_highfreq_map(reference_image)
            valid_hf = in_mask & torch.isfinite(highfreq_map)
            if valid_hf.any():
                hf_thresh = torch.quantile(highfreq_map[valid_hf], self.highfreq_quantile)
                hf_mask = highfreq_map >= hf_thresh
                score_map = torch.where(hf_mask & in_mask, score_map * (1.0 + self.highfreq_boost), score_map)

        selected_pixel_mask = torch.zeros_like(per_pixel_loss, dtype=torch.bool)

        # NDC → camera-space unprojection
        # Use camera FoV to build inverse projection
        import math
        tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

        # Camera → world
        # world_view_transform is W2C (4×4 row-major in this codebase).
        # Inverse gives C2W.
        w2c = viewpoint_cam.world_view_transform.T  # make column-major
        c2w = torch.inverse(w2c).to(device)

        # --- 1) Standard accumulation for pixels with valid depth ---
        if valid_surface.any():
            ys, xs = torch.where(valid_surface)
            depths = depth_map[ys, xs]  # [K]
            losses = score_map[ys, xs]  # [K]
            selected_pixel_mask[ys, xs] = True

            # Pixel to normalised screen coords  [-1, 1]
            ndc_x = (2.0 * xs.float() / W - 1.0)
            ndc_y = (2.0 * ys.float() / H - 1.0)

            # Sample around the surface depth to cover a small 3D neighborhood,
            # not only a single shell. This helps populate near-empty pockets.
            for _ in range(self.surface_samples):
                if self.surface_jitter > 0:
                    jitter = (torch.rand_like(depths) * 2.0 - 1.0) * self.surface_jitter
                    sample_depths = (depths * (1.0 + jitter)).clamp_min(1e-6)
                else:
                    sample_depths = depths

                cam_x = ndc_x * tanfovx * sample_depths
                cam_y = ndc_y * tanfovy * sample_depths
                cam_z = sample_depths

                pts_cam = torch.stack([cam_x, cam_y, cam_z, torch.ones_like(cam_z)], dim=-1)  # [K, 4]
                pts_world = (c2w @ pts_cam.T).T[:, :3]  # [K, 3]

                # Split the same loss mass across samples to keep scale stable.
                self._splat_points(pts_world, losses / float(self.surface_samples))

        # --- 2) Fill empty / hole pixels (depth==0) by sampling along the ray within the grid AABB ---
        if self.fill_empty and valid_holes.any() and self.zero_depth_samples > 0:
            ys0, xs0 = torch.where(valid_holes)

            # Subsample to keep this fast
            K0 = ys0.shape[0]
            if K0 > self.zero_depth_max_pixels:
                perm = torch.randperm(K0, device=device)[: self.zero_depth_max_pixels]
                ys0 = ys0[perm]
                xs0 = xs0[perm]
                K0 = ys0.shape[0]

            # Pixel to normalised screen coords  [-1, 1]
            ndc_x0 = (2.0 * xs0.float() / W - 1.0)
            ndc_y0 = (2.0 * ys0.float() / H - 1.0)

            # Ray direction in camera space (z forward)
            dir_cam = torch.stack([ndc_x0 * tanfovx, ndc_y0 * tanfovy, torch.ones_like(ndc_x0)], dim=-1)
            dir_cam = torch.nn.functional.normalize(dir_cam, dim=-1)
            dir_world = (c2w[:3, :3] @ dir_cam.T).T
            dir_world = torch.nn.functional.normalize(dir_world, dim=-1)

            origin = viewpoint_cam.camera_center.to(device).view(1, 3)

            # Ray-AABB intersection (slab method)
            inv_d = 1.0 / torch.where(dir_world.abs() < 1e-8, torch.full_like(dir_world, 1e-8), dir_world)
            t0s = (self.grid_min.view(1, 3) - origin) * inv_d
            t1s = (self.grid_max.view(1, 3) - origin) * inv_d
            tmin = torch.minimum(t0s, t1s).max(dim=-1).values
            tmax = torch.maximum(t0s, t1s).min(dim=-1).values

            hit = (tmax > torch.maximum(tmin, torch.zeros_like(tmin)))
            if hit.any():
                tmin = torch.maximum(tmin, torch.zeros_like(tmin))
                tlen = (tmax - tmin).clamp_min(1e-6)

                # Sample one or more points along the valid segment
                losses0 = score_map[ys0, xs0]
                hit_idx = torch.where(hit)[0]

                # Keep only higher-scoring hole pixels so empty-space densification
                # focuses on difficult regions instead of random flat areas.
                if hit_idx.numel() > 0:
                    losses_hit = losses0[hit_idx]
                    hole_thresh = torch.quantile(losses_hit, self.hole_score_quantile)
                    keep = losses_hit >= hole_thresh
                    hit_idx = hit_idx[keep]

                if hit_idx.numel() == 0:
                    return {
                        "error_map": per_pixel_loss.detach(),
                        "score_map": score_map.detach(),
                        "highfreq_map": highfreq_map.detach(),
                        "selected_mask": selected_pixel_mask.detach(),
                    }

                origin_h = origin.expand(hit_idx.shape[0], 3)
                dir_h = dir_world[hit_idx]
                tmin_h = tmin[hit_idx]
                tlen_h = tlen[hit_idx]
                loss_h = losses0[hit_idx]
                selected_pixel_mask[ys0[hit_idx], xs0[hit_idx]] = True

                for _ in range(self.zero_depth_samples):
                    t = tmin_h + torch.rand_like(tmin_h) * tlen_h
                    pts_world = origin_h + dir_h * t.unsqueeze(-1)
                    self._splat_points(pts_world, loss_h)

        return {
            "error_map": per_pixel_loss.detach(),
            "score_map": score_map.detach(),
            "highfreq_map": highfreq_map.detach(),
            "selected_mask": selected_pixel_mask.detach(),
        }

    def _compute_highfreq_map(self, image):
        """Compute a normalized high-frequency magnitude map from an RGB image [3,H,W]."""
        if image.dim() != 3 or image.shape[0] not in (1, 3):
            if image.dim() >= 2:
                return torch.zeros(image.shape[-2:], dtype=torch.float32, device=image.device)
            return torch.zeros((1, 1), dtype=torch.float32, device=self.loss_grid.device)

        img = image.float().unsqueeze(0)  # [1,C,H,W]
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img[:, 0:1]

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy).squeeze(0).squeeze(0)

        denom = torch.quantile(mag.view(-1), 0.99).clamp_min(1e-6)
        return (mag / denom).clamp(0.0, 1.0)

    def _splat_points(self, pts_world, losses):
        """Scatter-add a set of world points + scalar losses into the voxel grid."""
        device = self.loss_grid.device
        grid_size = (self.grid_max - self.grid_min).clamp_min(1e-8)
        rel = (pts_world - self.grid_min) / grid_size  # [K, 3] in [0,1]
        idx = (rel * self.grid_res).long().clamp(0, self.grid_res - 1)
        flat_idx = idx[:, 0] * (self.grid_res * self.grid_res) + idx[:, 1] * self.grid_res + idx[:, 2]
        self.loss_grid.view(-1).scatter_add_(0, flat_idx, losses.to(device))
        self.hit_count.view(-1).scatter_add_(0, flat_idx, torch.ones_like(losses, device=device))

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------
    def trigger_densification(self, gaussians, loss_thresh_quantile=0.7,
                              count_thresh=2, max_new_gaussians=512,
                              dens_everything=False,
                              dens_everything_per_cell=1,
                              dens_everything_max_gaussians=50000):
        """Count Gaussians per cell, call adaptive densify, then reset."""
        device = self.loss_grid.device
        center_mask = self._get_center_distance_mask(device)

        # Count Gaussians per cell
        grid_counts = self._count_gaussians_per_cell(gaussians)

        if dens_everything:
            # Debug mode (constrained): densify every *scene-relevant* grid cell,
            # not the whole padded AABB.
            observed_cells = self.hit_count > 0
            occupied_cells = grid_counts > 0

            if occupied_cells.any():
                occupied_dilated = F.max_pool3d(
                    occupied_cells.float().unsqueeze(0).unsqueeze(0),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ).squeeze(0).squeeze(0) > 0.5
            else:
                occupied_dilated = torch.zeros_like(occupied_cells, dtype=torch.bool)

            target_cells = observed_cells | occupied_dilated
            if center_mask is not None:
                target_cells = target_cells & center_mask
            if not target_cells.any():
                # If no scene evidence has been accumulated yet, avoid scattering
                # Gaussians everywhere in empty space.
                print("[Adaptive Densification][dens_everything] No observed/occupied cells yet; skipping this trigger.")
                self.loss_grid.zero_()
                self.hit_count.zero_()
                self._update_aabb(gaussians)
                return

            score_loss = torch.zeros_like(self.loss_grid, device=device, dtype=torch.float32)
            score_loss[target_cells] = 1.0

            n_cells = int(target_cells.sum().item())
            requested_budget = int(max(1, dens_everything_per_cell)) * n_cells
            budget = min(int(max(1, dens_everything_max_gaussians)), requested_budget)

            gaussians.adaptive_densify_from_loss_grid(
                score_loss,
                torch.zeros_like(grid_counts),
                self.grid_min,
                self.grid_max,
                loss_thresh_quantile=0.0,
                count_thresh=1,
                max_new_gaussians=budget,
            )
            print(f"[Adaptive Densification][dens_everything] Requested {requested_budget} ({dens_everything_per_cell}/cell over {n_cells} scene cells), capped to {budget}.")
        else:
            # Normalise accumulated loss by hit count (average loss per pixel-hit)
            avg_loss = self.loss_grid.clone()
            observed = self.hit_count > 0
            avg_loss[observed] /= self.hit_count[observed]

            # Temporal smoothing to make densification decisions more stable.
            self.loss_ema = self.ema_decay * self.loss_ema + (1.0 - self.ema_decay) * avg_loss
            score_loss = torch.maximum(avg_loss, self.loss_ema)
            if center_mask is not None:
                score_loss = torch.where(center_mask, score_loss, torch.zeros_like(score_loss))

            # Delegate to GaussianModel
            gaussians.adaptive_densify_from_loss_grid(
                score_loss, grid_counts,
                self.grid_min, self.grid_max,
                loss_thresh_quantile=loss_thresh_quantile,
                count_thresh=count_thresh,
                max_new_gaussians=max_new_gaussians,
            )

        # Reset accumulators for next window
        self.loss_grid.zero_()
        self.hit_count.zero_()

        # Update AABB to cover any new Gaussians
        self._update_aabb(gaussians)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _count_gaussians_per_cell(self, gaussians):
        """Return a [Gx, Gy, Gz] integer tensor counting Gaussians per cell."""
        device = self.loss_grid.device
        xyz = gaussians.get_xyz.detach()
        grid_size = self.grid_max - self.grid_min
        rel = (xyz - self.grid_min) / grid_size
        idx = (rel * self.grid_res).long().clamp(0, self.grid_res - 1)

        counts = torch.zeros(self.grid_res, self.grid_res, self.grid_res,
                             device=device, dtype=torch.float32)
        flat_idx = idx[:, 0] * (self.grid_res ** 2) + idx[:, 1] * self.grid_res + idx[:, 2]
        counts.view(-1).scatter_add_(0, flat_idx, torch.ones(xyz.shape[0], device=device))
        return counts

    def _update_aabb(self, gaussians, padding_factor=0.05):
        """Expand AABB if new Gaussians lie outside the current box."""
        xyz = gaussians.get_xyz.detach()
        pts_min = xyz.min(dim=0).values
        pts_max = xyz.max(dim=0).values
        extent = pts_max - pts_min
        pad = extent * padding_factor
        self.grid_min = torch.min(self.grid_min, pts_min - pad)
        self.grid_max = torch.max(self.grid_max, pts_max + pad)

    def _get_center_distance_mask(self, device):
        """Return voxel mask limited to a radius around mean camera center."""
        if self.camera_center is None or self.max_densify_distance <= 0:
            return None

        G = self.grid_res
        grid_shape = torch.tensor([G, G, G], device=device, dtype=torch.float32)
        cell_size = (self.grid_max - self.grid_min) / grid_shape

        ix = torch.arange(G, device=device, dtype=torch.float32)
        iy = torch.arange(G, device=device, dtype=torch.float32)
        iz = torch.arange(G, device=device, dtype=torch.float32)
        gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing='ij')

        centers = torch.stack([
            self.grid_min[0] + (gx + 0.5) * cell_size[0],
            self.grid_min[1] + (gy + 0.5) * cell_size[1],
            self.grid_min[2] + (gz + 0.5) * cell_size[2],
        ], dim=-1)

        dist = torch.norm(centers - self.camera_center.view(1, 1, 1, 3), dim=-1)
        return dist <= self.max_densify_distance
