"""3DGS renderer backend using `gsplat`.

Mirrors the legacy `SplatRenderer` interface (init_cameras / add_splats /
transform_many / render / render_raw) but stays close to the canonical
gsplat call shape used in `manipverse.reconstruction.splat_render`:

  - Load PLY directly with `plyfile` (skip GaussianModel, which truncates
    scales to [:2] for the surfel rasterizer).
  - Store *already-activated* tensors: scales = exp(scale_*), opacities =
    sigmoid(opacity), colors = clip(0.5 + SH_C0 * f_dc, 0, 1) (Marble + SAM3D
    PLYs are DC-only baked RGB).
  - Call `gsplat.rasterization` with `colors=` (no `sh_degree=`) and **no
    `backgrounds=` arg** — composite the bg color manually so we work
    against any gsplat 1.4+/1.5+ API.
  - 2DGS surfel PLYs (only scale_0/scale_1) are padded with a tiny z-scale
    so they render as flat disks under the 3D rasterizer.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from plyfile import PlyData

from gsplat import rasterization

import polaris.utils as utils
from polaris.splat_renderer.scene.cameras import Camera


SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# PLY loader (plain tensors, activations baked in)
# ---------------------------------------------------------------------------
def load_gaussian_ply(ply_path: str | Path, device) -> dict:
    """Load an inria-format 3DGS / 2DGS PLY into plain tensors.

    Returns a dict with keys: means [N,3], quats [N,4], scales [N,3] (post-exp),
    opacities [N] (post-sigmoid), colors [N,3] (RGB in [0,1] from f_dc).
    """
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    # Scales: read whatever scale_* dims are present, pad to 3 for gsplat.
    scale_props = sorted(
        [p.name for p in v.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if not scale_props:
        raise ValueError(f"{ply_path}: no scale_* properties")
    cols = [np.asarray(v[a], dtype=np.float32) for a in scale_props]
    if len(cols) < 3:
        # 2DGS surfel: pad missing axes with very-negative log-scale → flat disk.
        pad = np.full_like(cols[0], -20.0, dtype=np.float32)
        cols = cols + [pad] * (3 - len(cols))
    scales = np.stack(cols[:3], axis=-1)
    scales = np.exp(scales)

    quats = np.stack(
        [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
    ).astype(np.float32)
    quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)

    opacities = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))

    f_dc = np.stack(
        [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1
    ).astype(np.float32)
    colors = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    return {
        "means":     torch.from_numpy(xyz).to(dev),
        "quats":     torch.from_numpy(quats).to(dev),
        "scales":    torch.from_numpy(scales).to(dev),
        "opacities": torch.from_numpy(opacities).to(dev),
        "colors":    torch.from_numpy(colors).to(dev),
    }


# ---------------------------------------------------------------------------
# SplatRenderer
# ---------------------------------------------------------------------------
class SplatRenderer:
    """Drop-in replacement for the legacy SplatRenderer using gsplat.

    State layout (all on `self.device`):
      means     [N, 3]   # world positions, mutated by transform_many
      quats     [N, 4]   # world rotations, mutated by transform_many
      scales    [N, 3]   # post-exp, never mutated
      opacities [N]      # post-sigmoid, never mutated
      colors    [N, 3]   # RGB in [0,1], copied from original on transform

    Original (un-transformed) versions of mutable tensors are kept under
    `self._orig_*` so transform_many can replay any pose without drift.
    """

    def __init__(self, splats, bg_color=(0.5, 0.5, 0.5), device=0):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.bg_color = torch.tensor(list(bg_color), device=self.device, dtype=torch.float32)
        self.pcds: dict[str, str | Path] = dict(splats)
        self.splat_mapping: dict[str, tuple[int, int]] = {}

        self.means = torch.empty((0, 3), device=self.device)
        self.quats = torch.empty((0, 4), device=self.device)
        self.scales = torch.empty((0, 3), device=self.device)
        self.opacities = torch.empty((0,), device=self.device)
        self.colors = torch.empty((0, 3), device=self.device)

        for name, ply_path in self.pcds.items():
            self._append_one(name, ply_path)
        self._snapshot_original()
        print(
            f"[gsplat-renderer] loaded {len(self.pcds)} splats, "
            f"{self.means.shape[0]} gaussians (DC-only RGB, gsplat backend)"
        )

    # ---- PLY append ---------------------------------------------------------
    def _append_one(self, name: str, ply_path):
        d = load_gaussian_ply(ply_path, self.device)
        cur = self.means.shape[0]
        self.splat_mapping[name] = (cur, cur + d["means"].shape[0])
        self.means     = torch.cat([self.means,     d["means"]],     dim=0)
        self.quats     = torch.cat([self.quats,     d["quats"]],     dim=0)
        self.scales    = torch.cat([self.scales,    d["scales"]],    dim=0)
        self.opacities = torch.cat([self.opacities, d["opacities"]], dim=0)
        self.colors    = torch.cat([self.colors,    d["colors"]],    dim=0)

    def _snapshot_original(self):
        self._orig_means  = self.means.clone()
        self._orig_quats  = self.quats.clone()
        self._orig_colors = self.colors.clone()

    def add_splats(self, splats: dict):
        for name, ply_path in splats.items():
            self._append_one(name, ply_path)
            self.pcds[name] = ply_path
        self._snapshot_original()

    # ---- camera setup -------------------------------------------------------
    def init_cameras(self, cam_dict: dict):
        self.cameras: dict[str, Camera] = {}
        for name, p in cam_dict.items():
            self.cameras[name] = Camera(
                colmap_id=0,
                R=np.eye(3),
                T=np.array([0.0, 0.0, 0.0]),
                FoVy=p["fovy"],
                FoVx=p["fovx"],
                image=torch.zeros(3, p["res"][0], p["res"][1]),
                gt_alpha_mask=None,
                image_name="test",
                uid=0,
                data_device=self.device,
            )

    # ---- per-step transforms ------------------------------------------------
    def transform_many(self, all_transforms: dict):
        """Apply per-splat (translate, rotate-quat) updates from sim state."""
        if not all_transforms:
            return
        with torch.no_grad():
            indices_l, xyzs_l, quats_l = [], [], []
            for name, (translate, rotate) in all_transforms.items():
                translate = translate.to(self.device).float()
                rotate = rotate.to(self.device).float()
                start, end = self.splat_mapping[name]
                xyz0 = self._orig_means[start:end]
                q0   = self._orig_quats[start:end]
                xyzs_l.append(utils.rotate_vector_by_quaternion(rotate, xyz0) + translate)
                quats_l.append(utils.multiply_quaternions(rotate, q0))
                indices_l.append(torch.arange(start, end, device=self.device))
            idx = torch.cat(indices_l)
            self.means[idx]  = torch.cat(xyzs_l)
            self.quats[idx]  = torch.cat(quats_l)

    # ---- rasterization ------------------------------------------------------
    def _intrinsics_K(self, cam: Camera) -> torch.Tensor:
        W, H = int(cam.image_width), int(cam.image_height)
        fx = 0.5 * W / math.tan(0.5 * cam.FoVx)
        fy = 0.5 * H / math.tan(0.5 * cam.FoVy)
        return torch.tensor(
            [[fx, 0.0, 0.5 * W], [0.0, fy, 0.5 * H], [0.0, 0.0, 1.0]],
            dtype=torch.float32, device=self.device,
        )

    def _viewmat(self, cam: Camera) -> torch.Tensor:
        # cam.world_view_transform is stored column-major (transposed); undo
        # the transpose to recover the standard 4x4 world->cam viewmat.
        return cam.world_view_transform.T.contiguous().to(self.device).float()

    def _render_one(self, cam: Camera) -> torch.Tensor:
        """Returns RGB tensor of shape [H, W, 3] in [0,1]."""
        W, H = int(cam.image_width), int(cam.image_height)
        K = self._intrinsics_K(cam)
        V = self._viewmat(cam)
        # Quats are mutated by transform_many but never re-normalized — do it
        # at render time (cheap; matches load_gaussian_ply convention).
        quats = self.quats / (self.quats.norm(dim=-1, keepdim=True) + 1e-8)
        rendered, alpha, _info = rasterization(
            means=self.means,
            quats=quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,            # RGB; no sh_degree
            viewmats=V[None],
            Ks=K[None],
            width=W, height=H,
            render_mode="RGB",
            packed=False,
        )
        rgb = rendered[0, ..., :3]                 # [H, W, 3]
        a   = alpha[0, ..., 0:1] if alpha.dim() == 4 else alpha[0]   # [H, W, 1]
        # Composite background manually (gsplat 1.4 / 1.5 API agnostic).
        rgb = rgb + (1.0 - a) * self.bg_color.view(1, 1, 3)
        return rgb.clamp(0.0, 1.0)

    def render_raw(self, extrinsics_dict: dict):
        images = {}
        for name in self.cameras:
            if name in extrinsics_dict:
                self.cameras[name].set_extrinsics(
                    extrinsics_dict[name]["rot"], extrinsics_dict[name]["pos"]
                )
                images[name] = self._render_one(self.cameras[name]).clone()
        return images

    def render(self, extrinsics_dict: dict):
        # Match legacy axis-permutation so cam frames stay consistent with the
        # surfel renderer call site.
        p_mat = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        images = {}
        for name in self.cameras:
            if name in extrinsics_dict:
                cam_r = extrinsics_dict[name]["rot"] @ p_mat
                self.cameras[name].set_extrinsics(cam_r, extrinsics_dict[name]["pos"])
            images[name] = self._render_one(self.cameras[name]).clone()
        return images
