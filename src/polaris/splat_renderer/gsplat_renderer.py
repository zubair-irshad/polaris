"""3DGS renderer backend using `gsplat`.

Drop-in alternative to the diff-surfel-rasterization (2DGS) `render` in
`gaussian_renderer.py`. Used when assets come from 3DGS pipelines such as
Marble (background scene) or SAM-3D-Objects (rigid objects), which produce
PLYs with 3-dim scales and full 3D anisotropic covariances.

Returns the same dict keys as the surfel renderer so the rest of the stack
(`SplatRenderer.render`) can stay agnostic. Surfel-only outputs (rend_dist,
surf_normal) are returned as zeros to satisfy callers.
"""

import math
import torch

from gsplat.rendering import rasterization

from polaris.splat_renderer.scene.gaussian_model import GaussianModel
import polaris.splat_renderer.utils.sh_utils as sh_utils


def _fov_to_intrinsics(fovx: float, fovy: float, W: int, H: int) -> torch.Tensor:
    fx = 0.5 * W / math.tan(0.5 * fovx)
    fy = 0.5 * H / math.tan(0.5 * fovy)
    K = torch.tensor(
        [[fx, 0.0, 0.5 * W], [0.0, fy, 0.5 * H], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return K


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color=None,
):
    device = pc.get_xyz.device
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    # World->View transform stored transposed (col-major), undo the transpose
    # to get the standard 4x4 viewmat that gsplat expects.
    viewmat = viewpoint_camera.world_view_transform.T.contiguous().to(device)
    K = _fov_to_intrinsics(viewpoint_camera.FoVx, viewpoint_camera.FoVy, W, H).to(device)

    means = pc.get_xyz
    quats = pc.get_rotation  # already normalized via rotation_activation
    scales = pc.get_scaling * scaling_modifier
    if scales.shape[-1] == 2:
        # Surfel asset rendered with the 3DGS backend — pad the missing axis
        # with a tiny scale so the Gaussian behaves like a flat disk.
        pad = torch.full_like(scales[..., :1], 1e-6)
        scales = torch.cat([scales, pad], dim=-1)
    elif scales.shape[-1] != 3:
        raise ValueError(f"gsplat backend expects 2 or 3 scale dims, got {scales.shape[-1]}")

    opacities = pc.get_opacity.squeeze(-1)

    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        # Concatenate DC + rest SH coefficients into [N, K, 3] (gsplat layout).
        # gaussian_model stores features as [N, K, 3] after the transpose in
        # load_ply, so DC is [N, 1, 3] and rest is [N, K-1, 3].
        colors = pc.get_features  # [N, K, 3]
        sh_degree = int(pc.active_sh_degree)

    bg = bg_color.to(device).reshape(1, 3) if bg_color is not None else None

    rendered, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None],          # [1, 4, 4]
        Ks=K[None],                      # [1, 3, 3]
        width=W,
        height=H,
        sh_degree=sh_degree,
        backgrounds=bg,
        render_mode="RGB+ED",            # rgb + expected depth in last channel
        packed=False,
    )

    rgb = rendered[0, ..., :3].permute(2, 0, 1).contiguous()           # [3, H, W]
    depth = rendered[0, ..., 3:4].permute(2, 0, 1).contiguous()        # [1, H, W]
    alpha_chw = alpha[0].permute(2, 0, 1).contiguous()                 # [1, H, W]

    radii = info.get("radii", None)
    if radii is None:
        radii = torch.zeros(means.shape[0], device=device, dtype=torch.int32)
    else:
        radii = radii.squeeze(0).to(torch.int32)

    zeros_3hw = torch.zeros((3, H, W), device=device, dtype=rgb.dtype)
    zeros_1hw = torch.zeros((1, H, W), device=device, dtype=rgb.dtype)

    rets = {
        "render": rgb,
        "viewspace_points": means,
        "visibility_filter": radii > 0,
        "radii": radii,
        "rend_alpha": alpha_chw,
        "rend_normal": zeros_3hw,    # 3DGS backend does not produce normals
        "rend_dist": zeros_1hw,
        "surf_depth": depth,
        "surf_normal": zeros_3hw,
    }
    return rets


# ---------------------------------------------------------------------------
# SplatRenderer (gsplat backend)
#
# Mirrors the legacy SplatRenderer interface (init_cameras / add_splats /
# transform_many / render / render_raw) but uses the gsplat-based `render`
# defined above and loads 3DGS PLYs (3 scale dims, expected from Marble
# background + SAM-3D-Objects rigid splats). 2DGS PLYs (PolaRiS-Hub stock
# scenes) are rejected here on purpose — point those at the legacy
# `polaris.splat_renderer.SplatRenderer` instead.
# ---------------------------------------------------------------------------
import numpy as np
from plyfile import PlyData
from torch import nn

import polaris.utils as utils
from polaris.splat_renderer.scene.cameras import Camera


def _load_full_scales(ply_path: str, device) -> nn.Parameter:
    """Read scale_0/1[/2] from a PLY without upstream's [:2] truncation.

    - 3DGS PLY (scale_0/1/2)        -> [N, 3] log-scales as authored.
    - 2DGS surfel PLY (scale_0/1)   -> [N, 3] with the third axis padded to
                                       a very negative log-scale (~exp(-20)
                                       ≈ 2e-9) so the Gaussian renders as a
                                       flat disk under gsplat's 3D rasterizer.
    """
    ply = PlyData.read(ply_path)
    props = ply.elements[0]
    available = sorted(
        [p.name for p in props.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    cols = [np.asarray(props[a], dtype=np.float32) for a in available]
    if len(cols) == 0:
        raise ValueError(f"{ply_path}: no scale_* properties in PLY")
    if len(cols) < 3:
        # 2DGS surfel — pad missing axes with a very negative log-scale.
        pad = np.full_like(cols[0], -20.0, dtype=np.float32)
        cols = cols + [pad] * (3 - len(cols))
    scales = np.stack(cols[:3], axis=-1)
    t = torch.from_numpy(scales).to(device)
    return nn.Parameter(t.requires_grad_(True))


class _DummyPipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    depth_ratio = 0.0
    debug = False


class SplatRenderer:
    """3DGS / gsplat-backed renderer with the legacy SplatRenderer API."""

    def __init__(self, splats, bg_color=(0.5, 0.5, 0.5), device=0):
        self.device = device
        self.bg_color = torch.tensor(list(bg_color), device=self.device).float()
        self.pcds = dict(splats)
        self.splat_mapping: dict[str, tuple[int, int]] = {}
        self.pipe = _DummyPipe()

        # One big GaussianModel that all sub-splats are concatenated into.
        # Always sh_degree=3 to match the legacy SplatRenderer.
        self.big_model = GaussianModel(3)
        self.original_big_model = GaussianModel(3)
        self._init_models()
        n = self.big_model.get_xyz.shape[0]
        sh = self.big_model.active_sh_degree
        print(f"[gsplat-renderer] loaded {len(self.pcds)} splats, {n} gaussians, sh_degree={sh}")

    # ---- camera setup --------------------------------------------------------
    def init_cameras(self, cam_dict):
        self.cameras = {}
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

    # ---- model init / append -------------------------------------------------
    def _init_models(self):
        empty_attrs = ("_xyz", "_rotation", "_opacity", "_features_rest", "_features_dc", "_scaling")
        for a in empty_attrs:
            setattr(self.big_model, a, getattr(self.big_model, a).to(self.device))
        for name, ply_path in self.pcds.items():
            self._append_one(name, ply_path)
        self._snapshot_original()

    def _append_one(self, name, ply_path):
        # NB: upstream GaussianModel.load_ply truncates scales to [:2] for the
        # surfel rasterizer, which would lose the z-axis on 3DGS PLYs. So we
        # call load_ply for everything *except* `_scaling`, then overwrite
        # `_scaling` with all available scale dims read directly from the PLY.
        # Real 2DGS PLYs (only scale_0 / scale_1) are padded with a tiny z so
        # gsplat renders them as flat disks (visually equivalent to surfel).
        m = GaussianModel(3)
        m.load_ply(str(ply_path))
        m._scaling = _load_full_scales(str(ply_path), self.device)
        cur = self.big_model._xyz.shape[0]
        self.splat_mapping[name] = (cur, cur + m._xyz.shape[0])
        for a in ("_xyz", "_rotation", "_opacity", "_features_rest", "_features_dc", "_scaling"):
            cat = torch.cat([getattr(self.big_model, a), getattr(m, a).to(self.device)], dim=0)
            setattr(self.big_model, a, cat.requires_grad_())

    def _snapshot_original(self):
        for a in ("_xyz", "_rotation", "_opacity", "_features_rest", "_features_dc", "_scaling"):
            setattr(self.original_big_model, a, getattr(self.big_model, a).clone())

    def add_splats(self, splats):
        for name, ply_path in splats.items():
            self._append_one(name, ply_path)
            self.pcds[name] = ply_path
        self._snapshot_original()

    # ---- per-step transforms -------------------------------------------------
    def transform_many(self, all_transforms):
        with torch.no_grad():
            indices, xyzs, rots, frest = [], [], [], []
            for name, (translate, rotate) in all_transforms.items():
                translate = translate.to(self.device)
                rotate = rotate.to(self.device)
                start, end = self.splat_mapping[name]
                xyzs.append(
                    utils.rotate_vector_by_quaternion(
                        rotate, self.original_big_model._xyz[start:end]
                    ) + translate
                )
                rots.append(
                    utils.multiply_quaternions(
                        rotate, self.original_big_model._rotation[start:end]
                    )
                )
                frest.append(self.original_big_model._features_rest[start:end])
                indices.append(torch.arange(start, end))
            if not indices:
                return
            indices = torch.cat(indices).to(self.device)
            self.big_model._xyz[indices] = torch.cat(xyzs)
            self.big_model._rotation[indices] = torch.cat(rots)
            self.big_model._features_rest[indices] = torch.cat(frest)

    # ---- rasterization -------------------------------------------------------
    def _render_one(self, cam):
        return render(cam, self.big_model, self.pipe, self.bg_color)

    def render_raw(self, extrinsics_dict):
        images = {}
        for name in self.cameras:
            if name in extrinsics_dict:
                self.cameras[name].set_extrinsics(
                    extrinsics_dict[name]["rot"], extrinsics_dict[name]["pos"]
                )
                images[name] = self._render_one(self.cameras[name])["render"].permute(1, 2, 0).clone()
        return images

    def render(self, extrinsics_dict):
        # Match legacy axis-permutation so cam frames stay consistent.
        p_mat = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        images = {}
        for name in self.cameras:
            if name in extrinsics_dict:
                cam_r = extrinsics_dict[name]["rot"] @ p_mat
                self.cameras[name].set_extrinsics(cam_r, extrinsics_dict[name]["pos"])
            images[name] = self._render_one(self.cameras[name])["render"].permute(1, 2, 0).clone()
        return images
