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
