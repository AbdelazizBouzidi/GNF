import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import pysdf
import os

__all__ = ['SDFDataset']

class SDFDataset(Dataset):
    """A sampling SDF dataset compatible with main_sdf.py.

    Parameters
    ----------
    path : str
        Path to a watertight mesh file (e.g. .obj).
    size : int, default 100
        Number of ``__getitem__`` iterations the dataloader should yield. Each
        iteration resamples ``num_samples`` query points and their signed
        distances on-the-fly so the network sees new data every epoch.
    num_samples : int, default 2**18
        Number of spatial samples drawn per iteration; **must** be divisible by
        8 because of the sampling strategy (1/8 surface, 7/8 random).
    surface_only : bool, default False
        If True, only returns points sampled on the mesh surface.
    clip_sdf : float or None, default None
        If given, clamps the returned SDF values into [-clip_sdf, clip_sdf].
    """

    def __init__(self, path: str, size: int = 100, num_samples: int = 2 ** 18,
                 surface_only: bool = False, clip_sdf: float | None = None):
        super().__init__()

        assert num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.path = path
        self.size = size
        self.num_samples = num_samples
        self.surface_only = surface_only
        self.clip_sdf = clip_sdf

        # Load mesh ---------------------------------------------------------
        self.mesh: trimesh.Trimesh = trimesh.load(path, force='mesh')
        if not self.mesh.is_watertight:
            print("[WARN] mesh is not watertight! SDF values may be invalid.")

        # Normalize mesh to fit roughly within [-1, 1] bounding box ----------
        vs = self.mesh.vertices
        vmin, vmax = vs.min(0), vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        self.mesh.vertices = (vs - v_center[None, :]) * v_scale

        # Build fast SDF query object ---------------------------------------
        self._sdf = pysdf.SDF(self.mesh.vertices, self.mesh.faces)

    # ---------------------------------------------------------------------
    #  PyTorch mandatory methods
    # ---------------------------------------------------------------------
    def __len__(self):
        return self.size

    def __getitem__(self, _):
        # ``_`` is ignored – we re-sample every call so idx carries no meaning.
        n = self.num_samples
        sdf_values = np.zeros((n, 1), dtype=np.float32)

        if self.surface_only:
            points = self.mesh.sample(n).astype(np.float32)
        else:
            # 1/8 from surface, 7/8 uniformly in bounding box.
            surface_pts = self.mesh.sample(n // 8)
            uniform_pts = np.random.rand(7 * n // 8, 3) * 2 - 1  # [-1, 1]^3
            points = np.concatenate([surface_pts, uniform_pts], axis=0).astype(np.float32)

            # Evaluate SDF for non-surface points.  Surface samples keep 0.
            sdf_values[n // 8:] = -self._sdf(points[n // 8:])[:, None].astype(np.float32)

        if self.clip_sdf is not None:
            np.clip(sdf_values, -self.clip_sdf, self.clip_sdf, out=sdf_values)

        return {
            'points': points,    # (N, 3) float32
            'sdfs': sdf_values   # (N, 1) float32
        }