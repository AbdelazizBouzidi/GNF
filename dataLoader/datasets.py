import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from tqdm import tqdm
from torchvision import transforms as T

from .ray_utils import *
from .blender import BlenderDataset
from .tankstemple import TanksTempleDataset
from .llff import LLFFDataset
from .nsvf import NSVF
from .sdf import SDFDataset
from .dtu_objs import DTUDataset
from .colmap import ColmapDataset
from .google_objs import GoogleObjsDataset
from .your_own_data import YourOwnDataset
from .image import ImageDataset

def circle(radius=3.5, h=0.0, axis='z', t0=0, r=1):
    if axis == 'z':
        return lambda t: [radius * np.cos(r * t + t0), radius * np.sin(r * t + t0), h]
    elif axis == 'y':
        return lambda t: [radius * np.cos(r * t + t0), h, radius * np.sin(r * t + t0)]
    else:
        return lambda t: [h, radius * np.cos(r * t + t0), radius * np.sin(r * t + t0)]

def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)

def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2
    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2 == 0] = 1
        return x / l2,

def cat(x, axis=1):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=axis)
    return np.concatenate(x, axis=axis)

def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    if at is None:
        at = torch.zeros_like(camera_position)
    else:
        at = torch.tensor(at).type_as(camera_position)
    if up is None:
        up = torch.zeros_like(camera_position)
        up[2] = -1
    else:
        up = torch.tensor(up).type_as(camera_position)

    z_axis = normalize(at - camera_position)[0]
    x_axis = normalize(cross(up, z_axis))[0]
    y_axis = normalize(cross(z_axis, x_axis))[0]

    R = cat([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R

def gen_path(pos_gen, at=(0, 0, 0), up=(0, -1, 0), frames=180):
    c2ws = []
    for t in range(frames):
        c2w = torch.eye(4)
        cam_pos = torch.tensor(pos_gen(t * (360.0 / frames) / 180 * np.pi))
        cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
        c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
        c2ws.append(c2w)
    return torch.stack(c2ws)

def get_dataset(cfg, dataset_type='tanks', split='train'):
    """
    Factory function to get the appropriate dataset
    """
    dataset_classes = {
        'blender': BlenderDataset,
        'tanks': TanksTempleDataset,
        'temple': TanksTempleDataset,
        'llff': LLFFDataset,
        'nsvf': NSVF,
        'sdf': SDFDataset,
        'image': ImageDataset,
        'dtu': DTUDataset,
        'colmap': ColmapDataset,
        'google_objs': GoogleObjsDataset,
        'your_own': YourOwnDataset
    }
    
    if dataset_type not in dataset_classes:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
    return dataset_classes[dataset_type](
        cfg=cfg,
        split=split
    ) 