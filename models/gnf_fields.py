# GNF: Gaussian Neural Fields - Official Implementation
import torch, math
import torch.nn
import torch.nn.functional as F
import numpy as np
import time, skimage
from utils import N_to_reso, N_to_vm_reso
from models.sh import eval_sh_bases
from models.encodings import get_encoder


# import BasisCoding

def grid_mapping(positions, freq_bands, aabb, basis_mapping='sawtooth'):
    aabbSize = max(aabb[1] - aabb[0])
    scale = aabbSize[..., None] / freq_bands
    if basis_mapping == 'triangle':
        pts_local = (positions - aabb[0]).unsqueeze(-1) % scale
        pts_local_int = ((positions - aabb[0]).unsqueeze(-1) // scale) % 2
        pts_local = pts_local / (scale / 2) - 1
        pts_local = torch.where(pts_local_int == 1, -pts_local, pts_local)
    elif basis_mapping == 'sawtooth':
        pts_local = (positions - aabb[0])[..., None] % scale
        pts_local = pts_local / (scale / 2) - 1
        pts_local = pts_local.clamp(-1., 1.)
    elif basis_mapping == 'sinc':
        pts_local = torch.sin((positions - aabb[0])[..., None] / (scale / np.pi) - np.pi / 2)
    elif basis_mapping == 'trigonometric':
        pts_local = (positions - aabb[0])[..., None] / scale * 2 * np.pi
        pts_local = torch.cat((torch.sin(pts_local), torch.cos(pts_local)), dim=-1)
    elif basis_mapping == 'x':
        pts_local = (positions - aabb[0]).unsqueeze(-1) / scale
    # elif basis_mapping=='hash':
    #     pts_local = (positions - aabb[0])/max(aabbSize)

    return pts_local


def dct_dict(n_atoms_fre, size, n_selete, dim=2):
    """
    Create a dictionary using the Discrete Cosine Transform (DCT) basis. If n_atoms is
    not a perfect square, the returned dictionary will have ceil(sqrt(n_atoms))**2 atoms
    :param n_atoms:
        Number of atoms in dict
    :param size:
        Size of first patch dim
    :return:
        DCT dictionary, shape (size*size, ceil(sqrt(n_atoms))**2)
    """
    # todo flip arguments to match random_dictionary
    p = n_atoms_fre  # int(math.ceil(math.sqrt(n_atoms)))
    dct = np.zeros((p, size))

    for k in range(p):
        basis = np.cos(np.arange(size) * k * math.pi / p)
        if k > 0:
            basis = basis - np.mean(basis)

        dct[k] = basis

    kron = np.kron(dct, dct)
    if 3 == dim:
        kron = np.kron(kron, dct)

    if n_selete < kron.shape[0]:
        idx = [x[0] for x in np.array_split(np.arange(kron.shape[0]), n_selete)]
        kron = kron[idx]

    for col in range(kron.shape[0]):
        norm = np.linalg.norm(kron[col]) or 1
        kron[col] /= norm

    kron = torch.FloatTensor(kron)
    return kron


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma * dist)

    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)
    weights = alpha * T[..., :-1]  # [N_rays, N_samples]
    return alpha, weights, T[..., -1:]


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(
            self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1


class MLPMixer(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim=16,
                 num_layers=2,
                 hidden_dim=64, pe=0, with_dropout=False):
        super().__init__()

        self.with_dropout = with_dropout
        self.in_dim = in_dim + 2 * in_dim * pe
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pe = pe

        backbone = []
        for l in range(num_layers):
            if l == 0:
                layer_in_dim = self.in_dim
            else:
                layer_in_dim = self.hidden_dim

            if l == num_layers - 1:
                layer_out_dim, bias = out_dim, False
            else:
                layer_out_dim, bias = self.hidden_dim, True

            backbone.append(torch.nn.Linear(layer_in_dim, layer_out_dim, bias=bias))

        self.backbone = torch.nn.ModuleList(backbone)
        # torch.nn.init.constant_(backbone[0].weight.data, 1.0/self.in_dim)

    def forward(self, x, is_train=False):
        # x: [B, 3]
        h = x
        if self.pe > 0:
            h = torch.cat([h, positional_encoding(h, self.pe)], dim=-1)

        if self.with_dropout and is_train:
            h = F.dropout(h, p=0.1)

        for l in range(self.num_layers):
            h = self.backbone[l](h)
            if l != self.num_layers - 1:  # l!=0 and
                h = F.relu(h, inplace=True)
                # h = torch.sin(h)
        # sigma, feat = h[...,0], h[...,1:]
        return h


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, num_layers=3, hidden_dim=64, viewpe=6, feape=2):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 3 + inChanel + 2 * viewpe * 3 + 2 * feape * inChanel
        self.num_layers = num_layers
        self.viewpe = viewpe
        self.feape = feape

        mlp = []
        for l in range(num_layers):
            if l == 0:
                in_dim = 32
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim, bias = 49, True  # 3 rgb
            else:
                out_dim, bias = hidden_dim, True

            mlp.append(torch.nn.Linear(in_dim, out_dim, bias=bias))

        self.mlp = torch.nn.ModuleList(mlp)
        # torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, features):

        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        # if self.viewpe > 0:
        #     indata += [positional_encoding(viewdirs, self.viewpe)]

        h = torch.cat(indata, dim=-1)
        for l in range(self.num_layers):
            h = self.mlp[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        # h[1:] = torch.sigmoid(h[1:])
        return h

import torch
import torch.nn as nn

class RBFDecoder(nn.Module):
    """Radial Basis Function (RBF) decoder for neural field modeling.
    
    This decoder uses RBFs to approximate the target function, with support for:
    - Multiple basis functions (Gaussian, Multiquadric, etc.)
    - Multi-level feature processing
    - Adaptive basis function centers and widths
    """
    def __init__(self, n_rbfs, out_dim, num_features,
                 basis_function, per_level=True, device="cuda"):
        super(RBFDecoder, self).__init__()
        
        # Core parameters
        self.n_rbfs = n_rbfs
        self.output_dim = out_dim
        self.num_features = num_features
        self.per_level = per_level
        self.device = device
        
        # RBF centers and parameters
        self.centers = nn.ModuleList()
        self.betas = nn.ParameterList()  # Controls RBF widths
        self.betas_shifts = nn.ParameterList()  # Additional shift parameters
        self.weighted_sums = nn.ModuleList()  # Final MLP layers
        
        # Initialize RBF parameters
        if not per_level:
            # Single-level: all features processed together
            self._init_single_level(n_rbfs, num_features, out_dim)
        else:
            # Multi-level: features processed per level
            self._init_multi_level(n_rbfs, num_features, out_dim)
            
        # Set up basis function and numerical stability
        self.output_function = self._get_rbf_fn(basis_function)
        self.epsilon = 1e-9
        self.indices = torch.arange(n_rbfs).to(device)

    def _init_single_level(self, n_rbfs, feat_dim, out_dim):
        """Initialize parameters for single-level processing."""
        geom_centers = nn.Embedding(n_rbfs, feat_dim).to(self.device)
        rbf_betas_geom = nn.Parameter(torch.zeros((n_rbfs, feat_dim), device=self.device))
        rbf_betas_shift = nn.Parameter(torch.zeros((n_rbfs,), device=self.device))
        geom_out = nn.Linear(n_rbfs, out_dim, bias=True).to(self.device)
        
        torch.nn.init.uniform_(geom_centers.weight, -1e-4, 1e-4)
        
        self.centers.append(geom_centers)
        self.betas.append(rbf_betas_geom)
        self.betas_shifts.append(rbf_betas_shift)
        self.weighted_sums.append(geom_out)

    def _init_multi_level(self, n_rbfs, feat_dim, num_levels, out_dim):
        """Initialize parameters for multi-level processing."""
        for _ in range(num_levels):
            geom_centers = nn.Embedding(n_rbfs, feat_dim).to(self.device)
            rbf_betas_geom = nn.Parameter(torch.zeros((n_rbfs, feat_dim), device=self.device))
            rbf_betas_shift = nn.Parameter(torch.zeros((n_rbfs,), device=self.device))
            
            torch.nn.init.uniform_(geom_centers.weight, -1e-4, 1e-4)
            
            self.centers.append(geom_centers)
            self.betas.append(rbf_betas_geom)
            self.betas_shifts.append(rbf_betas_shift)
            
        geom_out = nn.Linear(n_rbfs * num_levels, out_dim, bias=True).to(self.device)
        self.weighted_sums.append(geom_out)

    def _get_rbf_fn(self, basis_fn):
        """Get the RBF basis function."""
        if basis_fn == 'gaussian':
            return lambda dist_sq: torch.exp(-0.5 * dist_sq)
        elif basis_fn == 'multiquadric':
            return lambda dist_sq: torch.sqrt(dist_sq + 1)
        elif basis_fn == 'inverse_multiquadric':
            return lambda dist_sq: 1.0 / torch.sqrt(dist_sq + 1)
        elif basis_fn == 'cubic':
            return lambda dist_sq: dist_sq.sqrt() ** 3
        elif basis_fn == 'thin_plate_spline':
            return lambda dist_sq: (dist_sq + self.epsilon).sqrt().pow(2) * torch.log((dist_sq + self.epsilon).sqrt())
        elif basis_fn == 'wendland_c2':
            def wendland_c2(dist_sq):
                r = torch.sqrt(dist_sq + self.epsilon)
                l = (10 // 2) + 2
                w = torch.clamp(1 - r, min=0.0)
                return w ** (l + 2) * ((l + 2) * r + 1)
            return wendland_c2
        else:
            raise ValueError(f"Unsupported basis function: {basis_fn}")

    def forward(self, x, feats):
        """Forward pass computing RBF approximation.
        
        Args:
            x: Input features [B, D] where D is feature dimension
            
        Returns:
            Output values [B, output_dim]
        """
        rbfs = []
        
        # Process each level
        for i, (centers, betas, shifts) in enumerate(zip(self.centers, self.betas, self.betas_shifts)):
            # Get centers and betas for current level
            centers = torch.cat([self.centers[j](self.indices) for j in range(i+1)], dim=-1)[..., :feats.shape[1]]
            betas = torch.exp(torch.cat([self.betas[j] for j in range(i+1)], dim=-1))[..., :feats.shape[1]]
            
            if not self.per_level:
                # Single-level: process all features together
                x_in = feats
            else:
                # Multi-level: process features for current level
                x_in = feats[:, 0:(i + 1) * self.num_features]
            # Compute RBF distances and outputs
            term1 = (x_in ** 2) @ betas.T
            term2 = x_in @ (betas * centers).T
            term3 = (betas * centers ** 2).sum(dim=1)
            dist_sq = term1 - 2 * term2 + term3.unsqueeze(0)
            outs = self.output_function(dist_sq)
            rbfs.append(outs)
        
        # Combine and transform outputs
        outs = torch.cat(rbfs, dim=-1)
        return self.weighted_sums[0](outs)

    def get_parameters(self):
        """Get all learnable parameters."""
        return {
            "centers": [c.weight for c in self.centers],
            "betas": list(self.betas),
            "shifts": list(self.betas_shifts),
            "mlp_params": [{"weight": mlp.weight, "bias": mlp.bias} for mlp in self.weighted_sums]
        }

    def extra_repr(self):
        """String representation of the module."""
        return (f"n_rbfs={self.n_rbfs}, out_dim={self.output_dim}, "
                f"feat_dim={self.num_features}, "
                f"per_level={self.per_level}")

class GNFRenderer(torch.nn.Module):
    def __init__(self, cfg, device):
        super(GNFRenderer, self).__init__()

        self.cfg = cfg
        self.device = device

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.n_scene, self.scene_idx = 1, 0

        self.alphaMask = None
        self.coeff_type, self.basis_type = cfg.model.coeff_type, cfg.model.basis_type

        # Calculate number of SH coefficients
        self.sh_degree = cfg.renderer.sh_degree
        self.sh_coeffs = (self.sh_degree + 1) ** 2  # Number of coefficients per channel
        self.total_sh_coeffs = self.sh_coeffs * 3  # Total coefficients for RGB

        # Total output dimension includes density
        self.total_output_dim = 1 + self.total_sh_coeffs

        # Initialize encoding
        encoding_kwargs = {
            'input_dim': cfg.model.in_dim,
        }

        if cfg.model.encoding_type == 'dif_grid':
            encoding_kwargs.update({
                'basis_dims': cfg.model.basis_dims,
                'basis_resos': cfg.model.basis_resos,
                'freq_bands': cfg.model.freq_bands,
                'basis_mapping': cfg.model.basis_mapping,
                'aabb': torch.FloatTensor(cfg.dataset.aabb).to(device)
            })
        elif cfg.model.encoding_type == 'hashgrid':
            encoding_kwargs.update({
                'num_levels': cfg.model.num_levels,
                'num_features': cfg.model.num_features,
                'base_resolution': cfg.model.base_resolution,
                'desired_resolution': cfg.model.desired_resolution,
                'log2_hashmap_size': cfg.model.log2_hashmap_size
            })
        elif cfg.model.encoding_type == 'frequency':
            encoding_kwargs.update({
                'num_frequencies': cfg.model.num_frequencies,
                'include_input': cfg.model.include_input
            })
        elif cfg.model.encoding_type == 'spherical_harmonics':
            encoding_kwargs.update({
                'degree': cfg.model.degree
            })
            
        self.encoding = get_encoder(cfg.model.encoding_type, **encoding_kwargs).to(device)

        # Initialize linear projection layer if enabled
        if cfg.model.use_linear_projection:
            self.linear_mat = MLPMixer(
                in_dim=self.encoding.output_dim,
                out_dim=cfg.model.projection_dim,
                num_layers=2,
                hidden_dim=64,
                pe=4
            ).to(device)
        else:
            self.linear_mat = None

        if 'reconstruction' in cfg.defaults.mode:
            # Initialize the decoder based on config
            decoder_type = cfg.renderer.decoder_type
            if decoder_type == 'rbf':
                self.renderModule = RBFDecoder(
                    n_rbfs=cfg.renderer.n_rbfs,
                    out_dim=self.total_output_dim,  # Output includes density and SH coefficients
                    num_levels=cfg.renderer.num_levels,
                    num_features=cfg.renderer.num_features,
                    basis_function=cfg.renderer.basis_function,
                    per_level=cfg.renderer.per_level
                ).to(device)
            elif decoder_type == 'mlp':
                self.renderModule = MLPRender_Fea(
                    inChanel=self.encoding.output_dim - 1,
                    num_layers=cfg.renderer.num_layers,
                    hidden_dim=cfg.renderer.hidden_dim,
                    viewpe=cfg.renderer.view_pe,
                    feape=cfg.renderer.fea_pe
                ).to(device)
            else:
                raise ValueError(f"Unknown decoder type: {decoder_type}")
            
            self.is_unbound = self.cfg.dataset.is_unbound
            if self.is_unbound:
                self.bg_len = 0.2
                self.inward_aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).to(device)
                self.aabb = self.inward_aabb * (1 + self.bg_len)
            else:
                self.inward_aabb = self.aabb

            self.cur_volumeSize = N_to_reso(cfg.training.volume_resoInit ** self.in_dim, self.aabb)
            self.update_renderParams(self.cur_volumeSize)

        print('=====> total parameters: ', self.n_parameters())

    def get_coding(self, x):
        # Get encoded features
        features = self.encoding(x)
        
        # Apply linear projection if enabled
        if self.linear_mat is not None:
            features = self.linear_mat(features)
            
        return features, features

    def n_parameters(self):
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_optparam_groups(self, lr_small=0.001, lr_large=0.02):
        """Get parameter groups for optimization with different learning rates."""
        grad_vars = []
        
        # Encoder parameters
        grad_vars += [{'params': self.encoding.parameters(), 'lr': lr_large}]
        
        # Linear projection layer if used
        if self.linear_mat is not None:
            grad_vars += [{'params': self.linear_mat.parameters(), 'lr': lr_small}]
        
        # Renderer (RBF decoder) parameters
        if hasattr(self, 'renderModule'):
            params = self.renderModule.get_parameters()
            grad_vars += [
                {"params": params["centers"], "lr": lr_large},
                {"params": params["betas"], "lr": lr_large},
                {"params": [p for mlp in params["mlp_params"] for p in mlp.values()], "lr": lr_large}
            ]
        
        return grad_vars

    def set_optimizable(self, items, statue):
        """Set which components are optimizable."""
        for item in items:
            if item == 'encoder':
                self.encoding.requires_grad = statue
            elif item == 'proj' and self.linear_mat is not None:
                self.linear_mat.requires_grad = statue
            elif item == 'renderer' and hasattr(self, 'renderModule'):
                self.renderModule.requires_grad = statue

    def TV_loss(self, reg):
        total = 0
        for idx in range(len(self.basises)):
            total = total + reg(self.basises[idx]) * 1e-2
        return total

    def sample_point_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.cfg.dataset.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0, :self.in_dim] > rays_pts) | (rays_pts > self.aabb[1, :self.in_dim])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_point(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1, :self.in_dim] - rays_o) / vec
        rate_b = (self.aabb[0, :self.in_dim] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=0.05, max=1e3)
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = self.stepSize * rng.to(rays_o.device)
        interpx = (t_min[..., None] + step)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0, :self.in_dim] > rays_pts) | (rays_pts > self.aabb[1, :self.in_dim])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def sample_point_unbound(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples

        N_inner, N_outer = 3 * N_samples // 4, N_samples // 4
        b_inner = torch.linspace(0, 2, N_inner + 1).to(self.device)
        b_outer = 2 / torch.linspace(1, 1 / 16, N_outer + 1).to(self.device)

        if is_train:
            rng = torch.rand((N_inner + N_outer), device=self.device)
            interpx = torch.cat([
                b_inner[1:] * rng[:N_inner] + b_inner[:-1] * (1 - rng[:N_inner]),
                b_outer[1:] * rng[N_inner:] + b_outer[:-1] * (1 - rng[N_inner:]),
            ])[None]
        else:
            interpx = torch.cat([
                (b_inner[1:] + b_inner[:-1]) * 0.5,
                (b_outer[1:] + b_outer[:-1]) * 0.5,
            ])[None]

        rays_pts = rays_o[:, None, :] + rays_d[:, None, :] * interpx[..., None]

        norm = rays_pts.abs().amax(dim=-1, keepdim=True)
        inner_mask = (norm <= 1)
        rays_pts = torch.where(
            inner_mask,
            rays_pts,
            rays_pts / norm * ((1 + self.bg_len) - self.bg_len / norm)
        )

        return rays_pts, interpx, inner_mask.squeeze(-1)

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) / (self.aabb[1] - self.aabb[0])

    @torch.no_grad()
    def basis2density(self, density_features):
        return F.softplus(density_features + self.cfg.renderer.density_shift)

    def save(self, path):
        ckpt = {'state_dict': self.state_dict(), 'cfg': self.cfg}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device),
                                           alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])
        volumeSize = N_to_reso(self.cfg.training.volume_resoFinal ** self.in_dim, self.aabb)
        self.update_renderParams(volumeSize)

    def update_renderParams(self, gridSize):
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(units) * self.cfg.renderer.step_ratio
        aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((aabbDiag / self.stepSize).item()) + 1

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.update_renderParams(res_target)

        if self.cfg.dataset.dataset_name == 'google_objs' and self.n_scene == 1 and self.cfg.model.coeff_type == 'grid':
            coeffs = [
                F.interpolate(self.coeffs[0].data, size=None, scale_factor=1.3, align_corners=True, mode='trilinear')]
            self.coeffs = torch.nn.ParameterList(coeffs)

    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            feats, _ = self.get_coding(xyz_locs[alpha_mask])
            feats = self.renderModule(feats)
            sigma[alpha_mask] = self.basis2density(feats[..., 0])

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return alpha

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None, times=16):

        gridSize = self.gridSize.tolist() if gridSize is None else gridSize

        aabbSize = self.inward_aabb[1] - self.inward_aabb[0]
        units = aabbSize / (torch.LongTensor(gridSize).to(self.device) - 1)
        units_half = 1.0 / (torch.LongTensor(gridSize) - 1) * 0.5
        stepSize = torch.mean(units)

        samples = torch.stack(torch.meshgrid(
            [torch.linspace(units_half[0], 1 - units_half[0], gridSize[0]),
             torch.linspace(units_half[1], 1 - units_half[1], gridSize[1]),
             torch.linspace(units_half[2], 1 - units_half[2], gridSize[2])], indexing='ij'
        ), -1).to(self.device)
        dense_xyz = self.inward_aabb[0] * (1 - samples) + self.inward_aabb[1] * samples

        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = torch.zeros_like(dense_xyz[..., 0])
        for _ in range(times):
            for i in range(gridSize[2]):
                shiftment = (torch.rand(dense_xyz[i].shape) * 2 - 1).to(self.device) * (
                            units / 2 * 1.2) if times > 1 else 0.0
                alpha[i] += self.compute_alpha((dense_xyz[i] + shiftment).view(-1, 3),
                                               stepSize * self.cfg.renderer.distance_scale).view(
                    (gridSize[1], gridSize[0]))
        return alpha / times, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200), is_update_alphaMask=False):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = alpha.clamp(0, 1)[None, None]
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])

        # filter floaters
        min_size = np.mean(alpha.shape[-3:]).item()
        alphaMask_thres = self.cfg.renderer.alphaMask_thres if is_update_alphaMask else 0.08
        if self.is_unbound:
            alphaMask_thres = 0.04
            alpha = (alpha >= alphaMask_thres).float()
        else:
            alpha = skimage.morphology.remove_small_objects(alpha.cpu().numpy() >= alphaMask_thres, min_size=min_size,
                                                            connectivity=1)
            alpha = torch.FloatTensor(alpha).to(self.device)

        if is_update_alphaMask:
            self.alphaMask = AlphaGridMask(self.device, self.inward_aabb, alpha)

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)
        if not self.is_unbound:
            pad = (xyz_max - xyz_min) / 20
            xyz_min -= pad
            xyz_max += pad

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        return new_aabb

    @torch.no_grad()
    def shrink(self, new_aabb):
        """Update the model's bounding box and reinitialize parameters.
        
        This method is called during training when the scene bounds change.
        For DiF-Grid encoder, it updates the AABB and reinitializes the encoder
        with the new bounds.
        
        Args:
            new_aabb: New axis-aligned bounding box [2, 3] tensor
        """
        # Update AABB
        self.aabb = self.inward_aabb = new_aabb
        self.cfg.dataset.aabb = self.aabb.tolist()
        
        # Reinitialize encoder with new AABB
        if self.cfg.model.encoding_type == 'dif_grid':
            # Get current encoder parameters
            encoding_kwargs = {
                'input_dim': self.cfg.model.in_dim,
                'basis_dims': self.cfg.model.basis_dims,
                'basis_resos': self.cfg.model.basis_resos,
                'freq_bands': self.cfg.model.freq_bands,
                'basis_mapping': self.cfg.model.basis_mapping,
                'aabb': new_aabb
            }
            # Create new encoder with updated AABB
            self.encoding = get_encoder(self.cfg.model.encoding_type, **encoding_kwargs).to(self.device)
        
        # Update render parameters for new bounds
        self.update_renderParams(self.gridSize.tolist())

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240 * 5, bbox_only=False):
        tt = time.time()
        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        length_current = 0
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)  # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)  # .clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _, _ = self.sample_point(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox = (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            # mask_filtered.append(mask_inbbox.cpu())
            length = torch.sum(mask_inbbox)
            all_rays[length_current:length_current + length], all_rgbs[length_current:length_current + length] = \
            rays_chunk[mask_inbbox].cpu(), all_rgbs[idx_chunk][mask_inbbox.cpu()]
            length_current += length

        return all_rays[:length_current], all_rgbs[:length_current]

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if self.is_unbound:
            xyz_sampled, z_vals, inner_mask = self.sample_point_unbound(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                        N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], z_vals[:, -1:] - z_vals[:, -2:-1]), dim=-1)
        elif ndc_ray:
            xyz_sampled, z_vals, inner_mask = self.sample_point_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                    N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, inner_mask = self.sample_point(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        ray_valid = torch.ones_like(xyz_sampled[..., 0]).bool() if self.is_unbound else inner_mask
        if self.alphaMask is not None:
            alpha_inner_valid = self.alphaMask.sample_alpha(xyz_sampled[inner_mask]) > 0.5
            ray_valid[inner_mask.clone()] = alpha_inner_valid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        coeffs = torch.zeros((1, sum(self.cfg.model.basis_dims)), device=xyz_sampled.device)
        if ray_valid.any():
            feats, coeffs = self.get_coding(xyz_sampled[ray_valid])
            feats = self.renderModule(feats)
            sigma[ray_valid] = self.basis2density(feats[..., 0])

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.cfg.renderer.distance_scale)
        app_mask = weight > self.cfg.renderer.rayMarch_weight_thres
        ray_valid_new = torch.logical_and(ray_valid, app_mask)
        app_mask = ray_valid_new[ray_valid]

        
        if app_mask.any():
            # Get output from decoder
            output = self.renderModule(feats[app_mask, :])  # [N, total_output_dim]
            
            # Separate density and SH coefficients
            density = output[..., 0]  # First position is density
            sh_coeffs = output[..., 1:]  # Remaining are SH coefficients
            
            # Reshape SH coefficients for RGB channels
            sh_coeffs = sh_coeffs.view(-1, 3, self.sh_coeffs)  # [N, 3, sh_coeffs]
            
            # Evaluate SH bases for view directions
            sh_bases = eval_sh_bases(self.sh_degree, viewdirs[ray_valid_new])  # [N, sh_coeffs]
            
            # Compute RGB values using SH
            rgb[ray_valid_new] = torch.sigmoid(torch.sum(
                sh_bases.unsqueeze(1) * sh_coeffs,  # [N, 3, sh_coeffs]
                dim=-1  # Sum over SH coefficients
            ))

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)

        return rgb_map, depth_map, coeffs