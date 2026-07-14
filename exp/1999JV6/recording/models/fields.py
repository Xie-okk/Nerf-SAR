import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder


# SDF Network: This implementation is borrowed from IDR
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 init_ellipsoid_scale=(1.0, 1.0, 1.0),
                 sdf_mode='mlp',
                 ellipsoid_radius=None,
                 ellipsoid_center=(0.0, 0.0, 0.0),
                 learn_ellipsoid=False,
                 residual_scale=1.0,
                 residual_init_weight=0.0,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        init_ellipsoid_scale = torch.tensor(init_ellipsoid_scale, dtype=torch.float32)
        if init_ellipsoid_scale.numel() != 3:
            raise ValueError('init_ellipsoid_scale must contain exactly 3 values')
        if torch.any(init_ellipsoid_scale <= 0):
            raise ValueError('init_ellipsoid_scale values must be positive')

        self.sdf_mode = str(sdf_mode)
        if self.sdf_mode not in ('mlp', 'ellipsoid_residual'):
            raise ValueError("sdf_mode must be 'mlp' or 'ellipsoid_residual'")
        self.use_ellipsoid_residual = self.sdf_mode == 'ellipsoid_residual'
        self.residual_scale = float(residual_scale)
        self.learn_ellipsoid = bool(learn_ellipsoid)

        if self.use_ellipsoid_residual:
            if d_in != 3:
                raise ValueError('ellipsoid_residual mode requires d_in == 3')
            if ellipsoid_radius is None:
                ellipsoid_radius = init_ellipsoid_scale
            ellipsoid_radius = torch.tensor(ellipsoid_radius, dtype=torch.float32)
            ellipsoid_center = torch.tensor(ellipsoid_center, dtype=torch.float32)
            if ellipsoid_radius.numel() != 3:
                raise ValueError('ellipsoid_radius must contain exactly 3 values')
            if ellipsoid_center.numel() != 3:
                raise ValueError('ellipsoid_center must contain exactly 3 values')
            if torch.any(ellipsoid_radius <= 0):
                raise ValueError('ellipsoid_radius values must be positive')

            if self.learn_ellipsoid:
                self.ellipsoid_log_radius = nn.Parameter(torch.log(ellipsoid_radius))
                self.ellipsoid_center = nn.Parameter(ellipsoid_center)
            else:
                self.register_buffer('ellipsoid_radius', ellipsoid_radius)
                self.register_buffer('ellipsoid_center', ellipsoid_center)
            self.residual_weight = nn.Parameter(torch.tensor(float(residual_init_weight), dtype=torch.float32))

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    lin.weight.data[:, :3] /= init_ellipsoid_scale.to(lin.weight.device)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                    skip_input_start = lin.weight.shape[1] - dims[0]
                    lin.weight.data[:, skip_input_start:skip_input_start + 3] /= init_ellipsoid_scale.to(lin.weight.device)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    if l == 0:
                        lin.weight.data[:, :3] /= init_ellipsoid_scale.to(lin.weight.device)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def get_ellipsoid_radius(self):
        if self.learn_ellipsoid:
            return torch.exp(self.ellipsoid_log_radius).clamp_min(1e-4)
        return self.ellipsoid_radius

    def ellipsoid_sdf(self, inputs):
        radius = self.get_ellipsoid_radius().to(device=inputs.device, dtype=inputs.dtype)
        center = self.ellipsoid_center.to(device=inputs.device, dtype=inputs.dtype)
        points = inputs[:, :3] - center[None, :]

        # Analytic ellipsoid SDF approximation. It is exact for a sphere and has
        # the correct signed zero level set for arbitrary positive semi-axes.
        k0 = torch.linalg.norm(points / radius[None, :], ord=2, dim=-1, keepdim=True)
        k1 = torch.linalg.norm(points / (radius[None, :] ** 2), ord=2, dim=-1, keepdim=True)
        sdf = k0 * (k0 - 1.0) / k1.clamp_min(1e-6)
        center_sdf = -torch.min(radius).expand_as(sdf)
        return torch.where(k1 > 1e-6, sdf, center_sdf)

    def forward(self, inputs):
        raw_inputs = inputs
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        mlp_output = torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)
        if not self.use_ellipsoid_residual:
            return mlp_output

        residual = self.residual_scale * self.residual_weight * mlp_output[:, :1]
        sdf = self.ellipsoid_sdf(raw_inputs) + residual
        return torch.cat([sdf, mlp_output[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        # 璁＄畻绌洪棿瀵兼暟锛岀敤浜庤幏鍙栬〃闈㈡硶绾垮拰绾︽潫 Eikonal Loss
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class ISARNeRFNetwork(nn.Module):
    """Position-encoded NeRF field with density and one ISAR intensity channel."""
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 multires=0,
                 skips=(4,),
                 density_mode='mlp',
                 ellipsoid_radius=None,
                 ellipsoid_center=(0.0, 0.0, 0.0),
                 prior_density=0.5,
                 surface_width=0.05,
                 sdf_residual_scale=0.1,
                 intensity_min=0.1):
        super().__init__()
        self.density_mode = str(density_mode)
        if self.density_mode not in ('mlp', 'ellipsoid_sdf_residual'):
            raise ValueError("density_mode must be 'mlp' or 'ellipsoid_sdf_residual'")
        self.use_ellipsoid_surface = self.density_mode == 'ellipsoid_sdf_residual'

        if self.use_ellipsoid_surface:
            if d_in != 3:
                raise ValueError('ellipsoid_sdf_residual mode requires d_in == 3')
            if ellipsoid_radius is None:
                raise ValueError('ellipsoid_radius is required for ellipsoid_sdf_residual mode')
            ellipsoid_radius = torch.tensor(ellipsoid_radius, dtype=torch.float32)
            ellipsoid_center = torch.tensor(ellipsoid_center, dtype=torch.float32)
            if ellipsoid_radius.numel() != 3 or ellipsoid_center.numel() != 3:
                raise ValueError('ellipsoid_radius and ellipsoid_center must contain exactly 3 values')
            if torch.any(ellipsoid_radius <= 0):
                raise ValueError('ellipsoid_radius values must be positive')
            if prior_density <= 0 or surface_width <= 0:
                raise ValueError('prior_density and surface_width must be positive')

            self.register_buffer('ellipsoid_radius', ellipsoid_radius)
            self.register_buffer('ellipsoid_center', ellipsoid_center)
            self.prior_density = float(prior_density)
            self.surface_width = float(surface_width)
            self.sdf_residual_scale = float(sdf_residual_scale)
            if self.sdf_residual_scale <= 0:
                raise ValueError('sdf_residual_scale must be positive')

        self.intensity_min = float(intensity_min)
        if not 0.0 <= self.intensity_min < 1.0:
            raise ValueError('intensity_min must be in [0, 1)')

        self.embed_fn = None
        input_ch = d_in
        if multires > 0:
            self.embed_fn, input_ch = get_embedder(multires, input_dims=d_in)

        self.skips = set(skips)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [
                nn.Linear(W + input_ch, W) if layer in self.skips else nn.Linear(W, W)
                for layer in range(1, D)
            ]
        )
        self.density_linear = nn.Linear(W, 1)
        self.intensity_linear = nn.Linear(W, 1)
        if self.use_ellipsoid_surface:
            nn.init.zeros_(self.density_linear.weight)
            nn.init.zeros_(self.density_linear.bias)

    def forward(self, points):
        embedded_points = self.embed_fn(points) if self.embed_fn is not None else points
        hidden = embedded_points
        for layer, linear in enumerate(self.pts_linears):
            if layer in self.skips:
                hidden = torch.cat([embedded_points, hidden], dim=-1)
            hidden = F.relu(linear(hidden))
        return self.density_linear(hidden), self.intensity_linear(hidden)

    def ellipsoid_sdf(self, points):
        radius = self.ellipsoid_radius.to(device=points.device, dtype=points.dtype)
        center = self.ellipsoid_center.to(device=points.device, dtype=points.dtype)
        shifted_points = points[:, :3] - center[None, :]
        k0 = torch.linalg.norm(shifted_points / radius[None, :], dim=-1, keepdim=True)
        k1 = torch.linalg.norm(shifted_points / (radius[None, :] ** 2), dim=-1, keepdim=True)
        sdf = k0 * (k0 - 1.0) / k1.clamp_min(1e-6)
        center_sdf = -torch.min(radius).expand_as(sdf)
        return torch.where(k1 > 1e-6, sdf, center_sdf)

    def density_intensity(self, points, use_learned_intensity=True, return_aux=False):
        raw_density, raw_intensity = self(points)
        aux = {}
        if not self.use_ellipsoid_surface:
            density = F.softplus(raw_density)
        else:
            sdf_residual = self.sdf_residual_scale * raw_density
            surface_sdf = self.ellipsoid_sdf(points) + sdf_residual
            density = self.prior_density * torch.exp(-(surface_sdf / self.surface_width) ** 2)
            aux = {
                'surface_sdf': surface_sdf,
                'sdf_residual': sdf_residual,
            }
        if use_learned_intensity:
            intensity = self.intensity_min + (1.0 - self.intensity_min) * torch.sigmoid(raw_intensity)
        else:
            intensity = torch.ones_like(raw_intensity)
        if return_aux:
            return density, intensity, aux
        return density, intensity

    def surface_sdf(self, points):
        if not self.use_ellipsoid_surface:
            raise RuntimeError('surface_sdf is only available in ellipsoid_sdf_residual mode')
        raw_density, _ = self(points)
        return self.ellipsoid_sdf(points) + self.sdf_residual_scale * raw_density

    def density(self, points):
        return self.density_intensity(points)[0]


class SingleVarianceNetwork(nn.Module):
    """
    NeuS Variance Network 
    """
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * torch.exp(self.variance * 10.0)

