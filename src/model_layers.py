import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network Layer with B-spline basis functions.

    Computes: output = W_base * SiLU(x) + W_spline * B-splines(x)

    Matches the reference KansformerEPI implementation (grid_size=5, spline_order=3).
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Build uniform B-spline knot grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # (in_features, grid_size + 2*spline_order + 1)

        # Parameters
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.spline_scaler = None

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x):
        """Compute B-spline bases. x: (batch, in_features) -> (batch, in_features, n_bases)"""
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid  # (in_features, n_knots)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x, y):
        """Fit B-spline coefficients via least squares. Used during initialization."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        A = self.b_splines(x).transpose(0, 1)  # (in, batch, n_bases)
        B = y.transpose(0, 1)                   # (in, batch, out)
        solution = torch.linalg.lstsq(A, B).solution  # (in, n_bases, out)
        result = solution.permute(2, 0, 1)             # (out, in, n_bases)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        # Base path: SiLU(x) @ W_base^T
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # Spline path: B-splines(x) @ W_spline^T
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        """Adaptively update grid based on input data distribution."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step + x_sorted[0] - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """L1 + entropy regularization on spline weights."""
        l1_fake = self.spline_weight.abs().mean(-1)
        reg_activation = l1_fake.sum()
        p = l1_fake / reg_activation
        reg_entropy = -torch.sum(p * p.log())
        return regularize_activation * reg_activation + regularize_entropy * reg_entropy


class KAN(nn.Module):
    """Multi-layer KAN network. Chains KANLinear layers sequentially."""
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_f, out_f,
                    grid_size=grid_size, spline_order=spline_order,
                    scale_noise=scale_noise, scale_base=scale_base,
                    scale_spline=scale_spline, base_activation=base_activation,
                    grid_eps=grid_eps, grid_range=grid_range,
                )
            )

    def forward(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


# ---------------------------------------------------------------------------
# Positional Encoding (sinusoidal, same as reference)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ---------------------------------------------------------------------------
# Self-Attention Pooling (structured attention, Lin et al. 2017)
# ---------------------------------------------------------------------------
class SelfAttentionPooling(nn.Module):
    def __init__(self, d_model, da=64, r=32):
        super().__init__()
        self.r = r
        self.ws1 = nn.Linear(d_model, da, bias=True)
        self.ws2 = nn.Linear(da, r, bias=True)

    def forward(self, H):
        A = torch.softmax(self.ws2(torch.tanh(self.ws1(H))), dim=1)  # (B, S, r)
        A = A.transpose(1, 2)  # (B, r, S)
        M = torch.bmm(A, H)   # (B, r, d_model)
        return M, A

    def penalization_term(self, A):
        AAT = torch.bmm(A, A.transpose(1, 2))
        I = torch.eye(self.r, device=A.device).unsqueeze(0)
        return torch.norm(AAT - I, p='fro', dim=(1, 2)).mean()
