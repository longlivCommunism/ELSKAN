import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import griddata
from scipy.stats import qmc

RESULT_PIC_DIR = 'result_pic'
os.makedirs(RESULT_PIC_DIR, exist_ok=True)


def plot(X, pred_vals=None, analy_vals=None, error=None, title_pred="Prediction", title_analy="Exact Solution",
         title_error="Absolute Error", save_name='result', save_dir=RESULT_PIC_DIR):
   
    os.makedirs(save_dir, exist_ok=True)
    x_coord, y_coord = X[:, 0], X[:, 1]

    fine_grid_res = 500
    gx = np.linspace(-1, 1, fine_grid_res)
    gy = np.linspace(-1, 1, fine_grid_res)
    GX, GY = np.meshgrid(gx, gy)

    def save_contour(data, filename, cmap='turbo'):
        if data is None:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)

        grid_data = griddata((x_coord, y_coord), data.squeeze(), (GX, GY), method='cubic')
        contour = ax.contourf(GX, GY, grid_data, levels=80, cmap=cmap, extend='both')
        plt.colorbar(contour, ax=ax)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.margins(0)

        fig.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
        plt.close(fig)

    save_contour(pred_vals, f"{save_name}_pred.png", cmap='turbo')

    print(f"[plot] Figure saved to '{save_dir}' with prefix '{save_name}'.")


def plot_log_error(X, error, title='', save_name='result', save_dir=RESULT_PIC_DIR, seed=None):
    import matplotlib.colors as mcolors
    
    os.makedirs(save_dir, exist_ok=True)
    x_coord, y_coord = X[:, 0], X[:, 1]
    
    fine_grid_res = 800
    gx = np.linspace(-1, 1, fine_grid_res)
    gy = np.linspace(-1, 1, fine_grid_res)
    GX, GY = np.meshgrid(gx, gy)
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('x', fontsize=18)
    ax_log.set_ylabel('y', fontsize=18)
    
    error_plot = error.squeeze()
    grid_error = griddata((x_coord, y_coord), error_plot, (GX, GY), method='cubic')
    
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(grid_error, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    cs = ax_log.contourf(GX, GY, error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
    cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cbar = plt.colorbar(cs, ax=ax_log, ticks=cbar_ticks, format='%.0e')
    cbar.set_label('Absolute Error (log scale)', fontsize=14)
    cbar.ax.set_ylim(e_min, e_max)
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontsize(10)
    
    ax_log.text(0.02, 0.98, f'Max: {np.max(error_plot):.2e}\nMean: {np.mean(error_plot):.2e}', 
                transform=ax_log.transAxes, fontsize=11, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_log.set_xlim(-1, 1)
    ax_log.set_ylim(-1, 1)
    ax_log.set_aspect('equal', adjustable='box')
    ax_log.margins(0)
    
    if seed is not None:
        filename = f"{save_name}_error_log_{seed}.png"
    else:
        filename = f"{save_name}_error_log.png"
    fig_log.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
    plt.close(fig_log)
    print(f"[plot_log_error] Log-scale error figure saved: '{os.path.join(save_dir, filename)}'.")



def square_sampler_sobol(n_interior=5000, n_bc=256):
    sampler_int = qmc.Sobol(d=2, scramble=False)
    sample_interior = sampler_int.random(n=n_interior)
    XY_interior = 2.0 * sample_interior - 1.0

    sampler_bc = qmc.Sobol(d=1, scramble=False)
    n_points_per_edge = n_bc // 4
    edge_coords = 2.0 * sampler_bc.random(n=n_points_per_edge).flatten() - 1.0

    bottom = np.vstack([edge_coords, np.full(n_points_per_edge, -1)]).T
    top = np.vstack([edge_coords, np.full(n_points_per_edge, 1)]).T
    left = np.vstack([np.full(n_points_per_edge, -1), edge_coords]).T
    right = np.vstack([np.full(n_points_per_edge, 1), edge_coords]).T

    XY_boundary = np.vstack([bottom, top, left, right])
    return XY_interior, XY_boundary


class KANLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.base_weight, gain=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order: -self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = (self.grid)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1] + (
                        grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.scaled_spline_weight.view(self.out_features, -1))
        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)


class efficentKAN(torch.nn.Module):
    def __init__(self, layers_hidden, grid_size=10, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=None, grid_eps=0.02, grid_range=[-1, 1]):
        super(efficentKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                          scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                          base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            )

    def forward(self, x: torch.Tensor):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class AL_PKAN(nn.Module):
    def __init__(self, input_dim=2, gru_hidden=64, gru_layers=2, kan_layers=[64, 64, 64, 1],
                 grid_size=10, spline_order=3, history_steps=3, device=torch.device('cpu'), dtype=torch.float64):
        super(AL_PKAN, self).__init__()
        self.history_steps = history_steps
        # Encoder: GRU
        self.gru = nn.GRU(input_dim, gru_hidden, gru_layers, batch_first=True, dtype=dtype, device=device)
        # Decoder: KAN
        self.kan = efficentKAN(kan_layers, grid_size=grid_size, spline_order=spline_order,
                               base_activation=lambda: nn.GELU(approximate='tanh'),
                               grid_range=[-1, 1]).to(device=device, dtype=dtype)

    def forward(self, x):
        x_seq = x.unsqueeze(1).repeat(1, self.history_steps, 1)

        # ========================================================
        # ========================================================
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.gru(x_seq)

        h = out[:, -1, :]
        y = self.kan(h)
        return y


class piCKANELM_SOV2_nodecay(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, degree1_high, degree1_low, degree1_high_need, degree2, coeff_scaled_high=1.,
                 coeff_scaled_low=1., device=torch.device('cpu'), dtype=torch.float32):
        super(piCKANELM_SOV2_nodecay, self).__init__()
        self.in_dim = in_dim
        self.hidden_nodes = hidden_nodes
        self.out_dim = out_dim
        self.degree1_high = degree1_high
        self.degree1_low = degree1_low
        self.degree1_high_need = degree1_high_need
        self.degree2 = degree2

        self.cheb_coeff_high = nn.Parameter(torch.empty(degree1_high_need, hidden_nodes).to(device=device, dtype=dtype), requires_grad=False)
        self.cheb_coeff_low = nn.Parameter(torch.empty(degree1_low + 1, hidden_nodes).to(device=device, dtype=dtype), requires_grad=False)
        self.beta = nn.Parameter(torch.ones(hidden_nodes * (degree2 + 1), out_dim, device=device, dtype=dtype), requires_grad=False)

        nn.init.normal_(self.cheb_coeff_high, mean=0.0, std=coeff_scaled_high / (in_dim * degree1_high_need))
        nn.init.normal_(self.cheb_coeff_low, mean=0.0, std=coeff_scaled_low / (in_dim * (degree1_low + 1)))

        # degrees = torch.arange(degree1_high - degree1_high_need + 1, degree1_high + 1, device=device, dtype=dtype)
        # self.cheb_coeff_high.data = self.cheb_coeff_high.data / (degrees[:, None] / degrees[0])

    def forward(self, X):
        H1_x = self.chebyshev(X[:, 0][:, None], 1, self.degree1_low).reshape(X.shape[0], -1) @ self.cheb_coeff_low
        H1_t = self.chebyshev(X[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(X.shape[0], -1) @ self.cheb_coeff_high
        output_x = self.chebyshev(F.tanh(H1_x), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1)
        output_t = self.chebyshev(F.tanh(H1_t), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1)
        output = (output_x * output_t) @ self.beta

        return output

    @staticmethod
    def chebyshev(x, dim, degree_max, degree_droped: int = None):
        cheby_poly = torch.ones(x.shape[0], dim, degree_max + 1).to(device=x.device, dtype=x.dtype)
        if degree_max >= 1:
            cheby_poly[:, :, 1] = x
        if degree_max >= 2:
            cheby_poly[:, :, 2] = 2 * x ** 2 - 1
        if degree_max >= 3:
            cheby_poly[:, :, 3] = 4 * x ** 3 - 3 * x

        for i in range(4, degree_max + 1):
            cheby_poly[:, :, i] = 2 * x * cheby_poly[:, :, i - 1].clone() - cheby_poly[:, :, i - 2].clone()

        if degree_droped is not None:
            start_cut = degree_max - degree_droped + 1
            cheby_poly = cheby_poly[:, :, start_cut:]
        return cheby_poly

    def chebyshev1(self, x, dim, degree_max, degree_droped: int = None):
        cheby_poly1 = torch.ones(x.shape[0], dim, degree_max + 1, device=x.device, dtype=x.dtype)
        cheby_poly1[:, :, 0] = cheby_poly1[:, :, 0].clone() * 0.
        cheby_poly = self.chebyshev(x, dim, degree_max)

        if degree_max >= 2:
            cheby_poly1[:, :, 2] = 4 * x
        if degree_max >= 3:
            cheby_poly1[:, :, 3] = 12 * x ** 2 - 3.

        for i in range(4, degree_max + 1):
            cheby_poly1[:, :, i] = (2 * cheby_poly[:, :, i - 1].clone() +
                                    2 * x * cheby_poly1[:, :, i - 1].clone() - cheby_poly1[:, :, i - 2].clone())

        if degree_droped is not None:
            start_cut = degree_max - degree_droped + 1
            cheby_poly1 = cheby_poly1[:, :, start_cut:]
        return cheby_poly1

    def chebyshev2(self, x, dim, degree_max, degree_droped: int = None):
        cheby_poly2 = torch.ones(x.shape[0], dim, degree_max + 1, dtype=x.dtype, device=x.device)
        cheby_poly2[:, :, 0], cheby_poly2[:, :, 1] = cheby_poly2[:, :, 0].clone() * 0., cheby_poly2[:, :, 1].clone() * 0.
        cheby_poly1 = self.chebyshev1(x, dim, degree_max)

        if degree_max >= 2:
            cheby_poly2[:, :, 2] = 4.
        if degree_max >= 3:
            cheby_poly2[:, :, 3] = 24 * x

        for i in range(4, degree_max + 1):
            cheby_poly2[:, :, i] = (4 * cheby_poly1[:, :, i - 1].clone() +
                                    2 * x * cheby_poly2[:, :, i - 1].clone() - cheby_poly2[:, :, i - 2].clone())

        if degree_droped is not None:
            start_cut = degree_max - degree_droped + 1
            cheby_poly2 = cheby_poly2[:, :, start_cut:]
        return cheby_poly2

    def generate_H(self, Xtf, Xbc, Xic):
        bcmid_x = self.chebyshev(Xbc[:, 0][:, None], 1, self.degree1_low).reshape(Xbc.shape[0], -1) @ self.cheb_coeff_low
        bcmid_t = self.chebyshev(Xbc[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xbc.shape[0], -1) @ self.cheb_coeff_high
        H_bc_x = self.chebyshev(F.tanh(bcmid_x), self.hidden_nodes, self.degree2)
        H_bc_t = self.chebyshev(F.tanh(bcmid_t), self.hidden_nodes, self.degree2)
        H_bc = (H_bc_x * H_bc_t).reshape(Xbc.shape[0], -1)

        icmid_x = self.chebyshev(Xic[:, 0][:, None], 1, self.degree1_low).reshape(Xic.shape[0], -1) @ self.cheb_coeff_low
        icmid_t = self.chebyshev(Xic[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xic.shape[0], -1) @ self.cheb_coeff_high
        H_ic_x = self.chebyshev(F.tanh(icmid_x), self.hidden_nodes, self.degree2)
        H_ic_t = self.chebyshev(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
        H_ic = (H_ic_x * H_ic_t).reshape(Xic.shape[0], -1)

        dZdy_ic = self.chebyshev1(Xic[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xic.shape[0], -1) @ self.cheb_coeff_high
        tanh1_y_ic = 1 - F.tanh(icmid_t) ** 2
        dHxdZ_ic = self.chebyshev1(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
        Ht_ic = dHxdZ_ic * dZdy_ic[:, :, None] * tanh1_y_ic[:, :, None] * H_ic_x
        Ht_ic = Ht_ic.clone().reshape(Xic.shape[0], -1)

        x, t = Xtf[:, 0].clone()[:, None], Xtf[:, 1].clone()[:, None]
        Z_x = self.chebyshev(x, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_low
        Z_t = self.chebyshev(t, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_high
        H_x = self.chebyshev(F.tanh(Z_x), self.hidden_nodes, self.degree2)
        H_t = self.chebyshev(F.tanh(Z_t), self.hidden_nodes, self.degree2)
        H = (H_x * H_t).reshape(Xtf.shape[0], -1)

        w_x = self.cheb_coeff_low
        w_y = self.cheb_coeff_high
        dZdx = self.chebyshev1(x, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_x
        dZdy = self.chebyshev1(t, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ w_y
        ddZdxx = self.chebyshev2(x, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_x
        ddZdyy = self.chebyshev2(t, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ w_y
        tanh1_x = 1 - F.tanh(Z_x) ** 2
        tanh1_y = 1 - F.tanh(Z_t) ** 2
        tanh2_x = -2 * F.tanh(Z_x) * tanh1_x
        tanh2_y = -2 * F.tanh(Z_t) * tanh1_y

        dHxdZ = self.chebyshev1(F.tanh(Z_x), self.hidden_nodes, self.degree2)
        ddHxdZZ = self.chebyshev2(F.tanh(Z_x), self.hidden_nodes, self.degree2)
        dHydZ = self.chebyshev1(F.tanh(Z_t), self.hidden_nodes, self.degree2)
        ddHydZZ = self.chebyshev2(F.tanh(Z_t), self.hidden_nodes, self.degree2)

        Hx = dHxdZ * dZdx[:, :, None] * tanh1_x[:, :, None] * H_t
        Hy = dHydZ * dZdy[:, :, None] * tanh1_y[:, :, None] * H_x
        Hx = Hx.clone().reshape(Xtf.shape[0], -1)
        Hy = Hy.clone().reshape(Xtf.shape[0], -1)
        gradH = Hx + Hy

        Hxx = (ddHxdZZ * (tanh1_x * dZdx)[:, :, None] ** 2 + dHxdZ * (tanh2_x * dZdx ** 2)[:, :, None] + dHxdZ * (tanh1_x * ddZdxx)[:, :, None]) * H_t
        Hyy = (ddHydZZ * (tanh1_y * dZdy)[:, :, None] ** 2 + dHydZ * (tanh2_y * dZdy ** 2)[:, :, None] + dHydZ * (tanh1_y * ddZdyy)[:, :, None]) * H_x
        Hxx = Hxx.clone().reshape(Xtf.shape[0], -1)
        Hyy = Hyy.clone().reshape(Xtf.shape[0], -1)
        LapH = Hxx + Hyy

        return H, Hx, Hy, gradH, Hxx, Hyy, LapH, H_bc, H_ic, Ht_ic


class piCKANELM_SOV2(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, degree1_high, degree1_low, degree1_high_need, degree2, coeff_scaled_high=1.,
                 coeff_scaled_low=1., device=torch.device('cpu'), dtype=torch.float32):
        super(piCKANELM_SOV2, self).__init__()
        self.in_dim = in_dim
        self.hidden_nodes = hidden_nodes
        self.out_dim = out_dim
        self.degree1_high = degree1_high
        self.degree1_low = degree1_low
        self.degree1_high_need = degree1_high_need
        self.degree2 = degree2

        self.cheb_coeff_high = nn.Parameter(torch.empty(degree1_high_need, hidden_nodes).to(device=device, dtype=dtype), requires_grad=False)
        self.cheb_coeff_low = nn.Parameter(torch.empty(degree1_low + 1, hidden_nodes).to(device=device, dtype=dtype), requires_grad=False)
        self.beta = nn.Parameter(torch.ones(hidden_nodes * (degree2 + 1), out_dim, device=device, dtype=dtype), requires_grad=False)

        nn.init.normal_(self.cheb_coeff_high, mean=0.0, std=coeff_scaled_high / (in_dim * degree1_high_need))
        nn.init.normal_(self.cheb_coeff_low, mean=0.0, std=coeff_scaled_low / (in_dim * (degree1_low + 1)))

        degrees = torch.arange(degree1_high - degree1_high_need + 1, degree1_high + 1, device=device, dtype=dtype)
        self.cheb_coeff_high.data = self.cheb_coeff_high.data / (degrees[:, None] / degrees[0])

    def forward(self, X):
        H1_x = self.chebyshev(X[:, 0][:, None], 1, self.degree1_low).reshape(X.shape[0], -1) @ self.cheb_coeff_low
        H1_t = self.chebyshev(X[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(X.shape[0], -1) @ self.cheb_coeff_high
        output_x = self.chebyshev(F.tanh(H1_x), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1)
        output_t = self.chebyshev(F.tanh(H1_t), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1)
        output = (output_x * output_t) @ self.beta

        return output

    @staticmethod
    def chebyshev(x, dim, degree_max, degree_droped: int = None):
        cheby_poly = torch.ones(x.shape[0], dim, degree_max + 1).to(device=x.device, dtype=x.dtype)
        if degree_max >= 1:
            cheby_poly[:, :, 1] = x
        if degree_max >= 2:
            cheby_poly[:, :, 2] = 2 * x ** 2 - 1
        if degree_max >= 3:
            cheby_poly[:, :, 3] = 4 * x ** 3 - 3 * x

        for i in range(4, degree_max + 1):
            cheby_poly[:, :, i] = 2 * x * cheby_poly[:, :, i - 1].clone() - cheby_poly[:, :, i - 2].clone()

        if degree_droped is not None:
            start_cut = degree_max - degree_droped + 1
            cheby_poly = cheby_poly[:, :, start_cut:]
        return cheby_poly

    def chebyshev1(self, x, dim, degree_max, degree_droped: int = None):
        cheby_poly1 = torch.ones(x.shape[0], dim, degree_max + 1, device=x.device, dtype=x.dtype)
        cheby_poly1[:, :, 0] = cheby_poly1[:, :, 0].clone() * 0.
        cheby_poly = self.chebyshev(x, dim, degree_max)

        if degree_max >= 2:
            cheby_poly1[:, :, 2] = 4 * x
        if degree_max >= 3:
            cheby_poly1[:, :, 3] = 12 * x ** 2 - 3.

        for i in range(4, degree_max + 1):
            cheby_poly1[:, :, i] = (2 * cheby_poly[:, :, i - 1].clone() +
                                    2 * x * cheby_poly1[:, :, i - 1].clone() - cheby_poly1[:, :, i - 2].clone())

        if degree_droped is not None:
            start_cut = degree_max - degree_droped + 1
            cheby_poly1 = cheby_poly1[:, :, start_cut:]
        return cheby_poly1

    def chebyshev2(self, x, dim, degree_max, degree_droped: int = None):
        cheby_poly2 = torch.ones(x.shape[0], dim, degree_max + 1, dtype=x.dtype, device=x.device)
        cheby_poly2[:, :, 0], cheby_poly2[:, :, 1] = cheby_poly2[:, :, 0].clone() * 0., cheby_poly2[:, :, 1].clone() * 0.
        cheby_poly1 = self.chebyshev1(x, dim, degree_max)

        if degree_max >= 2:
            cheby_poly2[:, :, 2] = 4.
        if degree_max >= 3:
            cheby_poly2[:, :, 3] = 24 * x

        for i in range(4, degree_max + 1):
            cheby_poly2[:, :, i] = (4 * cheby_poly1[:, :, i - 1].clone() +
                                    2 * x * cheby_poly2[:, :, i - 1].clone() - cheby_poly2[:, :, i - 2].clone())

        if degree_droped is not None:
            start_cut = degree_max - degree_droped + 1
            cheby_poly2 = cheby_poly2[:, :, start_cut:]
        return cheby_poly2

    def generate_H(self, Xtf, Xbc, Xic):
        bcmid_x = self.chebyshev(Xbc[:, 0][:, None], 1, self.degree1_low).reshape(Xbc.shape[0], -1) @ self.cheb_coeff_low
        bcmid_t = self.chebyshev(Xbc[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xbc.shape[0], -1) @ self.cheb_coeff_high
        H_bc_x = self.chebyshev(F.tanh(bcmid_x), self.hidden_nodes, self.degree2)
        H_bc_t = self.chebyshev(F.tanh(bcmid_t), self.hidden_nodes, self.degree2)
        H_bc = (H_bc_x * H_bc_t).reshape(Xbc.shape[0], -1)

        icmid_x = self.chebyshev(Xic[:, 0][:, None], 1, self.degree1_low).reshape(Xic.shape[0], -1) @ self.cheb_coeff_low
        icmid_t = self.chebyshev(Xic[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xic.shape[0], -1) @ self.cheb_coeff_high
        H_ic_x = self.chebyshev(F.tanh(icmid_x), self.hidden_nodes, self.degree2)
        H_ic_t = self.chebyshev(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
        H_ic = (H_ic_x * H_ic_t).reshape(Xic.shape[0], -1)

        dZdy_ic = self.chebyshev1(Xic[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xic.shape[0], -1) @ self.cheb_coeff_high
        tanh1_y_ic = 1 - F.tanh(icmid_t) ** 2
        dHxdZ_ic = self.chebyshev1(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
        Ht_ic = dHxdZ_ic * dZdy_ic[:, :, None] * tanh1_y_ic[:, :, None] * H_ic_x
        Ht_ic = Ht_ic.clone().reshape(Xic.shape[0], -1)

        x, t = Xtf[:, 0].clone()[:, None], Xtf[:, 1].clone()[:, None]
        Z_x = self.chebyshev(x, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_low
        Z_t = self.chebyshev(t, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_high
        H_x = self.chebyshev(F.tanh(Z_x), self.hidden_nodes, self.degree2)
        H_t = self.chebyshev(F.tanh(Z_t), self.hidden_nodes, self.degree2)
        H = (H_x * H_t).reshape(Xtf.shape[0], -1)

        w_x = self.cheb_coeff_low
        w_y = self.cheb_coeff_high
        dZdx = self.chebyshev1(x, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_x
        dZdy = self.chebyshev1(t, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ w_y
        ddZdxx = self.chebyshev2(x, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_x
        ddZdyy = self.chebyshev2(t, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ w_y
        tanh1_x = 1 - F.tanh(Z_x) ** 2
        tanh1_y = 1 - F.tanh(Z_t) ** 2
        tanh2_x = -2 * F.tanh(Z_x) * tanh1_x
        tanh2_y = -2 * F.tanh(Z_t) * tanh1_y

        dHxdZ = self.chebyshev1(F.tanh(Z_x), self.hidden_nodes, self.degree2)
        ddHxdZZ = self.chebyshev2(F.tanh(Z_x), self.hidden_nodes, self.degree2)
        dHydZ = self.chebyshev1(F.tanh(Z_t), self.hidden_nodes, self.degree2)
        ddHydZZ = self.chebyshev2(F.tanh(Z_t), self.hidden_nodes, self.degree2)

        Hx = dHxdZ * dZdx[:, :, None] * tanh1_x[:, :, None] * H_t
        Hy = dHydZ * dZdy[:, :, None] * tanh1_y[:, :, None] * H_x
        Hx = Hx.clone().reshape(Xtf.shape[0], -1)
        Hy = Hy.clone().reshape(Xtf.shape[0], -1)
        gradH = Hx + Hy

        Hxx = (ddHxdZZ * (tanh1_x * dZdx)[:, :, None] ** 2 + dHxdZ * (tanh2_x * dZdx ** 2)[:, :, None] + dHxdZ * (tanh1_x * ddZdxx)[:, :, None]) * H_t
        Hyy = (ddHydZZ * (tanh1_y * dZdy)[:, :, None] ** 2 + dHydZ * (tanh2_y * dZdy ** 2)[:, :, None] + dHydZ * (tanh1_y * ddZdyy)[:, :, None]) * H_x
        Hxx = Hxx.clone().reshape(Xtf.shape[0], -1)
        Hyy = Hyy.clone().reshape(Xtf.shape[0], -1)
        LapH = Hxx + Hyy

        return H, Hx, Hy, gradH, Hxx, Hyy, LapH, H_bc, H_ic, Ht_ic



class piELMKAN(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, grid_range=[-1, 1.], num_grid=5, coeff_scale=5., device=torch.device('cpu'), dtype=torch.float32):
        super(piELMKAN, self).__init__()
        self.in_dim = in_dim
        self.hidden_nodes = hidden_nodes
        self.out_dim = out_dim
        self.num_grid = num_grid

        self.grid1 = torch.linspace(grid_range[0], grid_range[1], steps=num_grid, device=device, dtype=dtype)[None, :].expand(in_dim, num_grid)
        self.grid2 = torch.linspace(grid_range[0], grid_range[1], steps=num_grid, device=device, dtype=dtype)[None, :].expand(hidden_nodes, num_grid)
        ctrl = np.random.uniform(-coeff_scale, coeff_scale, size=(in_dim, hidden_nodes, num_grid)).transpose(0, 2, 1).reshape(in_dim * num_grid, hidden_nodes)
        self.ctrl = nn.Parameter(torch.from_numpy(ctrl).to(device=device, dtype=dtype), requires_grad=False)

        self.dist = (grid_range[1] - grid_range[0]) / (num_grid + 1)
        self.act = lambda x, grid: torch.exp(-((x[:, :, None] - grid[None, :, :]) / self.dist) ** 2)

        self.beta = nn.Parameter(torch.zeros(hidden_nodes * num_grid, 1, device=device, dtype=dtype), requires_grad=False)

    def forward(self, x):
        H = self.act(x, self.grid1).reshape(x.shape[0], -1) @ self.ctrl
        output = self.act(H, self.grid2).reshape(x.shape[0], -1) @ self.beta
        return output

    def phi_1(self, x, grid, dim=None):
        if dim is not None:
            x_dim = x[:, dim][:, None]
            grid_dim = grid[dim, :][None, :]
        else:
            x_dim, grid_dim = x, grid

        return self.act(x_dim, grid_dim) * -(2/self.dist ** 2) * (x_dim[:, :, None] - grid_dim[None, :, :])

    def phi_2(self, x, grid, dim=None):
        if dim is not None:
            x_dim = x[:, dim][:, None]
            grid_dim = grid[dim, :][None, :]
        else:
            x_dim, grid_dim = x, grid

        part1 = self.act(x_dim, grid_dim) * (-(2/self.dist ** 2) * (x_dim[:, :, None] - grid_dim[None, :, :])) ** 2
        part2 = self.act(x_dim, grid_dim) * -(2/self.dist ** 2)
        return part1 + part2

    def generate_H(self, Xf, Xbc):
        H_bc = self.act(self.act(Xbc, self.grid1).reshape(Xbc.shape[0], -1) @ self.ctrl, self.grid2).reshape(Xbc.shape[0], -1)

        Z = self.act(Xf, self.grid1).reshape(Xf.shape[0], -1) @ self.ctrl
        H = self.act(Z, self.grid2).reshape(Xf.shape[0], -1)
        w_x = self.ctrl[:self.num_grid, :].T
        w_y = self.ctrl[self.num_grid:, :].T

        H_x = (self.phi_1(Z, self.grid2) * w_x[None, :, :] * self.phi_1(Xf, self.grid1, dim=0)).reshape(Xf.shape[0], -1)
        H_y = (self.phi_1(Z, self.grid2) * w_y[None, :, :] * self.phi_1(Xf, self.grid1, dim=1)).reshape(Xf.shape[0], -1)

        gradHf = H_x + H_y

        H_xx = self.phi_2(Z, self.grid2) * (w_x[None, :, :] * self.phi_1(Xf, self.grid1, dim=0)) ** 2 + self.phi_1(Z, self.grid2) * w_x[None, :, :] * self.phi_2(Xf, self.grid1, dim=0)
        H_yy = self.phi_2(Z, self.grid2) * (w_y[None, :, :] * self.phi_1(Xf, self.grid1, dim=1)) ** 2 + self.phi_1(Z, self.grid2) * w_y[None, :, :] * self.phi_2(Xf, self.grid1, dim=1)
        LapHf = (H_xx + H_yy).reshape(Xf.shape[0], -1)
        # print(f"LapHf.shape:{LapHf.shape}")

        return H, H_x, H_y, gradHf, LapHf, H_bc



class piCKANELM_single(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, degree1, degree2, coeff_scaled=1.,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(piCKANELM_single, self).__init__()
        self.in_dim = in_dim
        self.hidden_nodes = hidden_nodes
        self.out_dim = out_dim
        self.degree1 = degree1
        self.degree2 = degree2

        input_basis_size = in_dim * (degree1 + 1)
        self.cheby_coeff = nn.Parameter(
            torch.empty(input_basis_size, hidden_nodes, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.beta = nn.Parameter(
            torch.ones(hidden_nodes * (degree2 + 1), out_dim, device=device, dtype=dtype),
            requires_grad=False,
        )

        nn.init.normal_(
            self.cheby_coeff,
            mean=0.0,
            std=coeff_scaled / input_basis_size,
        )

    def forward(self, X):
        H = self.chebyshev(X, self.in_dim, self.degree1).reshape(X.shape[0], -1) @ self.cheby_coeff
        output = self.chebyshev(F.tanh(H), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1) @ self.beta
        return output

    @staticmethod
    def chebyshev(x, dim, degree):
        cheby_poly = torch.ones(x.shape[0], dim, degree + 1).to(device=device, dtype=dtype)

        if degree >= 1:
            cheby_poly[:, :, 1] = x
        if degree >= 2:
            cheby_poly[:, :, 2] = 2 * x ** 2 - 1
        if degree >= 3:
            cheby_poly[:, :, 3] = 4 * x ** 3 - 3 * x

        for i in range(4, degree + 1):
            cheby_poly[:, :, i] = 2 * x * cheby_poly[:, :, i - 1].clone() - cheby_poly[:, :, i - 2].clone()
        return cheby_poly

    def chebyshev1(self, x, dim, degree):
        cheby_poly1 = torch.ones(x.shape[0], dim, degree + 1).to(device=device, dtype=dtype)
        cheby_poly1[:, :, 0] = cheby_poly1[:, :, 0].clone() * 0.
        cheby_poly = self.chebyshev(x, dim, degree)

        if degree >= 2:
            cheby_poly1[:, :, 2] = 4 * x
        if degree >= 3:
            cheby_poly1[:, :, 3] = 12 * x ** 2 - 3.

        for i in range(4, degree + 1):
            cheby_poly1[:, :, i] = 2 * cheby_poly[:, :, i - 1].clone() + 2 * x * cheby_poly1[:, :, i - 1].clone() - cheby_poly1[:, :, i - 2].clone()

        return cheby_poly1

    def chebyshev2(self, x, dim, degree):
        cheby_poly2 = torch.ones(x.shape[0], dim, degree + 1, dtype=x.dtype, device=x.device)
        cheby_poly2[:, :, 0], cheby_poly2[:, :, 1] = cheby_poly2[:, :, 0].clone() * 0., cheby_poly2[:, :, 1].clone() * 0.
        cheby_poly1 = self.chebyshev1(x, dim, degree)

        if degree >= 2:
            cheby_poly2[:, :, 2] = 4.
        if degree >= 3:
            cheby_poly2[:, :, 3] = 24 * x

        for i in range(4, degree + 1):
            cheby_poly2[:, :, i] = 4 * cheby_poly1[:, :, i - 1].clone() + 2 * x * cheby_poly2[:, :, i - 1].clone() - cheby_poly2[:, :, i - 2].clone()

        return cheby_poly2

    def generate_H(self, Xf, Xbc, Xic, n=(0, 1)):
        Z_bc = self.chebyshev(Xbc, self.in_dim, self.degree1).reshape(Xbc.shape[0], -1) @ self.cheby_coeff
        Z_ic = self.chebyshev(Xic, self.in_dim, self.degree1).reshape(Xic.shape[0], -1) @ self.cheby_coeff
        H_bc = self.chebyshev(F.tanh(Z_bc), self.hidden_nodes, self.degree2).reshape(Xbc.shape[0], -1)
        H_ic = self.chebyshev(F.tanh(Z_ic), self.hidden_nodes, self.degree2).reshape(Xic.shape[0], -1)

        w_x = self.cheby_coeff[:self.degree1 + 1, :]
        w_y = self.cheby_coeff[self.degree1 + 1:, :]

        xic, yic = Xic[:, 0].clone()[:, None], Xic[:, 1].clone()[:, None]
        Zic = self.chebyshev(Xic, self.in_dim, self.degree1).reshape(Xic.shape[0], -1) @ self.cheby_coeff
        dZdx_ic = self.chebyshev1(xic, 1, self.degree1).reshape(Xic.shape[0], -1) @ w_x
        dZdy_ic = self.chebyshev1(yic, 1, self.degree1).reshape(Xic.shape[0], -1) @ w_y
        tanh1_ic = 1 - F.tanh(Zic) ** 2

        dHdZ_ic = self.chebyshev1(F.tanh(Zic), self.hidden_nodes, self.degree2)
        Hx_ic = dHdZ_ic * (dZdx_ic * tanh1_ic)[:, :, None]
        Hy_ic = dHdZ_ic * (dZdy_ic * tanh1_ic)[:, :, None]
        Hx_ic = Hx_ic.reshape(Xic.shape[0], -1)
        Hy_ic = Hy_ic.reshape(Xic.shape[0], -1)
        gradH_ic = Hx_ic * n[0] + Hy_ic * n[1]

        x, y = Xf[:, 0].clone()[:, None], Xf[:, 1].clone()[:, None]
        Z = self.chebyshev(Xf, self.in_dim, self.degree1).reshape(Xf.shape[0], -1) @ self.cheby_coeff
        H = self.chebyshev(F.tanh(Z), self.hidden_nodes, self.degree2).reshape(Xf.shape[0], -1)

        dZdx = self.chebyshev1(x, 1, self.degree1).reshape(Xf.shape[0], -1) @ w_x
        dZdy = self.chebyshev1(y, 1, self.degree1).reshape(Xf.shape[0], -1) @ w_y
        ddZdxx = self.chebyshev2(x, 1, self.degree1).reshape(Xf.shape[0], -1) @ w_x
        ddZdyy = self.chebyshev2(y, 1, self.degree1).reshape(Xf.shape[0], -1) @ w_y
        tanh1 = 1 - F.tanh(Z) ** 2
        tanh2 = -2 * F.tanh(Z) * tanh1

        dHdZ = self.chebyshev1(F.tanh(Z), self.hidden_nodes, self.degree2)
        ddHdZZ = self.chebyshev2(F.tanh(Z), self.hidden_nodes, self.degree2)

        Hx = dHdZ * (dZdx * tanh1)[:, :, None]
        Hy = dHdZ * (dZdy * tanh1)[:, :, None]
        Hx = Hx.clone().reshape(Xf.shape[0], -1)
        Hy = Hy.clone().reshape(Xf.shape[0], -1)
        gradH = Hx + Hy

        Hxx = ddHdZZ * (tanh1 * dZdx)[:, :, None] ** 2 + dHdZ * (tanh2 * dZdx ** 2)[:, :, None] + dHdZ * (tanh1 * ddZdxx)[:, :, None]
        Hyy = ddHdZZ * (tanh1 * dZdy)[:, :, None] ** 2 + dHdZ * (tanh2 * dZdy ** 2)[:, :, None] + dHdZ * (tanh1 * ddZdyy)[:, :, None]
        Hxx = Hxx.clone().reshape(Xf.shape[0], -1)
        Hyy = Hyy.clone().reshape(Xf.shape[0], -1)
        LapH = Hxx + Hyy

        return H, Hx, Hy, gradH, Hxx, Hyy, LapH, gradH_ic, H_bc, H_ic


class piCKANELM_SOV2_add_fusion(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, degree1_high, degree1_low, degree1_high_need, degree2, coeff_scaled_high=1.,
                 coeff_scaled_low=1., device=torch.device('cpu'), dtype=torch.float32):
        super(piCKANELM_SOV2_add_fusion, self).__init__()
        self.in_dim = in_dim
        self.hidden_nodes = hidden_nodes
        self.out_dim = out_dim
        self.degree1_high = degree1_high
        self.degree1_low = degree1_low
        self.degree1_high_need = degree1_high_need
        self.degree2 = degree2

        self.cheb_coeff_high = nn.Parameter(torch.empty(degree1_high_need, hidden_nodes).to(device=device, dtype=dtype), requires_grad=False)
        self.cheb_coeff_low = nn.Parameter(torch.empty(degree1_low + 1, hidden_nodes).to(device=device, dtype=dtype), requires_grad=False)
        self.beta = nn.Parameter(torch.ones(hidden_nodes * (degree2 + 1), out_dim, device=device, dtype=dtype), requires_grad=False)

        nn.init.normal_(self.cheb_coeff_high, mean=0.0, std=coeff_scaled_high / (in_dim * degree1_high_need))
        nn.init.normal_(self.cheb_coeff_low, mean=0.0, std=coeff_scaled_low / (in_dim * (degree1_low + 1)))

        degrees = torch.arange(degree1_high - degree1_high_need + 1, degree1_high + 1, device=device, dtype=dtype)
        self.cheb_coeff_high.data = self.cheb_coeff_high.data / (degrees[:, None] / degrees[0])

    def forward(self, X):
        H1_x = self.chebyshev(X[:, 0][:, None], 1, self.degree1_low).reshape(X.shape[0], -1) @ self.cheb_coeff_low
        H1_t = self.chebyshev(X[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(X.shape[0], -1) @ self.cheb_coeff_high
        output_x = self.chebyshev(F.tanh(H1_x), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1)
        output_t = self.chebyshev(F.tanh(H1_t), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1)
        output = (output_x + output_t) @ self.beta

        return output

    @staticmethod
    def chebyshev(x, dim, degree_max, degree_droped: int = None):
        cheby_poly = torch.ones(x.shape[0], dim, degree_max + 1).to(device=x.device, dtype=x.dtype)
        if degree_max >= 1:
            cheby_poly[:, :, 1] = x
        if degree_max >= 2:
            cheby_poly[:, :, 2] = 2 * x ** 2 - 1
        if degree_max >= 3:
            cheby_poly[:, :, 3] = 4 * x ** 3 - 3 * x

        for i in range(4, degree_max + 1):
            cheby_poly[:, :, i] = 2 * x * cheby_poly[:, :, i - 1].clone() - cheby_poly[:, :, i - 2].clone()

        if degree_droped is not None:
            start_cut = degree_max - degree_droped + 1
            cheby_poly = cheby_poly[:, :, start_cut:]
        return cheby_poly

    def chebyshev1(self, x, dim, degree_max, degree_droped: int = None):
        cheby_poly1 = torch.ones(x.shape[0], dim, degree_max + 1, device=x.device, dtype=x.dtype)
        cheby_poly1[:, :, 0] = cheby_poly1[:, :, 0].clone() * 0.
        cheby_poly = self.chebyshev(x, dim, degree_max)

        if degree_max >= 2:
            cheby_poly1[:, :, 2] = 4 * x
        if degree_max >= 3:
            cheby_poly1[:, :, 3] = 12 * x ** 2 - 3.

        for i in range(4, degree_max + 1):
            cheby_poly1[:, :, i] = (2 * cheby_poly[:, :, i - 1].clone() +
                                    2 * x * cheby_poly1[:, :, i - 1].clone() - cheby_poly1[:, :, i - 2].clone())

        if degree_droped is not None:
            start_cut = degree_max - degree_droped + 1
            cheby_poly1 = cheby_poly1[:, :, start_cut:]
        return cheby_poly1

    def chebyshev2(self, x, dim, degree_max, degree_droped: int = None):
        cheby_poly2 = torch.ones(x.shape[0], dim, degree_max + 1, dtype=x.dtype, device=x.device)
        cheby_poly2[:, :, 0], cheby_poly2[:, :, 1] = cheby_poly2[:, :, 0].clone() * 0., cheby_poly2[:, :, 1].clone() * 0.
        cheby_poly1 = self.chebyshev1(x, dim, degree_max)

        if degree_max >= 2:
            cheby_poly2[:, :, 2] = 4.
        if degree_max >= 3:
            cheby_poly2[:, :, 3] = 24 * x

        for i in range(4, degree_max + 1):
            cheby_poly2[:, :, i] = (4 * cheby_poly1[:, :, i - 1].clone() +
                                    2 * x * cheby_poly2[:, :, i - 1].clone() - cheby_poly2[:, :, i - 2].clone())

        if degree_droped is not None:
            start_cut = degree_max - degree_droped + 1
            cheby_poly2 = cheby_poly2[:, :, start_cut:]
        return cheby_poly2

    def generate_H(self, Xtf, Xbc, Xic):
        bcmid_x = self.chebyshev(Xbc[:, 0][:, None], 1, self.degree1_low).reshape(Xbc.shape[0], -1) @ self.cheb_coeff_low
        bcmid_t = self.chebyshev(Xbc[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xbc.shape[0], -1) @ self.cheb_coeff_high
        H_bc_x = self.chebyshev(F.tanh(bcmid_x), self.hidden_nodes, self.degree2)
        H_bc_t = self.chebyshev(F.tanh(bcmid_t), self.hidden_nodes, self.degree2)
        H_bc = (H_bc_x + H_bc_t).reshape(Xbc.shape[0], -1)

        icmid_x = self.chebyshev(Xic[:, 0][:, None], 1, self.degree1_low).reshape(Xic.shape[0], -1) @ self.cheb_coeff_low
        icmid_t = self.chebyshev(Xic[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xic.shape[0], -1) @ self.cheb_coeff_high
        H_ic_x = self.chebyshev(F.tanh(icmid_x), self.hidden_nodes, self.degree2)
        H_ic_t = self.chebyshev(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
        H_ic = (H_ic_x + H_ic_t).reshape(Xic.shape[0], -1)

        dZdy_ic = self.chebyshev1(Xic[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xic.shape[0], -1) @ self.cheb_coeff_high
        tanh1_y_ic = 1 - F.tanh(icmid_t) ** 2
        dHxdZ_ic = self.chebyshev1(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
        Ht_ic = dHxdZ_ic * dZdy_ic[:, :, None] * tanh1_y_ic[:, :, None]
        Ht_ic = Ht_ic.clone().reshape(Xic.shape[0], -1)

        # add fusion: each branch depends on only its own coordinate

        x, t = Xtf[:, 0].clone()[:, None], Xtf[:, 1].clone()[:, None]
        Z_x = self.chebyshev(x, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_low
        Z_t = self.chebyshev(t, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_high
        H_x = self.chebyshev(F.tanh(Z_x), self.hidden_nodes, self.degree2)
        H_t = self.chebyshev(F.tanh(Z_t), self.hidden_nodes, self.degree2)
        H = H_x + H_t

        w_x = self.cheb_coeff_low
        w_y = self.cheb_coeff_high
        dZdx = self.chebyshev1(x, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_x
        dZdy = self.chebyshev1(t, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ w_y
        ddZdxx = self.chebyshev2(x, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_x
        ddZdyy = self.chebyshev2(t, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ w_y
        tanh1_x = 1 - F.tanh(Z_x) ** 2
        tanh1_y = 1 - F.tanh(Z_t) ** 2
        tanh2_x = -2 * F.tanh(Z_x) * tanh1_x
        tanh2_y = -2 * F.tanh(Z_t) * tanh1_y

        dHxdZ = self.chebyshev1(F.tanh(Z_x), self.hidden_nodes, self.degree2)
        ddHxdZZ = self.chebyshev2(F.tanh(Z_x), self.hidden_nodes, self.degree2)
        dHydZ = self.chebyshev1(F.tanh(Z_t), self.hidden_nodes, self.degree2)
        ddHydZZ = self.chebyshev2(F.tanh(Z_t), self.hidden_nodes, self.degree2)

        Hx = dHxdZ * dZdx[:, :, None] * tanh1_x[:, :, None]
        Hy = dHydZ * dZdy[:, :, None] * tanh1_y[:, :, None]
        Hx = Hx.clone().reshape(Xtf.shape[0], -1)
        Hy = Hy.clone().reshape(Xtf.shape[0], -1)
        gradH = Hx + Hy

        Hxx = (ddHxdZZ * (tanh1_x * dZdx)[:, :, None] ** 2 + dHxdZ * (tanh2_x * dZdx ** 2)[:, :, None] + dHxdZ * (tanh1_x * ddZdxx)[:, :, None])
        Hyy = (ddHydZZ * (tanh1_y * dZdy)[:, :, None] ** 2 + dHydZ * (tanh2_y * dZdy ** 2)[:, :, None] + dHydZ * (tanh1_y * ddZdyy)[:, :, None])
        Hxx = Hxx.clone().reshape(Xtf.shape[0], -1)
        Hyy = Hyy.clone().reshape(Xtf.shape[0], -1)
        LapH = Hxx + Hyy

        return H, Hx, Hy, gradH, Hxx, Hyy, LapH, H_bc, H_ic, Ht_ic


class piELMHKAN(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, num_grid1=5, num_grid2=5, knot_grid1=[-1., 1.], knot_grid2=[-1, 1.], coeff1_sacle=1.5,
                 sigmoid_scale=1., device=torch.device('cpu'), dtype=torch.float32):
        super(piELMHKAN, self).__init__()
        self.in_dim = in_dim
        self.hidden_nodes = hidden_nodes
        self.out_dim = out_dim
        self.num_grid1 = num_grid1
        self.num_grid2 = num_grid2

        self.grid1 = torch.linspace(knot_grid1[0], knot_grid1[1], num_grid1+1, device=device, dtype=dtype)[None, :].expand(in_dim, num_grid1+1)
        self.grid2 = torch.linspace(knot_grid2[0], knot_grid2[1], num_grid2, device=device, dtype=dtype)[None, :].expand(hidden_nodes, num_grid2)
        ctrl = np.random.uniform(-coeff1_sacle, coeff1_sacle, size=(in_dim, hidden_nodes, num_grid1)).transpose(0, 2, 1).reshape(in_dim * num_grid1, hidden_nodes)
        self.ctrl = nn.Parameter(torch.from_numpy(ctrl).to(device=device, dtype=dtype), requires_grad=False)
        self.beta = nn.Parameter(torch.zeros(hidden_nodes * num_grid2, out_dim, device=device, dtype=dtype), requires_grad=False)

        self.act1 = lambda x, grid: torch.sigmoid(sigmoid_scale * (x[:, :, None] - grid[None, :, :]))
        self.sigmoid_scale = sigmoid_scale
        self.dist = (knot_grid2[1] - knot_grid2[0]) / (num_grid2 - 1)
        self.act2 = lambda x, grid: torch.exp(-((x[:, :, None] - grid[None, :, :]) / self.dist) ** 2)

    def forward(self, x):
        basis = self.sig(x, self.grid1)
        H1 = basis.reshape(x.shape[0], -1) @ self.ctrl
        output = self.act2(H1, self.grid2).reshape(x.shape[0], -1) @ self.beta
        return output

    def sig(self, x, grid):
        """DiffSigmoid"""
        sgm_1 = self.act1(x, grid)
        bumps = sgm_1[:, :, 1:] - sgm_1[:, :, :-1]
        return bumps

    def sig_1(self, x, grid):
        sgm_1 = self.sigmoid_scale * (self.act1(x, grid) - self.act1(x, grid) ** 2)
        bumps = sgm_1[:, :, 1:] - sgm_1[:, :, :-1]
        return bumps

    def sig_2(self, x, grid):
        sgm_1 = self.sigmoid_scale * (self.act1(x, grid) - self.act1(x, grid) ** 2)
        sgm_2 = self.sigmoid_scale * (sgm_1 - 2 * self.act1(x, grid) * sgm_1)
        bumps = sgm_2[:, :, 1:] - sgm_2[:, :, :-1]
        return bumps

    def phi_1(self, x):
        return self.act2(x, self.grid2) * -(2 / self.dist ** 2) * (x[:, :, None] - self.grid2[None, :, :])

    def phi_2(self, x):
        part1 = self.act2(x, self.grid2) * (-(2 / self.dist ** 2) * (x[:, :, None] - self.grid2[None, :, :])) ** 2
        part2 = self.act2(x, self.grid2) * -(2 / self.dist ** 2)
        return part1 + part2

    def generate_H(self, Xf, Xbc, Xic=None):
        H_bc = self.act2(self.sig(Xbc, self.grid1).reshape(Xbc.shape[0], -1) @ self.ctrl, self.grid2).reshape(Xbc.shape[0], -1)
        if Xic is not None:
            H_ic = self.act2(self.sig(Xic, self.grid1).reshape(Xic.shape[0], -1) @ self.ctrl, self.grid2).reshape(Xic.shape[0], -1)
        else:
            H_ic = None

        Z = self.sig(Xf, self.grid1).reshape(Xf.shape[0], -1) @ self.ctrl
        H = self.act2(Z, self.grid2).reshape(Xf.shape[0], -1)
        w_x = self.ctrl[:self.num_grid1, :].T  # [hidden_nodes, num_grid1]
        w_y = self.ctrl[self.num_grid1:, :].T  # [hidden_nodes, num_grid2]
        x = Xf[:, 0][:, None]
        y = Xf[:, 1][:, None]
        grid1_x = self.grid1[0, :][None, :]  # [1, num_grid1]
        grid1_y = self.grid1[1, :][None, :]  # [1, num_grid1]
        sig_1_x = self.sig_1(x, grid1_x).permute(0, 2, 1).squeeze()
        sig_1_y = self.sig_1(y, grid1_y).permute(0, 2, 1).squeeze()
        sig_2_x = self.sig_2(x, grid1_x).permute(0, 2, 1).squeeze()
        sig_2_y = self.sig_2(y, grid1_y).permute(0, 2, 1).squeeze()
        dZdx = sig_1_x @ w_x.T
        dZdy = sig_1_y @ w_y.T
        dZZdxx = sig_2_x @ w_x.T
        dZZdyy = sig_2_y @ w_y.T
        # print(dZdx.shape, dZZdxx.shape)
        Hx = (self.phi_1(Z) * dZdx[:, :, None]).reshape(x.shape[0], -1)
        Hy = (self.phi_1(Z) * dZdy[:, :, None]).reshape(y.shape[0], -1)
        gradHf = Hx + Hy
        Hxx = (self.phi_2(Z) * dZdx[:, :, None] ** 2 + self.phi_1(Z) * dZZdxx[:, :, None]).reshape(Xf.shape[0], -1)
        Hyy = (self.phi_2(Z) * dZdy[:, :, None] ** 2 + self.phi_1(Z) * dZZdyy[:, :, None]).reshape(Xf.shape[0], -1)
        LapHf = (Hxx + Hyy).reshape(Xf.shape[0], -1)

        return H, Hx, Hy, gradHf, Hxx, Hyy, LapHf, H_bc, H_ic




def solve_2d_helmholtz_piELMHKAN(n_interior=4096, n_bc=256, bc_weight=1.0):
    k, xf, xbc, xf_ori, rhs, T_bc, analy_xf = _helmholtz_data(
        n_interior, n_bc
    )

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)

    hidden_nodes = 200
    num_grid1 = 10
    num_grid2 = 10
    model = piELMHKAN(
        in_dim=2,
        hidden_nodes=hidden_nodes,
        out_dim=1,
        num_grid1=num_grid1,
        num_grid2=num_grid2,
        knot_grid1=[-1., 1.],
        knot_grid2=[-1., 1.],
        coeff1_sacle=1.5,
        sigmoid_scale=1.0,
        device=device,
        dtype=dtype,
    )
    param = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    H, _, _, _, _, _, LapH, H_bc, _ = model.generate_H(xf_t, xbc_t)
    A_pde = LapH + (k ** 2) * H
    A = torch.vstack([A_pde, math.sqrt(bc_weight) * H_bc])
    b = torch.vstack([rhs_t, math.sqrt(bc_weight) * T_bc_t])
    beta = torch.linalg.pinv(A) @ b
    model.beta.data = beta
    elapsed_time = time.time() - start_time

    pde_loss = torch.mean((A_pde @ beta - rhs_t) ** 2).item()
    bc_loss = mean_squared_error(
        (H_bc @ beta).detach().cpu().numpy(), T_bc
    )
    pred = model(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, pred)
    mae = mean_absolute_error(analy_xf, pred)
    error = np.abs(pred - analy_xf)

    print("\n[piELMHKAN] Optimization Finished!")
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}")
    print(f"relative_error: {relative_error:.6e}, MAE: {mae:.6e}, MSE: {mse:.6e}")
    print(f"Parameter: {param}, Time: {elapsed_time:.4f}s")
    plot(xf_ori, pred, analy_xf, error, save_name='2d_helmholtz_piELMHKAN')
    plot_log_error(xf_ori, error, save_name='2d_helmholtz_piELMHKAN')



def _helmholtz_data(n_interior, n_bc):
    k = 1.0
    a1, a2 = 1.0, 4.0
    pi = np.pi

    xf, xbc = square_sampler_sobol(n_interior, n_bc)
    xf_ori = np.concatenate([xf, xbc], axis=0)

    def u_exact(X):
        return np.sin(a1 * pi * X[:, 0]) * np.sin(a2 * pi * X[:, 1])

    def RHS(X):
        return (k ** 2 - (a1 * pi) ** 2 - (a2 * pi) ** 2) * u_exact(X)

    rhs = RHS(xf)[:, np.newaxis]
    T_bc = np.zeros((xbc.shape[0], 1))
    analy_xf = u_exact(xf_ori)[:, np.newaxis]
    return k, xf, xbc, xf_ori, rhs, T_bc, analy_xf


def solve_2d_helmholtz_piELMKAN(n_interior=4096, n_bc=256, bc_weight=1.0):
    k, xf, xbc, xf_ori, rhs, T_bc, analy_xf = _helmholtz_data(n_interior, n_bc)

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)

    hidden_nodes = 200
    num_grid = 10
    model = piELMKAN(in_dim=2, hidden_nodes=hidden_nodes, out_dim=1,
                     grid_range=[-1, 1], num_grid=num_grid, coeff_scale=3.,
                     device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    H, H_x, H_y, gradHf, LapHf, H_bc = model.generate_H(xf_t, xbc_t)
    # Helmholtz: (LapH + k² H) β = rhs;  H_bc β = 0
    A = torch.vstack([LapHf + (k ** 2) * H, math.sqrt(bc_weight) * H_bc])
    b = torch.vstack([rhs_t, math.sqrt(bc_weight) * T_bc_t])
    beta = torch.linalg.pinv(A) @ b
    model.beta.data = beta
    elapsed_time = time.time() - start_time

    pde_loss = torch.mean(((LapHf + (k ** 2) * H) @ beta - rhs_t) ** 2).item()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    pred = model(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, pred)
    mae = mean_absolute_error(analy_xf, pred)
    error = np.abs(pred - analy_xf)

    print("\n[piELMKAN] Optimization Finished!")
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}")
    print(f"relative_error: {relative_error:.6e}, MAE: {mae:.6e}, MSE: {mse:.6e}")
    print(f"Parameter: {param}, Time: {elapsed_time:.4f}s")

    plot(xf_ori, pred, analy_xf, error, save_name='2d_helmholtz_piELMKAN')
    plot_log_error(xf_ori, error, save_name='2d_helmholtz_piELMKAN')



def solve_2d_helmholtz_piCKANELM_SOV2(seed, n_interior=4096, n_bc=256, bc_weight=1.0):
    k, xf, xbc, xf_ori, rhs, T_bc, analy_xf = _helmholtz_data(n_interior, n_bc)

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    xic_dummy = xbc_t[:1, :].clone()

    hidden_nodes = 200
    degree1_low = 3
    degree1_high = 15
    degree1_high_need = 10
    degree2 = 10
    model = piCKANELM_SOV2(in_dim=2, hidden_nodes=hidden_nodes, out_dim=1,
                                   degree1_high=degree1_high, degree1_low=degree1_low,
                                   degree1_high_need=degree1_high_need, degree2=degree2,
                                   coeff_scaled_high=1.0, coeff_scaled_low=1.0,
                                   device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    H, Hx, Hy, gradH, Hxx, Hyy, LapH, H_bc, _, _ = model.generate_H(xf_t, xbc_t, xic_dummy)
    A = torch.vstack([LapH + (k ** 2) * H, math.sqrt(bc_weight) * H_bc])
    b = torch.vstack([rhs_t, math.sqrt(bc_weight) * T_bc_t])
    beta = torch.linalg.pinv(A) @ b
    model.beta.data = beta
    elapsed_time = time.time() - start_time

    pde_loss = torch.mean(((LapH + (k ** 2) * H) @ beta - rhs_t) ** 2).item()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    pred = model(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, pred)
    mae = mean_absolute_error(analy_xf, pred)
    error = np.abs(pred - analy_xf)

    print("\n[piCKANELM_SOV2] Optimization Finished!")
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}")
    print(f"relative_error: {relative_error:.6e}, MAE: {mae:.6e}, MSE: {mse:.6e}")
    print(f"Parameter: {param}, Time: {elapsed_time:.4f}s")

    plot(xf_ori, pred, analy_xf, error, save_name='2d_helmholtz_piCKANELM_SOV2')
    plot_log_error(xf_ori, error, save_name='2d_helmholtz_piCKANELM_SOV2', seed=seed)
    return float(relative_error)


def solve_2d_helmholtz_piCKANELM_SOV2_nodecay(n_interior=4096, n_bc=256, bc_weight=1.0):
    k, xf, xbc, xf_ori, rhs, T_bc, analy_xf = _helmholtz_data(n_interior, n_bc)

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    xic_dummy = xbc_t[:1, :].clone()

    hidden_nodes = 200
    degree1_low = 3
    degree1_high = 15
    degree1_high_need = 10
    degree2 = 10
    model = piCKANELM_SOV2_nodecay(in_dim=2, hidden_nodes=hidden_nodes, out_dim=1,
                                   degree1_high=degree1_high, degree1_low=degree1_low,
                                   degree1_high_need=degree1_high_need, degree2=degree2,
                                   coeff_scaled_high=1.0, coeff_scaled_low=1.0,
                                   device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    H, Hx, Hy, gradH, Hxx, Hyy, LapH, H_bc, _, _ = model.generate_H(xf_t, xbc_t, xic_dummy)
    A = torch.vstack([LapH + (k ** 2) * H, math.sqrt(bc_weight) * H_bc])
    b = torch.vstack([rhs_t, math.sqrt(bc_weight) * T_bc_t])
    beta = torch.linalg.pinv(A) @ b
    model.beta.data = beta
    elapsed_time = time.time() - start_time

    pde_loss = torch.mean(((LapH + (k ** 2) * H) @ beta - rhs_t) ** 2).item()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    pred = model(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, pred)
    mae = mean_absolute_error(analy_xf, pred)
    error = np.abs(pred - analy_xf)

    print("\n[piCKANELM_SOV2] Optimization Finished!")
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}")
    print(f"relative_error: {relative_error:.6e}, MAE: {mae:.6e}, MSE: {mse:.6e}")
    print(f"Parameter: {param}, Time: {elapsed_time:.4f}s")

    plot(xf_ori, pred, analy_xf, error, save_name='2d_helmholtz_piCKANELM_SOV2_nodecay')
    plot_log_error(xf_ori, error, save_name='2d_helmholtz_piCKANELM_SOV2_nodecay')



def solve_2d_helmholtz_piCKANELM_single(n_interior=4096, n_bc=256, bc_weight=1.0):
    k, xf, xbc, xf_ori, rhs, T_bc, analy_xf = _helmholtz_data(n_interior, n_bc)
    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    xic_dummy = xbc_t[:1].clone()

    model = piCKANELM_single(
        in_dim=2, hidden_nodes=200, out_dim=1, degree1=12, degree2=10,
        coeff_scaled=1.0, device=device, dtype=dtype,
    )
    param = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    H, _, _, _, _, _, LapH, _, H_bc, _ = model.generate_H(
        xf_t, xbc_t, xic_dummy
    )
    A_pde = LapH + (k ** 2) * H
    A = torch.vstack([A_pde, math.sqrt(bc_weight) * H_bc])
    b = torch.vstack([rhs_t, math.sqrt(bc_weight) * T_bc_t])
    beta = torch.linalg.pinv(A) @ b
    model.beta.data = beta
    elapsed_time = time.time() - start_time

    pde_loss = torch.mean((A_pde @ beta - rhs_t) ** 2).item()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    pred = model(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, pred)
    mae = mean_absolute_error(analy_xf, pred)
    error = np.abs(pred - analy_xf)

    print("\n[piCKANELM_single] Optimization Finished!")
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}")
    print(f"relative_error: {relative_error:.6e}, MAE: {mae:.6e}, MSE: {mse:.6e}")
    print(f"Parameter: {param}, Time: {elapsed_time:.4f}s")
    plot(xf_ori, pred, analy_xf, error, save_name='2d_helmholtz_piCKANELM_single')
    plot_log_error(xf_ori, error, save_name='2d_helmholtz_piCKANELM_single')



def solve_2d_helmholtz_piCKANELM_SOV2_add_fusion(
        n_interior=4096, n_bc=256, bc_weight=1.0):
    k, xf, xbc, xf_ori, rhs, T_bc, analy_xf = _helmholtz_data(n_interior, n_bc)
    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    xic_dummy = xbc_t[:1].clone()

    model = piCKANELM_SOV2_add_fusion(
        in_dim=2, hidden_nodes=200, out_dim=1,
        degree1_high=15, degree1_low=3, degree1_high_need=10, degree2=10,
        coeff_scaled_high=1.0, coeff_scaled_low=1.0,
        device=device, dtype=dtype,
    )
    param = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    H, _, _, _, Hxx, Hyy, _, H_bc, _, _ = model.generate_H(
        xf_t, xbc_t, xic_dummy
    )
    H = H.reshape(xf_t.shape[0], -1)
    A_pde =  Hxx + Hyy + (k ** 2) * H
    A = torch.vstack([A_pde, math.sqrt(bc_weight) * H_bc])
    b = torch.vstack([rhs_t, math.sqrt(bc_weight) * T_bc_t])
    beta = torch.linalg.pinv(A) @ b
    model.beta.data = beta
    elapsed_time = time.time() - start_time

    pde_loss = torch.mean((A_pde @ beta - rhs_t) ** 2).item()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    pred = model(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, pred)
    mae = mean_absolute_error(analy_xf, pred)
    error = np.abs(pred - analy_xf)

    print("\n[piCKANELM_SOV2_add_fusion] Optimization Finished!")
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}")
    print(f"relative_error: {relative_error:.6e}, MAE: {mae:.6e}, MSE: {mse:.6e}")
    print(f"Parameter: {param}, Time: {elapsed_time:.4f}s")
    plot(xf_ori, pred, analy_xf, error, save_name='2d_helmholtz_piCKANELM_SOV2_add_fusion')
    plot_log_error(xf_ori, error, save_name='2d_helmholtz_piCKANELM_SOV2_add_fusion')



def solve_2d_helmholtz_efficientKAN(n_interior=4096, n_bc=256, bc_weight=1.0, adam_iter=3000):
    k, xf, xbc, xf_ori, rhs, T_bc, analy_xf = _helmholtz_data(n_interior, n_bc)

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype).requires_grad_(True)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    analy_xf_t = torch.from_numpy(analy_xf).to(device=device, dtype=dtype)

    model = efficentKAN([2, 64, 64, 64, 1],
                        grid_size=10,
                        spline_order=3,
                        scale_noise=0.1,
                        scale_base=1.0,
                        scale_spline=1.0,
                        base_activation=nn.Tanh,
                        grid_eps=0.02,
                        grid_range=[-1, 1]).to(device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())

    criterion = nn.MSELoss()
    optimizer_Adam = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.time()
    print("[efficientKAN] Start phase 1 to train on Adam...")
    for i in range(adam_iter):
        y_f = model(xf_t)
        df = torch.autograd.grad(y_f.sum(), xf_t, create_graph=True)[0]
        dxx = torch.autograd.grad(df[:, 0].sum(), xf_t, create_graph=True)[0][:, 0:1]
        dyy = torch.autograd.grad(df[:, 1].sum(), xf_t, create_graph=True)[0][:, 1:2]
        lap = dxx + dyy
        pde_loss = torch.mean((lap + (k ** 2) * y_f - rhs_t) ** 2)

        y_bc = model(xbc_t)
        bc_loss = criterion(y_bc, T_bc_t)

        total_loss = pde_loss + bc_weight * bc_loss

        optimizer_Adam.zero_grad()
        total_loss.backward()
        optimizer_Adam.step()

        if i % 500 == 0:
            with torch.no_grad():
                pred = model(xf_ori_t)
                relative_error = torch.linalg.norm(pred - analy_xf_t) / torch.linalg.norm(analy_xf_t)
            print(f"It: {i:05d}, pde_loss: {pde_loss.item():.4e}, bc_loss: {bc_loss.item():.4e}, "
                  f"total_loss: {total_loss.item():.4e}, relative_error: {relative_error.item():.4e}")

    print("\n[efficientKAN] Start phase 2 to optimize on LBFGS...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=5000,
        max_eval=50000,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        y_f = model(xf_t)
        df = torch.autograd.grad(y_f.sum(), xf_t, create_graph=True)[0]
        dxx = torch.autograd.grad(df[:, 0].sum(), xf_t, create_graph=True)[0][:, 0:1]
        dyy = torch.autograd.grad(df[:, 1].sum(), xf_t, create_graph=True)[0][:, 1:2]
        lap = dxx + dyy
        pde_loss = torch.mean((lap + (k ** 2) * y_f - rhs_t) ** 2)

        y_bc = model(xbc_t)
        bc_loss = criterion(y_bc, T_bc_t)

        total_loss = pde_loss + bc_weight * bc_loss
        total_loss.backward()
        return total_loss

    optimizer_lbfgs.step(closure)
    elapsed_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        pred = model(xf_ori_t).detach().cpu().numpy()
        relative_error = np.linalg.norm(pred - analy_xf) / np.linalg.norm(analy_xf)
        mse = mean_squared_error(analy_xf, pred)
        mae = mean_absolute_error(analy_xf, pred)
        error = np.abs(pred - analy_xf)

    print("\n[efficientKAN] Optimization Finished!")
    print(f"relative_error: {relative_error:.6e}, MAE: {mae:.6e}, MSE: {mse:.6e}")
    print(f"Parameter: {param}, Time: {elapsed_time:.4f}s")

    plot(xf_ori, pred, analy_xf, error, save_name='2d_helmholtz_efficientKAN')
    plot_log_error(xf_ori, error, save_name='2d_helmholtz_efficientKAN')



def solve_2d_helmholtz_alpkan(n_interior=4096, n_bc=256):
    k = 1.0
    a1 = 1.0
    a2 = 4.0
    pi = np.pi

    lr_theta = 1e-3
    lr_lambda = 5e-2
    lr_sigma = 5e-2

    xf, xbc = square_sampler_sobol(n_interior, n_bc)
    xf_ori = np.concatenate([xf, xbc], axis=0)

    def u_exact(X):
        return np.sin(a1 * pi * X[:, 0]) * np.sin(a2 * pi * X[:, 1])

    def RHS(X):
        return (k ** 2 - (a1 * pi) ** 2 - (a2 * pi) ** 2) * u_exact(X)

    rhs = RHS(xf)[:, np.newaxis]
    T_bc = np.zeros((xbc.shape[0], 1))
    analy_xf = u_exact(xf_ori)[:, np.newaxis]

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype).requires_grad_(True)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    analy_xf_t = torch.from_numpy(analy_xf).to(device=device, dtype=dtype)
    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)

    model = AL_PKAN(input_dim=2, gru_hidden=64, gru_layers=2, kan_layers=[64, 64, 64, 1],
                    grid_size=10, spline_order=3, history_steps=3, device=device, dtype=dtype)

    lambda_bc = torch.zeros((n_bc, 1), device=device, dtype=dtype, requires_grad=False)
    sigma = torch.tensor([1.0], device=device, dtype=dtype, requires_grad=False)

    param = sum(p.numel() for p in model.parameters())
    start_time = time.time()

    adam_iteration = 50000
    optimizer_theta = torch.optim.AdamW(model.parameters(), lr=lr_theta)

    print("Starting AL-PKAN Optimization (AdamW + Augmented Lagrangian)...")
    for i in range(adam_iteration):
        model.train()
        optimizer_theta.zero_grad()

        y_xf = model(xf_t)
        df = torch.autograd.grad(y_xf.sum(), xf_t, create_graph=True)[0]
        dxx = torch.autograd.grad(df[:, 0].sum(), xf_t, create_graph=True)[0][:, 0:1]
        dyy = torch.autograd.grad(df[:, 1].sum(), xf_t, create_graph=True)[0][:, 1:2]
        lapd = dxx + dyy

        pde_res = lapd + (k ** 2) * y_xf - rhs_t
        pde_loss = torch.mean(pde_res ** 2)

        y_bc = model(xbc_t)
        bc_res = y_bc - T_bc_t
        bc_mse = torch.mean(bc_res ** 2)

        # L = PDE_loss + sigma * BC_mse + <lambda, BC_res>
        lagrangian_term = torch.mean(lambda_bc * bc_res)
        penalty_term = sigma * bc_mse

        total_loss = pde_loss + penalty_term + lagrangian_term

        total_loss.backward()
        optimizer_theta.step()

        with torch.no_grad():
            lambda_bc += lr_lambda * bc_res
            sigma += lr_sigma * bc_mse

        if i % 1000 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(xf_ori_t)
                relative_error = torch.linalg.norm(pred - analy_xf_t) / torch.linalg.norm(analy_xf_t)
            print(f"Epoch: {i:05d}, PDE_loss: {pde_loss.item():.4e}, BC_mse: {bc_mse.item():.4e}, "
                  f"Sigma: {sigma.item():.2f}, Rel_Error: {relative_error:.4e}")

    elapsed_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        pred = model(xf_ori_t).detach().cpu().numpy()
        relative_error = np.linalg.norm(pred - analy_xf) / np.linalg.norm(analy_xf)
        mse = mean_squared_error(analy_xf, pred)
        mae = mean_absolute_error(analy_xf, pred)
        error = np.abs(pred - analy_xf)

    print(f"\nOptimization Finished!")
    print(f"relative_error: {relative_error:.6e}")
    print(f"MAE: {mae:.6e}, MSE: {mse:.6e}, Parameter: {param}, Time: {elapsed_time:.4f}s")

    plot(xf_ori, pred, analy_xf, error, save_name='2d_helmholtz_alpkan')
    plot_log_error(xf_ori, error, save_name='2d_helmholtz_alpkan')



if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # torch.manual_seed(1024)
    # np.random.seed(1024)

    run = {
        'piELMKAN':              False,
        'piCKANELM_single':      False,
        'piCKANELM_SOV2':        True,
        'piCKANELM_SOV2_add':    False,
        'piCKANELM_SOV2_nodecay': False,
        'piELMHKAN':             False,
        'efficientKAN':          False,
        'alpkan':                False,
    }

    if run['piELMKAN']:
        print("=" * 60)
        print("=" * 60)
        solve_2d_helmholtz_piELMKAN()  

    if run['piELMHKAN']:
        print("=" * 60)
        print("=" * 60)
        solve_2d_helmholtz_piELMHKAN() 

    if run['piCKANELM_single']:
        print("\n" + "=" * 60)
        print("=" * 60)
        solve_2d_helmholtz_piCKANELM_single() 

    if run['piCKANELM_SOV2']:
        print("\n" + "=" * 60)
        print("=" * 60)
        RANDOM_SEEDS = [1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030]
        SEED_RESULT_DIR = "26-0921Ablation_randomseed"
        os.makedirs(SEED_RESULT_DIR, exist_ok=True)
        CASE_NAME = "solve_2d_helmholtz_piCKANELM_SOV2"
        rel_errors = []
        _orig_show = plt.show
        plt.show = lambda *a, **k: plt.close("all")
        try:
            for seed in RANDOM_SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                err = solve_2d_helmholtz_piCKANELM_SOV2(seed)  # pde_loss: 1.539819e-10, bc_loss: 2.232077e-11
                rel_errors.append(float(err))
                print(f"[{CASE_NAME}] seed={seed} relative_error={float(err):.6e}")
        finally:
            plt.show = _orig_show
            plt.close("all")
        arr = np.array(rel_errors, dtype=np.float64)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        min_v, max_v = float(np.min(arr)), float(np.max(arr))
        print(f"[{CASE_NAME}] over {len(arr)} seeds: Mean={mean_v:.6e} Std={std_v:.6e} "
              f"Min={min_v:.6e} Max={max_v:.6e}  =>  {mean_v:.6e} ± {std_v:.6e}")
        base = f"{CASE_NAME}_error_"
        txt_path = os.path.join(SEED_RESULT_DIR, base + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"case: {CASE_NAME}\nseeds: {RANDOM_SEEDS}\n")
            for s, e in zip(RANDOM_SEEDS, rel_errors):
                f.write(f"seed={s} relative_error={e:.10e}\n")
            f.write(f"Mean={mean_v:.10e}\nStd(ddof=1)={std_v:.10e}\n"
                    f"Min={min_v:.10e}\nMax={max_v:.10e}\n"
                    f"Mean+-Std: {mean_v:.6e} +- {std_v:.6e}\n")
        print(f"saved: {txt_path}")
        fig_path = os.path.join(SEED_RESULT_DIR, base + ".png")
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
        x = np.arange(len(RANDOM_SEEDS))
        ax.errorbar(x, arr, yerr=np.full_like(arr, std_v), fmt="o-", capsize=4,
                    label=f"Mean={mean_v:.3e} ± Std={std_v:.3e}")
        ax.axhline(mean_v, color="r", linestyle="--", linewidth=1.2, label="Mean")
        ax.axhspan(mean_v - std_v, mean_v + std_v, color="r", alpha=0.12, label="±Std")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in RANDOM_SEEDS], rotation=30)
        ax.set_xlabel("seed")
        ax.set_ylabel("relative_error")
        ax.set_title(f"{CASE_NAME} (n={len(RANDOM_SEEDS)})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"saved: {fig_path}")
        
    if run['piCKANELM_SOV2_add']:
        print("\n" + "=" * 60)
        print("=" * 60)
        solve_2d_helmholtz_piCKANELM_SOV2_add_fusion()  
    if run['piCKANELM_SOV2_nodecay']:
        print("\n" + "=" * 60)
        print("=" * 60)
        solve_2d_helmholtz_piCKANELM_SOV2_nodecay()

    if run['efficientKAN']:
        print("\n" + "=" * 60)
        print("=" * 60)
        solve_2d_helmholtz_efficientKAN()  


    if run['alpkan']:
        print("\n" + "=" * 60)
        print("=" * 60)
        solve_2d_helmholtz_alpkan()  # relative_error: 3.939387e-03 MAE: 1.480184e-03, MSE: 3.651475e-06, Parameter: 161856, Time: 21630.4125s



