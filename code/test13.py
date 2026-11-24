import os
import csv
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyDOE2 import lhs
from scipy.interpolate import griddata
from efficentKAN import efficentKAN

RESULT_PIC_DIR = 'result_pic'
os.makedirs(RESULT_PIC_DIR, exist_ok=True)


class piELM(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, weight_scale=5., bias_scale=2., device=torch.device('cpu'), dtype=torch.float32):
        super(piELM, self).__init__()
        self.in_dim = in_dim
        self.hidden_nodes = hidden_nodes
        self.out_dim = out_dim
        self.device = device

        w = np.random.uniform(-weight_scale, weight_scale, size=(hidden_nodes, in_dim))
        b = np.random.uniform(-bias_scale, bias_scale, size=hidden_nodes)

        m = 0.
        M = 0.
        for i in range(in_dim):
            m += w[:, i]
            M += w[:, i] ** 2
        self.M = torch.from_numpy(M).to(device=device, dtype=dtype)
        self.m = torch.from_numpy(m).to(device=device, dtype=dtype)

        self.W = nn.Parameter(torch.from_numpy(w).to(device=device, dtype=dtype), requires_grad=False)
        self.b = nn.Parameter(torch.from_numpy(b).to(device=device, dtype=dtype), requires_grad=False)

        self.beta = nn.Parameter(torch.zeros(hidden_nodes, 1, device=device, dtype=dtype), requires_grad=False)

    def forward(self, x):
        H = F.tanh(x @ self.W.T + self.b[None, :])
        output = H @ self.beta
        return output

    def generate_H_tanh(self, Xf, Xbc, Xic=None):
        H_bc = F.tanh(Xbc @ self.W.T + self.b[None, :])

        if Xic is not None:
            H_ic = F.tanh(Xic @ self.W.T + self.b[None, :])
        else:
            H_ic = None

        H = F.tanh(Xf @ self.W.T + self.b)
        # phi'(z) = 1 - tanh(z)^2
        phi_1 = 1. - H ** 2
        Hx = phi_1 * self.W[:, 0]
        Hy = phi_1 * self.W[:, 1]
        gradHf = self.m * phi_1
        # phi''(z) = -2 * tanh(z) * (1 - tanh(z)^2)
        phi_2 = -2. * H * (1. - H ** 2)
        Hxx = phi_2 * self.W[:, 0] ** 2
        Hyy = phi_2 * self.W[:, 1] ** 2
        LapHf = self.M * phi_2

        return H, Hx, Hy, gradHf, Hxx, Hyy, LapHf, H_bc, H_ic


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


class piNN(nn.Module):
    def __init__(self, device=torch.device('cpu'), dtype=torch.float32):
        super(piNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 300, True, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(300, 300, True, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            # nn.BatchNorm1d(300, device=device, dtype=dtype),
            nn.Linear(300, 300, True, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            # nn.BatchNorm1d(300, device=device, dtype=dtype),
        )
        self.final_layer = nn.Linear(300, 1, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = self.final_layer(x)
        return x


class piCKANELM(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, degree1, degree2, coeff_scaled=1.,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(piCKANELM, self).__init__()
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
        input_width = self.degree1 + 1
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


def plot(X, pred_vals=None, analy_vals=None, error=None, plot_type="2d", title="pred", title_pred="", title_analy="", title_error="",
         value_range=None, error_range=None, contour_levels=25, save_name='result', save_dir=RESULT_PIC_DIR):
    os.makedirs(save_dir, exist_ok=True)
    x_coord, y_coord, z_coord = 0., 0., 0.
    if X.ndim == 1:
        x_coord = X
    else:
        if X.ndim == 2:
            x_coord, y_coord = X[:, 0], X[:, 1]
        elif X.ndim == 3:
            x_coord, y_coord, z_coord = X[:, 0], X[:, 1], X[:, 2]

    if plot_type == '1d':
        plt.figure(num=1, figsize=(10, 8), dpi=120)
        plt.title(title)
        plt.plot(x_coord, analy_vals.squeeze(), label="analy_vals", color="red", linestyle="--", linewidth=5)
        plt.plot(x_coord, pred_vals.squeeze(), label="pred_vals", color="blue", linewidth=3)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{save_name}_pred.png"), bbox_inches='tight', dpi=150)
        plt.close()

        plt.figure(num=2, figsize=(10, 8), dpi=120)
        plt.title(title)
        plt.plot(x_coord, error.squeeze(), label="error", color="red")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{save_name}_error.png"), bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"[plot] 1D figures saved to '{save_dir}' with prefix '{save_name}'.")

    elif plot_type == '2d':
        fine_grid_res = 1000
        gx = np.linspace(-1, 1, fine_grid_res)
        gy = np.linspace(-1, 1, fine_grid_res)
        GX, GY = np.meshgrid(gx, gy)

        vertices = np.array([
            [0, 1], [-1 / 3, 1 / 2], [-1, 1 / 2], [-2 / 3, 0], [-1, -1 / 2], [-1 / 3, -1 / 2],
            [0, -1], [1 / 3, -1 / 2], [1, -1 / 2], [2 / 3, 0], [1, 1 / 2], [1 / 3, 1 / 2], [0, 1]
        ])
        star_path = mpath.Path(vertices)

        def get_range(data_range, data_list):
            if data_range is not None:
                if len(data_range) != 2 or data_range[0] >= data_range[1]:
                    raise ValueError("not (min, max)，且 min < max")
                return float(data_range[0]), float(data_range[1])

            valid_data = [np.asarray(data).squeeze() for data in data_list if data is not None]
            finite_data = np.concatenate([data[np.isfinite(data)] for data in valid_data])
            data_min, data_max = float(finite_data.min()), float(finite_data.max())
            if data_min == data_max:
                data_min -= 0.5
                data_max += 0.5
            return data_min, data_max

        value_range_resolved = get_range(value_range, [pred_vals, analy_vals]) if (pred_vals is not None or analy_vals is not None) else None
        error_range_resolved = get_range(error_range, [error]) if error is not None else None

        # ==========================================
        # ==========================================
        if pred_vals is not None:
            plt.figure(num=1, figsize=(6, 4.8), dpi=120)
            plt.xlabel('x', fontsize=18)
            plt.ylabel('y', fontsize=18)
            
            pred_vals_sq = pred_vals.squeeze()
            grid_data = griddata((x_coord, y_coord), pred_vals_sq, (GX, GY), method='linear')
            
            patch = mpatches.PathPatch(star_path, facecolor='none', edgecolor='none')
            ax = plt.gca()
            ax.add_patch(patch)
            
            contour_min, contour_max = value_range_resolved
            levels = np.linspace(contour_min, contour_max, contour_levels)
            contour = plt.contourf(GX, GY, grid_data, levels=levels, cmap='turbo',
                                   vmin=contour_min, vmax=contour_max, extend='both')
            
            try:
                contour.set_clip_path(patch)
            except AttributeError:
                for collection in contour.collections:
                    collection.set_clip_path(patch)
            
            plt.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=0.5)
            plt.colorbar(contour)
            ax.axis('equal')
            
            plt.savefig(os.path.join(save_dir, f"{save_name}_pred.png"), bbox_inches='tight', dpi=150)
            plt.close()

        # ==========================================
        # ==========================================
        if analy_vals is not None:
            plt.figure(num=2, figsize=(6, 4.8), dpi=120)
            plt.xlabel('x', fontsize=18)
            plt.ylabel('y', fontsize=18)
            
            analy_vals_sq = analy_vals.squeeze()
            grid_data = griddata((x_coord, y_coord), analy_vals_sq, (GX, GY), method='linear')
            
            patch = mpatches.PathPatch(star_path, facecolor='none', edgecolor='none')
            ax = plt.gca()
            ax.add_patch(patch)
            
            contour_min, contour_max = value_range_resolved
            levels = np.linspace(contour_min, contour_max, contour_levels)
            contour = plt.contourf(GX, GY, grid_data, levels=levels, cmap='turbo',
                                   vmin=contour_min, vmax=contour_max, extend='both')
            
            try:
                contour.set_clip_path(patch)
            except AttributeError:
                for collection in contour.collections:
                    collection.set_clip_path(patch)
            
            plt.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=0.5)
            plt.colorbar(contour)
            ax.axis('equal')
            
            plt.savefig(os.path.join(save_dir, f"{save_name}_exact.png"), bbox_inches='tight', dpi=150)
            plt.close()

        # ==========================================
        # ==========================================
        if error is not None:
            plt.figure(num=3, figsize=(6, 4.8), dpi=120)
            plt.xlabel('x', fontsize=18)
            plt.ylabel('y', fontsize=18)
            
            error_sq = error.squeeze()
            grid_data = griddata((x_coord, y_coord), error_sq, (GX, GY), method='linear')
            
            patch = mpatches.PathPatch(star_path, facecolor='none', edgecolor='none')
            ax = plt.gca()
            ax.add_patch(patch)
            
            contour_min, contour_max = error_range_resolved
            levels = np.linspace(contour_min, contour_max, contour_levels)
            contour = plt.contourf(GX, GY, grid_data, levels=levels, cmap='turbo',
                                   vmin=contour_min, vmax=contour_max, extend='both')
            
            try:
                contour.set_clip_path(patch)
            except AttributeError:
                for collection in contour.collections:
                    collection.set_clip_path(patch)
            
            plt.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=0.5)
            plt.colorbar(contour)
            ax.axis('equal')
            
            plt.savefig(os.path.join(save_dir, f"{save_name}_error.png"), bbox_inches='tight', dpi=150)
            plt.close()
        
        print(f"[plot] 2D figures saved to '{save_dir}' with prefix '{save_name}'.")



def hexagon_sampler(x_inter=921, x_bc=200):
    x = np.linspace(-1, 1, num=100)
    y = np.linspace(-1, 1, num=100)
    X, Y = np.meshgrid(x, y, indexing='ij')
    XY = np.c_[X.ravel(), Y.ravel()]

    y1 = lambda x: -3/2 * x - 1
    y2 = lambda x: 3/2 * x - 1
    y3 = lambda x: -3/2 * x + 1
    y4 = lambda x: 3/2 * x + 1
    mask1 = (XY[:, 0] >= 2 / 3) & (XY[:, 0] <= 1) & (XY[:, 1] <= y2(XY[:, 0])) & (XY[:, 1] >= y3(XY[:, 0]))
    XY1 = XY[~mask1]
    mask2 = (XY1[:, 0] >= 0) & (XY1[:, 0] <= 1 / 3) & ((XY1[:, 1] >= y3(XY1[:, 0])) | (XY1[:, 1] <= y2(XY1[:, 0])))
    XY2 = XY1[~mask2]
    mask3 = (XY2[:, 0] >= 1 / 3) & (XY2[:, 0] <= 1) & ((XY2[:, 1] >= 1 / 2) | (XY2[:, 1] <= -1 / 2))
    XY3 = XY2[~mask3]
    mask4 = (XY3[:, 0] >= -1) & (XY3[:, 0] <= -2 / 3) & (XY3[:, 1] >= y4(XY3[:, 0])) & (XY3[:, 1] <= y1(XY3[:, 0]))
    XY4 = XY3[~mask4]
    mask5 = (XY4[:, 0] >= -1 / 3) & (XY4[:, 0] <= 0) & ((XY4[:, 1] >= y4(XY4[:, 0])) | (XY4[:, 1] <= y1(XY4[:, 0])))
    XY5 = XY4[~mask5]
    mask6 = (XY5[:, 0] >= -1) & (XY5[:, 0] <= -1 / 3) & ((XY5[:, 1] <= -1 / 2) | (XY5[:, 1] >= 1 / 2))
    XY6 = XY5[~mask6]
    XY_interfior = XY6[np.random.choice(len(XY6), size=x_inter, replace=False)]

    n_points_per_edge = 100
    bcx1 = np.linspace(2 / 3, 1, num=n_points_per_edge)
    bcx11 = np.linspace(0, 1 / 3, num=n_points_per_edge)
    bcy1 = y2(bcx1)  # y = 3/2*x - 1
    bcy2 = y3(bcx1)  # y = -3/2*x + 1
    bcy11 = y2(bcx11)
    bcy22 = y3(bcx11)
    bc_xy1 = np.vstack([bcx1, bcy1]).T
    bc_xy2 = np.vstack([bcx1, bcy2]).T
    bc_xy11 = np.vstack([bcx11, bcy11]).T
    bc_xy22 = np.vstack([bcx11, bcy22]).T

    bcx2 = np.linspace(1 / 3, 1, num=n_points_per_edge)
    bcy3 = np.full_like(bcx2, 1 / 2)  # y = 1/2
    bcy4 = np.full_like(bcx2, -1 / 2)  # y = -1/2
    bc_xy3 = np.vstack([bcx2, bcy3]).T
    bc_xy4 = np.vstack([bcx2, bcy4]).T

    bcx3 = np.linspace(-1, -2 / 3, num=n_points_per_edge)
    bcx33 = np.linspace(-1 / 3, 0, num=n_points_per_edge)
    bcy5 = y4(bcx3)  # y = 3/2*x + 1
    bcy6 = y1(bcx3)  # y = -3/2*x - 1
    bcy55 = y4(bcx33)
    bcy66 = y1(bcx33)
    bc_xy5 = np.vstack([bcx3, bcy5]).T
    bc_xy6 = np.vstack([bcx3, bcy6]).T
    bc_xy55 = np.vstack([bcx33, bcy55]).T
    bc_xy66 = np.vstack([bcx33, bcy66]).T

    bcx4 = np.linspace(-1, -1 / 3, num=n_points_per_edge)
    bcy7 = np.full_like(bcx4, 1 / 2)  # y = 1/2
    bcy8 = np.full_like(bcx4, -1 / 2)  # y = -1/2
    bc_xy7 = np.vstack([bcx4, bcy7]).T
    bc_xy8 = np.vstack([bcx4, bcy8]).T

    b_XY = np.vstack([bc_xy1, bc_xy2, bc_xy3, bc_xy4, bc_xy5, bc_xy6, bc_xy7, bc_xy8, bc_xy11, bc_xy22, bc_xy55, bc_xy66])
    XY_boundary = b_XY[np.random.choice(len(b_XY), size=x_bc, replace=False)]
    return XY_interfior, XY_boundary,


def solve_1d_advection(x_interfior=100, bc_weight=1., l2_reg=0., hidden_nodes=None):
    input_dim = 1
    if hidden_nodes is None:
        hidden_nodes = (x_interfior + 2) * 1

    xf_ori = np.linspace(0, 1, num=x_interfior + 2)
    xf = xf_ori[1:-1][:, np.newaxis]
    xbc = np.array([[0], [1.]])
    T_bc = np.array([[1.], [1.]])
    rhs = 2 * np.pi * np.cos(2 * np.pi * xf) * np.cos(4 * np.pi * xf) - 4 * np.pi * np.sin(2 * np.pi * xf) * np.sin(4 * np.pi * xf)

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    tbc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)

    pielm = piELM(input_dim, hidden_nodes, out_dim=1, device=device, dtype=dtype)
    _, gradHf, _, H_bc = pielm.generate_H_tanh(xf_t, xbc_t)

    sqrt_w = np.sqrt(bc_weight)
    A = torch.vstack([gradHf, H_bc * sqrt_w])  # shape=[102, 102]
    b = torch.vstack([rhs_t, tbc_t * sqrt_w])  # shape=[102, 1]

    # reg = torch.eye(hidden_nodes, hidden_nodes, device=device) * l2_reg
    beta = torch.linalg.pinv(A) @ b
    pielm.beta.data = beta
    pde_loss = mean_squared_error((gradHf @ beta).detach().cpu().numpy(), rhs)
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    xf_ori_t = torch.from_numpy(xf_ori[:, np.newaxis]).to(device=device, dtype=dtype)
    train_pred = pielm(xf_ori_t).detach().cpu().numpy()

    analy_xf = np.sin(2 * np.pi * xf_ori[:, np.newaxis]) * np.cos(4 * np.pi * xf_ori[:, np.newaxis]) + 1
    relative_error = np.linalg.norm(train_pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, train_pred)
    mae = mean_absolute_error(analy_xf, train_pred)
    error = train_pred - analy_xf

    plot(xf_ori, train_pred, analy_xf, error, save_name='1d_advection')
    return float(relative_error)


def solve_1d_diffusion(x_interfior=100, bc_weight=1., l2_reg=0., hidden_nodes=None):
    input_dim = 1
    if hidden_nodes is None:
        hidden_nodes = x_interfior + 2

    xbc = np.array([[0], [1.]])
    T_bc = np.array([[1.], [2.]])
    xf_ori = np.linspace(0, 1, num=x_interfior + 2)
    xf = xf_ori[1:-1][:, np.newaxis]
    rhs = -((1/2 * np.pi) ** 2 + (2 * np.pi) ** 2) * np.sin((1/2) * np.pi * xf) * np.cos(2 * np.pi * xf) - 4 * np.pi * (1/2) * np.pi * np.cos((1/2) * np.pi * xf) * np.sin(2 * np.pi * xf)

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    tbc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)

    pielm = piELM(input_dim, hidden_nodes, out_dim=1, weight_scale=10., bias_scale=5., device=device, dtype=dtype)
    _, _, LapHf, H_bc = pielm.generate_H_tanh(xf_t, xbc_t)

    sqrt_w = np.sqrt(bc_weight)
    A = torch.vstack([LapHf, H_bc * sqrt_w])  # shape=[102, 102]
    b = torch.vstack([rhs_t, tbc_t * sqrt_w])  # shape=[102, 1]

    reg = torch.eye(hidden_nodes, hidden_nodes, device=device) * l2_reg
    # beta = torch.linalg.solve(A.T @ A + reg, A.T @ b)
    beta = torch.linalg.pinv(A) @ b
    pielm.beta.data = beta
    pde_loss = mean_squared_error((LapHf @ beta).detach().cpu().numpy(), rhs)
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    xf_ori_t = torch.from_numpy(xf_ori[:, np.newaxis]).to(device=device, dtype=dtype)
    train_pred = pielm(xf_ori_t).detach().cpu().numpy()

    analy_xf = np.sin((1/2) * np.pi * xf_ori[:, np.newaxis]) * np.cos(2 * np.pi * xf_ori[:, np.newaxis]) + 1
    relative_error = np.linalg.norm(train_pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, train_pred)
    mae = mean_absolute_error(analy_xf, train_pred)
    error = train_pred - analy_xf

    plot(xf_ori, train_pred, analy_xf, error, save_name='1d_diffusion')
    return float(relative_error)


def solve_1d_advection_diffusion(x_interfior=20, bc_weight=1., l2_reg=0., hidden_nodes=None):
    in_dim = 1
    if hidden_nodes is None:
        hidden_nodes = x_interfior + 2
    v = 0.2

    xbc = np.array([[0.], [1.]])
    T_bc = np.array([[0.], [1.]])
    xf_ori = np.linspace(0, 1, num=x_interfior + 2)
    xf = xf_ori[1:-1][:, np.newaxis]

    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    tbc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xf_ori_t = torch.from_numpy(xf_ori[:, np.newaxis]).to(device=device, dtype=dtype)

    pielm = piELM(in_dim, hidden_nodes, out_dim=1, weight_scale=8., bias_scale=4., device=device, dtype=dtype)
    _, gradHf, LapHf, H_bc = pielm.generate_H_tanh(xf_t, xbc_t)

    sqrt_w = np.sqrt(bc_weight)
    A = torch.vstack([gradHf - v * LapHf, H_bc * sqrt_w])  # [22, 22]
    b = torch.vstack([torch.zeros(20, 1, device=device, dtype=dtype), tbc_t * sqrt_w])  # [22, 1]
    beta = torch.linalg.pinv(A) @ b
    pielm.beta.data = beta
    pde_loss = torch.mean(((gradHf - v * LapHf) @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    analy_xf = (np.exp(xf_ori[:, np.newaxis] / v) - 1) / (np.exp(1 / v) - 1)
    train_pred = pielm(xf_ori_t).detach().cpu().numpy()

    relative_error = np.linalg.norm(train_pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, train_pred)
    mae = mean_absolute_error(analy_xf, train_pred)
    error = train_pred - analy_xf


    plot(xf_ori, train_pred, analy_xf, error, save_name='1d_advection_diffusion')
    return float(relative_error)


def solve_1d_unsteady_advection(x_inferior=100, t_inferior=50, bc_weight=1., ic_weight=1., l2_reg=1e-3, hidden_nodes=450):
    in_dim = 2
    f_a_xt = lambda x: 1 + x[:, 0]

    x = np.linspace(-1, 1, x_inferior)
    t = np.linspace(0, 0.5, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = np.sin(np.pi * ((1 + xt_ori[:, 0]) * np.exp(-xt_ori[:, 1]) - 1))[:, np.newaxis]
    x_in = x[1:-1]
    t_in = t[1:-1]
    xxin, ttin = np.meshgrid(x_in, t_in, indexing='ij')
    xt_in = np.c_[xxin.ravel(), ttin.ravel()]
    xtf = xt_in[np.random.choice(len(xt_in), size=420, replace=False)]
    x_bc_right = np.column_stack([np.ones(20), np.linspace(0, 0.5, 20)])[np.random.choice(20, 11, False)]
    x_bc_left = np.column_stack([-np.ones(20), np.linspace(0, 0.5, 20)])[np.random.choice(20, 11, False)]
    x_bc = np.vstack([x_bc_right, x_bc_left])
    T_bc_right = np.sin(np.pi * (2 * np.exp(-x_bc_right[:, 1]) - 1))
    T_bc_left = np.zeros(x_bc_left.shape[0])
    T_bc = np.concatenate([T_bc_right, T_bc_left], axis=0)[:, np.newaxis]
    x_ic = np.column_stack([np.linspace(-1, 1, 50), np.zeros(50)])[np.random.choice(50, 40, False)]
    T_ic = np.sin(np.pi * x_ic[:, 0])[:, np.newaxis]

    scaler = StandardScaler()
    all_input = np.vstack([xtf, x_bc, x_ic, xt_ori])
    scaler.fit(all_input)
    xtf_scaled = scaler.transform(xtf)
    x_bc_scaled = scaler.transform(x_bc)
    x_ic_scaled = scaler.transform(x_ic)
    xt_ori_scaled = scaler.transform(xt_ori)

    xtf_t = torch.from_numpy(xtf).to(device=device, dtype=dtype)
    x_bc_t = torch.from_numpy(x_bc).to(device=device, dtype=dtype)
    x_ic_t = torch.from_numpy(x_ic).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    pielm = piELM(in_dim, hidden_nodes, out_dim=1, weight_scale=3., bias_scale=0.8, device=device, dtype=dtype)
    param = sum(p.numel() for p in pielm.parameters())

    start_time = time.time()
    _, Hx, Ht, _, _, _, _, H_bc, H_ic = pielm.generate_H_tanh(xtf_t, x_bc_t, x_ic_t)
    af = f_a_xt(xtf_t)[:, None]
    rhs = torch.zeros(xtf_t.shape[0], 1, device=device, dtype=dtype)
    A = torch.vstack([Ht + af * Hx, math.sqrt(bc_weight) * H_bc, math.sqrt(ic_weight) * H_ic])
    b = torch.vstack([rhs, math.sqrt(bc_weight) * T_bc_t, math.sqrt(ic_weight) * T_ic_t])
    # beta = torch.linalg.pinv(A) @ b
    beta = torch.linalg.solve(A.T @ A, A.T @ b)
    # AtA = A.T @ A
    # Atb = A.T @ b
    # I_reg = torch.eye(AtA.shape[0], device=device, dtype=dtype) * l2_reg
    # beta = torch.linalg.solve(AtA + I_reg, Atb)
    # U, s, V = torch.linalg.svd(A, full_matrices=False)
    # s_reg = torch.diag(s / (s ** 2 + l2_reg))
    # beta = V.T @ s_reg @ U.T @ b
    pielm.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = torch.mean(((Ht + af * Hx) @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    pred = pielm(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, pred)
    mae = mean_absolute_error(analy_xtf, pred)
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}, ic_loss:{ic_loss:.6e}, relative_error:{relative_error:.6e}, MAE:{mae:.6e}, mse:{mse:.6e}\n"
          f"param:{param}, train_time:{elapsed_time:.4f}")

    error = np.abs(pred - analy_xtf)

    # ==========================================
    # ==========================================
    
    fig1, ax1 = plt.subplots(num=1, figsize=(6, 4.8), dpi=120)
    ax1.set_xlabel("t", fontsize=18)
    ax1.set_ylabel("x", fontsize=18)
    contour1 = ax1.tricontourf(xt_ori[:, 1], xt_ori[:, 0], pred.squeeze(-1), cmap="turbo", levels=100, alpha=1)
    fig1.colorbar(contour1, ax=ax1)
    fig1.tight_layout()

    import matplotlib.colors as mcolors
    fig_log = plt.figure(num=2, figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('t', fontsize=18)
    ax_log.set_ylabel('x', fontsize=18)
    
    error_plot = error.squeeze()
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(error_plot, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    cs = ax_log.tricontourf(xt_ori[:, 1], xt_ori[:, 0], error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
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
    
    plt.tight_layout()
    plt.show()
    return float(relative_error)



def solve_2d_advection(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=0., hidden_nodes=None):
    in_dim = 2
    if hidden_nodes is None:
        hidden_nodes = x_interfior + x_boundary # 2000
    coef_a, coef_b = 1., 1/2

    xf, xbc = hexagon_sampler(x_interfior, x_boundary)
    xf_ori = np.concatenate([xf, xbc], axis=0)
    rhs = coef_a * -1/2 * np.pi * np.sin(np.pi * xf[:, 0]) * np.sin(np.pi * xf[:, 1]) + coef_b * 1/2 * np.pi * np.cos(np.pi * xf[:, 0]) * np.cos(np.pi * xf[:, 1])
    T_bc = 1/2 * np.cos(np.pi * xbc[:, 0]) * np.sin(np.pi * xbc[:, 1])
    analy_xf = 1/2 * np.cos(np.pi * xf_ori[:, 0]) * np.sin(np.pi * xf_ori[:, 1])

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)

    pielm = piELM(in_dim, hidden_nodes, out_dim=1, weight_scale=10., bias_scale=5., device=device, dtype=dtype)
    pielm_parameters = sum(p.numel() for p in pielm.parameters())
    H, Hx, Hy, gradHf, Hxx, Hyy, LapHf, H_bc, H_ic = pielm.generate_H_tanh(xf_t, xbc_t)

    start_time = time.time()
    sqrt_w = np.sqrt(bc_weight)
    A = torch.cat([coef_a * Hx + coef_b * Hy, H_bc * sqrt_w], dim=0)
    b = torch.cat([rhs_t, T_bc_t * sqrt_w], dim=0)
    # beta = torch.linalg.pinv(A) @ b
    beta = torch.linalg.solve(A.T @ A, A.T @ b)
    pielm.beta.data = beta
    elpased_time = time.time() - start_time
    pde_loss = mean_squared_error((coef_a * Hx @ beta + coef_b * Hy @ beta).detach().cpu().numpy(), rhs)
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    train_pred = pielm(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(train_pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, train_pred)
    mae = mean_absolute_error(analy_xf, train_pred)
    error = np.abs(train_pred - analy_xf)

    plot(xf_ori_t.detach().cpu().numpy(), train_pred, analy_xf, error, "2d", save_name='2d_advection')
    
    import matplotlib.colors as mcolors
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    from scipy.interpolate import griddata
    
    fine_grid_res = 1000
    gx = np.linspace(-1, 1, fine_grid_res)
    gy = np.linspace(-1, 1, fine_grid_res)
    GX, GY = np.meshgrid(gx, gy)
    
    vertices = np.array([
        [0, 1], [-1/3, 1/2], [-1, 1/2], [-2/3, 0], [-1, -1/2], [-1/3, -1/2],
        [0, -1], [1/3, -1/2], [1, -1/2], [2/3, 0], [1, 1/2], [1/3, 1/2], [0, 1]
    ])
    star_path = mpath.Path(vertices)
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('x', fontsize=18)
    ax_log.set_ylabel('y', fontsize=18)
    ax_log.set_title('piELM Advection Error (Log Scale)', fontsize=16)
    
    grid_error = griddata((xf_ori[:, 0], xf_ori[:, 1]), error.squeeze(), (GX, GY), method='linear')
    
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(grid_error, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    patch = mpatches.PathPatch(star_path, facecolor='none', edgecolor='none')
    ax_log.add_patch(patch)
    
    cs = ax_log.contourf(GX, GY, error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
    try:
        cs.set_clip_path(patch)
    except AttributeError:
        for collection in cs.collections:
            collection.set_clip_path(patch)
    
    ax_log.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=0.5)
    
    cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cbar = plt.colorbar(cs, ax=ax_log, ticks=cbar_ticks, format='%.0e')
    cbar.set_label('Absolute Error (log scale)', fontsize=14)
    cbar.ax.set_ylim(e_min, e_max)
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontsize(10)
    
    ax_log.text(0.02, 0.98, f'Max: {np.max(error):.2e}\nMean: {np.mean(error):.2e}', 
                transform=ax_log.transAxes, fontsize=11, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_log.axis('equal')
    plt.tight_layout()
    plt.show()
    return float(relative_error)


def solve_2d_diffusion(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=0., hidden_nodes=None):
    in_dim = 2
    if hidden_nodes is None:
        hidden_nodes = (x_interfior + x_boundary)
    # coef_a, coef_b = 1., 1/2

    xf, xbc = hexagon_sampler(x_interfior, x_boundary)
    xf_ori = np.concatenate([xf, xbc], axis=0)

    def RHS(X):
        Uxx = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 16 * X[:, 0] ** 2 - 4 * np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        Uyy = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 64 * X[:, 1] ** 2 - 8 * np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        return Uxx + Uyy
    rhs = RHS(xf)
    T_bc = 1/2 + np.exp(-(2 * xbc[:, 0] ** 2 + 4 * xbc[:, 1] ** 2))
    analy_xf = 1/2 + np.exp(-(2 * xf_ori[:, 0] ** 2 + 4 * xf_ori[:, 1] ** 2))

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)

    pielm = piELM(in_dim, hidden_nodes, out_dim=1, weight_scale=10., bias_scale=5., device=device, dtype=dtype)
    pielm_parameters = sum(p.numel() for p in pielm.parameters())
    H, Hx, Hy, gradHf, Hxx, Hyy, LapHf, H_bc, H_ic = pielm.generate_H_tanh(xf_t, xbc_t)

    start_time = time.time()
    sqrt_w = np.sqrt(bc_weight)
    A = torch.cat([LapHf, H_bc * sqrt_w], dim=0)
    b = torch.cat([rhs_t, T_bc_t * sqrt_w], dim=0)
    # beta = torch.linalg.pinv(A) @ b
    beta = torch.linalg.solve(A.T @ A, A.T @ b)
    pielm.beta.data = beta
    elpased_time = time.time() - start_time
    pde_loss = mean_squared_error((LapHf @ beta).detach().cpu().numpy(), rhs)
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    train_pred = pielm(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(train_pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, train_pred)
    mae = mean_absolute_error(analy_xf, train_pred)
    error = np.abs(train_pred - analy_xf)

    plot(xf_ori_t.detach().cpu().numpy(), train_pred, analy_xf, error, "2d", save_name='2d_diffusion')
    
    import matplotlib.colors as mcolors
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    from scipy.interpolate import griddata
    
    fine_grid_res = 1000
    gx = np.linspace(-1, 1, fine_grid_res)
    gy = np.linspace(-1, 1, fine_grid_res)
    GX, GY = np.meshgrid(gx, gy)
    
    vertices = np.array([
        [0, 1], [-1/3, 1/2], [-1, 1/2], [-2/3, 0], [-1, -1/2], [-1/3, -1/2],
        [0, -1], [1/3, -1/2], [1, -1/2], [2/3, 0], [1, 1/2], [1/3, 1/2], [0, 1]
    ])
    star_path = mpath.Path(vertices)
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('x', fontsize=18)
    ax_log.set_ylabel('y', fontsize=18)
    ax_log.set_title('piELM Diffusion Error (Log Scale)', fontsize=16)
    
    grid_error = griddata((xf_ori[:, 0], xf_ori[:, 1]), error.squeeze(), (GX, GY), method='linear')
    
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(grid_error, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    patch = mpatches.PathPatch(star_path, facecolor='none', edgecolor='none')
    ax_log.add_patch(patch)
    
    cs = ax_log.contourf(GX, GY, error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
    try:
        cs.set_clip_path(patch)
    except AttributeError:
        for collection in cs.collections:
            collection.set_clip_path(patch)
    
    ax_log.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=0.5)
    
    cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cbar = plt.colorbar(cs, ax=ax_log, ticks=cbar_ticks, format='%.0e')
    cbar.set_label('Absolute Error (log scale)', fontsize=14)
    cbar.ax.set_ylim(e_min, e_max)
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontsize(10)
    
    ax_log.text(0.02, 0.98, f'Max: {np.max(error):.2e}\nMean: {np.mean(error):.2e}', 
                transform=ax_log.transAxes, fontsize=11, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_log.axis('equal')
    plt.tight_layout()
    plt.show()
    return float(relative_error)


def solve_2d_diffusion_pikan(x_interfior=921, x_boundary=240, bc_weight=100., l2_reg=0., hidden_nodes=None):
    width = [2, 10, 1]
    if hidden_nodes is None:
        hidden_nodes = (x_interfior + x_boundary)

    xf, xbc = hexagon_sampler(x_interfior, x_boundary)
    xf_ori = np.concatenate([xf, xbc], axis=0)

    def RHS(X):
        Uxx = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 16 * X[:, 0] ** 2 - 4 * np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        Uyy = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 64 * X[:, 1] ** 2 - 8 * np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        return Uxx + Uyy
    rhs = RHS(xf)[:, np.newaxis]
    T_bc = (1/2 + np.exp(-(2 * xbc[:, 0] ** 2 + 4 * xbc[:, 1] ** 2)))[:, np.newaxis]
    analy_xf = 1/2 + np.exp(-(2 * xf_ori[:, 0] ** 2 + 4 * xf_ori[:, 1] ** 2))[:, np.newaxis]

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype).requires_grad_(True)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    analy_xf_t = torch.from_numpy(analy_xf).to(device=device, dtype=dtype)

    model = efficentKAN(width,
                        grid_size=100,
                        spline_order=3,
                        scale_noise=0.1,
                        scale_base=1.0,
                        scale_spline=1.0,
                        base_activation=torch.nn.SiLU,
                        grid_eps=0.02,
                        grid_range=[-1, 1]).to(device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-6)
    iteration = 5000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=iteration // 10, gamma=0.8)

    criterion = nn.MSELoss()
    history = {
        "pde_loss": [], "bc_loss": [], "total_loss": []
    }
    start_time = time.time()
    for i in range(iteration):
        y_xf = model(xf_t)
        df = torch.autograd.grad(y_xf.sum(), xf_t, create_graph=True)[0]
        dx, dy = df[:, 0], df[:, 1]
        dxx = torch.autograd.grad(dx.sum(), xf_t, create_graph=True)[0][:, 0]
        dyy = torch.autograd.grad(dy.sum(), xf_t, create_graph=True)[0][:, 1]
        lapd = (dxx + dyy)[:, None]

        pde_loss = criterion(lapd, rhs_t)

        y_bc = model(xbc_t)
        bc_loss = criterion(y_bc, T_bc_t)

        l1_reg = model.regularization_loss(1., 1.)
        std_lap = torch.std(lapd, dim=0)
        std_bc = torch.std(y_bc, dim=0)
        w_bc = std_lap / std_bc

        total_loss = pde_loss + bc_weight * bc_loss  + 1e-8 * l1_reg #

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
        pred = model(xf_ori_t)
        relative_error = torch.linalg.norm(pred - analy_xf_t) / torch.linalg.norm(analy_xf_t)

        lr_scheduler.step()

        history["pde_loss"].append(pde_loss)
        history["bc_loss"].append(bc_loss)
        history["total_loss"].append(total_loss)

        print(f"Epoch: {i:04d}, pde_loss: {pde_loss.item():.6e}, bc_loss: {bc_loss.item():.6e}, total_loss: {total_loss.item():.6e}, relative_error:{relative_error:.6e}")
    elapsed_time = time.time() - start_time

    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    pred = model(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, pred)
    mae = mean_absolute_error(analy_xf, pred)
    error = np.abs(pred - analy_xf)

    plot(xf_ori_t.detach().cpu().numpy(), pred, analy_xf, error, "2d", save_name='2d_diffusion_pikan')
    return float(relative_error)


def solve_2d_diffusion_pinn(x_interfior=921, x_boundary=240, bc_weight=10., l2_reg=0.):
    xf, xbc = hexagon_sampler(x_interfior, x_boundary)
    xf_ori = np.concatenate([xf, xbc], axis=0)

    def RHS(X):
        Uxx = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 16 * X[:, 0] ** 2 - 4 * np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        Uyy = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 64 * X[:, 1] ** 2 - 8 * np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        return Uxx + Uyy
    rhs = RHS(xf)[:, np.newaxis]
    T_bc = (1/2 + np.exp(-(2 * xbc[:, 0] ** 2 + 4 * xbc[:, 1] ** 2)))[:, np.newaxis]
    analy_xf = 1/2 + np.exp(-(2 * xf_ori[:, 0] ** 2 + 4 * xf_ori[:, 1] ** 2))[:, np.newaxis]

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype).requires_grad_(True)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    analy_xf_t = torch.from_numpy(analy_xf).to(device=device, dtype=dtype)

    model = piNN(device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())

    iteration = 5000
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-6)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=iteration // 5, gamma=0.1)

    criterion = nn.MSELoss()
    history = {
        "pde_loss": [], "bc_loss": [], "total_loss": []
    }
    start_time = time.time()
    for i in range(iteration):
        y_xf = model(xf_t)
        df = torch.autograd.grad(y_xf.sum(), xf_t, create_graph=True)[0]
        dx, dy = df[:, 0], df[:, 1]
        dxx = torch.autograd.grad(dx.sum(), xf_t, create_graph=True)[0][:, 0]
        dyy = torch.autograd.grad(dy.sum(), xf_t, create_graph=True)[0][:, 1]
        lapd = (dxx + dyy)[:, None]

        pde_loss = criterion(lapd, rhs_t)

        y_bc = model(xbc_t)
        bc_loss = criterion(y_bc, T_bc_t)

        # l1_reg = model.regularization_loss(1., 1.)
        # std_lap = torch.std(lapd, dim=0)
        # std_bc = torch.std(y_bc, dim=0)
        # w_bc = std_lap / std_bc

        total_loss = pde_loss + bc_weight * bc_loss  # + 1e-8 * l1_reg #

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
        pred = model(xf_ori_t)
        relative_error = torch.linalg.norm(pred - analy_xf_t) / torch.linalg.norm(analy_xf_t)

        lr_scheduler.step()

        history["pde_loss"].append(pde_loss)
        history["bc_loss"].append(bc_loss)
        history["total_loss"].append(total_loss)

        print(f"Epoch: {i:04d}, pde_loss: {pde_loss.item():.6e}, bc_loss: {bc_loss.item():.6e}, total_loss: {total_loss.item():.6e}, relative_error:{relative_error:.6e}")
    elapsed_time = time.time() - start_time

    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    pred = model(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, pred)
    mae = mean_absolute_error(analy_xf, pred)
    error = np.abs(pred - analy_xf)

    plot(xf_ori_t.detach().cpu().numpy(), pred, analy_xf, error, "2d", save_name='2d_diffusion_pinn2')
    return float(relative_error)


def solve_2d_advection_kan(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=5e-3, hidden_nodes=None):
    in_dim = 2
    if hidden_nodes is None:
        hidden_nodes = (x_interfior + x_boundary) // 10  # 2 * in_dim + 1
    print("KAN_hidden_nodes: ", hidden_nodes)
    coef_a, coef_b = 1., 1 / 2

    xf, xbc = hexagon_sampler(x_interfior, x_boundary)
    xf_ori = np.concatenate([xf, xbc], axis=0)
    rhs = (coef_a * -1 / 2 * np.pi * np.sin(np.pi * xf[:, 0]) * np.sin(np.pi * xf[:, 1]) + coef_b * 1 / 2 * np.pi * np.cos(np.pi * xf[:, 0]) * np.cos(np.pi * xf[:, 1]))[:, np.newaxis]
    T_bc = (1 / 2 * np.cos(np.pi * xbc[:, 0]) * np.sin(np.pi * xbc[:, 1]))[:, np.newaxis]
    analy_xf = (1 / 2 * np.cos(np.pi * xf_ori[:, 0]) * np.sin(np.pi * xf_ori[:, 1]))[:, np.newaxis]

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)

    pikan = piELMKAN(in_dim, hidden_nodes, out_dim=1, grid_range=[-1., 1], num_grid=10, coeff_scale=2., device=device, dtype=dtype)
    _, Hx, Hy, _, _, H_bc = pikan.generate_H(xf_t, xbc_t)

    sqrt_w = np.sqrt(bc_weight)
    A = torch.vstack([coef_a * Hx + coef_b * Hy, H_bc * sqrt_w])
    print(rhs_t.shape, T_bc_t.shape)
    b = torch.vstack([rhs_t, T_bc_t * sqrt_w])
    beta = torch.linalg.solve(A.T @ A, A.T @ b)
    # dim = A.shape[1]
    # reg = torch.eye(dim, dim, device=device, dtype=dtype) * l2_reg
    # beta = torch.linalg.solve(A.T @ A + reg, A.T @ b)
    # U, s, V = torch.linalg.svd(A, full_matrices=False)
    # s_reg = torch.diag(s / (s ** 2 + l2_reg))
    # beta = V.T @ s_reg @ U.T @ b
    pikan.beta.data = beta
    pde_loss = mean_squared_error((coef_a * Hx @ beta + coef_b * Hy @ beta).detach().cpu().numpy(), rhs)
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    train_pred = pikan(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(train_pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, train_pred)
    mae = mean_absolute_error(analy_xf, train_pred)
    error = np.abs(train_pred - analy_xf)

    result_error = np.concatenate([xf_ori, np.asarray(error).reshape(xf_ori.shape[0], -1)], axis=1)

    plot(xf_ori_t.detach().cpu().numpy(), train_pred, analy_xf, error, "2d", save_name='2d_advection_kan')
    
    import matplotlib.colors as mcolors
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    from scipy.interpolate import griddata
    
    fine_grid_res = 1000
    gx = np.linspace(-1, 1, fine_grid_res)
    gy = np.linspace(-1, 1, fine_grid_res)
    GX, GY = np.meshgrid(gx, gy)
    
    vertices = np.array([
        [0, 1], [-1/3, 1/2], [-1, 1/2], [-2/3, 0], [-1, -1/2], [-1/3, -1/2],
        [0, -1], [1/3, -1/2], [1, -1/2], [2/3, 0], [1, 1/2], [1/3, 1/2], [0, 1]
    ])
    star_path = mpath.Path(vertices)
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('x', fontsize=18)
    ax_log.set_ylabel('y', fontsize=18)
    ax_log.set_title('piKAN Advection Error (Log Scale)', fontsize=16)
    
    grid_error = griddata((xf_ori[:, 0], xf_ori[:, 1]), error.squeeze(), (GX, GY), method='linear')
    
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(grid_error, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    patch = mpatches.PathPatch(star_path, facecolor='none', edgecolor='none')
    ax_log.add_patch(patch)
    
    cs = ax_log.contourf(GX, GY, error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
    try:
        cs.set_clip_path(patch)
    except AttributeError:
        for collection in cs.collections:
            collection.set_clip_path(patch)
    
    ax_log.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=0.5)
    
    cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cbar = plt.colorbar(cs, ax=ax_log, ticks=cbar_ticks, format='%.0e')
    cbar.set_label('Absolute Error (log scale)', fontsize=14)
    cbar.ax.set_ylim(e_min, e_max)
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontsize(10)
    
    ax_log.text(0.02, 0.98, f'Max: {np.max(error):.2e}\nMean: {np.mean(error):.2e}', 
                transform=ax_log.transAxes, fontsize=11, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_log.axis('equal')
    plt.tight_layout()
    plt.show()
    return float(relative_error)


def solve_2d_diffusion_kan(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=0., hidden_nodes=None):
    in_dim = 2
    if hidden_nodes is None:
        hidden_nodes = (x_interfior + x_boundary) // 20
    print(f"hidden_nodes: {hidden_nodes}")
    # coef_a, coef_b = 1., 1/2

    xf, xbc = hexagon_sampler(x_interfior, x_boundary)
    xf_ori = np.concatenate([xf, xbc], axis=0)

    def RHS(X):
        Uxx = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 16 * X[:, 0] ** 2 - 4 * np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        Uyy = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 64 * X[:, 1] ** 2 - 8 * np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        return Uxx + Uyy
    rhs = RHS(xf)
    T_bc = 1/2 + np.exp(-(2 * xbc[:, 0] ** 2 + 4 * xbc[:, 1] ** 2))
    analy_xf = 1/2 + np.exp(-(2 * xf_ori[:, 0] ** 2 + 4 * xf_ori[:, 1] ** 2))

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)

    pikan = piELMKAN(in_dim, hidden_nodes, out_dim=1, grid_range=[-1.5, 1.5], num_grid=5, coeff_scale=1.0, device=device, dtype=dtype)
    _, _, _, _, LapHf, H_bc = pikan.generate_H(xf_t, xbc_t)

    sqrt_w = np.sqrt(bc_weight)
    A = torch.cat([LapHf, H_bc * sqrt_w], dim=0)
    b = torch.cat([rhs_t, T_bc_t * sqrt_w], dim=0)
    # beta = torch.linalg.pinv(A) @ b
    # U, s, V = torch.linalg.svd(A, full_matrices=False)
    # s_reg = torch.diag(s / (s ** 2 + l2_reg))
    # beta = V.T @ s_reg @ U.T @ b
    dim = A.shape[1]
    reg = torch.eye(dim, dim, device=device, dtype=dtype) * l2_reg
    beta = torch.linalg.solve(A.T @ A + reg, A.T @ b)
    pikan.beta.data = beta
    pde_loss = mean_squared_error((LapHf @ beta).detach().cpu().numpy(), rhs)
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    train_pred = pikan(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(train_pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, train_pred)
    mae = mean_absolute_error(analy_xf, train_pred)
    error = train_pred - analy_xf

    plot(xf_ori_t.detach().cpu().numpy(), train_pred, analy_xf, error, "2d", save_name='2d_diffusion_kan')
    
    import matplotlib.colors as mcolors
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    from scipy.interpolate import griddata
    
    fine_grid_res = 1000
    gx = np.linspace(-1, 1, fine_grid_res)
    gy = np.linspace(-1, 1, fine_grid_res)
    GX, GY = np.meshgrid(gx, gy)
    
    vertices = np.array([
        [0, 1], [-1/3, 1/2], [-1, 1/2], [-2/3, 0], [-1, -1/2], [-1/3, -1/2],
        [0, -1], [1/3, -1/2], [1, -1/2], [2/3, 0], [1, 1/2], [1/3, 1/2], [0, 1]
    ])
    star_path = mpath.Path(vertices)
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('x', fontsize=18)
    ax_log.set_ylabel('y', fontsize=18)
    ax_log.set_title('piKAN Diffusion Error (Log Scale)', fontsize=16)
    
    grid_error = griddata((xf_ori[:, 0], xf_ori[:, 1]), error.squeeze(), (GX, GY), method='linear')
    
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(grid_error, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    patch = mpatches.PathPatch(star_path, facecolor='none', edgecolor='none')
    ax_log.add_patch(patch)
    
    cs = ax_log.contourf(GX, GY, error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
    try:
        cs.set_clip_path(patch)
    except AttributeError:
        for collection in cs.collections:
            collection.set_clip_path(patch)
    
    ax_log.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=0.5)
    
    cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cbar = plt.colorbar(cs, ax=ax_log, ticks=cbar_ticks, format='%.0e')
    cbar.set_label('Absolute Error (log scale)', fontsize=14)
    cbar.ax.set_ylim(e_min, e_max)
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontsize(10)
    
    ax_log.text(0.02, 0.98, f'Max: {np.max(error):.2e}\nMean: {np.mean(error):.2e}', 
                transform=ax_log.transAxes, fontsize=11, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_log.axis('equal')
    plt.tight_layout()
    plt.show()
    return float(relative_error)


def solve_2d_advection_hybridkan(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=5e-3, hidden_nodes=None):
    in_dim = 2
    if hidden_nodes is None:
        hidden_nodes = (x_interfior + x_boundary) // 20  # 2 * in_dim + 1
    print("KAN_hidden_nodes: ", hidden_nodes)
    coef_a, coef_b = 1., 1 / 2

    xf, xbc = hexagon_sampler(x_interfior, x_boundary)
    xf_ori = np.concatenate([xf, xbc], axis=0)
    rhs = coef_a * -1 / 2 * np.pi * np.sin(np.pi * xf[:, 0]) * np.sin(np.pi * xf[:, 1]) + coef_b * 1 / 2 * np.pi * np.cos(np.pi * xf[:, 0]) * np.cos(np.pi * xf[:, 1])
    T_bc = 1 / 2 * np.cos(np.pi * xbc[:, 0]) * np.sin(np.pi * xbc[:, 1])
    analy_xf = 1 / 2 * np.cos(np.pi * xf_ori[:, 0]) * np.sin(np.pi * xf_ori[:, 1])

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)

    pihkan = piELMHKAN(in_dim, hidden_nodes, out_dim=1, num_grid1=10, num_grid2=10, knot_grid1=[-1.5, 1.5], knot_grid2=[-1, 1.], coeff1_sacle=1.0,
                       sigmoid_scale=2.8, device=device, dtype=dtype)
    pihkan_paramters = sum(p.numel() for p in pihkan.parameters())
    H, Hx, Hy, gradHf, Hxx, Hyy, LapHf, H_bc, H_ic = pihkan.generate_H(xf_t, xbc_t)

    start_time = time.time()
    sqrt_w = np.sqrt(bc_weight)
    A = torch.cat([coef_a * Hx + coef_b * Hy, H_bc * sqrt_w], dim=0)
    b = torch.cat([rhs_t, T_bc_t * sqrt_w], dim=0)
    # beta = torch.linalg.pinv(A) @ b
    beta = torch.linalg.solve(A.T @ A, A.T @ b)
    # dim = A.shape[1]
    # reg = torch.eye(dim, dim, device=device, dtype=dtype) * l2_reg
    # beta = torch.linalg.solve(A.T @ A + reg, A.T @ b)
    # U, s, V = torch.linalg.svd(A, full_matrices=False)
    # s_reg = torch.diag(s / (s ** 2 + l2_reg))
    # beta = V.T @ s_reg @ U.T @ b
    pihkan.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = mean_squared_error((coef_a * Hx @ beta + coef_b * Hy @ beta).detach().cpu().numpy(), rhs)
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    train_pred = pihkan(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(train_pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, train_pred)
    mae = mean_absolute_error(analy_xf, train_pred)
    error = np.abs(train_pred - analy_xf)

    result_error = np.concatenate([xf_ori, np.asarray(error).reshape(xf_ori.shape[0], -1)], axis=1)

    with open("error.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(result_error)

    plot(xf_ori_t.detach().cpu().numpy(), train_pred, analy_xf, error, "2d", save_name='2d_advection_hybridkan')
    
    import matplotlib.colors as mcolors
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    from scipy.interpolate import griddata
    
    fine_grid_res = 1000
    gx = np.linspace(-1, 1, fine_grid_res)
    gy = np.linspace(-1, 1, fine_grid_res)
    GX, GY = np.meshgrid(gx, gy)
    
    vertices = np.array([
        [0, 1], [-1/3, 1/2], [-1, 1/2], [-2/3, 0], [-1, -1/2], [-1/3, -1/2],
        [0, -1], [1/3, -1/2], [1, -1/2], [2/3, 0], [1, 1/2], [1/3, 1/2], [0, 1]
    ])
    star_path = mpath.Path(vertices)
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('x', fontsize=18)
    ax_log.set_ylabel('y', fontsize=18)
    ax_log.set_title('piHKAN Advection Error (Log Scale)', fontsize=16)
    
    grid_error = griddata((xf_ori[:, 0], xf_ori[:, 1]), error.squeeze(), (GX, GY), method='linear')
    
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(grid_error, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    patch = mpatches.PathPatch(star_path, facecolor='none', edgecolor='none')
    ax_log.add_patch(patch)
    
    cs = ax_log.contourf(GX, GY, error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
    try:
        cs.set_clip_path(patch)
    except AttributeError:
        for collection in cs.collections:
            collection.set_clip_path(patch)
    
    ax_log.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=0.5)
    
    cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cbar = plt.colorbar(cs, ax=ax_log, ticks=cbar_ticks, format='%.0e')
    cbar.set_label('Absolute Error (log scale)', fontsize=14)
    cbar.ax.set_ylim(e_min, e_max)
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontsize(10)
    
    ax_log.text(0.02, 0.98, f'Max: {np.max(error):.2e}\nMean: {np.mean(error):.2e}', 
                transform=ax_log.transAxes, fontsize=11, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_log.axis('equal')
    plt.tight_layout()
    plt.show()
    return float(relative_error)


def solve_2d_diffusion_hybridkan(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=5e-3, hidden_nodes=None):
    in_dim = 2
    if hidden_nodes is None:
        hidden_nodes = (x_interfior + x_boundary) // 20  # 2 * in_dim + 1
    print(f"hidden_nodes: {hidden_nodes}")
    # coef_a, coef_b = 1., 1/2

    xf, xbc = hexagon_sampler(x_interfior, x_boundary)
    xf_ori = np.concatenate([xf, xbc], axis=0)

    def RHS(X):
        Uxx = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 16 * X[:, 0] ** 2 - 4 * np.exp(
            -(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        Uyy = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 64 * X[:, 1] ** 2 - 8 * np.exp(
            -(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        return Uxx + Uyy

    rhs = RHS(xf)
    T_bc = 1 / 2 + np.exp(-(2 * xbc[:, 0] ** 2 + 4 * xbc[:, 1] ** 2))
    analy_xf = 1 / 2 + np.exp(-(2 * xf_ori[:, 0] ** 2 + 4 * xf_ori[:, 1] ** 2))

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)

    pihkan = piELMHKAN(in_dim, hidden_nodes, out_dim=1, num_grid1=10, num_grid2=10, knot_grid1=[-1.5, 1.5], knot_grid2=[-1.0, 1.0], coeff1_sacle=1.5,
                       sigmoid_scale=2.8, device=device, dtype=dtype)
    pikan_parameters = sum(p.numel() for p in pihkan.parameters())
    H, Hx, Hy, gradHf, Hxx, Hyy, LapHf, H_bc, H_ic = pihkan.generate_H(xf_t, xbc_t)

    start_time = time.time()
    sqrt_w = np.sqrt(bc_weight)
    A = torch.cat([LapHf, H_bc * sqrt_w], dim=0)
    b = torch.cat([rhs_t, T_bc_t * sqrt_w], dim=0)
    # beta = torch.linalg.pinv(A) @ b
    beta = torch.linalg.solve(A.T @ A, A.T @ b)
    pihkan.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = mean_squared_error((LapHf @ beta).detach().cpu().numpy(), rhs)
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    train_pred = pihkan(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(train_pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, train_pred)
    mae = mean_absolute_error(analy_xf, train_pred)
    error = np.abs(train_pred - analy_xf)

    plot(xf_ori_t.detach().cpu().numpy(), train_pred, analy_xf, error, "2d")
    
    import matplotlib.colors as mcolors
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    from scipy.interpolate import griddata
    
    fine_grid_res = 1000
    gx = np.linspace(-1, 1, fine_grid_res)
    gy = np.linspace(-1, 1, fine_grid_res)
    GX, GY = np.meshgrid(gx, gy)
    
    vertices = np.array([
        [0, 1], [-1/3, 1/2], [-1, 1/2], [-2/3, 0], [-1, -1/2], [-1/3, -1/2],
        [0, -1], [1/3, -1/2], [1, -1/2], [2/3, 0], [1, 1/2], [1/3, 1/2], [0, 1]
    ])
    star_path = mpath.Path(vertices)
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('x', fontsize=18)
    ax_log.set_ylabel('y', fontsize=18)
    ax_log.set_title('piHKAN Diffusion Error (Log Scale)', fontsize=16)
    
    grid_error = griddata((xf_ori[:, 0], xf_ori[:, 1]), error.squeeze(), (GX, GY), method='linear')
    
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(grid_error, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    patch = mpatches.PathPatch(star_path, facecolor='none', edgecolor='none')
    ax_log.add_patch(patch)
    
    cs = ax_log.contourf(GX, GY, error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
    try:
        cs.set_clip_path(patch)
    except AttributeError:
        for collection in cs.collections:
            collection.set_clip_path(patch)
    
    ax_log.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=0.5)
    
    cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cbar = plt.colorbar(cs, ax=ax_log, ticks=cbar_ticks, format='%.0e')
    cbar.set_label('Absolute Error (log scale)', fontsize=14)
    cbar.ax.set_ylim(e_min, e_max)
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontsize(10)
    
    ax_log.text(0.02, 0.98, f'Max: {np.max(error):.2e}\nMean: {np.mean(error):.2e}', 
                transform=ax_log.transAxes, fontsize=11, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_log.axis('equal')
    plt.tight_layout()
    plt.show()
    return float(relative_error)


def solve_2d_diffusion_pickanelm(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=5e-3, hidden_nodes=None):
    in_dim = 2
    if hidden_nodes is None:
        hidden_nodes = (x_interfior + x_boundary) // 20  # 2 * in_dim + 1
    print(f"hidden_nodes: {hidden_nodes}")
    # coef_a, coef_b = 1., 1/2

    xf, xbc = hexagon_sampler(x_interfior, x_boundary)
    xf_ori = np.concatenate([xf, xbc], axis=0)

    def RHS(X):
        Uxx = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 16 * X[:, 0] ** 2 - 4 * np.exp(
            -(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        Uyy = np.exp(-(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2)) * 64 * X[:, 1] ** 2 - 8 * np.exp(
            -(2 * X[:, 0] ** 2 + 4 * X[:, 1] ** 2))
        return Uxx + Uyy

    rhs = RHS(xf)
    T_bc = 1 / 2 + np.exp(-(2 * xbc[:, 0] ** 2 + 4 * xbc[:, 1] ** 2))
    analy_xf = 1 / 2 + np.exp(-(2 * xf_ori[:, 0] ** 2 + 4 * xf_ori[:, 1] ** 2))

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)

    pihkan = piCKANELM(in_dim, hidden_nodes, out_dim=1, degree1=10, degree2=10, coeff_scaled=2.,
                        device=device, dtype=dtype)
    pikan_parameters = sum(p.numel() for p in pihkan.parameters())
    H, Hx, Hy, gradHf, Hxx, Hyy, LapHf, _, H_bc, H_ic = pihkan.generate_H(xf_t, xbc_t, xbc_t)
    # H, Hx, Hy, gradH, Hxx, Hyy, LapH, gradH_ic, H_bc, H_ic
    start_time = time.time()
    sqrt_w = np.sqrt(bc_weight)
    A = torch.cat([LapHf, H_bc * sqrt_w], dim=0)
    b = torch.cat([rhs_t, T_bc_t * sqrt_w], dim=0)
    beta = torch.linalg.pinv(A) @ b
    pihkan.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = mean_squared_error((LapHf @ beta).detach().cpu().numpy(), rhs)
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)

    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    train_pred = pihkan(xf_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(train_pred - analy_xf) / np.linalg.norm(analy_xf)
    mse = mean_squared_error(analy_xf, train_pred)
    mae = mean_absolute_error(analy_xf, train_pred)
    error = np.abs(train_pred - analy_xf)

    plot(xf_ori_t.detach().cpu().numpy(), train_pred, analy_xf, error, "2d")
    return float(relative_error)



def solve_1d_unsteady_advection_hybridkan(x_inferior=100, t_inferior=50, bc_weight=1., ic_weight=1., l2_reg=1e-5, hidden_nodes=25):
    in_dim = 2
    G1 = 10
    G2 = 10
    f_a_xt = lambda x: 1 + x[:, 0]

    x = np.linspace(-1, 1, x_inferior)
    t = np.linspace(0, 0.5, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = np.sin(np.pi * ((1 + xt_ori[:, 0]) * np.exp(-xt_ori[:, 1]) - 1))[:, np.newaxis]
    x_in = x[1:-1]
    t_in = t[1:-1]
    xxin, ttin = np.meshgrid(x_in, t_in, indexing='ij')
    xt_in = np.c_[xxin.ravel(), ttin.ravel()]
    xtf = xt_in[np.random.choice(len(xt_in), size=420, replace=False)]
    x_bc_right = np.column_stack([np.ones(20), np.linspace(0, 0.5, 20)])[np.random.choice(20, 11, False)]
    x_bc_left = np.column_stack([-np.ones(20), np.linspace(0, 0.5, 20)])[np.random.choice(20, 11, False)]
    x_bc = np.vstack([x_bc_right, x_bc_left])
    T_bc_right = np.sin(np.pi * (2 * np.exp(-x_bc_right[:, 1]) - 1))
    T_bc_left = np.zeros(x_bc_left.shape[0])
    T_bc = np.concatenate([T_bc_right, T_bc_left], axis=0)[:, np.newaxis]
    x_ic = np.column_stack([np.linspace(-1, 1, 50), np.zeros(50)])[np.random.choice(50, 40, False)]
    T_ic = np.sin(np.pi * x_ic[:, 0])[:, np.newaxis]

    xtf_t = torch.from_numpy(xtf).to(device=device, dtype=dtype)
    x_bc_t = torch.from_numpy(x_bc).to(device=device, dtype=dtype)
    x_ic_t = torch.from_numpy(x_ic).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    model = piELMHKAN(in_dim, hidden_nodes, out_dim=1, num_grid1=G1, num_grid2=G2, knot_grid1=[-1.5, 1.5], knot_grid2=[-1, 1.], coeff1_sacle=1.,
                      sigmoid_scale=2.8, device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    _, Hx, Ht, _, _, _, _, H_bc, H_ic = model.generate_H(xtf_t, x_bc_t, x_ic_t)
    af = f_a_xt(xtf_t)[:, None]
    rhs = torch.zeros(xtf_t.shape[0], 1, device=device, dtype=dtype)
    A = torch.vstack([Ht + af * Hx, math.sqrt(bc_weight) * H_bc, math.sqrt(ic_weight) * H_ic])
    b = torch.vstack([rhs, math.sqrt(bc_weight) * T_bc_t, math.sqrt(ic_weight) * T_ic_t])
    # beta = torch.linalg.pinv(A) @ b
    beta = torch.linalg.solve(A.T @ A + torch.eye(A.shape[1]).to(device=device, dtype=dtype) * 0., A.T @ b)
    # AtA = A.T @ A
    # Atb = A.T @ b
    # I_reg = torch.eye(AtA.shape[0], device=device, dtype=dtype) * l2_reg
    # beta = torch.linalg.solve(AtA + I_reg, Atb)
    # U, s, V = torch.linalg.svd(A, full_matrices=False)
    # s_reg = torch.diag(s / (s ** 2 + l2_reg))
    # beta = V.T @ s_reg @ U.T @ b
    model.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = torch.mean(((Ht + af * Hx) @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    pred = model(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, pred)
    mae = mean_absolute_error(analy_xtf, pred)
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}, ic_loss:{ic_loss:.6e}, relative_error:{relative_error:.6e}, MAE:{mae:.6e}, mse:{mse:.6e}\n"
          f"param:{param}, train_time:{elapsed_time:.4f}")

    error = np.abs(pred - analy_xtf)
    
    import matplotlib.colors as mcolors
    
    fig_pred = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_pred = fig_pred.add_subplot(111)
    ax_pred.set_xlabel('t', fontsize=18)
    ax_pred.set_ylabel('x', fontsize=18)
    contour_turbo = ax_pred.tricontourf(xt_ori[:, 1], xt_ori[:, 0], pred.squeeze(-1), cmap='turbo', levels=100, alpha=1)
    fig_pred.colorbar(contour_turbo, ax=ax_pred)
    plt.tight_layout()
    plt.show()
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('t', fontsize=18)
    ax_log.set_ylabel('x', fontsize=18)
    
    error_plot = error.squeeze()
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(error_plot, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    cs = ax_log.tricontourf(xt_ori[:, 1], xt_ori[:, 0], error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
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
    
    plt.tight_layout()
    plt.show()
    return float(relative_error)



def TC_9():
    np.random.seed(1024)
    torch.manual_seed(1024)
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    xy_ori = np.c_[xx.ravel(), yy.ravel()]

    bounds = [(-1, 1)] * 2
    samples = lhs(2, samples=2000, criterion='center')
    scaled_samples = np.zeros_like(samples)
    for i, (low, high) in enumerate(bounds):
        scaled_samples[:, i] = low + (high - low) * samples[:, i]
    xyf = scaled_samples

    analy_xyf = np.exp(-(xyf[:, 0] ** 2 + xyf[:, 1] ** 2))[:, np.newaxis]
    analy_xy_ori = np.exp(-(xy_ori[:, 0] ** 2 + xy_ori[:, 1] ** 2))[:, np.newaxis]

    w = np.random.uniform(-5, 5, size=(1000, 2))
    b = np.random.uniform(-2, 2, size=1000)
    W = torch.from_numpy(w).to(device=device, dtype=dtype)
    b = torch.from_numpy(b).to(device=device, dtype=dtype)
    train_X = torch.from_numpy(xyf).to(device=device, dtype=dtype)
    train_Y = torch.from_numpy(analy_xyf).to(device=device, dtype=dtype)
    test_x = torch.from_numpy(xy_ori).to(device=device, dtype=dtype)

    H = F.tanh(train_X @ W.T + b[None, :])
    beta = torch.linalg.pinv(H) @ train_Y
    # print(beta.shape, (H @ beta).shape, analy_xyf.shape)
    train_loss = mean_squared_error((H @ beta).detach().cpu().numpy(), analy_xyf)

    pred = (F.tanh(test_x @ W.T + b[None, :]) @ beta).detach().cpu().numpy()

    test_loss = mean_squared_error(pred, analy_xy_ori)

    relative_error = np.linalg.norm(pred - analy_xy_ori) / np.linalg.norm(analy_xy_ori)


    import matplotlib.colors as mcolors
    
    plt.figure(num=1, figsize=(10, 8), dpi=120)
    plt.xlabel("x", fontsize=18)
    plt.ylabel("y", fontsize=18)
    plt.tricontourf(xy_ori[:, 0], xy_ori[:, 1], pred.squeeze(-1), cmap="turbo", levels=100, alpha=1)
    plt.colorbar()
    plt.tight_layout()
    
    plt.figure(num=2, figsize=(10, 8), dpi=120)
    plt.xlabel("x", fontsize=18)
    plt.ylabel("y", fontsize=18)
    error = np.abs(pred - analy_xy_ori).squeeze()
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(error, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    cs = plt.tricontourf(xy_ori[:, 0], xy_ori[:, 1], error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cbar = plt.colorbar(cs, ticks=cbar_ticks, format='%.0e')
    cbar.set_label('Absolute Error (log scale)', fontsize=14)
    plt.tight_layout()
    plt.show()



def solve_1d_unsteady_advection_diffusion(x_inferior=100, t_inferior=50, bc_weight=1., ic_weight=1., l2_reg=1e-5, hidden_nodes=2000):
    in_dim = 2
    v = 0.005
    a = 1

    x = np.linspace(0., 1, x_inferior)
    t = np.linspace(0, 0.5, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = (1 / np.sqrt(4 * xt_ori[:, 1] + 1) * np.exp(-(1 / (v * (4 * xt_ori[:, 1] + 1))) * (xt_ori[:, 0] - 0.2 - a * xt_ori[:, 1]) ** 2))[:, np.newaxis]

    bounds = [(0, 1), (0, 0.5)]
    samples = lhs(2, samples=2000, criterion='center')
    scaled_samples = np.zeros_like(samples)
    for i, (low, high) in enumerate(bounds):
        scaled_samples[:, i] = low + (high - low) * samples[:, i]
    xtf = scaled_samples

    x_bc_left = np.column_stack([np.zeros(300), np.linspace(0, 0.5, 300)])[np.random.choice(300, 100, False)]
    x_bc_right = np.column_stack([np.ones(300), np.linspace(0, 0.5, 300)])[np.random.choice(300, 100, False)]
    x_bc = np.vstack([x_bc_left, x_bc_right])
    T_bc = 1 / np.sqrt(4 * x_bc[:, 1] + 1) * np.exp(-(1 / (v * (4 * x_bc[:, 1] + 1))) * (x_bc[:, 0] - 0.2 - a * x_bc[:, 1]) ** 2)
    x_ic = np.column_stack([np.linspace(0, 1, 600), np.zeros(600)])[np.random.choice(600, 200, False)]
    T_ic = 1 / np.sqrt(4 * x_ic[:, 1] + 1) * np.exp(-(1 / (v * (4 * x_ic[:, 1] + 1))) * (x_ic[:, 0] - 0.2 - a * x_ic[:, 1]) ** 2)

    xtf_t = torch.from_numpy(xtf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(x_bc).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(x_ic).to(device=device, dtype=dtype)
    Tbc_t = torch.from_numpy(T_bc[:, np.newaxis]).to(device=device, dtype=dtype)
    Tic_t = torch.from_numpy(T_ic[:, np.newaxis]).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori).to(device=device, dtype=dtype)

    pielm = piELM(in_dim, hidden_nodes, out_dim=1, weight_scale=5., bias_scale=1., device=device, dtype=dtype)
    param = sum(p.numel() for p in pielm.parameters())

    start_time = time.time()
    _, Hx, Ht, _, Hxx, _, _, H_bc, H_ic = pielm.generate_H_tanh(xtf_t, xbc_t, xic_t)

    sqrt_w_bc = np.sqrt(bc_weight)
    sqrt_w_ic = np.sqrt(ic_weight)
    rhs = torch.zeros(xtf_t.shape[0], 1, device=device, dtype=dtype)
    A = torch.vstack([Ht + a * Hx - v * Hxx, sqrt_w_bc * H_bc, sqrt_w_ic * H_ic])
    b = torch.vstack([rhs, sqrt_w_bc * Tbc_t, sqrt_w_ic * Tic_t])
    # beta = torch.linalg.pinv(A) @ b
    # U, s, V = torch.linalg.svd(A, full_matrices=False)
    # s_reg = torch.diag(s / (s ** 2 + l2_reg))
    # beta = V.T @ s_reg @ U.T @ b
    beta = torch.linalg.solve(A.T @ A, A.T @ b)
    pielm.beta.data = beta
    elapsed_time = time.time() - start_time

    pde_loss = torch.mean(((Ht + a * Hx - v * Hxx) @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    # test_x = np.column_stack([np.linspace(0, 1, 300), np.ones(300) * 0.3])
    # test_y = (1 / np.sqrt(4 * test_x[:, 1] + 1) * np.exp(-(1 / (v * (4 * test_x[:, 1] + 1))) * (test_x[:, 0] - 0.2 - a * test_x[:, 1]) ** 2))[:, np.newaxis]
    # test_x_t = torch.from_numpy(test_x).to(device=device, dtype=dtype)
    # pred = pielm(test_x_t).detach().cpu().numpy()
    # relative_error = np.linalg.norm(pred - test_y) / np.linalg.norm(test_y)
    # mse = mean_squared_error(test_y, pred)
    # mae = mean_absolute_error(test_y, pred)
    pred = pielm(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, pred)
    mae = mean_absolute_error(analy_xtf, pred)
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}, ic_loss:{ic_loss:.6e}, relative_error:{relative_error:.6e}, MAE:{mae:.6e}, mse:{mse:.6e}\n"
          f"param:{param}, train_time:{elapsed_time:.4f}")
    error = np.abs(pred - analy_xtf)

    # ==========================================
    # ==========================================
    fig1, ax1 = plt.subplots(num=1, figsize=(6, 4.8))
    # ax1.set_title("Exact Solution")
    ax1.set_xlabel("t", fontsize=18)
    ax1.set_ylabel("x", fontsize=18)
    contour1 = ax1.tricontourf(xt_ori[:, 1], xt_ori[:, 0], analy_xtf.squeeze(-1), cmap="rainbow", levels=25, alpha=1)
    fig1.colorbar(contour1, ax=ax1)

    # ==========================================
    # ==========================================
    fig2, axes = plt.subplots(nrows=2, ncols=1, num=2, figsize=(6, 9.6))

    # axes[0].set_title("Prediction")
    axes[0].set_xlabel("t", fontsize=18)
    axes[0].set_ylabel("x", fontsize=18)
    contour_pred = axes[0].tricontourf(xt_ori[:, 1], xt_ori[:, 0], pred.squeeze(-1), cmap="rainbow", levels=25, alpha=1)
    fig2.colorbar(contour_pred, ax=axes[0])

    # axes[1].set_title("Absolute Error")
    axes[1].set_xlabel("t", fontsize=18)
    axes[1].set_ylabel("x", fontsize=18)
    contour_err = axes[1].tricontourf(xt_ori[:, 1], xt_ori[:, 0], error.squeeze(-1), cmap="rainbow", levels=25, alpha=1)
    fig2.colorbar(contour_err, ax=axes[1])

    fig2.tight_layout()

    plt.show()
    
    import matplotlib.colors as mcolors
    
    fig_pred = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_pred = fig_pred.add_subplot(111)
    ax_pred.set_xlabel('t', fontsize=18)
    ax_pred.set_ylabel('x', fontsize=18)
    contour_turbo = ax_pred.tricontourf(xt_ori[:, 1], xt_ori[:, 0], pred.squeeze(-1), cmap='turbo', levels=100, alpha=1)
    fig_pred.colorbar(contour_turbo, ax=ax_pred)
    plt.tight_layout()
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('t', fontsize=18)
    ax_log.set_ylabel('x', fontsize=18)
    
    error_plot = error.squeeze()
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(error_plot, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    cs = ax_log.tricontourf(xt_ori[:, 1], xt_ori[:, 0], error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
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
    
    plt.tight_layout()
    plt.show()
    return float(relative_error)



def solve_1d_unsteady_advection_diffusion_hybridkan(x_inferior=100, t_inferior=50, bc_weight=1., ic_weight=1., l2_reg=1., hidden_nodes=50):
    in_dim = 2
    G1 = 20
    G2 = 20
    v = 0.005
    a = 1

    x = np.linspace(0., 1, x_inferior)
    t = np.linspace(0, 0.5, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = (1 / np.sqrt(4 * xt_ori[:, 1] + 1) * np.exp(-(1 / (v * (4 * xt_ori[:, 1] + 1))) * (xt_ori[:, 0] - 0.2 - a * xt_ori[:, 1]) ** 2))[:, np.newaxis]

    bounds = [(0, 1), (0, 0.5)]
    samples = lhs(2, samples=2000, criterion='center')
    scaled_samples = np.zeros_like(samples)
    for i, (low, high) in enumerate(bounds):
        scaled_samples[:, i] = low + (high - low) * samples[:, i]
    xtf = scaled_samples

    x_bc_left = np.column_stack([np.zeros(300), np.linspace(0, 0.5, 300)])[np.random.choice(300, 100, False)]
    x_bc_right = np.column_stack([np.ones(300), np.linspace(0, 0.5, 300)])[np.random.choice(300, 100, False)]
    x_bc = np.vstack([x_bc_left, x_bc_right])
    T_bc = 1 / np.sqrt(4 * x_bc[:, 1] + 1) * np.exp(-(1 / (v * (4 * x_bc[:, 1] + 1))) * (x_bc[:, 0] - 0.2 - a * x_bc[:, 1]) ** 2)
    x_ic = np.column_stack([np.linspace(0, 1, 600), np.zeros(600)])[np.random.choice(600, 200, False)]
    T_ic = 1 / np.sqrt(4 * x_ic[:, 1] + 1) * np.exp(-(1 / (v * (4 * x_ic[:, 1] + 1))) * (x_ic[:, 0] - 0.2 - a * x_ic[:, 1]) ** 2)

    xtf_t = torch.from_numpy(xtf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(x_bc).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(x_ic).to(device=device, dtype=dtype)
    Tbc_t = torch.from_numpy(T_bc[:, np.newaxis]).to(device=device, dtype=dtype)
    Tic_t = torch.from_numpy(T_ic[:, np.newaxis]).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori).to(device=device, dtype=dtype)

    model = piELMHKAN(in_dim, hidden_nodes, out_dim=1, num_grid1=G1, num_grid2=G2, knot_grid1=[-1.0, 1.], knot_grid2=[-1, 1.], coeff1_sacle=1.,
                      sigmoid_scale=6.5, device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    _, Hx, Ht, _, Hxx, _, _, H_bc, H_ic = model.generate_H(xtf_t, xbc_t, xic_t)

    sqrt_w_bc = np.sqrt(bc_weight)
    sqrt_w_ic = np.sqrt(ic_weight)
    rhs = torch.zeros(xtf_t.shape[0], 1, device=device, dtype=dtype)
    A = torch.vstack([Ht + a * Hx - v * Hxx, sqrt_w_bc * H_bc, sqrt_w_ic * H_ic])
    b = torch.vstack([rhs, sqrt_w_bc * Tbc_t, sqrt_w_ic * Tic_t])
    # beta = torch.linalg.pinv(A) @ b
    # cond_A = torch.linalg.cond(A)
    # cond_AtA = torch.linalg.cond(A.T @ A)
    # print(f"cond(A) = {cond_A.item():.3e}, cond(A.T@A) = {cond_AtA.item():.3e}")
    lambda_reg = 1e-8
    beta = torch.linalg.solve(A.T @ A + lambda_reg * torch.eye(A.shape[1]).to(device=device, dtype=dtype), A.T @ b)
    model.beta.data = beta
    elapsed_time = time.time() - start_time

    pde_loss = torch.mean(((Ht + a * Hx - v * Hxx) @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    # test_x = np.column_stack([np.linspace(0, 1, 300), np.ones(300) * 0.3])
    # test_y = (1 / np.sqrt(4 * test_x[:, 1] + 1) * np.exp(-(1 / (v * (4 * test_x[:, 1] + 1))) * (test_x[:, 0] - 0.2 - a * test_x[:, 1]) ** 2))[:, np.newaxis]
    # test_x_t = torch.from_numpy(test_x).to(device=device, dtype=dtype)
    # pred = pielm(test_x_t).detach().cpu().numpy()
    # relative_error = np.linalg.norm(pred - test_y) / np.linalg.norm(test_y)
    # mse = mean_squared_error(test_y, pred)
    # mae = mean_absolute_error(test_y, pred)
    pred = model(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, pred)
    mae = mean_absolute_error(analy_xtf, pred)
    print(
        f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}, ic_loss:{ic_loss:.6e}, relative_error:{relative_error:.6e}, MAE:{mae:.6e}, mse:{mse:.6e}\n"
        f"param:{param}, train_time:{elapsed_time:.4f}")
    error = np.abs(pred - analy_xtf)

    # ==========================================
    # ==========================================
    fig1, ax1 = plt.subplots(num=1, figsize=(6, 4.8))
    # ax1.set_title("Exact Solution")
    ax1.set_xlabel("t", fontsize=18)
    ax1.set_ylabel("x", fontsize=18)
    contour1 = ax1.tricontourf(xt_ori[:, 1], xt_ori[:, 0], analy_xtf.squeeze(-1), cmap="rainbow", levels=25, alpha=1)
    fig1.colorbar(contour1, ax=ax1)

    # ==========================================
    # ==========================================
    
    fig_pred = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_pred = fig_pred.add_subplot(111)
    ax_pred.set_xlabel('t', fontsize=18)
    ax_pred.set_ylabel('x', fontsize=18)
    contour_turbo = ax_pred.tricontourf(xt_ori[:, 1], xt_ori[:, 0], pred.squeeze(-1), cmap='turbo', levels=100, alpha=1)
    fig_pred.colorbar(contour_turbo, ax=ax_pred)
    plt.tight_layout()
    
    import matplotlib.colors as mcolors
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('t', fontsize=18)
    ax_log.set_ylabel('x', fontsize=18)
    
    error_plot = error.squeeze()
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(error_plot, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    cs = ax_log.tricontourf(xt_ori[:, 1], xt_ori[:, 0], error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
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
    
    plt.tight_layout()
    plt.show()
    return float(relative_error)



def solve_1d_unsteady_advection_diffusion_pickanelm(x_inferior=100, t_inferior=50, bc_weight=1., ic_weight=1., l2_reg=1., hidden_nodes=110):
    in_dim = 2
    G1 = 20
    G2 = 20
    v = 0.005
    a = 1

    x = np.linspace(0., 1, x_inferior)
    t = np.linspace(0, 0.5, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = (1 / np.sqrt(4 * xt_ori[:, 1] + 1) * np.exp(-(1 / (v * (4 * xt_ori[:, 1] + 1))) * (xt_ori[:, 0] - 0.2 - a * xt_ori[:, 1]) ** 2))[:, np.newaxis]

    bounds = [(0, 1), (0, 0.5)]
    samples = lhs(2, samples=2000, criterion='center')
    scaled_samples = np.zeros_like(samples)
    for i, (low, high) in enumerate(bounds):
        scaled_samples[:, i] = low + (high - low) * samples[:, i]
    xtf = scaled_samples

    x_bc_left = np.column_stack([np.zeros(300), np.linspace(0, 0.5, 300)])[np.random.choice(300, 100, False)]
    x_bc_right = np.column_stack([np.ones(300), np.linspace(0, 0.5, 300)])[np.random.choice(300, 100, False)]
    x_bc = np.vstack([x_bc_left, x_bc_right])
    T_bc = 1 / np.sqrt(4 * x_bc[:, 1] + 1) * np.exp(-(1 / (v * (4 * x_bc[:, 1] + 1))) * (x_bc[:, 0] - 0.2 - a * x_bc[:, 1]) ** 2)
    x_ic = np.column_stack([np.linspace(0, 1, 600), np.zeros(600)])[np.random.choice(600, 200, False)]
    T_ic = 1 / np.sqrt(4 * x_ic[:, 1] + 1) * np.exp(-(1 / (v * (4 * x_ic[:, 1] + 1))) * (x_ic[:, 0] - 0.2 - a * x_ic[:, 1]) ** 2)

    xtf_t = torch.from_numpy(xtf).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(x_bc).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(x_ic).to(device=device, dtype=dtype)
    Tbc_t = torch.from_numpy(T_bc[:, np.newaxis]).to(device=device, dtype=dtype)
    Tic_t = torch.from_numpy(T_ic[:, np.newaxis]).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori).to(device=device, dtype=dtype)

    model = piCKANELM(in_dim, hidden_nodes, out_dim=1, degree1=5, degree2=2, coeff_scaled=1.,
                        device=device, dtype=dtype)

    param = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    _, Hx, Ht, _, Hxx, _, _, _, H_bc, H_ic = model.generate_H(xtf_t, xbc_t, xic_t)
    # H, Hx, Hy, gradH, Hxx, Hyy, LapH, gradH_ic, H_bc, H_ic

    sqrt_w_bc = np.sqrt(bc_weight)
    sqrt_w_ic = np.sqrt(ic_weight)
    rhs = torch.zeros(xtf_t.shape[0], 1, device=device, dtype=dtype)
    print(f"Ht.shape={Ht.shape}, Hx.shape={Hx.shape}, Hxx.shape={Hxx.shape}, H_bc.shape={H_bc.shape}, H_ic.shape={H_ic.shape}")
    A = torch.vstack([Ht + a * Hx - v * Hxx, sqrt_w_bc * H_bc, sqrt_w_ic * H_ic])
    b = torch.vstack([rhs, sqrt_w_bc * Tbc_t, sqrt_w_ic * Tic_t])
    I_reg = torch.eye(A.shape[1], device=device, dtype=dtype) * l2_reg
    beta = torch.linalg.solve(A.T @ A + I_reg, A.T @ b)
    model.beta.data = beta
    elapsed_time = time.time() - start_time

    pde_loss = torch.mean(((Ht + a * Hx - v * Hxx) @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    # test_x = np.column_stack([np.linspace(0, 1, 300), np.ones(300) * 0.3])
    # test_y = (1 / np.sqrt(4 * test_x[:, 1] + 1) * np.exp(-(1 / (v * (4 * test_x[:, 1] + 1))) * (test_x[:, 0] - 0.2 - a * test_x[:, 1]) ** 2))[:, np.newaxis]
    # test_x_t = torch.from_numpy(test_x).to(device=device, dtype=dtype)
    # pred = pielm(test_x_t).detach().cpu().numpy()
    # relative_error = np.linalg.norm(pred - test_y) / np.linalg.norm(test_y)
    # mse = mean_squared_error(test_y, pred)
    # mae = mean_absolute_error(test_y, pred)
    pred = model(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, pred)
    mae = mean_absolute_error(analy_xtf, pred)
    print(
        f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}, ic_loss:{ic_loss:.6e}, relative_error:{relative_error:.6e}, MAE:{mae:.6e}, mse:{mse:.6e}\n"
        f"param:{param}, train_time:{elapsed_time:.4f}")
    error = np.abs(pred - analy_xtf)
    
    import matplotlib.colors as mcolors
    
    fig_pred = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_pred = fig_pred.add_subplot(111)
    ax_pred.set_xlabel('t', fontsize=18)
    ax_pred.set_ylabel('x', fontsize=18)
    contour_turbo = ax_pred.tricontourf(xt_ori[:, 1], xt_ori[:, 0], pred.squeeze(-1), cmap='turbo', levels=100, alpha=1)
    fig_pred.colorbar(contour_turbo, ax=ax_pred)
    plt.tight_layout()
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('t', fontsize=18)
    ax_log.set_ylabel('x', fontsize=18)
    
    error_plot = error.squeeze()
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(error_plot, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    cs = ax_log.tricontourf(xt_ori[:, 1], xt_ori[:, 0], error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
    cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cbar = plt.colorbar(cs, ax=ax_log, ticks=cbar_ticks, format='%.0e')
    cbar.set_label('Absolute Error (log scale)', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    return float(relative_error)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    RANDOM_SEEDS = [1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030]
    SEED_RESULT_DIR = "20-NeuComp-Original_PIELM_randomseed"
    os.makedirs(SEED_RESULT_DIR, exist_ok=True)

    CASE_REGISTRY = {
        "solve_1d_advection": solve_1d_advection,
        "solve_1d_diffusion": solve_1d_diffusion,
        "solve_1d_advection_diffusion": solve_1d_advection_diffusion,
        "solve_1d_unsteady_advection": solve_1d_unsteady_advection,
        "solve_2d_advection": solve_2d_advection,
        "solve_2d_diffusion": solve_2d_diffusion,
        "solve_2d_diffusion_pikan": solve_2d_diffusion_pikan,
        "solve_2d_diffusion_pinn": solve_2d_diffusion_pinn,
        "solve_2d_advection_kan": solve_2d_advection_kan,
        "solve_2d_diffusion_kan": solve_2d_diffusion_kan,
        "solve_2d_advection_hybridkan": solve_2d_advection_hybridkan,
        "solve_2d_diffusion_hybridkan": solve_2d_diffusion_hybridkan,
        "solve_2d_diffusion_pickanelm": solve_2d_diffusion_pickanelm,
        "solve_1d_unsteady_advection_hybridkan": solve_1d_unsteady_advection_hybridkan,
        "solve_1d_unsteady_advection_diffusion": solve_1d_unsteady_advection_diffusion,
        "solve_1d_unsteady_advection_diffusion_hybridkan": solve_1d_unsteady_advection_diffusion_hybridkan,
        "solve_1d_unsteady_advection_diffusion_pickanelm": solve_1d_unsteady_advection_diffusion_pickanelm,
    }

    ACTIVE_CASE = "solve_1d_unsteady_advection_hybridkan"
    # ACTIVE_CASE = "solve_1d_advection"
    # ACTIVE_CASE = "solve_1d_diffusion"
    # ACTIVE_CASE = "solve_1d_advection_diffusion"
    # ACTIVE_CASE = "solve_1d_unsteady_advection"
    # ACTIVE_CASE = "solve_2d_advection"
    # ACTIVE_CASE = "solve_2d_diffusion"
    # ACTIVE_CASE = "solve_2d_diffusion_pikan"
    # ACTIVE_CASE = "solve_2d_diffusion_pinn"
    # ACTIVE_CASE = "solve_2d_advection_kan"
    # ACTIVE_CASE = "solve_2d_diffusion_kan"
    # ACTIVE_CASE = "solve_2d_advection_hybridkan"
    # ACTIVE_CASE = "solve_2d_diffusion_hybridkan"
    # ACTIVE_CASE = "solve_2d_diffusion_pickanelm"
    # ACTIVE_CASE = "solve_1d_unsteady_advection_diffusion"
    # ACTIVE_CASE = "solve_1d_unsteady_advection_diffusion_hybridkan"
    # ACTIVE_CASE = "solve_1d_unsteady_advection_diffusion_pickanelm"

    def run_multi_seed(case_name, seeds):
        solve_fn = CASE_REGISTRY[case_name]
        rel_errors = []
        _orig_show = plt.show
        plt.show = lambda *a, **k: plt.close("all")
        try:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                err = solve_fn()
                rel_errors.append(float(err))
                print(f"[{case_name}] seed={seed} relative_error={float(err):.6e}")
        finally:
            plt.show = _orig_show
            plt.close("all")
        arr = np.array(rel_errors, dtype=np.float64)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        min_v, max_v = float(np.min(arr)), float(np.max(arr))
        print(f"[{case_name}] over {len(arr)} seeds: Mean={mean_v:.6e} Std={std_v:.6e} "
              f"Min={min_v:.6e} Max={max_v:.6e}  =>  {mean_v:.6e} ± {std_v:.6e}")

        base = f"{case_name}_error_"
        txt_path = os.path.join(SEED_RESULT_DIR, base + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"case: {case_name}\nseeds: {list(seeds)}\n")
            for s, e in zip(seeds, rel_errors):
                f.write(f"seed={s} relative_error={e:.10e}\n")
            f.write(f"Mean={mean_v:.10e}\nStd(ddof=1)={std_v:.10e}\n"
                    f"Min={min_v:.10e}\nMax={max_v:.10e}\n"
                    f"Mean+-Std: {mean_v:.6e} +- {std_v:.6e}\n")
        print(f"saved: {txt_path}")

        fig_path = os.path.join(SEED_RESULT_DIR, base + ".png")
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
        x = np.arange(len(seeds))
        ax.errorbar(x, arr, yerr=np.full_like(arr, std_v), fmt="o-", capsize=4,
                    label=f"Mean={mean_v:.3e} ± Std={std_v:.3e}")
        ax.axhline(mean_v, color="r", linestyle="--", linewidth=1.2, label="Mean")
        ax.axhspan(mean_v - std_v, mean_v + std_v, color="r", alpha=0.12, label="±Std")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds], rotation=30)
        ax.set_xlabel("seed")
        ax.set_ylabel("relative_error")
        ax.set_title(f"{case_name} (n={len(seeds)})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"saved: {fig_path}")
        return rel_errors

    # run_multi_seed(ACTIVE_CASE, RANDOM_SEEDS)

    torch.manual_seed(1024); np.random.seed(1024)

    # solve_1d_advection()
    # solve_1d_diffusion()
    # solve_1d_advection_diffusion()
    # solve_2d_advection(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=0.)
    # solve_2d_diffusion(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=0., hidden_nodes=464)
    # solve_2d_advection_kan(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=0.)
    # solve_2d_diffusion_kan(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=5e-3)
    # solve_2d_advection_hybridkan(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=1e-5)
    # solve_2d_diffusion_hybridkan(x_interfior=921, x_boundary=240, bc_weight=1., l2_reg=1e-5)
    # solve_2d_diffusion_pikan()
    # solve_2d_diffusion_pinn()
    solve_1d_unsteady_advection(hidden_nodes=185)
    # solve_1d_unsteady_advection_hybridkan()
    # TC_9()
    # solve_1d_unsteady_advection_diffusion()
    # solve_1d_unsteady_advection_diffusion_hybridkan()
    # solve_2d_diffusion_pickanelm()
    # solve_1d_unsteady_advection_diffusion_pickanelm()
    # hexagon_sampler()
