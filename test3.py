import os

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from efficentKAN import efficentKAN
from pinn import PINN

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

        self.beta = nn.Parameter(torch.tensor([0.]), requires_grad=False)

    def forward(self, x):
        H = F.tanh(x @ self.W.T + self.b[None, :])
        output = H @ self.beta
        return output

    def generate_H_tanh(self, Xf, Xbc, xic):
        H_bc = F.tanh(Xbc @ self.W.T + self.b[None, :])
        H_ic = F.tanh(xic @ self.W.T + self.b[None, :])

        H = F.tanh(Xf @ self.W.T + self.b[None, :])
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

        self.dist = (grid_range[1] - grid_range[0]) / (num_grid - 1) * 1.8
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

    def generate_H(self, Xf, Xbc, Xic):
        H_bc = self.act(self.act(Xbc, self.grid1).reshape(Xbc.shape[0], -1) @ self.ctrl, self.grid2).reshape(Xbc.shape[0], -1)
        H_ic = self.act(self.act(Xic, self.grid1).reshape(Xic.shape[0], -1) @ self.ctrl, self.grid2).reshape(Xic.shape[0], -1)

        Z = self.act(Xf, self.grid1).reshape(Xf.shape[0], -1) @ self.ctrl
        H = self.act(Z, self.grid2).reshape(Xf.shape[0], -1)
        w_x = self.ctrl[:self.num_grid, :].T
        w_y = self.ctrl[self.num_grid:, :].T

        H_x = (self.phi_1(Z, self.grid2) * w_x[None, :, :] * self.phi_1(Xf, self.grid1, dim=0)).reshape(Xf.shape[0], -1)
        H_y = (self.phi_1(Z, self.grid2) * w_y[None, :, :] * self.phi_1(Xf, self.grid1, dim=1)).reshape(Xf.shape[0], -1)

        gradHf = H_x + H_y

        H_xx = (self.phi_2(Z, self.grid2) * (w_x[None, :, :] * self.phi_1(Xf, self.grid1, dim=0)) ** 2 + self.phi_1(Z, self.grid2) * w_x[None, :, :] * self.phi_2(Xf, self.grid1, dim=0)).reshape(Xf.shape[0], -1)
        H_yy = (self.phi_2(Z, self.grid2) * (w_y[None, :, :] * self.phi_1(Xf, self.grid1, dim=1)) ** 2 + self.phi_1(Z, self.grid2) * w_y[None, :, :] * self.phi_2(Xf, self.grid1, dim=1)).reshape(Xf.shape[0], -1)
        LapHf = H_xx + H_yy
        # print(f"LapHf.shape:{LapHf.shape}")

        return H, H_x, H_y, gradHf, H_xx, H_yy, LapHf, H_bc, H_ic


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
        ctrl = np.random.uniform(-coeff1_sacle, coeff1_sacle, size=(in_dim * num_grid1, hidden_nodes)) # * np.pi
        # print(ctrl)
        self.ctrl = nn.Parameter(torch.from_numpy(ctrl).to(device=device, dtype=dtype), requires_grad=False)
        self.beta = nn.Parameter(torch.zeros(hidden_nodes * num_grid2, out_dim, device=device, dtype=dtype), requires_grad=False)

        self.act1 = lambda x, grid: torch.sigmoid(sigmoid_scale * (x[:, :, None] - grid[None, :, :]))
        self.sigmoid_scale = sigmoid_scale
        self.dist = ((knot_grid2[1] - knot_grid2[0]) / (num_grid2 - 1)) * 1.
        self.act2 = lambda x, grid: torch.exp(-((x[:, :, None] - grid[None, :, :]) / self.dist) ** 2)

    def forward(self, x):
        H1_norm = F.tanh(self.cal_H(x))
        output = self.act2(H1_norm, self.grid2).reshape(x.shape[0], -1) @ self.beta
        return output

    def cal_H(self, x):
        basis = self.sig(x, self.grid1)
        H1 = basis.reshape(x.shape[0], -1) @ self.ctrl
        return H1

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

    def generate_H(self, Xf, Xbc, Xic):
        H_bc = self.act2(F.tanh(self.cal_H(Xbc)), self.grid2).reshape(Xbc.shape[0], -1)
        H_ic = self.act2(F.tanh(self.cal_H(Xic)), self.grid2).reshape(Xic.shape[0], -1)

        Z = self.cal_H(Xf)
        H = self.act2(F.tanh(Z), self.grid2).reshape(Xf.shape[0], -1)

        w_x = self.ctrl[:self.num_grid1, :]  # [hidden_nodes, num_grid1]
        w_y = self.ctrl[self.num_grid1:, :]  # [hidden_nodes, num_grid1]
        x = Xf[:, 0][:, None]
        y = Xf[:, 1][:, None]
        grid1_x = self.grid1[0, :][None, :]  # [1, num_grid1]
        grid1_y = self.grid1[1, :][None, :]  # [1, num_grid1]
        dZdx = self.sig_1(x, grid1_x).permute(0, 2, 1).squeeze() @ w_x
        dZdy = self.sig_1(y, grid1_y).permute(0, 2, 1).squeeze() @ w_y
        dZZdxx = self.sig_2(x, grid1_x).permute(0, 2, 1).squeeze() @ w_x
        dZZdyy = self.sig_2(y, grid1_y).permute(0, 2, 1).squeeze() @ w_y
        dHdZ = self.phi_1(F.tanh(Z))
        ddHdZZ = self.phi_2(F.tanh(Z))
        tanh1 = 1 - F.tanh(Z) ** 2
        tanh2 = -2 * F.tanh(Z) * tanh1
        # print(dZdx.shape, dZZdxx.shape)
        Hx = dHdZ * (dZdx * tanh1)[:, :, None]
        Hy = dHdZ * (dZdy * tanh1)[:, :, None]
        Hx = Hx.clone().reshape(x.shape[0], -1)
        Hy = Hy.clone().reshape(y.shape[0], -1)
        gradHf = Hx + Hy
        Hxx = ddHdZZ * (dZdx * tanh1)[:, :, None] ** 2 + dHdZ * (tanh2 * dZdx ** 2)[:, :, None] + dHdZ * (tanh1 * dZZdxx)[:, :, None]
        Hyy = ddHdZZ * (dZdy * tanh1)[:, :, None] ** 2 + dHdZ * (tanh2 * dZdy ** 2)[:, :, None] + dHdZ * (tanh1 * dZZdyy)[:, :, None]
        Hxx = Hxx.clone().reshape(x.shape[0], -1)
        Hyy = Hyy.clone().reshape(y.shape[0], -1)
        LapHf = Hxx + Hyy

        return H, Hx, Hy, gradHf, Hxx, Hyy, LapHf, H_bc, H_ic


class adapt_piELMHKAN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_nodes,
                 out_dim,
                 num_grid_diff_x=30,
                 num_grid_diff_t=5,
                 num_grid_rbf=100,
                 grid_range_diff=[-1.5, 1.5],
                 grid_range_rbf=[-1., 1],
                 sigmoid_scale=10.,
                 coeff_scale=1.5,
                 device=torch.device('cpu'),
                 dtype=torch.float32):
        super(adapt_piELMHKAN, self).__init__()
        self.in_dim = in_dim
        self.hidden_nodes = hidden_nodes
        self.out_dim = out_dim
        self.sigmoid_scale = sigmoid_scale
        self.grid_range_diff = grid_range_diff
        self.grid_range_rbf = grid_range_rbf
        self.num_grid_rbf = num_grid_rbf
        self.num_grid_diff_x = num_grid_diff_x
        self.num_grid_diff_t = num_grid_diff_t

        self.grid_x = torch.linspace(grid_range_diff[0], grid_range_diff[1], num_grid_diff_x + 1, device=device, dtype=dtype)
        self.grid_t = torch.linspace(grid_range_diff[0], grid_range_diff[1], num_grid_diff_t + 1, device=device, dtype=dtype)
        self.grid2 = torch.linspace(grid_range_rbf[0], grid_range_rbf[1], num_grid_rbf, device=device, dtype=dtype)[None, :].expand(hidden_nodes, num_grid_rbf)

        ctrl = np.random.uniform(-coeff_scale, coeff_scale, size=(num_grid_diff_x + num_grid_diff_t, hidden_nodes))
        self.ctrl = nn.Parameter(torch.from_numpy(ctrl).to(device=device, dtype=dtype), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(hidden_nodes * num_grid_rbf, out_dim, device=device, dtype=dtype), requires_grad=False)

        regular_dist = (self.grid_range_rbf[1] - self.grid_range_rbf[0]) / (self.num_grid_rbf - 1)
        n_high, n_medi = int(self.num_grid_rbf * 0.6), int(self.num_grid_rbf * 0.3)
        n_smth = self.num_grid_rbf - (n_high + n_medi)
        highF_d = regular_dist * 0.3
        mediF_d = regular_dist * 0.5
        smthF_d = regular_dist * 0.8
        self.dist_discret = torch.Tensor([highF_d] * n_high + [mediF_d] * n_medi + [smthF_d] * n_smth).to(device=device, dtype=dtype)[None, None, :]

    def forward(self, X: torch.Tensor):
        x, t = X[:, 0], X[:, 1]
        x_diff = self.DiffSigmoid(x, self.grid_x)
        t_diff = self.DiffSigmoid(t, self.grid_t)

        mid = torch.hstack([x_diff, t_diff]) @ self.ctrl  # (N, hidden_nodes)
        # mid_mean = mid.mean(dim=0, keepdim=True)
        # mid_std = mid.std(dim=0, unbiased=False, keepdim=True) + 1e-12
        # Z = (mid - mid_mean) / mid_std
        output = self.GaussianRBF(mid).reshape(X.shape[0], -1) @ self.beta
        return output

    def DiffSigmoid(self, x, grid):
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        if grid.dim() == 1:
            grid = grid.unsqueeze(0)
        sgm = F.sigmoid(self.sigmoid_scale * (x[:, :, None] - grid[None, :, :]))
        bumps = sgm[:, :, 1:] - sgm[:, :, :-1]
        return bumps.reshape(x.shape[0], -1)

    def DiffSigmoid_1(self, x, grid):
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        if grid.dim() == 1:
            grid = grid.unsqueeze(0)
        sgm = F.sigmoid(self.sigmoid_scale * (x[:, :, None] - grid[None, :, :]))
        sgm_1 = self.sigmoid_scale * (sgm - sgm ** 2)
        bumps = sgm_1[:, :, 1:] - sgm_1[:, :, :-1]
        return bumps.reshape(x.shape[0], -1)

    def DiffSigmoid_2(self, x, grid):
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        if grid.dim() == 1:
            grid = grid.unsqueeze(0)
        sgm = F.sigmoid(self.sigmoid_scale * (x[:, :, None] - grid[None, :, :]))
        sgm_1 = self.sigmoid_scale * (sgm - sgm ** 2)
        sgm_2 = self.sigmoid_scale * (sgm_1 - 2 * sgm * sgm_1)
        bumps = sgm_2[:, :, 1:] - sgm_2[:, :, :-1]
        return bumps.reshape(x.shape[0], -1)

    def GaussianRBF(self, x):
        minus = x[:, :, None] - self.grid2[None, :, :]
        rbf = torch.exp(- (minus / self.dist_discret) ** 2)
        return rbf

    def GaussianRBF_1(self, x):
        output = self.GaussianRBF(x) * -(2 / self.dist_discret ** 2) * (x[:, :, None] - self.grid2[None, :, :])
        return output

    def GaussianRBF_2(self, x):
        part1 = self.GaussianRBF(x) * (-(2 / self.dist_discret ** 2) * (x[:, :, None] - self.grid2[None, :, :])) ** 2
        part2 = self.GaussianRBF(x) * -(2 / self.dist_discret ** 2)
        return part1 + part2

    def Generate_Mat(self, Xf, Xbc, Xic=None):
        xbc_x, xbc_t = Xbc[:, 0], Xbc[:, 1]
        xbc_diff_x = self.DiffSigmoid(xbc_x, self.grid_x)
        xbc_diff_t = self.DiffSigmoid(xbc_t, self.grid_t)
        xbc_mid = torch.hstack([xbc_diff_x, xbc_diff_t]) @ self.ctrl
        H_bc = self.GaussianRBF(xbc_mid).reshape(Xbc.shape[0], -1)

        if Xic is not None:
            xic_x, xic_t = Xic[:, 0], Xic[:, 1]
            xic_diff_x = self.DiffSigmoid(xic_x, self.grid_x)
            xic_diff_t = self.DiffSigmoid(xic_t, self.grid_t)
            xic_mid = torch.hstack([xic_diff_x, xic_diff_t]) @ self.ctrl
            H_ic = self.GaussianRBF(xic_mid).reshape(Xic.shape[0], -1)
        else:
            H_ic = None

        x, t = Xf[:, 0], Xf[:, 1]
        x_diff = self.DiffSigmoid(x, self.grid_x)
        t_diff = self.DiffSigmoid(t, self.grid_t)

        Z = torch.hstack([x_diff, t_diff]) @ self.ctrl  # (N, hidden_nodes)
        # mid_mean = mid.mean(dim=0, keepdim=True)
        # mid_std = mid.std(dim=0, unbiased=False, keepdim=True) + 1e-12
        # Z = (mid - mid_mean) / mid_std
        H = self.GaussianRBF(Z).reshape(Xf.shape[0], -1)

        w_x = self.ctrl[:self.num_grid_diff_x, :]  # [num_gridx, hidden_nodes]
        w_y = self.ctrl[self.num_grid_diff_x:, :]  # [num_gridt, hidden_nodes]

        sig_1_x = self.DiffSigmoid_1(x, self.grid_x)
        sig_1_y = self.DiffSigmoid_1(t, self.grid_t)
        sig_2_x = self.DiffSigmoid_2(x, self.grid_x)
        sig_2_y = self.DiffSigmoid_2(t, self.grid_t)
        dZdx = sig_1_x @ w_x
        dZdy = sig_1_y @ w_y
        dZZdxx = sig_2_x @ w_x
        dZZdyy = sig_2_y @ w_y
        # print(dZdx.shape, dZZdxx.shape)
        Hx = (self.GaussianRBF_1(Z) * dZdx[:, :, None]).reshape(x.shape[0], -1)
        Hy = (self.GaussianRBF_1(Z) * dZdy[:, :, None]).reshape(t.shape[0], -1)
        gradHf = Hx + Hy
        Hxx = (self.GaussianRBF_2(Z) * dZdx[:, :, None] ** 2 + self.GaussianRBF_1(Z) * dZZdxx[:, :, None]).reshape(Xf.shape[0], -1)
        Hyy = (self.GaussianRBF_2(Z) * dZdy[:, :, None] ** 2 + self.GaussianRBF_1(Z) * dZZdyy[:, :, None]).reshape(Xf.shape[0], -1)
        LapHf = Hxx + Hyy

        return H, Hx, Hy, gradHf, Hxx, Hyy, LapHf, H_bc, H_ic


class piCKANELM_HighLowMixed(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, degree1_high, degree1_low, degree2, coeff_scaled_high=1., coeff_scaled_low=1.,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(piCKANELM_HighLowMixed, self).__init__()
        self.in_dim = in_dim
        self.hidden_nodes = hidden_nodes
        self.out_dim = out_dim
        self.degree1_high = degree1_high
        self.degree1_low = degree1_low
        self.degree2 = degree2

        self.cheb_coeff_high = nn.Parameter(torch.empty(degree1_high + 1, hidden_nodes).to(device=device, dtype=dtype), requires_grad=False)
        self.cheb_coeff_low = nn.Parameter(torch.empty(degree1_low + 1, hidden_nodes).to(device=device, dtype=dtype),requires_grad=False)
        self.beta = nn.Parameter(torch.ones(hidden_nodes * (degree2 + 1), out_dim, device=device, dtype=dtype), requires_grad=False)

        nn.init.normal_(self.cheb_coeff_high, mean=0.0, std=coeff_scaled_high / (in_dim * (degree1_high + 1)))
        nn.init.normal_(self.cheb_coeff_low, mean=0.0, std=coeff_scaled_low / (in_dim * (degree1_low + 1)))

    def forward(self, X):
        H1_x = self.chebyshev(X[:, 0][:, None], 1, self.degree1_high).reshape(X.shape[0], -1) @ self.cheb_coeff_high
        H1_t = self.chebyshev(X[:, 1][:, None], 1, self.degree1_low).reshape(X.shape[0], -1) @ self.cheb_coeff_low
        output_x = self.chebyshev(F.tanh(H1_x), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1)
        output_t = self.chebyshev(F.tanh(H1_t), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1)
        output = (output_x * output_t) @ self.beta

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

    def generate_H(self, Xtf, Xbc, Xic):
        bcmid_x = self.chebyshev(Xbc[:, 0][:, None], 1, self.degree1_high).reshape(Xbc.shape[0], -1) @ self.cheb_coeff_high
        bcmid_t = self.chebyshev(Xbc[:, 1][:, None], 1, self.degree1_low).reshape(Xbc.shape[0], -1) @ self.cheb_coeff_low
        H_bc_x = self.chebyshev(F.tanh(bcmid_x), self.hidden_nodes, self.degree2).reshape(Xbc.shape[0], -1)
        H_bc_t = self.chebyshev(F.tanh(bcmid_t), self.hidden_nodes, self.degree2).reshape(Xbc.shape[0], -1)
        H_bc = H_bc_x * H_bc_t

        icmid_x = self.chebyshev(Xic[:, 0][:, None], 1, self.degree1_high).reshape(Xic.shape[0], -1) @ self.cheb_coeff_high
        icmid_t = self.chebyshev(Xic[:, 1][:, None], 1, self.degree1_low).reshape(Xic.shape[0], -1) @ self.cheb_coeff_low
        H_ic_x = self.chebyshev(F.tanh(icmid_x), self.hidden_nodes, self.degree2).reshape(Xic.shape[0], -1)
        H_ic_t = self.chebyshev(F.tanh(icmid_t), self.hidden_nodes, self.degree2).reshape(Xic.shape[0], -1)
        H_ic = H_ic_x * H_ic_t

        x, t = Xtf[:, 0].clone()[:, None], Xtf[:, 1].clone()[:, None]
        Z_x = self.chebyshev(x, 1, self.degree1_high).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_high
        Z_t = self.chebyshev(t, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_low
        H_x = self.chebyshev(F.tanh(Z_x), self.hidden_nodes, self.degree2)
        H_t = self.chebyshev(F.tanh(Z_t), self.hidden_nodes, self.degree2)
        H = H_x * H_t

        w_x = self.cheb_coeff_high
        w_y = self.cheb_coeff_low
        dZdx = self.chebyshev1(x, 1, self.degree1_high).reshape(Xtf.shape[0], -1) @ w_x
        dZdy = self.chebyshev1(t, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_y
        ddZdxx = self.chebyshev2(x, 1, self.degree1_high).reshape(Xtf.shape[0], -1) @ w_x
        ddZdyy = self.chebyshev2(t, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_y
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

        return H, Hx, Hy, gradH, Hxx, Hyy, LapH, H_bc, H_ic


class piCKANELM(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, degree1, degree2, coeff_scaled=1.,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(piCKANELM, self).__init__()
        self.in_dim = in_dim
        self.hidden_nodes = hidden_nodes
        self.out_dim = out_dim
        self.degree1 = degree1
        self.degree2 = degree2

        self.cheb_coeff = nn.Parameter(torch.empty(degree1 + 1, hidden_nodes).to(device=device, dtype=dtype), requires_grad=False)
        self.beta = nn.Parameter(torch.ones(hidden_nodes * (degree2 + 1), out_dim, device=device, dtype=dtype), requires_grad=False)

        nn.init.normal_(self.cheb_coeff, mean=0.0, std=coeff_scaled / (in_dim * (degree1 + 1)))

    def forward(self, X):
        H = self.chebyshev(X, self.in_dim, self.degree1).reshape(X.shape[0], -1) @ self.cheb_coeff
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

    def generate_H(self, Xtf, Xbc, Xic):
        bcmid = self.chebyshev(Xbc, self.in_dim, self.degree1_high).reshape(Xbc.shape[0], -1) @ self.cheb_coeff
        H_bc = self.chebyshev(F.tanh(bcmid), self.hidden_nodes, self.degree2).reshape(Xbc.shape[0], -1)

        icmid = self.chebyshev(Xic, self.in_dim, self.degree1_high).reshape(Xic.shape[0], -1) @ self.cheb_coeff
        H_ic = self.chebyshev(F.tanh(icmid), self.hidden_nodes, self.degree2).reshape(Xic.shape[0], -1)

        x, t = Xtf[:, 0].clone()[:, None], Xtf[:, 1].clone()[:, None]
        Z = self.chebyshev(Xtf, self.in_dim, self.degree1_high).reshape(Xtf.shape[0], -1) @ self.cheb_coeff
        H = self.chebyshev(F.tanh(Z), self.hidden_nodes, self.degree2)

        w_x = self.cheb_coeff
        w_y = self.cheb_coeff_low
        dZdx = self.chebyshev1(x, 1, self.degree1_high).reshape(Xtf.shape[0], -1) @ w_x
        dZdy = self.chebyshev1(t, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_y
        ddZdxx = self.chebyshev2(x, 1, self.degree1_high).reshape(Xtf.shape[0], -1) @ w_x
        ddZdyy = self.chebyshev2(t, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_y
        tanh1 = 1 - F.tanh(Z) ** 2
        tanh2 = -2 * F.tanh(Z) * tanh1

        dHdZ = self.chebyshev1(F.tanh(Z), self.hidden_nodes, self.degree2)
        ddHdZZ = self.chebyshev2(F.tanh(Z), self.hidden_nodes, self.degree2)

        Hx = dHdZ * (dZdx * tanh1)[:, :, None]
        Hy = dHdZ * (dZdy * tanh1)[:, :, None]
        Hx = Hx.clone().reshape(Xtf.shape[0], -1)
        Hy = Hy.clone().reshape(Xtf.shape[0], -1)
        gradH = Hx + Hy

        Hxx = ddHdZZ * (tanh1 * dZdx)[:, :, None] ** 2 + dHdZ * (tanh2 * dZdx ** 2)[:, :, None] + dHdZ * (tanh1 * ddZdxx)[:, :, None]
        Hyy = ddHdZZ * (tanh1 * dZdy)[:, :, None] ** 2 + dHdZ * (tanh2 * dZdy ** 2)[:, :, None] + dHdZ * (tanh1 * ddZdyy)[:, :, None]
        Hxx = Hxx.clone().reshape(Xtf.shape[0], -1)
        Hyy = Hyy.clone().reshape(Xtf.shape[0], -1)
        LapH = Hxx + Hyy

        return H, Hx, Hy, gradH, Hxx, Hyy, LapH, H_bc, H_ic


def plot(X, pred_vals=None, analy_vals=None, error=None, epoch=None, plot_type='1d', title='', title_pred="", title_analy="", title_error="", save_name='result', save_dir=RESULT_PIC_DIR):
    os.makedirs(save_dir, exist_ok=True)
    x_coord, y_coord, z_coord = 0., 0., 0.
    if X.shape[1] == 1:
        x_coord = X
    else:
        if X.shape[1] == 2:
            x_coord, y_coord = X[:, 0], X[:, 1]
        elif X.shape[1] == 3:
            x_coord, y_coord, z_coord = X[:, 0], X[:, 1], X[:, 2]

    if plot_type == '1d':
        plt.figure(num=1, figsize=(10, 8))
        plt.title(title)
        plt.plot(x_coord, pred_vals, label='Prediction', color='red', linestyle='--', linewidth=4)
        plt.plot(x_coord, analy_vals, label='Analyte', color='blue', linewidth=2)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{save_name}_pred.png"), bbox_inches='tight', dpi=150)
        plt.close()

    elif plot_type == '2d':
        import matplotlib.colors as mcolors
        
        fig1, ax1 = plt.subplots(num=1, figsize=(6, 4.8), dpi=120)
        ax1.set_xlabel('x', fontsize=18)
        ax1.set_ylabel('y', fontsize=18)
        pred_vals = pred_vals.squeeze()
        contour1 = ax1.tricontourf(x_coord, y_coord, pred_vals, levels=100, cmap='turbo', alpha=1)
        fig1.colorbar(contour1, ax=ax1)
        fig1.tight_layout()
        fig1.savefig(os.path.join(save_dir, f"{save_name}_pred.png"), bbox_inches='tight', dpi=150)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(num=2, figsize=(6, 4.8), dpi=120)
        ax2.set_xlabel('x', fontsize=18)
        ax2.set_ylabel('y', fontsize=18)
        error = error.squeeze()
        e_min, e_max = 1e-8, 1
        error_clipped = np.clip(error, e_min, e_max)
        norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
        levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
        cs = ax2.tricontourf(x_coord, y_coord, error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
        cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
        cbar = fig2.colorbar(cs, ax=ax2, ticks=cbar_ticks, format='%.0e')
        cbar.set_label('Absolute Error (log scale)', fontsize=14)
        cbar.ax.set_ylim(e_min, e_max)
        for label in cbar.ax.get_yticklabels():
            label.set_rotation(0)
            label.set_fontsize(10)
        
        ax2.text(0.02, 0.98, f'Max: {np.max(error):.2e}\nMean: {np.mean(error):.2e}', 
                 transform=ax2.transAxes, fontsize=11, 
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig2.tight_layout()
        fig2.savefig(os.path.join(save_dir, f"{save_name}_error_log.png"), bbox_inches='tight', dpi=150)
        plt.close(fig2)
    
    print(f"[plot] Figure saved to '{save_dir}' with prefix '{save_name}'.")


def fit_multifrequency_function_mlpELM(x_inferior=1000, x_bc=300):
    x = np.linspace(0, 1, x_inferior)
    f = lambda x: np.sin(2 * np.pi * x) + 0.1 * np.sin(50 * np.pi * x)
    y = f(x)

    train_x = torch.from_numpy(x[:, None]).to(device=device, dtype=dtype)
    train_y = torch.from_numpy(y[:, None]).to(device=device, dtype=dtype)

    pielm = piELM(1, 500, 1, weight_scale=50., bias_scale=40., device=device, dtype=dtype)
    param = sum(p.numel() for p in pielm.parameters())
    start_tiem = time.time()
    H = F.tanh(train_x @ pielm.W.T + pielm.b[None, :])
    beta = torch.linalg.pinv(H) @ train_y
    pielm.beta.data = beta
    elpased_time = time.time() - start_tiem

    pred = pielm(train_x).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - y[:, np.newaxis]) / np.linalg.norm(y[:, np.newaxis])
    print(f"relative error: {relative_error:.6e}, elpased_time: {elpased_time:.4f}, params: {param}")

    plot(x[:, np.newaxis], pred_vals=pred, analy_vals=y, title='elm')


def fit_multifrequency_function_kanELM(x_inferior=1000, x_bc=300):
    x = np.linspace(0, 1, x_inferior)
    f = lambda x: np.sin(2 * np.pi * x) + 0.1 * np.sin(50 * np.pi * x)
    y = f(x)

    train_x = torch.from_numpy(x[:, None]).to(device=device, dtype=dtype)
    train_y = torch.from_numpy(y[:, None]).to(device=device, dtype=dtype)

    pielmhkan = piELMHKAN(1, 10, 1, num_grid1=20, num_grid2=20, knot_grid1=[-1.5, 1.5], knot_grid2=[-1, 1.], coeff1_sacle=1.5,
                          sigmoid_scale=20., device=device, dtype=dtype)
    param = sum(p.numel() for p in pielmhkan.parameters())
    start_tiem = time.time()
    basis = pielmhkan.sig(train_x, pielmhkan.grid1)
    H1 = basis.reshape(x.shape[0], -1) @ pielmhkan.ctrl
    H = pielmhkan.act2(H1, pielmhkan.grid2).reshape(train_x.shape[0], -1)
    beta = torch.linalg.pinv(H) @ train_y
    pielmhkan.beta.data = beta
    elpased_time = time.time() - start_tiem

    pred = pielmhkan(train_x).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - y[:, np.newaxis]) / np.linalg.norm(y[:, np.newaxis])
    print(f"relative error: {relative_error:.6e}, elpased_time: {elpased_time:.4f}, params: {param}")

    plot(x[:, np.newaxis], pred_vals=pred, analy_vals=y, title='kan')


def fit_F_heat_conduct_kanELM(x_inferior=100, t_inferior=10, bc_weight=1., l2_reg=0.):
    in_dim = 2
    hidden_nodes = 100
    Frequency = 50
    x_ori = np.linspace(0, 1, x_inferior + 2)
    t_ori = np.linspace(0, 1, t_inferior + 2)
    xx, tt = np.meshgrid(x_ori, t_ori, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = np.exp(-xt_ori[:, 1]) * np.sin(Frequency * np.pi * xt_ori[:, 0])

    train_X = torch.from_numpy(xt_ori).to(device=device, dtype=dtype)
    train_y = torch.from_numpy(analy_xtf).to(device=device, dtype=dtype)

    pielmhkan = piELMHKAN(in_dim, hidden_nodes, out_dim=1, num_grid1=50, num_grid2=50, knot_grid1=[-1.5, 1.5], knot_grid2=[-1.0, 1.0], coeff1_sacle=1.0,
                          sigmoid_scale=30., device=device, dtype=dtype)
    pielmhkan_param = sum(p.numel() for p in pielmhkan.parameters())
    start_time = time.time()
    basis = pielmhkan.sig(train_X, pielmhkan.grid1)
    H1 = basis.reshape(xt_ori.shape[0], -1) @ pielmhkan.ctrl
    H = pielmhkan.act2(H1, pielmhkan.grid2).reshape(train_X.shape[0], -1)
    beta = torch.linalg.pinv(H) @ train_y
    pielmhkan.beta.data = beta
    elpased_time = time.time() - start_time

    pred_pielmhkan = pielmhkan(train_X).detach().cpu().numpy()
    relative_error_hkan = np.linalg.norm(pred_pielmhkan - analy_xtf) / np.linalg.norm(analy_xtf)
    print(f"relative error hkan: {relative_error_hkan:.6e}, param:{pielmhkan_param}, train_time:{elpased_time:.4f}")
    error = pred_pielmhkan - analy_xtf

    plot(xt_ori, pred_vals=pred_pielmhkan, analy_vals=analy_xtf, error=error, plot_type='2d', title_pred="kan_pred", title_analy="exact", title_error="kan_error")


def fit_F_heat_conduct_mlpELM(x_inferior=100, t_inferior=10, bc_weight=1., l2_reg=0.):
    in_dim = 2
    hidden_nodes = 5000
    Frequency = 50

    x = np.linspace(0, 1, x_inferior + 2)
    t = np.linspace(0, 1, t_inferior + 2)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = np.exp(-xt_ori[:, 1]) * np.sin(Frequency * np.pi * xt_ori[:, 0])

    train_X = torch.from_numpy(xt_ori).to(device=device, dtype=dtype)
    train_y = torch.from_numpy(analy_xtf).to(device=device, dtype=dtype)

    pielm = piELM(in_dim, hidden_nodes, out_dim=1, weight_scale=30, bias_scale=20., device=device, dtype=dtype)
    param = sum(p.numel() for p in pielm.parameters())
    start_time = time.time()
    H = F.tanh(train_X @ pielm.W.T + pielm.b[None, :])
    beta = torch.linalg.pinv(H) @ train_y
    pielm.beta.data = beta
    elpased_time = time.time() - start_time

    pred = pielm(train_X).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    print(f"relative error: {relative_error:.6e}, param:{param}, train_time:{elpased_time:.4f}")
    error = pred - analy_xtf

    plot(xt_ori, pred_vals=pred, analy_vals=analy_xtf, error=error, plot_type='2d', title_pred="elm_pred", title_analy="exact", title_error="elm_error")


def solve_F_heat_conduct_adaptkanELM(x_inferior=200, t_inferior=20, bc_weight=1., ic_weight=1., l2_reg=1e-5):
    in_dim = 2
    hidden_nodes = 10
    Frequency = 50
    G1_x = 15
    G1_t = 5
    G2 = 10

    x = np.linspace(0, 1, x_inferior)
    t = np.linspace(0, 1, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = np.exp(-xt_ori[:, 1]) * np.sin(Frequency * np.pi * xt_ori[:, 0])
    # xtf = lhs(2, samples=2000, criterion='center')
    x_in, t_in = x[1:-1], t[1:-1]
    xxin, ttin = np.meshgrid(x_in, t_in, indexing='ij')
    xt_in = np.c_[xxin.ravel(), ttin.ravel()]
    xtf = xt_in[np.random.choice(len(xt_in), 1000, False)]
    x_bc_right = np.column_stack([np.ones(300), np.linspace(0, 1, 300)])[np.random.choice(300, 50, replace=False)]
    x_bc_left = np.column_stack([np.zeros(300), np.linspace(0, 1, 300)])[np.random.choice(300, 50, replace=False)]
    x_bc = np.vstack([x_bc_right, x_bc_left])
    T_bc = np.zeros((x_bc.shape[0], 1))
    x_ic = np.column_stack([np.linspace(0, 1, 300), np.zeros(300)])[np.random.choice(300, 50, replace=False)]
    T_ic = np.sin(Frequency * np.pi * np.linspace(0, 1, 300)[:, np.newaxis])[np.random.choice(300, 50, replace=False)]

    scaler = MinMaxScaler((-1, 1))
    all_input = np.vstack([xtf, x_bc, x_ic, xt_ori])
    scaler.fit(all_input)
    xtf_scaled = scaler.transform(xtf)
    x_bc_scaled = scaler.transform(x_bc)
    x_ic_scaled = scaler.transform(x_ic)
    xt_ori_scaled = scaler.transform(xt_ori)

    xtf_t = torch.from_numpy(xtf).to(device=device, dtype=dtype).requires_grad_(True)
    xbc_t = torch.from_numpy(x_bc).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(x_ic).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    model = adapt_piELMHKAN(in_dim,
                            hidden_nodes,
                            out_dim=1,
                            num_grid_diff_x=G1_x,
                            num_grid_diff_t=G1_t,
                            num_grid_rbf=G2,
                            grid_range_diff=[-1.0, 1.0],
                            grid_range_rbf=[-1., 1],
                            sigmoid_scale=2.8,
                            coeff_scale=1.5,
                            device=device,
                            dtype=dtype)

    param = sum(p.numel() for p in model.parameters())
    _, _, Ht, _, Hxx, _, _, H_bc, H_ic = model.Generate_Mat(xtf_t, xbc_t, xic_t)
    start_time = time.time()
    sqrt_w_b = np.sqrt(bc_weight)
    sqrt_w_i = np.sqrt(ic_weight)
    rhs = torch.zeros(xtf.shape[0], 1, device=device, dtype=dtype)
    A = torch.vstack([Ht - (1 / (Frequency * torch.pi) ** 2) * Hxx, sqrt_w_b * H_bc, sqrt_w_i * H_ic])
    # A_norm = torch.linalg.norm(A, dim=0, keepdim=True) + 1e-12
    # A_ = A / A_norm
    b = torch.vstack([rhs, sqrt_w_b * T_bc_t, sqrt_w_i * T_ic_t])
    # beta = torch.linalg.pinv(A) @ b
    U, s, V = torch.linalg.svd(A, full_matrices=False)
    s_reg = torch.diag(s / (s ** 2 + l2_reg))
    beta = (V.T @ s_reg @ U.T @ b) # / A_norm.T
    model.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = torch.mean(((Ht - (1 / (Frequency * torch.pi) ** 2) * Hxx) @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    pred = model(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, pred)
    mae = mean_absolute_error(analy_xtf, pred)
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}, ic_loss:{ic_loss:.6e}, relative_error:{relative_error:.6e}, MAE:{mae:.6e}, mse:{mse:.6e}\n"
          f"param:{param}, train_time:{elapsed_time:.4f}")

    error = pred - analy_xtf[:, np.newaxis]

    x, t = xt_ori_t[:, 0], xt_ori_t[:, 1]
    x_diff = model.DiffSigmoid(x, model.grid_x)
    t_diff = model.DiffSigmoid(t, model.grid_t)

    mid = torch.hstack([x_diff, t_diff]) @ model.ctrl  # (N, hidden_nodes)
    H_full = model.GaussianRBF(mid).reshape(xt_ori_t.shape[0], -1)
    analy_xf_t = torch.from_numpy(analy_xtf).to(device=device, dtype=dtype)
    alpha = torch.linalg.pinv(H_full) @ analy_xf_t
    proj = (H_full @ alpha).detach().cpu().numpy()
    proj_err = np.linalg.norm(proj - analy_xtf) / np.linalg.norm(analy_xtf)
    # compare norms
    print("||beta||, ||alpha||:", torch.norm(beta).item(), torch.norm(alpha).item())
    print("||beta - alpha|| / ||alpha||:", torch.norm(beta - alpha).item() / (torch.norm(alpha).item() + 1e-30))
    A_cpu = A.detach().cpu()
    u, s, v = torch.linalg.svd(A_cpu, full_matrices=False)
    print(f"A sv min/max:{s.min().item():.6e}, {s.max().item():.6e}, cond:{(s.max() / (s.min() + 1e-16)).item():.6e}")
    # also H_full svd
    H_cpu = H_full.detach().cpu()
    uh, sh, vh = torch.linalg.svd(H_cpu, full_matrices=False)
    print("H_full sv min/max:", sh.min().item(), sh.max().item(), "cond:", (sh.max() / (sh.min() + 1e-16)).item())
    C = np.corrcoef(H_cpu.numpy().T)
    off_diag = C.copy()
    np.fill_diagonal(off_diag, 0)
    print("mean abs column-corr:", np.mean(np.abs(off_diag)))

    plot(xt_ori, pred_vals=pred, analy_vals=analy_xtf, error=error, plot_type='2d',title_pred='kan_pred', title_analy='exact', title_error='kan_error')
    return float(relative_error)


def solve_F_heat_conduct_hybridkanELM(x_inferior=300, t_inferior=100, bc_weight=4., ic_weight=4., l2_reg=0.):
    in_dim = 2
    hidden_nodes = 100  # (x_inferior + t_inferior) // 10
    G1 = 100
    G2 = 10
    Frequency = 50
    factor = (1 / (Frequency * torch.pi) ** 2)

    x = np.linspace(0, 1, x_inferior)
    t = np.linspace(0, 1, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = np.exp(-xt_ori[:, 1]) * np.sin(Frequency * np.pi * xt_ori[:, 0])
    # xtf = lhs(2, samples=2000, criterion='center')
    x_in, t_in = x[1:-1], t[1:-1]
    xxin, ttin = np.meshgrid(x_in, t_in, indexing='ij')
    xt_in = np.c_[xxin.ravel(), ttin.ravel()]
    xtf = xt_in[np.random.choice(len(xt_in), 20000, False)]
    x_bc_right = np.column_stack([np.ones(3000), np.linspace(0, 1, 3000)])[np.random.choice(3000, 1000, replace=False)]
    x_bc_left = np.column_stack([np.zeros(3000), np.linspace(0, 1, 3000)])[np.random.choice(3000, 1000, replace=False)]
    x_bc = np.vstack([x_bc_right, x_bc_left])
    T_bc = np.zeros((x_bc.shape[0], 1))
    x_ic = np.column_stack([np.linspace(0, 1, 3000), np.zeros(3000)])[np.random.choice(3000, 2000, replace=False)]
    T_ic = np.sin(Frequency * np.pi * x_ic[:, 0][:, np.newaxis])

    scaler = lambda x: 2 * x - 1
    xtf_scaled = scaler(xtf)
    x_bc_scaled = scaler(x_bc)
    x_ic_scaled = scaler(x_ic)
    xt_ori_scaled = scaler(xt_ori)

    xtf_t = torch.from_numpy(xtf_scaled).to(device=device, dtype=dtype).requires_grad_(True)
    x_bc_t = torch.from_numpy(x_bc_scaled).to(device=device, dtype=dtype)
    x_ic_t = torch.from_numpy(x_ic_scaled).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    pielmhkan = piELMHKAN(in_dim, hidden_nodes, out_dim=1, num_grid1=G1, num_grid2=G2, knot_grid1=[-1, 1], knot_grid2=[-1.0, 1.0], coeff1_sacle=1.0,
                          sigmoid_scale=10.8, device=device, dtype=dtype)
    param = sum(p.numel() for p in pielmhkan.parameters())
    _, _, Ht, _, Hxx, _, _, H_bc, H_ic = pielmhkan.generate_H(xtf_t, x_bc_t, x_ic_t)

    start_time = time.time()
    rhs = torch.zeros(xtf_t.shape[0], 1, device=device, dtype=dtype)
    sqrt_bw = np.sqrt(bc_weight)
    sqrt_iw = np.sqrt(ic_weight)

    A_pde = 2. * Ht - 4. * factor * Hxx
    std_pde = torch.std(A_pde).item() + 1e-8
    std_bc = torch.std(H_bc).item() + 1e-8
    std_ic = torch.std(H_ic).item() + 1e-8
    w_bc = std_pde / std_bc
    w_ic = std_pde / std_ic
    print(f"w_bc={w_bc}, w_ic={w_ic}")

    A = torch.vstack([A_pde, H_bc * w_bc, H_ic * w_ic])
    b = torch.vstack([rhs, T_bc_t * w_bc, T_ic_t * w_ic])
    col_norm = torch.linalg.norm(A, dim=0, keepdim=True)
    col_norm = torch.clamp(col_norm, min=1e-8)
    A_scaled = A / col_norm
    print(A_scaled.shape, b.shape)
    # beta = torch.linalg.pinv(A) @ b
    # U, s, V = torch.linalg.svd(A, full_matrices=False)
    # s_reg = torch.diag(s / (s ** 2 + l2_reg))
    # beta = V.T @ s_reg @ U.T @ b
    I_reg = torch.eye(A.shape[1], device=device, dtype=dtype) * l2_reg
    beta = torch.linalg.solve(A_scaled.T @ A_scaled + I_reg, A_scaled.T @ b) / col_norm.T
    pielmhkan.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = (torch.mean(((Ht - (1 / (Frequency * torch.pi) ** 2) * Hxx) @ beta) ** 2)).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    train_pred = pielmhkan(xt_ori_t).squeeze(-1).detach().cpu().numpy()
    relative_error = np.linalg.norm(train_pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, train_pred)
    mae = mean_absolute_error(analy_xtf, train_pred)
    error = train_pred - analy_xtf
    print(f"error.shape={error.shape}")
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}, ic_loss:{ic_loss:.6e}, relative_error:{relative_error:.6e}, MAE:{mae:.6e}, mse:{mse:.6e}\n"
          f"param:{param}, train_time:{elapsed_time:.4f}")
    #  H_full = pielmhkan.act2(pielmhkan.sig(xt_ori_t, pielmhkan.grid1).reshape(xt_ori_t.shape[0], -1) @ pielmhkan.ctrl,
    #                          pielmhkan.grid2).reshape(xt_ori_t.shape[0], -1)
    #  analy_xf_t = torch.from_numpy(analy_xtf).to(device=device, dtype=dtype)
    #  alpha = torch.linalg.pinv(H_full) @ analy_xf_t
    #  proj = (H_full @ alpha).detach().cpu().numpy()
    #  proj_err = np.linalg.norm(proj - analy_xtf) / np.linalg.norm(analy_xtf)
    #  # compare norms
    #  print("||beta||, ||alpha||:", torch.norm(beta).item(), torch.norm(alpha).item())
    #  print("||beta - alpha|| / ||alpha||:", torch.norm(beta - alpha).item() / (torch.norm(alpha).item() + 1e-30))
    #  A_cpu = A.detach().cpu()
    #  u, s, v = torch.linalg.svd(A_cpu, full_matrices=False)
    #  print(f"A sv min/max:{s.min().item():.6e}, {s.max().item():.6e}, cond:{(s.max() / (s.min() + 1e-16)).item():.6e}")
    #  # also H_full svd
    #  H_cpu = H_full.detach().cpu()
    #  uh, sh, vh = torch.linalg.svd(H_cpu, full_matrices=False)
    #  print("H_full sv min/max:", sh.min().item(), sh.max().item(), "cond:", (sh.max() / (sh.min() + 1e-16)).item())
#
    # C = np.corrcoef(H_cpu.numpy().T)
    # off_diag = C.copy()
    # np.fill_diagonal(off_diag, 0)
    # print("mean abs column-corr:", np.mean(np.abs(off_diag)))
#
    plot(xt_ori, train_pred, analy_xtf, error, plot_type='2d',title_pred='kan_pred', title_analy='exact', title_error='kan_error')
    return float(relative_error)


def solve_F_heat_conduct_kanELM(x_inferior=100, t_inferior=20, bc_weight=1., ic_weight=1., l2_reg=1e-5):
    in_dim = 2
    hidden_nodes = 160
    G = 10
    Frequency = 50

    x = np.linspace(0, 1, x_inferior)
    t = np.linspace(0, 1, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = np.exp(-xt_ori[:, 1]) * np.sin(Frequency * np.pi * xt_ori[:, 0])
    # xtf = lhs(2, samples=2000, criterion='center')
    x_in, t_in = x[1:-1], t[1:-1]
    xxin, ttin = np.meshgrid(x_in, t_in, indexing='ij')
    xt_in = np.c_[xxin.ravel(), ttin.ravel()]
    xtf = xt_in[np.random.choice(len(xt_in), 1000, False)]
    x_bc_right = np.column_stack([np.ones(300), np.linspace(0, 1, 300)])[np.random.choice(300, 200, replace=False)]
    x_bc_left = np.column_stack([np.zeros(300), np.linspace(0, 1, 300)])[np.random.choice(300, 200, replace=False)]
    x_bc = np.vstack([x_bc_right, x_bc_left])
    T_bc = np.zeros((x_bc.shape[0], 1))
    x_ic = np.column_stack([np.linspace(0, 1, 300), np.zeros(300)])[np.random.choice(300, 200, replace=False)]
    T_ic = np.sin(Frequency * np.pi * np.linspace(0, 1, 300)[:, np.newaxis])[np.random.choice(300, 200, replace=False)]

    scaler = MinMaxScaler((-1, 1))
    all_input = np.vstack([xtf, x_bc, x_ic, xt_ori])
    scaler.fit(all_input)
    xtf_scaled = scaler.transform(xtf)
    x_bc_scaled = scaler.transform(x_bc)
    x_ic_scaled = scaler.transform(x_ic)
    xt_ori_scaled = scaler.transform(xt_ori)

    xtf_t = torch.from_numpy(xtf).to(device=device, dtype=dtype).requires_grad_(True)
    x_bc_t = torch.from_numpy(x_bc).to(device=device, dtype=dtype)
    x_ic_t = torch.from_numpy(x_ic).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    print(f"xtf_f.shape={xtf_t.shape}, model=[2, {hidden_nodes * G}, {hidden_nodes * G}, 1]")
    pielmhkan = piELMKAN(in_dim, hidden_nodes, 1, grid_range=[-1, 1.], num_grid=G, coeff_scale=1.5, device=device, dtype=dtype)
    param = sum(p.numel() for p in pielmhkan.parameters())
    _, _, Ht, _, Hxx, _, _, H_bc, H_ic = pielmhkan.generate_H(xtf_t, x_bc_t, x_ic_t)

    start_time = time.time()
    rhs = torch.zeros(xtf_t.shape[0], 1, device=device, dtype=dtype)
    sqrt_bw = np.sqrt(bc_weight)
    sqrt_iw = np.sqrt(ic_weight)
    A = torch.vstack([Ht - (1 / (Frequency * torch.pi) ** 2) * Hxx, H_bc * sqrt_bw, H_ic * sqrt_iw])
    b = torch.vstack([rhs, T_bc_t * sqrt_bw, T_ic_t * sqrt_iw])
    # beta = torch.linalg.pinv(A) @ b
    U, s, V = torch.linalg.svd(A, full_matrices=False)
    s_reg = torch.diag(s / (s ** 2 + l2_reg))
    beta = V.T @ s_reg @ U.T @ b
    pielmhkan.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = (torch.mean(((Ht - (1 / (Frequency * torch.pi) ** 2) * Hxx) @ beta) ** 2)).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    train_pred = pielmhkan(xt_ori_t).squeeze(-1).detach().cpu().numpy()
    relative_error = np.linalg.norm(train_pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, train_pred)
    mae = mean_absolute_error(analy_xtf, train_pred)
    error = train_pred - analy_xtf
    print(f"error.shape={error.shape}")
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}, ic_loss:{ic_loss:.6e}, relative_error:{relative_error:.6e}, MAE:{mae:.6e}, mse:{mse:.6e}\n"
          f"param:{param}, train_time:{elapsed_time:.4f}")
    # H_full = pielmhkan.act2(pielmhkan.sig(xt_ori_t, pielmhkan.grid1).reshape(xt_ori_t.shape[0], -1) @ pielmhkan.ctrl,
    #                         pielmhkan.grid2).reshape(xt_ori_t.shape[0], -1)
    # analy_xf_t = torch.from_numpy(analy_xtf).to(device=device, dtype=dtype)
    # alpha = torch.linalg.pinv(H_full) @ analy_xf_t
    # proj = (H_full @ alpha).detach().cpu().numpy()
    # proj_err = np.linalg.norm(proj - analy_xtf) / np.linalg.norm(analy_xtf)
    # # compare norms
    # print("||beta||, ||alpha||:", torch.norm(beta).item(), torch.norm(alpha).item())
    # print("||beta - alpha|| / ||alpha||:", torch.norm(beta - alpha).item() / (torch.norm(alpha).item() + 1e-30))
    # A_cpu = A.detach().cpu()
    # u, s, v = torch.linalg.svd(A_cpu, full_matrices=False)
    # print(f"A sv min/max:{s.min().item():.6e}, {s.max().item():.6e}, cond:{(s.max() / (s.min() + 1e-16)).item():.6e}")
    # # also H_full svd
    # H_cpu = H_full.detach().cpu()
    # uh, sh, vh = torch.linalg.svd(H_cpu, full_matrices=False)
    # print("H_full sv min/max:", sh.min().item(), sh.max().item(), "cond:", (sh.max() / (sh.min() + 1e-16)).item())
##
    # C = np.corrcoef(H_cpu.numpy().T)
    # off_diag = C.copy()
    # np.fill_diagonal(off_diag, 0)
    # print("mean abs column-corr:", np.mean(np.abs(off_diag)))
#
    plot(xt_ori, train_pred, analy_xtf, error, plot_type='2d',title_pred='kan_pred', title_analy='exact', title_error='kan_error')
    return float(relative_error)


def solve_F_heat_conduct_mlpELM(x_inferior=100, t_inferior=50, bc_weight=1., ic_weight=1., l2_reg=0.):
    in_dim = 2
    hidden_nodes = 5000
    Frequency = 50

    x = np.linspace(0, 1, x_inferior)
    t = np.linspace(0, 1, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = np.exp(-xt_ori[:, 1]) * np.sin(Frequency * np.pi * xt_ori[:, 0])
    # xtf = lhs(2, samples=2000, criterion='center')
    x_in, t_in = x[1:-1], t[1:-1]
    xxin, ttin = np.meshgrid(x_in, t_in, indexing='ij')
    xt_in = np.c_[xxin.ravel(), ttin.ravel()]
    xtf = xt_in[np.random.choice(len(xt_in), 1000, False)]
    x_bc_right = np.column_stack([np.ones(300), np.linspace(0, 1, 300)])[np.random.choice(300, 200, replace=False)]
    x_bc_left = np.column_stack([np.zeros(300), np.linspace(0, 1, 300)])[np.random.choice(300, 200, replace=False)]
    x_bc = np.vstack([x_bc_right, x_bc_left])
    T_bc = np.zeros((x_bc.shape[0], 1))
    x_ic = np.column_stack([np.linspace(0, 1, 300), np.zeros(300)])[np.random.choice(300, 200, replace=False)]
    T_ic = np.sin(Frequency * np.pi * np.linspace(0, 1, 300)[:, np.newaxis])[np.random.choice(300, 200, replace=False)]

    scaler = MinMaxScaler((-1, 1))
    all_input = np.vstack([xtf, x_bc, x_ic, xt_ori])
    scaler.fit(all_input)
    xtf_scaled = scaler.transform(xtf)
    x_bc_scaled = scaler.transform(x_bc)
    x_ic_scaled = scaler.transform(x_ic)
    xt_ori_scaled = scaler.transform(xt_ori)

    xtf_t = torch.from_numpy(xtf).to(device=device, dtype=dtype).requires_grad_(True)
    x_bc_t = torch.from_numpy(x_bc).to(device=device, dtype=dtype)
    x_ic_t = torch.from_numpy(x_ic).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    pielm = piELM(in_dim, hidden_nodes, out_dim=1, weight_scale=30, bias_scale=20, device=device, dtype=dtype)
    param = sum(p.numel() for p in pielm.parameters())
    _, _, Ht, _, Hxx, _, _, H_bc, H_ic = pielm.generate_H_tanh(xtf_t, x_bc_t, x_ic_t)

    start_time = time.time()
    rhs = torch.zeros(xtf_t.shape[0], 1, device=device, dtype=dtype)
    sqrt_w_b = np.sqrt(bc_weight)
    sqrt_w_i = np.sqrt(ic_weight)
    # print(Ht.shape, Hxx.shape, x_bc_t.shape, x_ic_t.shape)
    A = torch.vstack([Ht - (1 / (Frequency * torch.pi) ** 2) * Hxx, H_bc * sqrt_w_b, H_ic ** sqrt_w_i])
    b = torch.vstack([rhs, T_bc_t * sqrt_w_b, T_ic_t * sqrt_w_i])
    beta = torch.linalg.pinv(A) @ b
    pielm.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = torch.mean(((Ht - (1 / (Frequency * torch.pi) ** 2) * Hxx) @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    pred = pielm(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, pred)
    mae = mean_absolute_error(analy_xtf, pred)
    error = pred - analy_xtf
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}, ic_loss:{ic_loss:.6e}, relative_error:{relative_error:.6e}, MAE:{mae:.6e}, mse:{mse:.6e}\n"
          f"param:{param}, train_time:{elapsed_time:.4f}")

    plot(xt_ori, pred_vals=pred, analy_vals=analy_xtf, error=error, plot_type='2d', title_pred='elm_pred', title_analy='exact', title_error='elm_error')
    return float(relative_error)


def solve_F_heat_conduct_pikan(x_inferior=300, t_inferior=100, bc_weight=1., ic_weight=1., l2_reg=1e-8):
    width = [2, 10, 10, 1]
    Frequency = 50

    x = np.linspace(0, 1, x_inferior)
    t = np.linspace(0, 1, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = (np.exp(-xt_ori[:, 1]) * np.sin(Frequency * np.pi * xt_ori[:, 0]))[:, np.newaxis]
    N_x_train = 300
    N_t_train = 100
    x_cgl = 0.5 * (1 - np.cos(np.pi * np.arange(N_x_train) / (N_x_train - 1)))
    t_uni = np.linspace(0, 1, N_t_train)
    xx_train, tt_train = np.meshgrid(x_cgl, t_uni, indexing='ij')
    xtf = np.c_[xx_train.ravel(), tt_train.ravel()]
    N_bc = 2000
    x_cgl_bc = 0.5 * (1 - np.cos(np.pi * np.arange(N_bc) / (N_bc - 1)))
    x_bc_right = np.column_stack([np.ones(2000), x_cgl_bc])
    x_bc_left = np.column_stack([np.zeros(2000), x_cgl_bc])
    x_bc = np.vstack([x_bc_right, x_bc_left])
    T_bc = np.zeros((x_bc.shape[0], 1))
    N_ic = 2000
    x_cgl_ic = 0.5 * (1 - np.cos(np.pi * np.arange(N_ic) / (N_ic - 1)))
    x_ic = np.c_[x_cgl_ic, np.zeros(N_ic)]
    T_ic = np.sin(Frequency * np.pi * x_ic[:, 0][:, np.newaxis])

    scaler = lambda x: 2 * x - 1
    xtf_scaled = scaler(xtf)
    x_bc_scaled = scaler(x_bc)
    x_ic_scaled = scaler(x_ic)
    xt_ori_scaled = scaler(xt_ori)

    xtf_t = torch.from_numpy(xtf_scaled).to(device=device, dtype=dtype).requires_grad_(True)
    x_bc_t = torch.from_numpy(x_bc_scaled).to(device=device, dtype=dtype)
    x_ic_t = torch.from_numpy(x_ic_scaled).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    pikan = efficentKAN(width,
                        grid_size=100,
                        spline_order=3,
                        scale_noise=0.1,
                        scale_base=1,
                        scale_spline=1.0,
                        base_activation=torch.nn.SiLU,
                        grid_eps=0.02,
                        grid_range=[-1, 1]).to(device=device, dtype=dtype)
    optimizer1 = torch.optim.Adam(pikan.parameters(), lr=1e-2)
    # optimizer2 = torch.optim.LBFGS(pikan.parameters(), lr=0.1)
    iteration = 5000
    loss_fn = nn.MSELoss()

    history = {
        'pde_loss': [], "bc_loss": [], 'ic_loss': [], 'total_loss': [], 'l1_reg':[]
    }

    start_time = time.time()

    for it in range(iteration):
        optimizer1.zero_grad()
        output_xtf = pikan(xtf_t)
        dH = torch.autograd.grad(output_xtf.sum(), xtf_t, create_graph=True)[0]
        Ht, Hx = dH[:, 1], dH[:, 0]
        Hxx = torch.autograd.grad(dH.sum(), xtf_t, create_graph=True)[0][:, 0]
        pde_loss = torch.mean((2. * Ht - 4. * (1 / (Frequency * torch.pi) ** 2) * Hxx) ** 2)
        output_b = pikan(x_bc_t)
        bc_loss = loss_fn(output_b, T_bc_t)
        output_i = pikan(x_ic_t)
        ic_loss = loss_fn(output_i, T_ic_t)

        l1_reg = pikan.regularization_loss(1., 1.)
        total_loss = pde_loss + bc_weight * bc_loss + ic_weight * ic_loss + 1e-5 * l1_reg
        total_loss.backward()
        optimizer1.step()

        history['pde_loss'].append(pde_loss.item())
        history['bc_loss'].append(bc_loss.item())
        history['ic_loss'].append(ic_loss.item())
        history['total_loss'].append(total_loss.item())
        history['l1_reg'].append(l1_reg.item())
        print(f"Iteration:{it}, total_loss:{total_loss:.6e},  pde_loss:{pde_loss:.6e}, bc_loss:{bc_loss:.6e}, ic_loss:{ic_loss:.6e}, ")

    elapsed_time = time.time() - start_time
    pred = pikan(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, pred)
    mae = mean_absolute_error(analy_xtf, pred)
    print(f"Relative error: {relative_error:.6e}, mse: {mse:.6e}, mae: {mae:.6e}, train_time:{elapsed_time:.4f}")
    error = np.abs(pred - analy_xtf)

    epoch_line = np.linspace(1, iteration + 1, iteration)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.plot(epoch_line, history['pde_loss'], color='red', label='pde_loss')
    plt.plot(epoch_line, history['bc_loss'], color='blue', label='bc_loss')
    plt.plot(epoch_line, history['ic_loss'], color='green', label='ic_loss')
    plt.plot(epoch_line, history['total_loss'], color='orange', label='total_loss')
    # plt.plot(epoch_line, history['l1_reg'], color='black', label='l1_reg')
    plt.legend()
    plt.show()

    plot(xt_ori, pred_vals=pred, analy_vals=analy_xtf, error=error, plot_type='2d', save_name='heat_conduct_pikan')
    return float(relative_error)


def solve_F_heat_conduct_pinn(x_inferior=300, t_inferior=100, bc_weight=1., ic_weight=1., l2_reg=1e-8):
    Frequency = 50

    x = np.linspace(0, 1, x_inferior)
    t = np.linspace(0, 1, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = (np.exp(-xt_ori[:, 1]) * np.sin(Frequency * np.pi * xt_ori[:, 0]))[:, np.newaxis]
    N_x_train = 300
    N_t_train = 100
    x_cgl = 0.5 * (1 - np.cos(np.pi * np.arange(N_x_train) / (N_x_train - 1)))
    t_uni = np.linspace(0, 1, N_t_train)
    xx_train, tt_train = np.meshgrid(x_cgl, t_uni, indexing='ij')
    xtf = np.c_[xx_train.ravel(), tt_train.ravel()]
    N_bc = 2000
    x_cgl_bc = 0.5 * (1 - np.cos(np.pi * np.arange(N_bc) / (N_bc - 1)))
    x_bc_right = np.column_stack([np.ones(2000), x_cgl_bc])
    x_bc_left = np.column_stack([np.zeros(2000), x_cgl_bc])
    x_bc = np.vstack([x_bc_right, x_bc_left])
    T_bc = np.zeros((x_bc.shape[0], 1))
    N_ic = 2000
    x_cgl_ic = 0.5 * (1 - np.cos(np.pi * np.arange(N_ic) / (N_ic - 1)))
    x_ic = np.c_[x_cgl_ic, np.zeros(N_ic)]
    T_ic = np.sin(Frequency * np.pi * x_ic[:, 0][:, np.newaxis])

    scaler = lambda x: 2 * x - 1
    xtf_scaled = scaler(xtf)
    x_bc_scaled = scaler(x_bc)
    x_ic_scaled = scaler(x_ic)
    xt_ori_scaled = scaler(xt_ori)

    xtf_t = torch.from_numpy(xtf_scaled).to(device=device, dtype=dtype).requires_grad_(True)
    x_bc_t = torch.from_numpy(x_bc_scaled).to(device=device, dtype=dtype)
    x_ic_t = torch.from_numpy(x_ic_scaled).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    pinn = PINN(2, 1, [30, 30, 30, 30], act='relu', device=device, dtype=dtype)
    optimizer1 = torch.optim.Adam(pinn.parameters(), lr=1e-2)
    # optimizer2 = torch.optim.LBFGS(pikan.parameters(), lr=0.1)
    iteration = 3000
    loss_fn = nn.MSELoss()

    start_time = time.time()

    for it in range(iteration):
        optimizer1.zero_grad()
        output_xtf = pinn(xtf_t)
        dH = torch.autograd.grad(output_xtf.sum(), xtf_t, create_graph=True)[0]
        Ht, Hx = dH[:, 1], dH[:, 0]
        Hxx = torch.autograd.grad(dH.sum(), xtf_t, create_graph=True)[0][:, 0]
        pde_loss = torch.mean((2. * Ht - 4. * (1 / (Frequency * torch.pi) ** 2) * Hxx) ** 2)
        output_b = pinn(x_bc_t)
        bc_loss = loss_fn(output_b, T_bc_t)
        output_i = pinn(x_ic_t)
        ic_loss = loss_fn(output_i, T_ic_t)

        # l1_reg = pinn.regularization_loss(1., 1.)
        total_loss = pde_loss + bc_weight * bc_loss + ic_weight * ic_loss  # + 1e-5 * l1_reg
        total_loss.backward()
        optimizer1.step()
        print(f"Iteration:{it}, total_loss:{total_loss:.6e},  pde_loss:{pde_loss:.6e}, bc_loss:{bc_loss:.6e}, ic_loss:{ic_loss:.6e}, ")

    elapsed_time = time.time() - start_time
    pred = pinn(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, pred)
    mae = mean_absolute_error(analy_xtf, pred)
    print(f"Relative error: {relative_error:.6e}, mse: {mse:.6e}, mae: {mae:.6e}, train_time:{elapsed_time:.4f}")
    error = np.abs(pred - analy_xtf)

    plot(xt_ori, analy_xtf, error, plot_type='2d', )
    return float(relative_error)


def solve_crack_PIKAN(x_inferior=1000, y_inferior=50, bc_weight=1., ic_weight=1.):
    width = [2, 10, 10, 1]

    def getSourceItem(x):
        r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        theta = np.arctan2(x[:, 1], x[:, 0])
        return np.sqrt(r) * np.sin(1 / 2 * theta)

    x_ori = np.linspace(-1, 1, 100)
    y_ori_up = np.linspace(0, 1, 100)
    y_ori_down = np.linspace(-1, -1e-8, 100)
    xx, yyup = np.meshgrid(x_ori, y_ori_up, indexing='ij')
    xy_ori_up = np.c_[xx.ravel(), yyup.ravel()]
    xx, yydown = np.meshgrid(x_ori, y_ori_down, indexing='ij')
    xy_ori_down = np.c_[xx.ravel(), yydown.ravel()]
    xy_ori = np.vstack([xy_ori_up, xy_ori_down])
    analy_xy = getSourceItem(xy_ori)[:, np.newaxis]
    y_up = np.linspace(0, 1, 100)
    x_up_in, y_up_in = x_ori[1:-1], y_up[1:-1]
    xui, yui = np.meshgrid(x_up_in, y_up_in, indexing='ij')
    xy_up_in = np.c_[xui.ravel(), yui.ravel()]
    xy_up_sample = xy_up_in[np.random.choice(len(xy_up_in), size=2048, replace=False)]
    y_low = np.linspace(-1, 0, 100)
    x_low_in, y_low_in = x_ori[1:-1], y_low[1:-1]
    xli, yli = np.meshgrid(x_low_in, y_low_in, indexing='ij')
    xy_low_in = np.c_[xli.ravel(), yli.ravel()]
    xy_low_sample = xy_low_in[np.random.choice(len(xy_low_in), size=2048, replace=False)]
    bc_up_up = np.column_stack([np.linspace(-1, 1, 500), np.ones(500)])[np.random.choice(500, 256, replace=False)]
    bc_up_down = np.column_stack([np.linspace(-1, 0, 500), np.full(500, 1e-8)])[
        np.random.choice(500, 256, replace=False)]
    bc_up_right = np.column_stack([np.ones(500), np.linspace(0, 1, 500)])[np.random.choice(500, 128, replace=False)]
    bc_up_left = np.column_stack([-np.ones(500), np.linspace(0, 1, 500)])[np.random.choice(500, 128, replace=False)]
    bc_up = np.vstack([bc_up_up, bc_up_down, bc_up_right, bc_up_left])
    T_bc_up = getSourceItem(bc_up)[:, np.newaxis]
    bc_down_up = np.column_stack([np.linspace(-1, 1, 500), -np.ones(500)])[np.random.choice(500, 256, replace=False)]
    bc_down_down = np.column_stack([np.linspace(-1, 0, 500), np.full(500, -1e-8)])[
        np.random.choice(500, 256, replace=False)]
    bc_down_right = np.column_stack([np.ones(500), np.linspace(-1, 0, 500)])[np.random.choice(500, 128, replace=False)]
    bc_down_left = np.column_stack([-np.ones(500), np.linspace(-1, 0, 500)])[np.random.choice(500, 128, replace=False)]
    bc_down = np.vstack([bc_down_up, bc_down_down, bc_down_right, bc_down_left])
    T_bc_down = getSourceItem(bc_down)[:, np.newaxis]
    random_id_inter = np.random.choice(3000, 1000, replace=False)
    xy_intface_sample = np.column_stack([np.linspace(0, 1, 3000), np.zeros(3000)])[random_id_inter]

    xy_up_sample_t = torch.from_numpy(xy_up_sample).to(device=device, dtype=dtype).requires_grad_(True)
    xy_down_sample_t = torch.from_numpy(xy_low_sample).to(device=device, dtype=dtype).requires_grad_(True)
    xy_bc_up_sample_t = torch.from_numpy(bc_up).to(device=device, dtype=dtype).requires_grad_(True)
    xy_bc_down_sample_t = torch.from_numpy(bc_down).to(device=device, dtype=dtype).requires_grad_(True)
    xy_intface_sample_t = torch.from_numpy(xy_intface_sample).to(device=device, dtype=dtype).requires_grad_(True)
    T_bc_up_t = torch.from_numpy(T_bc_up).to(device=device, dtype=dtype)
    T_bc_down_t = torch.from_numpy(T_bc_down).to(device=device, dtype=dtype)
    xy_ori_up_t = torch.from_numpy(xy_ori_up).to(device=device, dtype=dtype)
    xy_ori_down_t = torch.from_numpy(xy_ori_down).to(device=device, dtype=dtype)

    model1 = efficentKAN(width,
                         grid_size=10,
                         spline_order=3,
                         scale_noise=0.1,
                         scale_base=1.0,
                         scale_spline=1.0,
                         base_activation=torch.nn.SiLU,
                         grid_eps=0.02,
                         grid_range=[-1, 1]).to(device=device, dtype=dtype)

    model2 = efficentKAN(width,
                         grid_size=10,
                         spline_order=3,
                         scale_noise=0.1,
                         scale_base=1.0,
                         scale_spline=1.0,
                         base_activation=torch.nn.SiLU,
                         grid_eps=0.02,
                         grid_range=[-1, 1]).to(device=device, dtype=dtype)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    iteration = 3000
    loss_fn = nn.MSELoss()
    n_vec = (0, 1)

    start_time = time.time()
   
    def get_pde_bc_loss(model, xy, xy_bc, Tbc):
        output_xtf = model(xy)
        dH = torch.autograd.grad(output_xtf.sum(), xy, create_graph=True)[0]
        Hx, Hy = dH[:, 0], dH[:, 1]
        Hxx = torch.autograd.grad(Hx.sum(), xy, create_graph=True)[0][:, 0]
        Hyy = torch.autograd.grad(Hy.sum(), xy, create_graph=True)[0][:, 1]
        LapH = Hxx + Hyy
        pde_loss = torch.mean(LapH ** 2)  # pde loss
        output_b = model(xy_bc)
        bc_loss = loss_fn(output_b, Tbc)  # bc_loss
        return pde_loss, bc_loss

    for it in range(iteration):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        # discrete domain
        pde_loss_up, bc_loss_up = get_pde_bc_loss(model1, xy_up_sample_t, xy_bc_up_sample_t, T_bc_up_t)
        pde_loss_down, bc_loss_down = get_pde_bc_loss(model2, xy_down_sample_t, xy_bc_down_sample_t, T_bc_down_t)
        pde_loss = pde_loss_up + pde_loss_down
        bc_loss = bc_loss_up + bc_loss_down
        # interface continuity
        output_intf_up = model1(xy_intface_sample_t) 
        output_intf_down = model2(xy_intface_sample_t)
        avg_loss = loss_fn(output_intf_up, output_intf_down)  # avg loss
        dH_up = torch.autograd.grad(output_intf_up.sum(), xy_intface_sample_t, create_graph=True)[0]
        Hx_up, Hy_up = dH_up[:, 0], dH_up[:, 1]
        gradH_up = Hx_up * n_vec[0] + Hy_up * n_vec[1]
        dH_down = torch.autograd.grad(output_intf_down.sum(), xy_intface_sample_t, create_graph=True)[0]
        Hx_down, Hy_down = dH_down[:, 0], dH_down[:, 1]
        gradH_down = Hx_down * n_vec[0] + Hy_down * n_vec[1]
        flux_loss = loss_fn(gradH_up, gradH_down)  # flux loss

        l1_reg_up = model1.regularization_loss(1., 1.)
        l1_reg_down = model2.regularization_loss(1., 1.)
        total_loss = pde_loss + 50 * bc_loss + 10 * (avg_loss + flux_loss) + 1e-5 * (l1_reg_up + l1_reg_down)
        total_loss.backward()
        optimizer1.step()
        optimizer2.step()
        if it % 500 == 0:
            print(
                f"Iteration:{it}, total_loss:{total_loss:.6e},  pde_loss:{pde_loss:.6e}, bc_loss:{bc_loss:.6e}, avg_loss:{avg_loss:.6e}, flux_loss: {flux_loss:.6e} ")

    elapsed_time = time.time() - start_time
    pred_up = model1(xy_ori_up_t).detach().cpu().numpy()
    pred_down = model2(xy_ori_down_t).detach().cpu().numpy()
    pred = np.concatenate([pred_up, pred_down], axis=0)
    relative_error = np.linalg.norm(pred - analy_xy) / np.linalg.norm(analy_xy)
    mse = mean_squared_error(analy_xy, pred)
    mae = mean_absolute_error(analy_xy, pred)
    print(f"Relative error: {relative_error:.6e}, mse: {mse:.6e}, mae: {mae:.6e}, train_time:{elapsed_time:.4f}")
    error = np.abs(pred - analy_xy)

    plot(xy_ori, pred_vals=pred, analy_vals=analy_xy, error=error, plot_type='2d', save_name='crack_pikan')
    return float(relative_error)


def solve_wave_efficientKAN(x_inferior=100, t_inferior=100, bc_weight=1., ic_weight=1., icgrad_weight=1.):
    width = [2, 10, 10, 1]

    # ==========================
    # ==========================
    x = np.linspace(0, 1, x_inferior)
    t = np.linspace(0, 1, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    # u(x, t) = sin(πx) cos(10πt) + sin(2πx) cos(20πt)
    analy_xtf = (np.sin(np.pi * xt_ori[:, 0]) * np.cos(np.pi * 10. * xt_ori[:, 1]) + np.sin(
        np.pi * 2 * xt_ori[:, 0]) * np.cos(np.pi * 20. * xt_ori[:, 1]))[:, np.newaxis]

    N_x_train = 100
    N_t_train = 300
    t_cgl = 0.5 * (1 - np.cos(np.pi * np.arange(N_t_train) / (N_t_train - 1)))
    x_uni = np.linspace(0, 1, N_x_train)
    xx_train, tt_train = np.meshgrid(x_uni, t_cgl, indexing='ij')
    xtf = np.c_[xx_train.ravel(), tt_train.ravel()]

    N_bc = 2000
    t_cgl_bc = 0.5 * (1 - np.cos(np.pi * np.arange(N_bc) / (N_bc - 1)))
    x_bc_left = np.column_stack([np.zeros(N_bc), t_cgl_bc])
    x_bc_right = np.column_stack([np.ones(N_bc), t_cgl_bc])
    x_bc = np.vstack([x_bc_left, x_bc_right])
    T_bc = np.zeros((x_bc.shape[0], 1))

    N_ic = 2000
    x_cgl_ic = 0.5 * (1 - np.cos(np.pi * np.arange(N_ic) / (N_ic - 1)))
    x_ic = np.column_stack([x_cgl_ic, np.zeros(N_ic)])
    T_ic = np.sin(np.pi * x_ic[:, 0][:, np.newaxis]) + np.sin(np.pi * 2 * x_ic[:, 0][:, np.newaxis])

    scaler = lambda x: 2 * x - 1
    xtf_scaled = scaler(xtf)
    xbc_scaled = scaler(x_bc)
    xic_scaled = scaler(x_ic)
    xt_ori_scaled = scaler(xt_ori)

    xtf_t = torch.from_numpy(xtf_scaled).to(device=device, dtype=dtype).requires_grad_(True)
    xbc_t = torch.from_numpy(xbc_scaled).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(xic_scaled).to(device=device, dtype=dtype).requires_grad_(True)
    Tbc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    Tic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)
    analy_xtf_t = torch.from_numpy(analy_xtf).to(device=device, dtype=dtype)

    # ==========================
    # ==========================
    model = efficentKAN(width,
                        grid_size=50,
                        spline_order=3,
                        scale_noise=0.01,
                        scale_base=1,
                        scale_spline=1.0,
                        base_activation=torch.nn.SiLU,
                        grid_eps=0.02,
                        grid_range=[-1, 1]).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    loss_fn = nn.MSELoss()
    start_time = time.time()
    print("[efficientKAN] Start phase 1 to train on Adam...")
    for it in range(3000):
        optimizer.zero_grad()

        output_xtf = model(xtf_t)
        dH = torch.autograd.grad(output_xtf.sum(), xtf_t, create_graph=True)[0]
        Hxx = torch.autograd.grad(dH[:, 0].sum(), xtf_t, create_graph=True)[0][:, 0]
        Htt = torch.autograd.grad(dH[:, 1].sum(), xtf_t, create_graph=True)[0][:, 1]

        pde_loss = torch.mean((4. * Htt - 4. * 100. * Hxx) ** 2)

        output_b = model(xbc_t)
        bc_loss = loss_fn(output_b, Tbc_t)

        output_i = model(xic_t)
        ic_loss = loss_fn(output_i, Tic_t)

        dH_ic = torch.autograd.grad(output_i.sum(), xic_t, create_graph=True)[0]
        Ht_ic = dH_ic[:, 1]
        icgrad_loss = torch.mean((2. * Ht_ic) ** 2)

        l1_reg = model.regularization_loss(1., 1.)

        total_loss = pde_loss + bc_weight * bc_loss + ic_weight * ic_loss + icgrad_weight * icgrad_loss + 1e-8 * l1_reg
        total_loss.backward()
        optimizer.step()

        if it % 500 == 0:
            with torch.no_grad():
                pred = model(xt_ori_t)
                relative_error = torch.linalg.norm(pred - analy_xtf_t) / torch.linalg.norm(analy_xtf_t)
            print(f"It: {it:05d}, pde_loss: {pde_loss.item():.4e}, bc_loss: {bc_loss.item():.4e}, "
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
        output_xtf = model(xtf_t)
        dH = torch.autograd.grad(output_xtf.sum(), xtf_t, create_graph=True)[0]
        Hxx = torch.autograd.grad(dH[:, 0].sum(), xtf_t, create_graph=True)[0][:, 0]
        Htt = torch.autograd.grad(dH[:, 1].sum(), xtf_t, create_graph=True)[0][:, 1]
        pde_loss = torch.mean((1 / 100. * Htt - Hxx) ** 2)
        output_b = model(xbc_t)
        bc_loss = loss_fn(output_b, Tbc_t)
        output_i = model(xic_t)
        ic_loss = loss_fn(output_i, Tic_t)
        dH_ic = torch.autograd.grad(output_i.sum(), xic_t, create_graph=True)[0]
        Ht_ic = dH_ic[:, 1]
        icgrad_loss = torch.mean((2. * Ht_ic) ** 2)
        l1_reg = model.regularization_loss(1., 1.)
        total_loss = pde_loss + bc_weight * bc_loss + ic_weight * ic_loss + icgrad_weight * icgrad_loss + 1e-5 * l1_reg
        total_loss.backward()

        return total_loss

    optimizer_lbfgs.step(closure)

    elapsed_time = time.time() - start_time

    # ==========================
    # ==========================
    pred = model(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(analy_xtf, pred)
    mae = mean_absolute_error(analy_xtf, pred)
    error = np.abs(pred - analy_xtf)

    print("-" * 50)
    print(f"Training Time: {elapsed_time:.4f}s")
    print(f"Relative error: {relative_error:.6e}")
    print(f"MSE: {mse:.6e}, MAE: {mae:.6e}")
    print("-" * 50)

    plot(xt_ori, pred_vals=pred, analy_vals=analy_xtf, error=error, plot_type='2d', save_name='wave_pikan')
    return float(relative_error)



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # seed = 1024
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    RANDOM_SEEDS = [1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030]
    SEED_RESULT_DIR = "25-CMAME-Original_PIKAN_randomseed"
    os.makedirs(SEED_RESULT_DIR, exist_ok=True)

    CASE_REGISTRY = {
        "solve_F_heat_conduct_adaptkanELM": solve_F_heat_conduct_adaptkanELM,
        "solve_F_heat_conduct_hybridkanELM": solve_F_heat_conduct_hybridkanELM,
        "solve_F_heat_conduct_kanELM": solve_F_heat_conduct_kanELM,
        "solve_F_heat_conduct_mlpELM": solve_F_heat_conduct_mlpELM,
        "solve_F_heat_conduct_pikan": solve_F_heat_conduct_pikan,
        "solve_F_heat_conduct_pinn": solve_F_heat_conduct_pinn,
        "solve_crack_PIKAN": solve_crack_PIKAN,
        "solve_wave_efficientKAN": solve_wave_efficientKAN,
    }

    ACTIVE_CASE = "solve_F_heat_conduct_pikan"
    # ACTIVE_CASE = "solve_F_heat_conduct_adaptkanELM"
    # ACTIVE_CASE = "solve_F_heat_conduct_hybridkanELM"
    # ACTIVE_CASE = "solve_F_heat_conduct_kanELM"
    # ACTIVE_CASE = "solve_F_heat_conduct_mlpELM"
    # ACTIVE_CASE = "solve_F_heat_conduct_pinn"
    # ACTIVE_CASE = "solve_crack_PIKAN"
    # ACTIVE_CASE = "solve_wave_efficientKAN"

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

    # fit_multifrequency_function_mlpELM(200)
    # fit_multifrequency_function_kanELM(200)
    # fit_F_heat_conduct_kanELM(x_inferior=500, t_inferior=10, bc_weight=1., l2_reg=1e-3)
    # fit_F_heat_conduct_mlpELM(x_inferior=500, t_inferior=10, bc_weight=1., l2_reg=0.)
    # solve_F_heat_conduct_adaptkanELM()
    # solve_F_heat_conduct_kanELM()
    # solve_F_heat_conduct_hybridkanELM()
    # solve_F_heat_conduct_mlpELM(x_inferior=500, t_inferior=10, bc_weight=1., ic_weight=1., l2_reg=0.)
    # fit_nonlinear_case_CKAN(x_inferior=20 * 20, )
    # solve_F_heat_conduct_pikan()
    # solve_F_heat_conduct_pinn()
    solve_crack_PIKAN()
    # solve_wave_efficientKAN()
