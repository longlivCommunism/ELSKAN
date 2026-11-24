import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from pyDOE2 import lhs


RESULT_PIC_DIR = 'result_pic'
os.makedirs(RESULT_PIC_DIR, exist_ok=True)


def plot(X, pred_vals=None, analy_vals=None, error=None, loss=None, epoch=None, plot_type='1d', title='', title_pred="", title_analy="", title_error="",
         value_range=None, error_range=None):
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

        plt.show()

    elif plot_type == '2d':
        plt.figure(num=1, figsize=(6, 4.8))
        plt.xlabel('t', fontsize=18)
        plt.ylabel('x', fontsize=18)
        pred_vals = pred_vals.squeeze()
        if value_range:
            v_min, v_max = value_range
            levels = np.linspace(v_min, v_max, 26)
            cs = plt.tricontourf(y_coord, x_coord, pred_vals, levels=levels, cmap='rainbow', alpha=1, vmin=v_min, vmax=v_max, extend='both')
            cbar = plt.colorbar(cs, ticks=[v_min, v_max])
            cbar.ax.set_ylim(v_min, v_max)
        else:
            plt.tricontourf(y_coord, x_coord, pred_vals, levels=25, cmap='rainbow', alpha=1)
            plt.colorbar()

        plt.figure(num=2, figsize=(6, 4.8))
        plt.xlabel('t', fontsize=18)
        plt.ylabel('x', fontsize=18)
        analy_vals = analy_vals.squeeze()
        if value_range:
            cs = plt.tricontourf(y_coord, x_coord, analy_vals, levels=levels, alpha=1, cmap='rainbow', vmin=v_min, vmax=v_max, extend='both')
            cbar = plt.colorbar(cs, ticks=[v_min, v_max])
            cbar.ax.set_ylim(v_min, v_max)
        else:
            plt.tricontourf(y_coord, x_coord, analy_vals, levels=25, alpha=1, cmap='rainbow')
            plt.colorbar()

        plt.figure(num=3, figsize=(6, 4.8))
        plt.xlabel('t', fontsize=18)
        plt.ylabel('x', fontsize=18)
        error = error.squeeze()
        if error_range:
            e_min, e_max = error_range
            levels = np.linspace(e_min, e_max, 26)
            cs = plt.tricontourf(y_coord, x_coord, error, levels=levels, alpha=1, cmap='rainbow', vmin=e_min, vmax=e_max, extend='max')
            cbar = plt.colorbar(cs, ticks=[e_min, e_max])
            cbar.ax.set_ylim(e_min, e_max)
        else:
            plt.tricontourf(y_coord, x_coord, error, levels=25, alpha=1, cmap='rainbow')
            plt.colorbar()

        plt.show()

    elif plot_type == 'loss':
        epoches = np.linspace(1, epoch + 1, epoch)
        plt.figure(figzise=(10, 8))
        plt.title('loss curve', fontsize=14, encoding='utf-8')
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.plot(epoches, loss, label='loss', color='blue')
        plt.show()


def plot_for_crack(X, pred_vals=None, analy_vals=None, error=None, loss=None, epoch=None, plot_type='1d', title='', title_pred="", title_analy="", title_error="",
         value_range=None, error_range=None):
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

        plt.show()

    elif plot_type == '2d':
        plt.figure(num=1, figsize=(6, 4.8))
        plt.xlabel('x', fontsize=18)
        plt.ylabel('y', fontsize=18)
        pred_vals = pred_vals.squeeze()
        if value_range:
            v_min, v_max = value_range
            levels = np.linspace(v_min, v_max, 26)
            cs = plt.tricontourf(x_coord, y_coord, pred_vals, levels=levels, cmap='rainbow', alpha=1, vmin=v_min, vmax=v_max, extend='both')
            cbar = plt.colorbar(cs, ticks=[v_min, v_max])
            cbar.ax.set_ylim(v_min, v_max)
        else:
            plt.tricontourf(x_coord, y_coord, pred_vals, levels=25, cmap='rainbow', alpha=1)
            plt.colorbar()

        plt.figure(num=2, figsize=(6, 4.8))
        plt.xlabel('x', fontsize=18)
        plt.ylabel('y', fontsize=18)
        analy_vals = analy_vals.squeeze()
        if value_range:
            cs = plt.tricontourf(y_coord, x_coord, analy_vals, levels=levels, alpha=1, cmap='rainbow', vmin=v_min, vmax=v_max, extend='both')
            cbar = plt.colorbar(cs, ticks=[v_min, v_max])
            cbar.ax.set_ylim(v_min, v_max)
        else:
            plt.tricontourf(x_coord,y_coord,  analy_vals, levels=25, alpha=1, cmap='rainbow')
            plt.colorbar()

        plt.figure(num=3, figsize=(6, 4.8))
        plt.xlabel('x', fontsize=18)
        plt.ylabel('y', fontsize=18)
        error = error.squeeze()
        if error_range:
            e_min, e_max = error_range
            levels = np.linspace(e_min, e_max, 26)
            cs = plt.tricontourf(y_coord, x_coord, error, levels=levels, alpha=1, cmap='rainbow', vmin=e_min, vmax=e_max, extend='max')
            cbar = plt.colorbar(cs, ticks=[e_min, e_max])
            cbar.ax.set_ylim(e_min, e_max)
        else:
            plt.tricontourf(x_coord, y_coord, error, levels=25, alpha=1, cmap='rainbow')
            plt.colorbar()

        plt.show()

    elif plot_type == 'loss':
        epoches = np.linspace(1, epoch + 1, epoch)
        plt.figure(figzise=(10, 8))
        plt.title('loss curve', fontsize=14, encoding='utf-8')
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.plot(epoches, loss, label='loss', color='blue')
        plt.show()


def plot_with_log(X, pred_vals=None, analy_vals=None, error=None, title='', 
                  title_error="", value_range=(1e-8, 1e-1)):
    import matplotlib.colors as mcolors
    
    # Extract coordinates
    if X.shape[1] == 2:
        x_coord, y_coord = X[:, 0], X[:, 1]
    elif X.shape[1] == 3:
        x_coord, y_coord, z_coord = X[:, 0], X[:, 1], X[:, 2]
    else:
        raise ValueError("X must have 2 or 3 columns for 2D plotting")
    
    # Plot error with logarithmic colorbar
    plt.figure(figsize=(6, 4.8), dpi=120)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.title(title_error, fontsize=16)
    
    error = error.squeeze()
    
    # Use fixed error range from value_range parameter
    e_min, e_max = value_range
    if e_min <= 0 or e_max <= 0:
        raise ValueError("value_range must contain positive values for log scale")
    
    # Clip errors to [e_min, e_max] to avoid extend='both' extreme colors
    error_clipped = np.clip(error, e_min, e_max)
    
    # Create logarithmic normalization with fixed range
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    
    # Generate logarithmically-spaced levels for uniform color distribution
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    cs = plt.tricontourf(x_coord, y_coord, error_clipped, levels=levels, 
                        cmap='hot', norm=norm, alpha=1)
    
    cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cbar = plt.colorbar(cs, ticks=cbar_ticks, format='%.0e')
    cbar.set_label('Absolute Error (log scale)', fontsize=14)
    cbar.ax.set_ylim(e_min, e_max)
    
    # Rotate tick labels for better readability
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontsize(10)
    
    # Annotate actual max error on the plot
    plt.text(0.02, 0.98, f'Max: {np.max(error):.2e}\nMean: {np.mean(error):.2e}', 
            transform=plt.gca().transAxes, fontsize=11, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


class piELM_SOV(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, w_scaled_h, b_scaled_h, w_scaled_l, b_scaled_l,
                 device=torch.device('cpu'), dtype=torch.float32):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(piELM_SOV, self).__init__()
        self.in_dim = in_dim

        self.weight_h = nn.Parameter(torch.empty(1, hidden_dim, **factory_kwargs), requires_grad=False)
        self.weight_l = nn.Parameter(torch.empty(1, hidden_dim, **factory_kwargs), requires_grad=False)
        self.bias_h = nn.Parameter(torch.empty(1, hidden_dim, **factory_kwargs), requires_grad=False)
        self.bias_l = nn.Parameter(torch.empty(1, hidden_dim, **factory_kwargs), requires_grad=False)

        nn.init.uniform_(self.weight_h, -w_scaled_h, w_scaled_h)
        nn.init.uniform_(self.weight_l, -w_scaled_l, w_scaled_l)
        nn.init.uniform_(self.bias_h, -b_scaled_h, b_scaled_h)
        nn.init.uniform_(self.bias_l, -b_scaled_l, b_scaled_l)

        self.beta = nn.Parameter(torch.ones(hidden_dim, out_dim, **factory_kwargs), requires_grad=False)

    def forward(self, X):
        x_h, x_l = X[:, 1][:, None], X[:, 0][:, None]
        H_h = F.tanh(x_h @ self.weight_h + self.bias_h)
        H_l = F.tanh(x_l @ self.weight_l + self.bias_l)
        H = H_h * H_l
        output = H @ self.beta
        return output

    def generate_H(self, Xf, Xbc, Xic):
        H_bc_h = F.tanh(Xbc[:, 1][:, None] @ self.weight_h + self.bias_h)
        H_bc_l = F.tanh(Xbc[:, 0][:, None] @ self.weight_l + self.bias_l)
        H_bc = H_bc_h * H_bc_l
        H_ic_h = F.tanh(Xic[:, 1][:, None] @ self.weight_h + self.bias_h)
        H_ic_l = F.tanh(Xic[:, 0][:, None] @ self.weight_l + self.bias_l)
        H_ic = H_ic_h * H_ic_l

        Ht_ic = (1. - H_ic_h ** 2) * H_ic_l * self.weight_h

        H_f_h = F.tanh(Xf[:, 1][:, None] @ self.weight_h + self.bias_h)
        H_f_l = F.tanh(Xf[:, 0][:, None] @ self.weight_l + self.bias_l)
        Hy = (1. - H_f_h ** 2) * H_f_l * self.weight_h
        Hx = (1. - H_f_l ** 2) * H_f_h * self.weight_l
        gradH = Hx + Hy
        Hyy = -2. * H_f_h * (1. - H_f_h ** 2) * H_f_l * self.weight_h ** 2
        Hxx = -2. * H_f_l * (1. - H_f_l ** 2) * H_f_h * self.weight_l ** 2
        lapH = Hxx + Hyy

        return Hx, Hy, gradH, Hxx, Hyy, lapH, H_bc, H_ic, Ht_ic


class piELM_SOV_4Heat(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, w_scaled_h, b_scaled_h, w_scaled_l, b_scaled_l,
                 device=torch.device('cpu'), dtype=torch.float32):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(piELM_SOV_4Heat, self).__init__()
        self.in_dim = in_dim

        self.weight_h = nn.Parameter(torch.empty(1, hidden_dim, **factory_kwargs), requires_grad=False)
        self.weight_l = nn.Parameter(torch.empty(1, hidden_dim, **factory_kwargs), requires_grad=False)
        self.bias_h = nn.Parameter(torch.empty(1, hidden_dim, **factory_kwargs), requires_grad=False)
        self.bias_l = nn.Parameter(torch.empty(1, hidden_dim, **factory_kwargs), requires_grad=False)

        nn.init.uniform_(self.weight_h, -w_scaled_h, w_scaled_h)
        nn.init.uniform_(self.weight_l, -w_scaled_l, w_scaled_l)
        nn.init.uniform_(self.bias_h, -b_scaled_h, b_scaled_h)
        nn.init.uniform_(self.bias_l, -b_scaled_l, b_scaled_l)

        self.beta = nn.Parameter(torch.ones(hidden_dim, out_dim, **factory_kwargs), requires_grad=False)

    def forward(self, X):
        x_h, x_l = X[:, 0][:, None], X[:, 1][:, None]
        H_h = F.tanh(x_h @ self.weight_h + self.bias_h)
        H_l = F.tanh(x_l @ self.weight_l + self.bias_l)
        H = H_h * H_l
        output = H @ self.beta
        return output

    def generate_H(self, Xf, Xbc, Xic):
        H_bc_h = F.tanh(Xbc[:, 0][:, None] @ self.weight_h + self.bias_h)
        H_bc_l = F.tanh(Xbc[:, 1][:, None] @ self.weight_l + self.bias_l)
        H_bc = H_bc_h * H_bc_l
        H_ic_h = F.tanh(Xic[:, 0][:, None] @ self.weight_h + self.bias_h)
        H_ic_l = F.tanh(Xic[:, 1][:, None] @ self.weight_l + self.bias_l)
        H_ic = H_ic_h * H_ic_l

        Ht_ic = (1. - H_ic_h ** 2) * H_ic_l * self.weight_h

        H_f_h = F.tanh(Xf[:, 0][:, None] @ self.weight_h + self.bias_h)
        H_f_l = F.tanh(Xf[:, 1][:, None] @ self.weight_l + self.bias_l)
        Hx = (1. - H_f_h ** 2) * H_f_l * self.weight_h
        Hy = (1. - H_f_l ** 2) * H_f_h * self.weight_l
        gradH = Hx + Hy
        Hxx = -2. * H_f_h * (1. - H_f_h ** 2) * H_f_l * self.weight_h ** 2
        Hyy = -2. * H_f_l * (1. - H_f_l ** 2) * H_f_h * self.weight_l ** 2
        lapH = Hxx + Hyy

        return Hx, Hy, gradH, Hxx, Hyy, lapH, H_bc, H_ic


class sub_piELM_fixed(nn.Module):
    def __init__(self, in_dim, hidden_dim, w_scaled, b_scaled, device=torch.device('cpu'), dtype=torch.float32):
        super(sub_piELM_fixed, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        weight = np.random.uniform(-w_scaled, w_scaled, size=(in_dim, hidden_dim))
        bias = np.random.uniform(-b_scaled, b_scaled, size=(1, hidden_dim))

        self.w = nn.Parameter(torch.from_numpy(weight).to(device=device, dtype=dtype), requires_grad=False)
        self.b = nn.Parameter(torch.from_numpy(bias).to(device=device, dtype=dtype), requires_grad=False)

    def forward(self, x):
        return F.tanh(x @ self.w + self.b)

    def generate_mat(self, Xf, Xbc, Xif):
        H_bc = self.forward(Xbc)
        Z_if = self.forward(Xif)
        Zf = self.forward(Xf)

        phi_1 = 1. - Zf ** 2
        Hx = phi_1 * self.w[0, :]
        Hy = phi_1 * self.w[1, :]
        phi_2 = -2. * Zf * phi_1
        Hxx = phi_2 * self.w[0, :] ** 2
        Hyy = phi_2 * self.w[1, :] ** 2
        gradH = Hx + Hy
        LapH = Hxx + Hyy

        phi_1_if = 1. - Z_if ** 2
        Hx_if = phi_1_if * self.w[0, :]
        Hy_if = phi_1_if * self.w[1, :]
        gradH_if = Hx_if + Hy_if

        return Zf, Hx, Hy, gradH, Hxx, Hyy, LapH, Z_if, gradH_if, H_bc


class piELM_decomp(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, w_scaled: list, b_scaled: list, device=torch.device('cpu'),
                 dtype=torch.float32):
        super(piELM_decomp, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype

        self.up_model = sub_piELM_fixed(in_dim=in_dim, hidden_dim=hidden_dim, w_scaled=w_scaled[0], b_scaled=b_scaled[0],
                                        device=device, dtype=dtype)
        self.down_model = sub_piELM_fixed(in_dim=in_dim, hidden_dim=hidden_dim, w_scaled=w_scaled[0], b_scaled=b_scaled[0],
                                        device=device, dtype=dtype)

        self.beta = nn.Parameter(torch.ones(2 * hidden_dim, out_dim, device=device, dtype=dtype), requires_grad=False)

    def forward(self, xu, xd):
        H_up = self.up_model(xu) @ self.beta[:self.hidden_dim, :]
        H_down = self.down_model(xd) @ self.beta[self.hidden_dim:, :]
        output = torch.vstack([H_up, H_down])
        return output

    def fit(self, Xf_up, Xf_down, Xbc_up, Xbc_down, Xinterface, Tbc_u, Tbc_d, l2_reg=1e-8):
        start_time = time.time()

        _, _, _, _, _, _, LapH_u, Z_if_u, gradH_if_u, H_bc_u = self.up_model.generate_mat(Xf_up, Xbc_up, Xinterface)
        _, _, _, _, _, _, LapH_d, Z_if_d, gradH_if_d, H_bc_d = self.down_model.generate_mat(Xf_down, Xbc_down, Xinterface)

        row1 = torch.hstack([LapH_u, torch.zeros(LapH_u.shape[0], LapH_d.shape[1], device=self.device, dtype=self.dtype)])
        row2 = torch.hstack([torch.zeros(LapH_d.shape[0], LapH_u.shape[1], device=self.device, dtype=self.dtype), LapH_d])
        row3 = torch.hstack([H_bc_u, torch.zeros(H_bc_u.shape[0], H_bc_d.shape[1], device=self.device, dtype=self.dtype)])
        row4 = torch.hstack([torch.zeros(H_bc_d.shape[0], H_bc_u.shape[1], device=self.device, dtype=self.dtype), H_bc_d])
        row5 = torch.hstack([-Z_if_u, Z_if_d])
        row6 = torch.hstack([gradH_if_u, -gradH_if_d])

        rhs1 = torch.zeros(row1.shape[0], 1, device=self.device, dtype=self.dtype)
        rhs2 = torch.zeros(row2.shape[0], 1, device=self.device, dtype=self.dtype)
        rhs5 = torch.zeros(row5.shape[0], 1, device=self.device, dtype=self.dtype)
        rhs6 = torch.zeros(row6.shape[0], 1, device=self.device, dtype=self.dtype)

        A = torch.vstack([row1, row2, np.sqrt(50) * row3, np.sqrt(50) * row4, np.sqrt(10) * row5, np.sqrt(10) * row6])
        b = torch.vstack([rhs1, rhs2, np.sqrt(50) * Tbc_u, np.sqrt(50) * Tbc_d, np.sqrt(10) * rhs5, np.sqrt(10) * rhs6])

        I = torch.eye(A.shape[1], device=self.device, dtype=self.dtype) * l2_reg
        beta = torch.linalg.solve(A.T @ A + I, A.T @ b)
        self.beta.data = beta
        elapsed_time = time.time() - start_time

        pde_loss = torch.mean((row1 @ beta) ** 2) + torch.mean((row2 @ beta) ** 2)
        bc_loss = torch.mean((row3 @ beta - Tbc_u) ** 2) + torch.mean((row4 @ beta - Tbc_d) ** 2)
        if_loss = torch.mean((row5 @ beta) ** 2) + torch.mean((row6 @ beta) ** 2)

        return pde_loss.detach().cpu().numpy(), bc_loss.detach().cpu().numpy(), if_loss.detach().cpu().numpy(), elapsed_time


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
        H = H_x * H_t

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


class piCKANELM_SOV2_4Heat(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, degree1_high, degree1_low, degree1_high_need, degree2, coeff_scaled_high=1.,
                 coeff_scaled_low=1., device=torch.device('cpu'), dtype=torch.float32):
        super(piCKANELM_SOV2_4Heat, self).__init__()
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
        H1_t = self.chebyshev(X[:, 1][:, None], 1, self.degree1_low).reshape(X.shape[0], -1) @ self.cheb_coeff_low
        H1_x = self.chebyshev(X[:, 0][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(X.shape[0], -1) @ self.cheb_coeff_high
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
        bcmid_t = self.chebyshev(Xbc[:, 1][:, None], 1, self.degree1_low).reshape(Xbc.shape[0], -1) @ self.cheb_coeff_low
        bcmid_x = self.chebyshev(Xbc[:, 0][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xbc.shape[0], -1) @ self.cheb_coeff_high
        H_bc_x = self.chebyshev(F.tanh(bcmid_x), self.hidden_nodes, self.degree2)
        H_bc_t = self.chebyshev(F.tanh(bcmid_t), self.hidden_nodes, self.degree2)
        H_bc = (H_bc_x * H_bc_t).reshape(Xbc.shape[0], -1)

        icmid_t = self.chebyshev(Xic[:, 1][:, None], 1, self.degree1_low).reshape(Xic.shape[0], -1) @ self.cheb_coeff_low
        icmid_x = self.chebyshev(Xic[:, 0][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xic.shape[0], -1) @ self.cheb_coeff_high
        H_ic_x = self.chebyshev(F.tanh(icmid_x), self.hidden_nodes, self.degree2)
        H_ic_t = self.chebyshev(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
        H_ic = (H_ic_x * H_ic_t).reshape(Xic.shape[0], -1)

        dZdt_ic = self.chebyshev1(
            Xic[:, 1][:, None], 1, self.degree1_low
        ).reshape(Xic.shape[0], -1) @ self.cheb_coeff_low
        tanh1_t_ic = 1 - F.tanh(icmid_t) ** 2
        dHtdZ_ic = self.chebyshev1(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
        Ht_ic = dHtdZ_ic * dZdt_ic[:, :, None] * tanh1_t_ic[:, :, None] * H_ic_x
        Ht_ic = Ht_ic.clone().reshape(Xic.shape[0], -1)

        x, t = Xtf[:, 0].clone()[:, None], Xtf[:, 1].clone()[:, None]
        Z_t = self.chebyshev(t, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_low
        Z_x = self.chebyshev(x, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_high
        H_x = self.chebyshev(F.tanh(Z_x), self.hidden_nodes, self.degree2)
        H_t = self.chebyshev(F.tanh(Z_t), self.hidden_nodes, self.degree2)
        H = H_x * H_t

        w_y = self.cheb_coeff_low
        w_x = self.cheb_coeff_high
        dZdy = self.chebyshev1(t, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_y
        dZdx = self.chebyshev1(x, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ w_x
        ddZdyy = self.chebyshev2(t, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ w_y
        ddZdxx = self.chebyshev2(x, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0], -1) @ w_x
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


class sub_piCKAN_fixed(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim=1, degree1=3, degree2=3, coeff_scaled=1.,device=torch.device('cpu'), dtype=torch.float32):
        super(sub_piCKAN_fixed, self).__init__()
        self.in_dim = in_dim
        self.hidden_nodes = hidden_nodes
        self.out_dim = out_dim
        self.degree1 = degree1
        self.degree2 = degree2

        self.cheby_coeff = nn.Parameter(torch.empty(in_dim * (degree1 + 1), hidden_nodes).to(device=device, dtype=dtype), requires_grad=False)
        nn.init.normal_(self.cheby_coeff, mean=0.0, std=coeff_scaled / (in_dim * (degree1 + 1)))

    def forward(self, X):
        H = self.chebyshev(X, self.in_dim, self.degree1).reshape(X.shape[0], -1) @ self.cheby_coeff
        output = self.chebyshev(F.tanh(H), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1)
        return output

    @staticmethod
    def chebyshev(x, dim, degree):
        cheby_poly = torch.ones(x.shape[0], dim, degree + 1).to(device=x.device, dtype=x.dtype)

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
        H_bc = self.forward(Xbc)
        H_ic = self.forward(Xic)

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
        H = self.chebyshev(F.tanh(Z), self.hidden_nodes, self.degree2)

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


class piCKANELM_decomp(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, degree_sub_up, degree_sub_down, degree2, coeff_scaled_up,
                 coeff_scaled_down, device=torch.device('cpu'), dtype=torch.float32):
        super(piCKANELM_decomp, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.degree2 = degree2

        self.ckan_up = sub_piCKAN_fixed(in_dim, hidden, out_dim, degree_sub_up, degree2, coeff_scaled=coeff_scaled_up, device=device, dtype=dtype)
        self.ckan_down = sub_piCKAN_fixed(in_dim, hidden, out_dim, degree_sub_down, degree2, coeff_scaled=coeff_scaled_down, device=device, dtype=dtype)

        self.beta = nn.Parameter(torch.ones(2 * hidden * (degree2 + 1), device=device, dtype=dtype), requires_grad=False)

    def forward(self, x_up, x_down):
        H_up = self.ckan_up(x_up)
        H_down = self.ckan_down(x_down)
        output_up = H_up @ self.beta[:self.hidden * (self.degree2 + 1), :]
        output_down = H_down @ self.beta[self.hidden * (self.degree2 + 1):, :]
        return torch.vstack([output_up, output_down])

    def fit(self, up, down, bc_up, bc_down, interface, Tbc_up, Tbc_down, l2_reg=1e-8):
        start_time = time.time()

        Mat_up = self.ckan_up.generate_H(up, bc_up, interface)
        Mat_down = self.ckan_down.generate_H(down, bc_down, interface)

        _, _, _, _, _, _, LapH_up, gradH_ic_up, H_bc_up, H_ic_up = Mat_up
        _, _, _, _, _, _, LapH_down, gradH_ic_down, H_bc_down, H_ic_down = Mat_down

        row1 = torch.hstack([LapH_up, torch.zeros(LapH_up.shape[0], LapH_down.shape[1], device=device, dtype=dtype)])
        row2 = torch.hstack([torch.zeros(LapH_down.shape[0], LapH_up.shape[1], device=device, dtype=dtype), LapH_down])
        row3 = torch.hstack([H_bc_up, torch.zeros(H_bc_up.shape[0], H_bc_down.shape[1], device=device, dtype=dtype)])
        row4 = torch.hstack([torch.zeros(H_bc_down.shape[0], H_bc_up.shape[1], device=device, dtype=dtype), H_bc_down])
        row5 = torch.hstack([-H_ic_up, H_ic_down])
        row6 = torch.hstack([gradH_ic_up, -gradH_ic_down])

        rhs_1 = torch.zeros(row1.shape[0], 1, device=device, dtype=dtype)
        rhs_2 = torch.zeros(row2.shape[0], 1, device=device, dtype=dtype)
        rhs_5 = torch.zeros(row5.shape[0], 1, device=device, dtype=dtype)
        rhs_6 = torch.zeros(row6.shape[0], 1, device=device, dtype=dtype)

        A = torch.vstack([row1, row2, np.sqrt(50) * row3, np.sqrt(50) * row4, np.sqrt(10) * row5, np.sqrt(10) * row6])
        b = torch.vstack([rhs_1, rhs_2, np.sqrt(50) * Tbc_up, np.sqrt(50) * Tbc_down, np.sqrt(10) * rhs_5, np.sqrt(10) * rhs_6])
        # beta = torch.linalg.pinv(A) @ b
        I_reg = torch.eye(A.shape[1], device=device, dtype=dtype) * l2_reg
        beta = torch.linalg.solve(A.T @ A + I_reg, A.T @ b)
        self.beta.data = beta
        elapsed_time = time.time() - start_time

        pde_loss = (torch.mean((row1 @ beta) ** 2) + torch.mean((row2 @ beta) ** 2)).detach().cpu().numpy()
        bc_loss = (torch.mean(((row3 @ beta) - Tbc_up) ** 2) + torch.mean(((row4 @ beta) - Tbc_down) ** 2)).detach().cpu().numpy()
        inter_loss = torch.mean((row5 @ beta) ** 2).detach().cpu().numpy() + torch.mean((row6 @ beta) ** 2).detach().cpu().numpy()

        return pde_loss, bc_loss, inter_loss, elapsed_time



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



class piCKANELM_SOV(nn.Module):
    def __init__(self, in_dim, hidden_nodes, out_dim, degree1_high, degree1_low, degree2, coeff_scaled_high=1., coeff_scaled_low=1.,
                 device=torch.device('cpu'), dtype=torch.float32):
        super(piCKANELM_SOV, self).__init__()
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
        # nn.init.uniform_(self.cheb_coeff_high, -coeff_scaled_high, coeff_scaled_high)
        # nn.init.uniform_(self.cheb_coeff_low, -coeff_scaled_low, coeff_scaled_low)

    def forward(self, X):
        H1_x = self.chebyshev(X[:, 0][:, None], 1, self.degree1_high).reshape(X.shape[0], -1) @ self.cheb_coeff_high
        H1_t = self.chebyshev(X[:, 1][:, None], 1, self.degree1_low).reshape(X.shape[0], -1) @ self.cheb_coeff_low
        output_x = self.chebyshev(F.tanh(H1_x), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1)
        output_t = self.chebyshev(F.tanh(H1_t), self.hidden_nodes, self.degree2).reshape(X.shape[0], -1)
        output = (output_x * output_t) @ self.beta

        return output

    @staticmethod
    def chebyshev(x, dim, degree, dropout: bool = False):
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


# ========================================================= #

def solve_F_heat_conduct_cKANELM(x_inferior=100, t_inferior=100, bc_weight=1., ic_weight=1., l2_reg=1e-8, hidden_nodes=1000):
    in_dim = 2
    degree1_high = 45
    degree1_low = 5
    degree2 = 2
    Frequency = 50
    factor = (1 / (Frequency * np.pi)) ** 2

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
    # x_ic = np.column_stack([np.linspace(0, 1, 3000), np.zeros(3000)])[np.random.choice(3000, 2000, replace=False)]
    # T_ic = np.sin(Frequency * np.pi * x_ic[:, 0][:, np.newaxis])
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
    xbc_t = torch.from_numpy(x_bc_scaled).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(x_ic_scaled).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    model = piCKANELM_SOV(in_dim, hidden_nodes, out_dim=1, degree1_high=degree1_high, degree1_low=degree1_low, degree2=degree2,
                          coeff_scaled_high=4.3, coeff_scaled_low=1.0, device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())
    _, _, Ht, _, Hxx, _, _, H_bc, H_ic = model.generate_H(xtf_t, xbc_t, xic_t)
    start_time = time.time()
    # sqrt_w_b = np.sqrt(bc_weight)
    # sqrt_w_i = np.sqrt(ic_weight)  * w_Ht
    rhs = torch.zeros(xtf.shape[0], 1, device=device, dtype=dtype)

    A_pde_raw = 2. * Ht - 4. * factor * Hxx
    std_Ht = torch.std(2. * Ht).item() + 1e-8
    std_Hxx = torch.std(4 * factor * Hxx).item() + 1e-8

    std_pde = torch.std(A_pde_raw).item() + 1e-8
    std_bc = torch.std(H_bc).item() + 1e-8
    std_ic = torch.std(H_ic).item() + 1e-8
    w_bc_dyn = std_pde / std_bc
    w_ic_dyn = std_pde / std_ic
    print(f"w_bc={w_bc_dyn}, w_ic={w_ic_dyn}")

    A = torch.vstack([A_pde_raw * 1., w_bc_dyn * H_bc, w_ic_dyn * H_ic])
    b = torch.vstack([rhs, w_bc_dyn * T_bc_t, w_ic_dyn * T_ic_t])
    col_norm = torch.linalg.norm(A, dim=0, keepdim=True)
    col_norm = torch.clamp(col_norm, min=1e-8)
    A_scaled = A / col_norm
    # beta = torch.linalg.pinv(A) @ b
    # U, s, V = torch.linalg.svd(A, full_matrices=False)
    # s_reg = torch.diag(s / (s ** 2 + l2_reg))
    # beta = (V.T @ s_reg @ U.T @ b) # / A_norm.T
    I_reg = torch.eye(min(A.shape[0], A.shape[1]), device=device, dtype=dtype) * l2_reg
    if A.shape[0] >= A.shape[1]:
        beta = torch.linalg.solve(A_scaled.T @ A_scaled + I_reg, A_scaled.T @ b)
    else:
        alpha = torch.linalg.solve(A_scaled @ A_scaled.T + I_reg, b)
        beta = A_scaled.T @ alpha
    beta = beta / col_norm.T
    # beta = torch.linalg.lstsq(A_scaled, b, rcond=l2_reg).solution / col_norm.T
    model.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = torch.mean((A_pde_raw @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    pred = model(xt_ori_t).detach().cpu().numpy()
    print(f"pred.shape={pred.shape}, analy_xtf.shape={analy_xtf.shape}")
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


def solve_F_heat_conduct_cKANELM2(x_inferior=100, t_inferior=100, bc_weight=1., ic_weight=1., l2_reg=1e-8, solver="normal", hidden_nodes=600):
    in_dim = 2
    degree1_high = 35
    degree1_high_need = 15
    degree1_low = 1
    degree2 = 5
    Frequency = 50
    factor = (1 / (Frequency * np.pi)) ** 2

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
    # x_ic = np.column_stack([np.linspace(0, 1, 3000), np.zeros(3000)])[np.random.choice(3000, 2000, replace=False)]
    # T_ic = np.sin(Frequency * np.pi * x_ic[:, 0][:, np.newaxis])
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
    xbc_t = torch.from_numpy(x_bc_scaled).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(x_ic_scaled).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    model = piCKANELM_SOV2_4Heat(in_dim, hidden_nodes, 1, degree1_high, degree1_low, degree1_high_need, degree2,
                           coeff_scaled_high=5.8, coeff_scaled_low=2.2, device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())
    _, _, Ht, _, Hxx, _, _, H_bc, H_ic = model.generate_H(xtf_t, xbc_t, xic_t)
    start_time = time.time()
    # sqrt_w_b = np.sqrt(bc_weight)
    # sqrt_w_i = np.sqrt(ic_weight)  * w_Ht
    rhs = torch.zeros(xtf.shape[0], 1, device=device, dtype=dtype)

    A_pde_raw = 2. * Ht - 4. * factor * Hxx
    std_Ht = torch.std(2. * Ht).item() + 1e-8
    std_Hxx = torch.std(4 * factor * Hxx).item() + 1e-8

    std_pde = torch.std(A_pde_raw).item() + 1e-8
    std_bc = torch.std(H_bc).item() + 1e-8
    std_ic = torch.std(H_ic).item() + 1e-8
    w_bc_dyn = std_pde / std_bc
    w_ic_dyn = std_pde / std_ic
    print(f"w_bc={w_bc_dyn}, w_ic={w_ic_dyn}")

    A = torch.vstack([A_pde_raw * 1., w_bc_dyn * H_bc, w_ic_dyn * H_ic])
    b = torch.vstack([rhs, w_bc_dyn * T_bc_t, w_ic_dyn * T_ic_t])
    col_norm = torch.linalg.norm(A, dim=0, keepdim=True)
    col_norm = torch.clamp(col_norm, min=1e-8)
    A_scaled = A / col_norm
    # beta = torch.linalg.pinv(A) @ b
    # U, s, V = torch.linalg.svd(A, full_matrices=False)
    # s_reg = torch.diag(s / (s ** 2 + l2_reg))
    # beta = (V.T @ s_reg @ U.T @ b) # / A_norm.T
    I_reg = torch.eye(min(A.shape[0], A.shape[1]), device=device, dtype=dtype) * l2_reg
    # regular_matrix = A_scaled.T @ A_scaled + I_reg
    # R = A_scaled.T @ b
    # cond_number = torch.linalg.cond(regular_matrix)
    if solver == "normal":
        if A.shape[0] >= A.shape[1]:
            beta = torch.linalg.solve(A_scaled.T @ A_scaled + I_reg, A_scaled.T @ b)
        else:
            alpha = torch.linalg.solve(A_scaled @ A_scaled.T + I_reg, b)
            beta = A_scaled.T @ alpha
    elif solver == "qr":
        _n = A_scaled.shape[1]
        _sqrt_reg = float(l2_reg) ** 0.5 if float(l2_reg) > 0 else 0.0
        if _sqrt_reg > 0:
            _A_aug = torch.vstack([A_scaled, _sqrt_reg * torch.eye(_n, device=device, dtype=dtype)])
            _b_aug = torch.vstack([b, torch.zeros((_n, b.shape[1]), device=device, dtype=dtype)])
            Q, R = torch.linalg.qr(_A_aug, mode="reduced")
            beta = torch.linalg.solve(R, Q.T @ _b_aug)
        else:
            Q, R = torch.linalg.qr(A_scaled, mode="reduced")
            beta = torch.linalg.solve(R, Q.T @ b)
    elif solver == "svd":
        U, s, Vh = torch.linalg.svd(A_scaled, full_matrices=False)
        s_reg = s / (s ** 2 + l2_reg)
        beta = (Vh.T * s_reg) @ (U.T @ b)
    else:
        raise ValueError(f"unknown solver={solver}, options: normal/qr/svd")
    # print(f"[solver={solver}] cond(A_scaled)={torch.linalg.cond(A_scaled).item():.3e}")
    # residual = torch.linalg.norm(regular_matrix @ beta - R  ) / torch.linalg.norm(R)
    beta = beta / col_norm.T
    
    # beta = torch.linalg.lstsq(A_scaled, b, rcond=l2_reg).solution / col_norm.T
    model.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = torch.mean((A_pde_raw @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    pred = model(xt_ori_t).detach().cpu().numpy()
    print(f"pred.shape={pred.shape}, analy_xtf.shape={analy_xtf.shape}")
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


def solve_F_heat_conduct_ELMSOV(x_inferior=100, t_inferior=100, l2_reg=1e-8, hidden_num=5000):
    # hp
    in_dim = 2
    Frequency = 50
    factor = (1 / (Frequency * np.pi)) ** 2

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
    t_cgl_bc = 0.5 * (1 - np.cos(np.pi * np.arange(N_bc) / (N_bc - 1)))
    x_bc_right = np.column_stack([np.ones(2000), t_cgl_bc])
    x_bc_left = np.column_stack([np.zeros(2000), t_cgl_bc])
    x_bc = np.vstack([x_bc_right, x_bc_left])
    T_bc = np.zeros((x_bc.shape[0], 1))
    # x_ic = np.column_stack([np.linspace(0, 1, 3000), np.zeros(3000)])[np.random.choice(3000, 2000, replace=False)]
    # T_ic = np.sin(Frequency * np.pi * x_ic[:, 0][:, np.newaxis])
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
    xbc_t = torch.from_numpy(x_bc_scaled).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(x_ic_scaled).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    model = piELM_SOV_4Heat(in_dim, hidden_dim=hidden_num, out_dim=1, w_scaled_h=50, b_scaled_h=30, w_scaled_l=10, b_scaled_l=5,
                            device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())
    _, Ht, _, Hxx, _, _, H_bc, H_ic = model.generate_H(xtf_t, xbc_t, xic_t)
    start_time = time.time()
    # sqrt_w_b = np.sqrt(bc_weight)
    # sqrt_w_i = np.sqrt(ic_weight)  * w_Ht
    rhs = torch.zeros(xtf.shape[0], 1, device=device, dtype=dtype)

    A_pde_raw = 2. * Ht - 4. * factor * Hxx
    std_Ht = torch.std(2. * Ht).item() + 1e-8
    std_Hxx = torch.std(4 * factor * Hxx).item() + 1e-8

    std_pde = torch.std(A_pde_raw).item() + 1e-8
    std_bc = torch.std(H_bc).item() + 1e-8
    std_ic = torch.std(H_ic).item() + 1e-8
    w_bc_dyn = std_pde / std_bc
    w_ic_dyn = std_pde / std_ic
    print(f"w_bc={w_bc_dyn}, w_ic={w_ic_dyn}")

    A = torch.vstack([A_pde_raw * 1., w_bc_dyn * H_bc, w_ic_dyn * H_ic])
    b = torch.vstack([rhs, w_bc_dyn * T_bc_t, w_ic_dyn * T_ic_t])
    col_norm = torch.linalg.norm(A, dim=0, keepdim=True)
    col_norm = torch.clamp(col_norm, min=1e-8)
    A_scaled = A / col_norm
    # beta = torch.linalg.pinv(A) @ b
    # U, s, V = torch.linalg.svd(A, full_matrices=False)
    # s_reg = torch.diag(s / (s ** 2 + l2_reg))
    # beta = (V.T @ s_reg @ U.T @ b) # / A_norm.T
    I_reg = torch.eye(min(A.shape[0], A.shape[1]), device=device, dtype=dtype) * l2_reg
    # regular_matrix = A_scaled.T @ A_scaled + I_reg
    # R = A_scaled.T @ b
    # cond_number = torch.linalg.cond(regular_matrix)
    if A.shape[0] >= A.shape[1]:
        beta = torch.linalg.solve(A_scaled.T @ A_scaled + I_reg, A_scaled.T @ b)
    else:
        alpha = torch.linalg.solve(A_scaled @ A_scaled.T + I_reg, b)
        beta = A_scaled.T @ alpha
    # residual = torch.linalg.norm(regular_matrix @ beta - R) / torch.linalg.norm(R)
    beta = beta / col_norm.T
    model.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = torch.mean((A_pde_raw @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    pred = model(xt_ori_t).detach().cpu().numpy()
    print(f"pred.shape={pred.shape}, analy_xtf.shape={analy_xtf.shape}")
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


def solve_wave_cKANELM2(x_inferior=100, t_inferior=100, l2_reg=1e-8, hidden_nodes=600):
    in_dim = 2
    degree1_high = 15
    degree1_high_need = 10
    degree1_low = 3
    degree2 = 10

    x = np.linspace(0, 1, x_inferior)
    t = np.linspace(0, 1, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    # u(x, t) = sin(πx) cos(10πt) + sin(2πx) cos(20πt)
    analy_xtf = (np.sin(np.pi * xt_ori[:, 0]) * np.cos(np.pi * 10. * xt_ori[:, 1]) + np.sin(np.pi * 2 * xt_ori[:, 0]) * np.cos(np.pi * 20. * xt_ori[:, 1]))[:, np.newaxis]
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

    xtf_t = torch.from_numpy(xtf_scaled).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc_scaled).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(xic_scaled).to(device=device, dtype=dtype)
    Tbc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    Tic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)

    model = piCKANELM_SOV2(in_dim, hidden_nodes, 1, degree1_high, degree1_low, degree1_high_need, degree2,
                           coeff_scaled_high=2.3, coeff_scaled_low=1.2, device=device, dtype=dtype)
    params = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    _, _, _, _, Hxx, Htt, _, H_bc, H_ic, Ht_ic = model.generate_H(xtf_t, xbc_t, xic_t)

    rhs1 = torch.zeros((xtf_t.shape[0], 1), device=device, dtype=dtype)
    rhs2 = torch.zeros((xic_t.shape[0], 1), device=device, dtype=dtype)
    pde_factor = (4 * Htt - 4 * 100. * Hxx) / 400
    icgrad_factor = 2. * Ht_ic
    std_htt = torch.std(4 * Htt).item() + 1e-8
    std_hxx = torch.std(4 * 100 * Hxx).item() + 1e-8
    std_pde = torch.std(pde_factor).item() + 1e-8
    std_bc = torch.std(H_bc).item() + 1e-8
    std_ic = torch.std(H_ic).item() + 1e-8
    std_icgrad = torch.std(icgrad_factor).item() + 1e-8
    w_bc = std_pde / std_bc
    w_ic = std_pde / std_ic
    w_icgrad = std_pde / std_icgrad
    print(f"w_bc=:{w_bc:.4f}, w_ic:={w_ic:.4f}, w_icgrad:={w_icgrad:.4f}")

    A = torch.vstack([pde_factor, w_bc * H_bc, w_ic * H_ic, w_icgrad * icgrad_factor])
    b = torch.vstack([rhs1, w_bc * Tbc_t, w_ic * Tic_t, w_icgrad * rhs2])

    col_norm = torch.linalg.norm(A, dim=0, keepdim=True)
    col_norm = torch.clamp(col_norm, min=l2_reg)
    A_scaled = A / col_norm
    I_reg = torch.eye(min(A.shape[0], A.shape[1]), device=device, dtype=dtype) * l2_reg
    if A.shape[0] > A.shape[1]:
        beta = torch.linalg.solve(A_scaled.T @ A_scaled + I_reg, A_scaled.T @ b)
    else:
        alpha = torch.linalg.solve(A_scaled @ A_scaled.T + I_reg, b)
        beta = A_scaled.T @ alpha
    beta = beta / col_norm.T
    model.beta.data = beta
    elpased_time = time.time() - start_time
    pde_loss = torch.mean((pde_factor @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)
    icgrad_loss = torch.mean((icgrad_factor @ beta) ** 2)

    pred = model(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(pred, analy_xtf)
    mae = mean_absolute_error(pred, analy_xtf)
    print(f"pde_loss:{pde_loss:.6e}, bc_loss:{bc_loss:.6e}, ic_loss:{ic_loss:.6e}, icgrad_loss:{icgrad_loss:.6e}, \n"
          f"relative_error:{relative_error:.6e}, mse:{mse:.6e}, mae:{mae:.6e},param:{params}, elpased_time:{elpased_time:.4f}")
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


def solve_wave_ELMSOV(x_inferior=100, t_inferior=100, l2_reg=1e-8, hidden_nodes=8000):
    in_dim = 2

    x = np.linspace(0, 1, x_inferior)
    t = np.linspace(0, 1, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    # u(x, t) = sin(πx) cos(10πt) + sin(2πx) cos(20πt)
    analy_xtf = (np.sin(np.pi * xt_ori[:, 0]) * np.cos(np.pi * 10. * xt_ori[:, 1]) + np.sin(np.pi * 2 * xt_ori[:, 0]) * np.cos(np.pi * 20. * xt_ori[:, 1]))[:, np.newaxis]
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

    xtf_t = torch.from_numpy(xtf_scaled).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc_scaled).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(xic_scaled).to(device=device, dtype=dtype)
    Tbc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    Tic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)

    model = piELM_SOV(in_dim, hidden_dim=hidden_nodes, out_dim=1, w_scaled_h=80, b_scaled_h=40, w_scaled_l=8, b_scaled_l=4,
                      device=device, dtype=dtype)
    params = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    _, _, _, Hxx, Htt, _, H_bc, H_ic, Ht_ic = model.generate_H(xtf_t, xbc_t, xic_t)

    rhs1 = torch.zeros((xtf_t.shape[0], 1), device=device, dtype=dtype)
    rhs2 = torch.zeros((xic_t.shape[0], 1), device=device, dtype=dtype)
    pde_factor = (4 * Htt - 4 * 100. * Hxx) / 400
    icgrad_factor = 2. * Ht_ic
    std_htt = torch.std(4 * Htt).item() + 1e-8
    std_hxx = torch.std(4 * 100 * Hxx).item() + 1e-8
    std_pde = torch.std(pde_factor).item() + 1e-8
    std_bc = torch.std(H_bc).item() + 1e-8
    std_ic = torch.std(H_ic).item() + 1e-8
    std_icgrad = torch.std(icgrad_factor).item() + 1e-8
    w_bc = std_pde / std_bc
    w_ic = std_pde / std_ic
    w_icgrad = std_pde / std_icgrad
    print(f"w_bc=:{w_bc:.4f}, w_ic:={w_ic:.4f}, w_icgrad:={w_icgrad:.4f}")

    A = torch.vstack([pde_factor, w_bc * H_bc, w_ic * H_ic, w_icgrad * icgrad_factor])
    b = torch.vstack([rhs1, w_bc * Tbc_t, w_ic * Tic_t, w_icgrad * rhs2])

    col_norm = torch.linalg.norm(A, dim=0, keepdim=True)
    col_norm = torch.clamp(col_norm, min=l2_reg)
    A_scaled = A / col_norm
    I_reg = torch.eye(min(A.shape[0], A.shape[1]), device=device, dtype=dtype) * l2_reg
    if A.shape[0] > A.shape[1]:
        beta = torch.linalg.solve(A_scaled.T @ A_scaled + I_reg, A_scaled.T @ b)
    else:
        alpha = torch.linalg.solve(A_scaled @ A_scaled.T + I_reg, b)
        beta = A_scaled.T @ alpha
    beta = beta / col_norm.T
    model.beta.data = beta
    elpased_time = time.time() - start_time
    pde_loss = torch.mean((pde_factor @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)
    icgrad_loss = torch.mean((icgrad_factor @ beta) ** 2)

    pred = model(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(pred, analy_xtf)
    mae = mean_absolute_error(pred, analy_xtf)
    print(f"pde_loss:{pde_loss:.6e}, bc_loss:{bc_loss:.6e}, ic_loss:{ic_loss:.6e}, icgrad_loss:{icgrad_loss:.6e}, \n"
          f"relative_error:{relative_error:.6e}, mse:{mse:.6e}, mae:{mae:.6e},param:{params}, elpased_time:{elpased_time:.4f}")
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


def solve_crack_kanELM(x_inferior=1000, y_inferior=50, bc_weight=1., ic_weight=1., hidden=110):
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
    bc_up_down = np.column_stack([np.linspace(-1, 0, 500), np.full(500, 1e-8)])[np.random.choice(500, 50, replace=False)]
    bc_up_right = np.column_stack([np.ones(500), np.linspace(0, 1, 500)])[np.random.choice(500, 128, replace=False)]
    bc_up_left = np.column_stack([-np.ones(500), np.linspace(0, 1, 500)])[np.random.choice(500, 128, replace=False)]
    bc_up = np.vstack([bc_up_up, bc_up_down, bc_up_right, bc_up_left])
    T_bc_up = getSourceItem(bc_up)[:, np.newaxis]
    bc_down_up = np.column_stack([np.linspace(-1, 1, 500), -np.ones(500)])[np.random.choice(500, 256, replace=False)]
    bc_down_down = np.column_stack([np.linspace(-1, 0, 500), np.full(500, -1e-8)])[np.random.choice(500, 50, replace=False)]
    bc_down_right = np.column_stack([np.ones(500), np.linspace(-1, 0, 500)])[np.random.choice(500, 128, replace=False)]
    bc_down_left = np.column_stack([-np.ones(500), np.linspace(-1, 0, 500)])[np.random.choice(500, 128, replace=False)]
    bc_down = np.vstack([bc_down_up, bc_down_down, bc_down_right, bc_down_left])
    T_bc_down = getSourceItem(bc_down)[:, np.newaxis]
    random_id_inter = np.random.choice(3000, 100, replace=False)
    xy_intface_sample = np.column_stack([np.linspace(0, 1, 3000), np.zeros(3000)])[random_id_inter]
    '''
    plt.figure(figsize=(6, 4.8))
    plt.xlabel("x", fontsize=18)
    plt.ylabel("y", fontsize=18)
    plt.scatter(xy_up_sample[:, 0], xy_up_sample[:, 1], c='blue', s=1)
    plt.scatter(xy_low_sample[:, 0], xy_low_sample[:, 1], c='green', s=1)
    plt.scatter(bc_up[:, 0], bc_up[:, 1], c='red', s=1)
    plt.scatter(bc_down[:, 0], bc_down[:, 1], c='red', s=1)
    plt.scatter(xy_intface_sample[:, 0], xy_intface_sample[:, 1], c='orange', s=1)
    plt.savefig('results_pics/decomp_sample.png')
    plt.show()
    '''

    xy_up_sample_t = torch.from_numpy(xy_up_sample).to(device=device, dtype=dtype)
    xy_down_sample_t = torch.from_numpy(xy_low_sample).to(device=device, dtype=dtype)
    xy_bc_up_sample_t = torch.from_numpy(bc_up).to(device=device, dtype=dtype)
    xy_bc_down_sample_t = torch.from_numpy(bc_down).to(device=device, dtype=dtype)
    xy_intface_sample_t = torch.from_numpy(xy_intface_sample).to(device=device, dtype=dtype)
    T_bc_up_t = torch.from_numpy(T_bc_up).to(device=device, dtype=dtype)
    T_bc_down_t = torch.from_numpy(T_bc_down).to(device=device, dtype=dtype)
    xy_ori_up_t = torch.from_numpy(xy_ori_up).to(device=device, dtype=dtype)
    xy_ori_down_t = torch.from_numpy(xy_ori_down).to(device=device, dtype=dtype)

    model = piCKANELM_decomp(in_dim=2, out_dim=1, hidden=110, degree_sub_up=5, degree_sub_down=5,
                             degree2=5, coeff_scaled_up=1.0, coeff_scaled_down=1.0, device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())

    pde_loss, bc_loss, inter_loss, elapsed_time = model.fit(xy_up_sample_t, xy_down_sample_t, xy_bc_up_sample_t, xy_bc_down_sample_t, xy_intface_sample_t, T_bc_up_t, T_bc_down_t)

    pred = model(xy_ori_up_t, xy_ori_down_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xy) / np.linalg.norm(analy_xy)
    mse = mean_squared_error(analy_xy, pred)
    mae = mean_absolute_error(analy_xy, pred)

    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}, ic_loss:{inter_loss:.6e}, relative_error:{relative_error:.6e}, MAE:{mae:.6e}, mse:{mse:.6e}\n"
          f"param:{param}, train_time:{elapsed_time:.4f}")
    error = np.abs(pred - analy_xy)

    import matplotlib.colors as mcolors
    
    fig_pred = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_pred = fig_pred.add_subplot(111)
    ax_pred.set_xlabel('x', fontsize=18)
    ax_pred.set_ylabel('y', fontsize=18)
    contour_turbo = ax_pred.tricontourf(xy_ori[:, 0], xy_ori[:, 1], pred.squeeze(-1), cmap='turbo', levels=100, alpha=1)
    fig_pred.colorbar(contour_turbo, ax=ax_pred)
    plt.tight_layout()
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('x', fontsize=18)
    ax_log.set_ylabel('y', fontsize=18)
    
    error_plot = error.squeeze()
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(error_plot, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    cs = ax_log.tricontourf(xy_ori[:, 0], xy_ori[:, 1], error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
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


def solve_crack_ELM(x_inferior=1000, y_inferior=50, bc_weight=1., ic_weight=1., hidden_dim=1000):
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
    bc_up_down = np.column_stack([np.linspace(-1, 0, 500), np.full(500, 1e-8)])[np.random.choice(500, 256, replace=False)]
    bc_up_right = np.column_stack([np.ones(500), np.linspace(0, 1, 500)])[np.random.choice(500, 128, replace=False)]
    bc_up_left = np.column_stack([-np.ones(500), np.linspace(0, 1, 500)])[np.random.choice(500, 128, replace=False)]
    bc_up = np.vstack([bc_up_up, bc_up_down, bc_up_right, bc_up_left])
    T_bc_up = getSourceItem(bc_up)[:, np.newaxis]
    bc_down_up = np.column_stack([np.linspace(-1, 1, 500), -np.ones(500)])[np.random.choice(500, 256, replace=False)]
    bc_down_down = np.column_stack([np.linspace(-1, 0, 500), np.full(500, -1e-8)])[np.random.choice(500, 256, replace=False)]
    bc_down_right = np.column_stack([np.ones(500), np.linspace(-1, 0, 500)])[np.random.choice(500, 128, replace=False)]
    bc_down_left = np.column_stack([-np.ones(500), np.linspace(-1, 0, 500)])[np.random.choice(500, 128, replace=False)]
    bc_down = np.vstack([bc_down_up, bc_down_down, bc_down_right, bc_down_left])
    T_bc_down = getSourceItem(bc_down)[:, np.newaxis]
    random_id_inter = np.random.choice(3000, 1000, replace=False)
    xy_intface_sample = np.column_stack([np.linspace(0, 1, 3000), np.zeros(3000)])[random_id_inter]

    xy_up_sample_t = torch.from_numpy(xy_up_sample).to(device=device, dtype=dtype)
    xy_down_sample_t = torch.from_numpy(xy_low_sample).to(device=device, dtype=dtype)
    xy_bc_up_sample_t = torch.from_numpy(bc_up).to(device=device, dtype=dtype)
    xy_bc_down_sample_t = torch.from_numpy(bc_down).to(device=device, dtype=dtype)
    xy_intface_sample_t = torch.from_numpy(xy_intface_sample).to(device=device, dtype=dtype)
    T_bc_up_t = torch.from_numpy(T_bc_up).to(device=device, dtype=dtype)
    T_bc_down_t = torch.from_numpy(T_bc_down).to(device=device, dtype=dtype)
    xy_ori_up_t = torch.from_numpy(xy_ori_up).to(device=device, dtype=dtype)
    xy_ori_down_t = torch.from_numpy(xy_ori_down).to(device=device, dtype=dtype)

    model = piELM_decomp(in_dim=2, out_dim=1, hidden_dim=hidden_dim, w_scaled=[7., 7.], b_scaled=[3., 3.], device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())

    pde_loss, bc_loss, inter_loss, elapsed_time = model.fit(xy_up_sample_t, xy_down_sample_t, xy_bc_up_sample_t, xy_bc_down_sample_t, xy_intface_sample_t, T_bc_up_t, T_bc_down_t)

    pred = model(xy_ori_up_t, xy_ori_down_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xy) / np.linalg.norm(analy_xy)
    mse = mean_squared_error(analy_xy, pred)
    mae = mean_absolute_error(analy_xy, pred)

    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}, ic_loss:{inter_loss:.6e}, relative_error:{relative_error:.6e}, MAE:{mae:.6e}, mse:{mse:.6e}\n"
          f"param:{param}, train_time:{elapsed_time:.4f}")
    error = np.abs(pred - analy_xy)

    import matplotlib.colors as mcolors
    
    fig_pred = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_pred = fig_pred.add_subplot(111)
    ax_pred.set_xlabel('x', fontsize=18)
    ax_pred.set_ylabel('y', fontsize=18)
    contour_turbo = ax_pred.tricontourf(xy_ori[:, 0], xy_ori[:, 1], pred.squeeze(-1), cmap='turbo', levels=100, alpha=1)
    fig_pred.colorbar(contour_turbo, ax=ax_pred)
    plt.tight_layout()
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('x', fontsize=18)
    ax_log.set_ylabel('y', fontsize=18)
    
    error_plot = error.squeeze()
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(error_plot, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    cs = ax_log.tricontourf(xy_ori[:, 0], xy_ori[:, 1], error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
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


# ============ ablation ============ #


def solve_crack_cKANELM_single(
        x_inferior=100, y_inferior=100, bc_weight=1.0, l2_reg=1e-8, hidden_nodes=1000):
    def get_source_item(x):
        r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        theta = np.arctan2(x[:, 1], x[:, 0])
        return np.sqrt(r) * np.sin(0.5 * theta)

    x_ori = np.linspace(-1., 1., x_inferior)
    y_ori_up = np.linspace(0., 1., y_inferior)
    y_ori_down = np.linspace(-1., -1e-8, y_inferior)
    xx, yy_up = np.meshgrid(x_ori, y_ori_up, indexing='ij')
    xy_ori_up = np.c_[xx.ravel(), yy_up.ravel()]
    xx, yy_down = np.meshgrid(x_ori, y_ori_down, indexing='ij')
    xy_ori_down = np.c_[xx.ravel(), yy_down.ravel()]
    xy_ori = np.vstack([xy_ori_up, xy_ori_down])
    analy_xy = get_source_item(xy_ori)[:, None]

    N_x_train, N_y_train = 100, 100
    x_train = np.linspace(-1., 1., N_x_train)[1:-1]
    y_train_up = np.linspace(0., 1., N_y_train)[1:-1]
    y_train_down = np.linspace(-1., 0., N_y_train)[1:-1]
    xx, yy = np.meshgrid(x_train, y_train_up, indexing='ij')
    xy_up_in = np.c_[xx.ravel(), yy.ravel()]
    xx, yy = np.meshgrid(x_train, y_train_down, indexing='ij')
    xy_down_in = np.c_[xx.ravel(), yy.ravel()]
    xy_interior = np.vstack([xy_up_in, xy_down_in])

    N_edge = 500
    x_edge = np.linspace(-1., 1., N_edge)
    y_edge_up = np.linspace(0., 1., N_edge)
    y_edge_down = np.linspace(-1., 0., N_edge)
    bc = np.vstack([
        np.column_stack([x_edge, np.ones(N_edge)]),
        np.column_stack([x_edge, -np.ones(N_edge)]),
        np.column_stack([np.ones(N_edge), y_edge_up]),
        np.column_stack([-np.ones(N_edge), y_edge_up]),
        np.column_stack([np.ones(N_edge), y_edge_down]),
        np.column_stack([-np.ones(N_edge), y_edge_down]),
        np.column_stack([x_edge[x_edge <= 0.], np.full(np.count_nonzero(x_edge <= 0.), 1e-8)]),
        np.column_stack([x_edge[x_edge <= 0.], np.full(np.count_nonzero(x_edge <= 0.), -1e-8)]),
    ])
    T_bc = get_source_item(bc)[:, None]

    xy_ic_dummy = torch.zeros((1, 2), device=device, dtype=dtype)
    xy_interior_t = torch.from_numpy(xy_interior).to(device=device, dtype=dtype)
    xy_bc_t = torch.from_numpy(bc).to(device=device, dtype=dtype)
    xy_ori_t = torch.from_numpy(xy_ori).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)

    model = piCKANELM_single(
        in_dim=2, hidden_nodes=1000, out_dim=1, degree1=15, degree2=15,
        coeff_scaled=1.0, device=device, dtype=dtype,
    )
    param = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    H, _, _, _, _, _, LapH, _, H_bc, _ = model.generate_H(
        xy_interior_t, xy_bc_t, xy_ic_dummy
    )
    rhs = torch.zeros((xy_interior_t.shape[0], 1), device=device, dtype=dtype)
    A_pde = LapH
    A = torch.vstack([A_pde, np.sqrt(bc_weight) * H_bc])
    b = torch.vstack([rhs, np.sqrt(bc_weight) * T_bc_t])

    col_norm = torch.linalg.norm(A, dim=0, keepdim=True).clamp_min(l2_reg)
    A_scaled = A / col_norm
    I_reg = torch.eye(A.shape[1], device=device, dtype=dtype) * l2_reg
    beta = torch.linalg.solve(
        A_scaled.T @ A_scaled + I_reg, A_scaled.T @ b
    ) / col_norm.T
    model.beta.data = beta
    elapsed_time = time.time() - start_time

    pde_loss = torch.mean((A_pde @ beta) ** 2).item()
    bc_loss = mean_squared_error(
        (H_bc @ beta).detach().cpu().numpy(), T_bc
    )
    pred = model(xy_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xy) / np.linalg.norm(analy_xy)
    mse = mean_squared_error(analy_xy, pred)
    mae = mean_absolute_error(analy_xy, pred)
    error = np.abs(pred - analy_xy)

    print("\n[crack_cKANELM_single] Optimization Finished!")
    print(f"pde_loss: {pde_loss:.6e}, bc_loss: {bc_loss:.6e}")
    print(f"relative_error: {relative_error:.6e}, MAE: {mae:.6e}, MSE: {mse:.6e}")
    print(f"Parameter: {param}, Time: {elapsed_time:.4f}s")
    
    import matplotlib.colors as mcolors
    
    fig_pred = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_pred = fig_pred.add_subplot(111)
    ax_pred.set_xlabel('x', fontsize=18)
    ax_pred.set_ylabel('y', fontsize=18)
    contour_turbo = ax_pred.tricontourf(xy_ori[:, 0], xy_ori[:, 1], pred.squeeze(-1), cmap='turbo', levels=100, alpha=1)
    fig_pred.colorbar(contour_turbo, ax=ax_pred)
    plt.tight_layout()
    
    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('x', fontsize=18)
    ax_log.set_ylabel('y', fontsize=18)
    
    error_plot = error.squeeze()
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(error_plot, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    
    cs = ax_log.tricontourf(xy_ori[:, 0], xy_ori[:, 1], error_clipped, levels=levels, cmap='hot', norm=norm, alpha=1)
    
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


def solve_wave_cKANELM_single(x_inferior=100, t_inferior=100, l2_reg=1e-8, hidden_nodes=600):
    in_dim = 2
    degree1 = 15
    degree2 = 10

    x = np.linspace(0, 1, x_inferior)
    t = np.linspace(0, 1, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    # u(x, t) = sin(πx) cos(10πt) + sin(2πx) cos(20πt)
    analy_xtf = (np.sin(np.pi * xt_ori[:, 0]) * np.cos(np.pi * 10. * xt_ori[:, 1]) + np.sin(np.pi * 2 * xt_ori[:, 0]) * np.cos(np.pi * 20. * xt_ori[:, 1]))[:, np.newaxis]
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

    xtf_t = torch.from_numpy(xtf_scaled).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc_scaled).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(xic_scaled).to(device=device, dtype=dtype)
    Tbc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    Tic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)

    model = piCKANELM_single(in_dim, hidden_nodes, out_dim=1, degree1=degree1, degree2=degree2, coeff_scaled=1.,
                            device=device, dtype=dtype)
    params = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    _, _, _, _, Hxx, Htt, _, gradH_ic, H_bc, H_ic = model.generate_H(xtf_t, xbc_t, xic_t)

    rhs1 = torch.zeros((xtf_t.shape[0], 1), device=device, dtype=dtype)
    rhs2 = torch.zeros((xic_t.shape[0], 1), device=device, dtype=dtype)
    pde_factor = (4 * Htt - 4 * 100. * Hxx) / 400
    icgrad_factor = 2. * gradH_ic
    std_htt = torch.std(4 * Htt).item() + 1e-8
    std_hxx = torch.std(4 * 100 * Hxx).item() + 1e-8
    std_pde = torch.std(pde_factor).item() + 1e-8
    std_bc = torch.std(H_bc).item() + 1e-8
    std_ic = torch.std(H_ic).item() + 1e-8
    std_icgrad = torch.std(icgrad_factor).item() + 1e-8
    w_bc = std_pde / std_bc
    w_ic = std_pde / std_ic
    w_icgrad = std_pde / std_icgrad
    print(f"w_bc=:{w_bc:.4f}, w_ic:={w_ic:.4f}, w_icgrad:={w_icgrad:.4f}")

    A = torch.vstack([pde_factor, w_bc * H_bc, w_ic * H_ic, w_icgrad * icgrad_factor])
    b = torch.vstack([rhs1, w_bc * Tbc_t, w_ic * Tic_t, w_icgrad * rhs2])

    col_norm = torch.linalg.norm(A, dim=0, keepdim=True)
    col_norm = torch.clamp(col_norm, min=l2_reg)
    A_scaled = A / col_norm
    I_reg = torch.eye(A.shape[1], device=device, dtype=dtype) * l2_reg
    if A.shape[0] > A.shape[1]:
        beta = torch.linalg.solve(A_scaled.T @ A_scaled + I_reg, A_scaled.T @ b)
    else:
        alpha = torch.linalg.solve(A_scaled @ A_scaled.T + I_reg, b)
        beta = A_scaled.T @ alpha
    beta = beta / col_norm.T
    model.beta.data = beta
    elpased_time = time.time() - start_time
    pde_loss = torch.mean((pde_factor @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)
    icgrad_loss = torch.mean((icgrad_factor @ beta) ** 2)

    pred = model(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(pred, analy_xtf)
    mae = mean_absolute_error(pred, analy_xtf)
    print(f"pde_loss:{pde_loss:.6e}, bc_loss:{bc_loss:.6e}, ic_loss:{ic_loss:.6e}, icgrad_loss:{icgrad_loss:.6e}, \n"
          f"relative_error:{relative_error:.6e}, mse:{mse:.6e}, mae:{mae:.6e},param:{params}, elpased_time:{elpased_time:.4f}")
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


def solve_wave_cKANELM_add_fusion(x_inferior=100, t_inferior=100, l2_reg=1e-8, hidden_nodes=600):
    in_dim = 2
    degree1_high = 15
    degree1_high_need = 10
    degree1_low = 3
    degree2 = 10

    x = np.linspace(0, 1, x_inferior)
    t = np.linspace(0, 1, t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    # u(x, t) = sin(πx) cos(10πt) + sin(2πx) cos(20πt)
    analy_xtf = (np.sin(np.pi * xt_ori[:, 0]) * np.cos(np.pi * 10. * xt_ori[:, 1]) + np.sin(np.pi * 2 * xt_ori[:, 0]) * np.cos(np.pi * 20. * xt_ori[:, 1]))[:, np.newaxis]
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

    xtf_t = torch.from_numpy(xtf_scaled).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc_scaled).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(xic_scaled).to(device=device, dtype=dtype)
    Tbc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    Tic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)

    model = piCKANELM_SOV2_add_fusion(in_dim, hidden_nodes, 1, degree1_high, degree1_low, degree1_high_need, degree2,
                                    coeff_scaled_high=2.3, coeff_scaled_low=1.2, device=device, dtype=dtype)
    params = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    _, _, _, _, Hxx, Htt, _, H_bc, H_ic, Ht_ic = model.generate_H(xtf_t, xbc_t, xic_t)

    rhs1 = torch.zeros((xtf_t.shape[0], 1), device=device, dtype=dtype)
    rhs2 = torch.zeros((xic_t.shape[0], 1), device=device, dtype=dtype)
    pde_factor = (4 * Htt - 4 * 100. * Hxx) / 400
    icgrad_factor = 2. * Ht_ic
    std_htt = torch.std(4 * Htt).item() + 1e-8
    std_hxx = torch.std(4 * 100 * Hxx).item() + 1e-8
    std_pde = torch.std(pde_factor).item() + 1e-8
    std_bc = torch.std(H_bc).item() + 1e-8
    std_ic = torch.std(H_ic).item() + 1e-8
    std_icgrad = torch.std(icgrad_factor).item() + 1e-8
    w_bc = std_pde / std_bc
    w_ic = std_pde / std_ic
    w_icgrad = std_pde / std_icgrad
    print(f"w_bc=:{w_bc:.4f}, w_ic:={w_ic:.4f}, w_icgrad:={w_icgrad:.4f}")

    A = torch.vstack([pde_factor, w_bc * H_bc, w_ic * H_ic, w_icgrad * icgrad_factor])
    b = torch.vstack([rhs1, w_bc * Tbc_t, w_ic * Tic_t, w_icgrad * rhs2])

    col_norm = torch.linalg.norm(A, dim=0, keepdim=True)
    col_norm = torch.clamp(col_norm, min=l2_reg)
    A_scaled = A / col_norm
    I_reg = torch.eye(min(A.shape[0], A.shape[1]), device=device, dtype=dtype) * l2_reg
    if A.shape[0] > A.shape[1]:
        beta = torch.linalg.solve(A_scaled.T @ A_scaled + I_reg, A_scaled.T @ b)
    else:
        alpha = torch.linalg.solve(A_scaled @ A_scaled.T + I_reg, b)
        beta = A_scaled.T @ alpha
    beta = beta / col_norm.T
    model.beta.data = beta
    elpased_time = time.time() - start_time
    pde_loss = torch.mean((pde_factor @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)
    icgrad_loss = torch.mean((icgrad_factor @ beta) ** 2)

    pred = model(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(pred, analy_xtf)
    mae = mean_absolute_error(pred, analy_xtf)
    print(f"pde_loss:{pde_loss:.6e}, bc_loss:{bc_loss:.6e}, ic_loss:{ic_loss:.6e}, icgrad_loss:{icgrad_loss:.6e}, \n"
          f"relative_error:{relative_error:.6e}, mse:{mse:.6e}, mae:{mae:.6e},param:{params}, elpased_time:{elpased_time:.4f}")
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

    # relative_error:1.527595e+00, mse:1.166656e+00, mae:8.210335e-01,param:15000, elpased_time:3.8486


def solve_F_heat_conduct_cKANELM2_uniformsample(x_inferior=100, t_inferior=100, bc_weight=1., ic_weight=1., l2_reg=1e-8, hidden_nodes=600):
    in_dim = 2
    degree1_high = 35
    degree1_high_need = 15
    degree1_low = 1
    degree2 = 5
    Frequency = 50
    factor = (1 / (Frequency * np.pi)) ** 2

    # Uniform sampling for the interior, boundary, and initial-condition points.
    x = np.linspace(0., 1., x_inferior)
    t = np.linspace(0., 1., t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    analy_xtf = (np.exp(-xt_ori[:, 1]) * np.sin(Frequency * np.pi * xt_ori[:, 0]))[:, np.newaxis]

    N_x_train, N_t_train = 300, 100
    x_train = np.linspace(0., 1., N_x_train)
    t_train = np.linspace(0., 1., N_t_train)
    xx_train, tt_train = np.meshgrid(x_train, t_train, indexing='ij')
    xtf = np.c_[xx_train.ravel(), tt_train.ravel()]

    N_bc = 2000
    t_bc = np.linspace(0., 1., N_bc)
    x_bc = np.vstack([
        np.column_stack([np.ones(N_bc), t_bc]),
        np.column_stack([np.zeros(N_bc), t_bc]),
    ])
    T_bc = np.zeros((x_bc.shape[0], 1))

    N_ic = 2000
    x_ic_coord = np.linspace(0., 1., N_ic)
    x_ic = np.column_stack([x_ic_coord, np.zeros(N_ic)])
    T_ic = np.sin(Frequency * np.pi * x_ic_coord)[:, None]

    scaler = lambda x: 2 * x - 1
    xtf_scaled = scaler(xtf)
    x_bc_scaled = scaler(x_bc)
    x_ic_scaled = scaler(x_ic)
    xt_ori_scaled = scaler(xt_ori)

    xtf_t = torch.from_numpy(xtf_scaled).to(device=device, dtype=dtype).requires_grad_(True)
    xbc_t = torch.from_numpy(x_bc_scaled).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(x_ic_scaled).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    T_ic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)

    model = piCKANELM_SOV2_4Heat(in_dim, hidden_nodes, 1, degree1_high, degree1_low, degree1_high_need, degree2,
                           coeff_scaled_high=5.8, coeff_scaled_low=1.2, device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())
    _, _, Ht, _, Hxx, _, _, H_bc, H_ic = model.generate_H(xtf_t, xbc_t, xic_t)
    start_time = time.time()
    # sqrt_w_b = np.sqrt(bc_weight)
    # sqrt_w_i = np.sqrt(ic_weight)  * w_Ht
    rhs = torch.zeros(xtf.shape[0], 1, device=device, dtype=dtype)

    A_pde_raw = 2. * Ht - 4. * factor * Hxx
    std_Ht = torch.std(2. * Ht).item() + 1e-8
    std_Hxx = torch.std(4 * factor * Hxx).item() + 1e-8

    std_pde = torch.std(A_pde_raw).item() + 1e-8
    std_bc = torch.std(H_bc).item() + 1e-8
    std_ic = torch.std(H_ic).item() + 1e-8
    w_bc_dyn = std_pde / std_bc
    w_ic_dyn = std_pde / std_ic
    print(f"w_bc={w_bc_dyn}, w_ic={w_ic_dyn}")

    A = torch.vstack([A_pde_raw * 1., w_bc_dyn * H_bc, w_ic_dyn * H_ic])
    b = torch.vstack([rhs, w_bc_dyn * T_bc_t, w_ic_dyn * T_ic_t])
    col_norm = torch.linalg.norm(A, dim=0, keepdim=True)
    col_norm = torch.clamp(col_norm, min=1e-8)
    A_scaled = A / col_norm
    # beta = torch.linalg.pinv(A) @ b
    # U, s, V = torch.linalg.svd(A, full_matrices=False)
    # s_reg = torch.diag(s / (s ** 2 + l2_reg))
    # beta = (V.T @ s_reg @ U.T @ b) # / A_norm.T
    I_reg = torch.eye(min(A.shape[0], A.shape[1]), device=device, dtype=dtype) * l2_reg
    if A.shape[0] >= A.shape[1]:
        beta = torch.linalg.solve(A_scaled.T @ A_scaled + I_reg, A_scaled.T @ b)
    else:
        alpha = torch.linalg.solve(A_scaled @ A_scaled.T + I_reg, b)
        beta = A_scaled.T @ alpha
    beta = beta / col_norm.T
    # beta = torch.linalg.lstsq(A_scaled, b, rcond=l2_reg).solution / col_norm.T
    model.beta.data = beta
    elapsed_time = time.time() - start_time
    pde_loss = torch.mean((A_pde_raw @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)

    pred = model(xt_ori_t).detach().cpu().numpy()
    print(f"pred.shape={pred.shape}, analy_xtf.shape={analy_xtf.shape}")
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


def solve_wave_cKANELM2_uniformsample(x_inferior=100, t_inferior=100, l2_reg=1e-8, hidden_nodes=600):
    in_dim = 2
    degree1_high = 15
    degree1_high_need = 10
    degree1_low = 3
    degree2 = 10

    # Uniform sampling for the interior, boundary, and initial-condition points.
    x = np.linspace(0., 1., x_inferior)
    t = np.linspace(0., 1., t_inferior)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xt_ori = np.c_[xx.ravel(), tt.ravel()]
    # u(x, t) = sin(pi*x) cos(10*pi*t) + sin(2*pi*x) cos(20*pi*t)
    analy_xtf = (
        np.sin(np.pi * xt_ori[:, 0]) * np.cos(10. * np.pi * xt_ori[:, 1])
        + np.sin(2. * np.pi * xt_ori[:, 0]) * np.cos(20. * np.pi * xt_ori[:, 1])
    )[:, None]

    N_x_train, N_t_train = 100, 300
    x_train = np.linspace(0., 1., N_x_train)
    t_train = np.linspace(0., 1., N_t_train)
    xx_train, tt_train = np.meshgrid(x_train, t_train, indexing='ij')
    xtf = np.c_[xx_train.ravel(), tt_train.ravel()]

    N_bc = 2000
    t_bc = np.linspace(0., 1., N_bc)
    x_bc = np.vstack([
        np.column_stack([np.zeros(N_bc), t_bc]),
        np.column_stack([np.ones(N_bc), t_bc]),
    ])
    T_bc = np.zeros((x_bc.shape[0], 1))

    N_ic = 2000
    x_ic_coord = np.linspace(0., 1., N_ic)
    x_ic = np.column_stack([x_ic_coord, np.zeros(N_ic)])
    T_ic = (
        np.sin(np.pi * x_ic_coord) + np.sin(2. * np.pi * x_ic_coord)
    )[:, None]

    scaler = lambda x: 2 * x - 1
    xtf_scaled = scaler(xtf)
    xbc_scaled = scaler(x_bc)
    xic_scaled = scaler(x_ic)
    xt_ori_scaled = scaler(xt_ori)

    xtf_t = torch.from_numpy(xtf_scaled).to(device=device, dtype=dtype)
    xbc_t = torch.from_numpy(xbc_scaled).to(device=device, dtype=dtype)
    xic_t = torch.from_numpy(xic_scaled).to(device=device, dtype=dtype)
    Tbc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    Tic_t = torch.from_numpy(T_ic).to(device=device, dtype=dtype)
    xt_ori_t = torch.from_numpy(xt_ori_scaled).to(device=device, dtype=dtype)

    model = piCKANELM_SOV2(in_dim, hidden_nodes, 1, degree1_high, degree1_low, degree1_high_need, degree2,
                           coeff_scaled_high=2.3, coeff_scaled_low=1.2, device=device, dtype=dtype)
    params = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    _, _, _, _, Hxx, Htt, _, H_bc, H_ic, Ht_ic = model.generate_H(xtf_t, xbc_t, xic_t)

    rhs1 = torch.zeros((xtf_t.shape[0], 1), device=device, dtype=dtype)
    rhs2 = torch.zeros((xic_t.shape[0], 1), device=device, dtype=dtype)
    pde_factor = (4 * Htt - 4 * 100. * Hxx) / 400
    icgrad_factor = 2. * Ht_ic
    std_htt = torch.std(4 * Htt).item() + 1e-8
    std_hxx = torch.std(4 * 100 * Hxx).item() + 1e-8
    std_pde = torch.std(pde_factor).item() + 1e-8
    std_bc = torch.std(H_bc).item() + 1e-8
    std_ic = torch.std(H_ic).item() + 1e-8
    std_icgrad = torch.std(icgrad_factor).item() + 1e-8
    w_bc = std_pde / std_bc
    w_ic = std_pde / std_ic
    w_icgrad = std_pde / std_icgrad
    print(f"w_bc=:{w_bc:.4f}, w_ic:={w_ic:.4f}, w_icgrad:={w_icgrad:.4f}")

    A = torch.vstack([pde_factor, w_bc * H_bc, w_ic * H_ic, w_icgrad * icgrad_factor])
    b = torch.vstack([rhs1, w_bc * Tbc_t, w_ic * Tic_t, w_icgrad * rhs2])

    col_norm = torch.linalg.norm(A, dim=0, keepdim=True)
    col_norm = torch.clamp(col_norm, min=l2_reg)
    A_scaled = A / col_norm
    I_reg = torch.eye(min(A.shape[0], A.shape[1]), device=device, dtype=dtype) * l2_reg
    if A.shape[0] > A.shape[1]:
        beta = torch.linalg.solve(A_scaled.T @ A_scaled + I_reg, A_scaled.T @ b)
    else:
        alpha = torch.linalg.solve(A_scaled @ A_scaled.T + I_reg, b)
        beta = A_scaled.T @ alpha
    beta = beta / col_norm.T
    model.beta.data = beta
    elpased_time = time.time() - start_time
    pde_loss = torch.mean((pde_factor @ beta) ** 2).detach().cpu().numpy()
    bc_loss = mean_squared_error((H_bc @ beta).detach().cpu().numpy(), T_bc)
    ic_loss = mean_squared_error((H_ic @ beta).detach().cpu().numpy(), T_ic)
    icgrad_loss = torch.mean((icgrad_factor @ beta) ** 2)

    pred = model(xt_ori_t).detach().cpu().numpy()
    relative_error = np.linalg.norm(pred - analy_xtf) / np.linalg.norm(analy_xtf)
    mse = mean_squared_error(pred, analy_xtf)
    mae = mean_absolute_error(pred, analy_xtf)
    print(f"pde_loss:{pde_loss:.6e}, bc_loss:{bc_loss:.6e}, ic_loss:{ic_loss:.6e}, icgrad_loss:{icgrad_loss:.6e}, \n"
          f"relative_error:{relative_error:.6e}, mse:{mse:.6e}, mae:{mae:.6e},param:{params}, elpased_time:{elpased_time:.4f}")
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




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # seed = 1024
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # l2_regs = [1e-4, 1e-6, 1e-8, 1e-10]  # 1021, 1022, 1023, 1024, 1025]
    # for l2_reg in l2_regs:
    #     torch.manual_seed(1024)
    #     np.random.seed(1024)
    #     solve_F_heat_conduct_cKANELM2(l2_reg=l2_reg)

    
    RANDOM_SEEDS = [1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030]
    SEED_RESULT_DIR = "25-JCP-Scaled_cPIKAN_randomseed"
    os.makedirs(SEED_RESULT_DIR, exist_ok=True)

    CASE_REGISTRY = {
        "solve_F_heat_conduct_cKANELM": solve_F_heat_conduct_cKANELM,
        "solve_F_heat_conduct_cKANELM2": solve_F_heat_conduct_cKANELM2,
        "solve_F_heat_conduct_ELMSOV": solve_F_heat_conduct_ELMSOV,
        "solve_wave_cKANELM2": solve_wave_cKANELM2,
        "solve_wave_ELMSOV": solve_wave_ELMSOV,
        "solve_crack_kanELM": solve_crack_kanELM,
        "solve_crack_ELM": solve_crack_ELM,
        "solve_crack_cKANELM_single": solve_crack_cKANELM_single,
        "solve_wave_cKANELM_single": solve_wave_cKANELM_single,
        "solve_wave_cKANELM_add_fusion": solve_wave_cKANELM_add_fusion,
        "solve_F_heat_conduct_cKANELM2_uniformsample": solve_F_heat_conduct_cKANELM2_uniformsample,
        "solve_wave_cKANELM2_uniformsample": solve_wave_cKANELM2_uniformsample,
    }

    ACTIVE_CASE = "solve_F_heat_conduct_cKANELM2"
    # ACTIVE_CASE = "solve_F_heat_conduct_cKANELM"
    # ACTIVE_CASE = "solve_F_heat_conduct_ELMSOV"
    # ACTIVE_CASE = "solve_wave_cKANELM2"
    # ACTIVE_CASE = "solve_wave_ELMSOV"
    # ACTIVE_CASE = "solve_crack_kanELM"
    # ACTIVE_CASE = "solve_crack_ELM"
    # ACTIVE_CASE = "solve_crack_cKANELM_single"
    # ACTIVE_CASE = "solve_wave_cKANELM_single"
    # ACTIVE_CASE = "solve_wave_cKANELM_add_fusion"
    # ACTIVE_CASE = "solve_F_heat_conduct_cKANELM2_uniformsample"
    # ACTIVE_CASE = "solve_wave_cKANELM2_uniformsample"

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
                                                        
      # solve_wave_cKANELM2_uniformsample()         
      # solve_crack_cKANELM_single()  
    torch.manual_seed(1024); np.random.seed(1024)
    # solve_F_heat_conduct_cKANELM2(solver="qr")      # QR
    # solve_F_heat_conduct_cKANELM2(solver="svd")     # SVD

    # solve_F_heat_conduct_cKANELM2(hidden_nodes=600)
    # solve_F_heat_conduct_ELMSOV(hidden_num=600)
    # print("--------------------------------")
    
   
    # solve_wave_cKANELM2(hidden_nodes=1000)
    # solve_wave_ELMSOV(hidden_nodes=3000)
    # print("--------------------------------")
    # solve_crack_kanELM(hidden=110)
    solve_crack_ELM(hidden_dim=500)
    # solve_crack_cKANELM_single(hidden_nodes=1000)
    # solve_wave_cKANELM_single(hidden_nodes=1000)
    # solve_wave_cKANELM_add_fusion(hidden_nodes=1000)
    # solve_F_heat_conduct_cKANELM2_uniformsample(hidden_nodes=1000)
    # solve_wave_cKANELM2_uniformsample(hidden_nodes=1000)

    # solve_F_heat_conduct_cKANELM()
    # solve_F_heat_conduct_cKANELM2()
    # solve_F_heat_conduct_ELMSOV()
    # solve_crack_kanELM()
    # solve_crack_ELM()
    # solve_crack_cKANELM_single()
    # solve_wave_cKANELM2()
    # solve_wave_ELMSOV()
