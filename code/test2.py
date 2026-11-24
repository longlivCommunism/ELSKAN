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


def plot_log_error(X, error, title='Error (Log Scale)', save_name='result', save_dir=RESULT_PIC_DIR):
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
    
    filename = f"{save_name}_error_log.png"
    fig_log.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
    plt.close(fig_log)
    print(f"[plot_log_error] Log-scale error figure saved: '{os.path.join(save_dir, filename)}'.")



class WavKANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(WavKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # Linear weights for combining outputs
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))  # not used; you may like to use it for wieghting base activation and adding it like Spl-KAN paper
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2) - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)

        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1 / 2, torch.ones_like(v),
                                   torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype,
                                          device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            # You can try many more wavelet types ...
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        # You may like test the cases like Spl-KAN
        # wav_output = F.linear(wavelet_output, self.weight)
        # base_output = F.linear(self.base_activation(x), self.weight1)

        base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output  # + base_output

        # Apply batch normalization
        return self.bn(combined_output)


class WavKAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_type='mexican_hat'):
        super(WavKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(WavKANLinear(in_features, out_features, wavelet_type))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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


def solve_2d_helmholtz_WavKAN(n_interior=4096, n_bc=256, bc_weight=1.0, adam_iter=3000,
                              wavelet_type='mexican_hat'):
    k, xf, xbc, xf_ori, rhs, T_bc, analy_xf = _helmholtz_data(n_interior, n_bc)

    xf_t = torch.from_numpy(xf).to(device=device, dtype=dtype).requires_grad_(True)
    xbc_t = torch.from_numpy(xbc).to(device=device, dtype=dtype)
    rhs_t = torch.from_numpy(rhs).to(device=device, dtype=dtype)
    T_bc_t = torch.from_numpy(T_bc).to(device=device, dtype=dtype)
    xf_ori_t = torch.from_numpy(xf_ori).to(device=device, dtype=dtype)
    analy_xf_t = torch.from_numpy(analy_xf).to(device=device, dtype=dtype)

    model = WavKAN([2, 64, 64, 64, 1], wavelet_type=wavelet_type).to(device=device, dtype=dtype)
    param = sum(p.numel() for p in model.parameters())

    criterion = nn.MSELoss()
    optimizer_Adam = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.time()
    print(f"[WavKAN-{wavelet_type}] Start phase 1 to train on Adam...")
    model.train()
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
            model.eval()
            with torch.no_grad():
                pred = model(xf_ori_t)
                relative_error = torch.linalg.norm(pred - analy_xf_t) / torch.linalg.norm(analy_xf_t)
            model.train()
            print(f"It: {i:05d}, pde_loss: {pde_loss.item():.4e}, bc_loss: {bc_loss.item():.4e}, "
                  f"total_loss: {total_loss.item():.4e}, relative_error: {relative_error.item():.4e}")

    print(f"\n[WavKAN-{wavelet_type}] Start phase 2 to optimize on LBFGS...")
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

    print(f"\n[WavKAN-{wavelet_type}] Optimization Finished!")
    print(f"relative_error: {relative_error:.6e}, MAE: {mae:.6e}, MSE: {mse:.6e}")
    print(f"Parameter: {param}, Time: {elapsed_time:.4f}s")

    plot(xf_ori, pred, analy_xf, error, save_name=f'2d_helmholtz_WavKAN_{wavelet_type}')
    plot_log_error(xf_ori, error, title=f'WavKAN-{wavelet_type} Helmholtz Error (Log Scale)', 
                   save_name=f'2d_helmholtz_WavKAN_{wavelet_type}')



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.manual_seed(1024)
    np.random.seed(1024)

    solve_2d_helmholtz_WavKAN()
