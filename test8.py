
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from Models import piELM_decomp
import sys
import atexit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64

# All figures and logs are saved here
OUT_DIR = 'nonlinear_results'
os.makedirs(OUT_DIR, exist_ok=True)

# Log all console output to nonlinear_results/
LOG_PATH = os.path.join(OUT_DIR, 'regenerate_pielm_plots_log.txt')
_log_file = open(LOG_PATH, 'w', encoding='utf-8')
atexit.register(_log_file.close)


class Tee:
    """Write to both the console and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


sys.stdout = Tee(sys.__stdout__, _log_file)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)  # piELM_decomp draws its ELM hidden weights with np.random
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Deterministic settings (same as test_newton_solver.py __main__)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass  # Older PyTorch versions

def generate_domain_2regions(nx_up, ny_up, nx_down, ny_down, nx_bc, ny_bc, nx_ic):
    x_up = torch.linspace(0, 1, nx_up, device=device, dtype=dtype)
    y_up = torch.linspace(0.5, 1.0, ny_up, device=device, dtype=dtype)
    X_up, Y_up = torch.meshgrid(x_up, y_up, indexing='ij')
    domain_up = torch.stack([X_up.flatten(), Y_up.flatten()], dim=1)
    
    x_down = torch.linspace(0, 1, nx_down, device=device, dtype=dtype)
    y_down = torch.linspace(0, 0.5, ny_down, device=device, dtype=dtype)
    X_down, Y_down = torch.meshgrid(x_down, y_down, indexing='ij')
    domain_down = torch.stack([X_down.flatten(), Y_down.flatten()], dim=1)
    
    x_bc = torch.linspace(0, 1, nx_bc, device=device, dtype=dtype)
    y_bc = torch.linspace(0.5, 1.0, ny_bc, device=device, dtype=dtype)
    bc_top = torch.stack([x_bc, torch.ones_like(x_bc)], dim=1)
    bc_left_up = torch.stack([torch.zeros_like(y_bc), y_bc], dim=1)
    bc_right_up = torch.stack([torch.ones_like(y_bc), y_bc], dim=1)
    bc_up = torch.vstack([bc_top, bc_left_up, bc_right_up])
    
    y_bc_down = torch.linspace(0, 0.5, ny_bc, device=device, dtype=dtype)
    bc_bottom = torch.stack([x_bc, torch.zeros_like(x_bc)], dim=1)
    bc_left_down = torch.stack([torch.zeros_like(y_bc_down), y_bc_down], dim=1)
    bc_right_down = torch.stack([torch.ones_like(y_bc_down), y_bc_down], dim=1)
    bc_down = torch.vstack([bc_bottom, bc_left_down, bc_right_down])
    
    x_ic = torch.linspace(0, 1, nx_ic, device=device, dtype=dtype)
    y_ic = 0.5 * torch.ones_like(x_ic)
    interface = torch.stack([x_ic, y_ic], dim=1)
    
    return domain_up, domain_down, bc_up, bc_down, interface

def generate_test_points(n_up=2000, n_down=2000, seed=12345):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    up = torch.rand(n_up - 2, 2, generator=gen, dtype=dtype)
    up[:, 1] = 0.5 + 0.5 * up[:, 1]
    down = torch.rand(n_down - 2, 2, generator=gen, dtype=dtype)
    down[:, 1] = 0.5 * down[:, 1]
    # Add the four vertices of the square domain so the plot covers the corners
    corners_up = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=dtype)
    corners_down = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    up = torch.cat([up, corners_up], dim=0)
    down = torch.cat([down, corners_down], dim=0)
    return up.to(device), down.to(device)


def plot_test_fields(test_all, u_pred_all, u_true_all, case_name='case', arch_name='CKAN'):
   
    import matplotlib.colors as mcolors
    import matplotlib.tri as mtri
    U_pred = u_pred_all.reshape(-1).detach().cpu().numpy()
    U_exact = u_true_all.reshape(-1).detach().cpu().numpy()
    X = test_all[:, 0].cpu().numpy()
    Y = test_all[:, 1].cpu().numpy()
    U_error = np.abs(U_pred - U_exact)
    tri = mtri.Triangulation(X, Y)

    fig_pred = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_pred = fig_pred.add_subplot(111)
    ax_pred.set_xlabel('x', fontsize=18)
    ax_pred.set_ylabel('y', fontsize=18)
    contour_turbo = ax_pred.tricontourf(tri, U_pred, levels=100, cmap='turbo')
    ax_pred.axhline(y=0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    fig_pred.colorbar(contour_turbo, ax=ax_pred)
    plt.tight_layout()
    pred_filename = os.path.join(OUT_DIR, f'nonlinear_{case_name}_{arch_name}_prediction.jpg')
    fig_pred.savefig(pred_filename, dpi=150, bbox_inches='tight')
    print(f"Prediction figure saved: {pred_filename}")
    plt.close(fig_pred)

    fig_log = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_log = fig_log.add_subplot(111)
    ax_log.set_xlabel('x', fontsize=18)
    ax_log.set_ylabel('y', fontsize=18)
    e_min, e_max = 1e-8, 1
    error_clipped = np.clip(U_error, e_min, e_max)
    norm = mcolors.LogNorm(vmin=e_min, vmax=e_max)
    levels = np.logspace(np.log10(e_min), np.log10(e_max), 100)
    cs = ax_log.tricontourf(tri, error_clipped, levels=levels, cmap='hot', norm=norm)
    ax_log.axhline(y=0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    cbar_ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    cbar = plt.colorbar(cs, ax=ax_log, ticks=cbar_ticks, format='%.0e')
    cbar.set_label('Absolute Error (log scale)', fontsize=14)
    cbar.ax.set_ylim(e_min, e_max)
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontsize(10)
    ax_log.text(0.02, 0.98, f'Max: {U_error.max():.2e}\nMean: {U_error.mean():.2e}',
                transform=ax_log.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    error_filename = os.path.join(OUT_DIR, f'nonlinear_{case_name}_{arch_name}_error.jpg')
    fig_log.savefig(error_filename, dpi=150, bbox_inches='tight')
    print(f"Error figure saved: {error_filename}")
    plt.close(fig_log)


print("="*80)
print("Regenerating piELM Newton-LLSQ Test Results")
print("="*80)

domain_up, domain_down, bc_up, bc_down, interface = generate_domain_2regions(
    nx_up=25, ny_up=25, nx_down=25, ny_down=25,
    nx_bc=50, ny_bc=50, nx_ic=50
)

test_up, test_down = generate_test_points()

# =============================================================================
# Test 1: Burgers Equation
# =============================================================================
print("\n" + "="*80)
print("Test 1: Burgers Equation")
print("="*80)

def u_exact_burgers(points):
    x, y = points[:, 0:1], points[:, 1:2]
    return 0.1 * torch.sin(np.pi * x) * torch.sin(np.pi * y)

def compute_source_burgers(points):
    x, y = points[:, 0:1], points[:, 1:2]
    u = u_exact_burgers(points)
    lap_u = -2 * np.pi**2 * u
    u_x = 0.1 * np.pi * torch.cos(np.pi * x) * torch.sin(np.pi * y)
    u_y = 0.1 * np.pi * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    return lap_u + u * (u_x + u_y)

model1 = piELM_decomp(in_dim=2, out_dim=1, hidden_dim=200,
                      w_scaled=[2.0, 2.0], b_scaled=[1.0, 1.0],
                      device=device, dtype=dtype)

pde_loss, bc_loss, if_loss, elapsed, iters = model1.fit_newton_simple(
    domain_up, domain_down, bc_up, bc_down, interface,
    u_exact_burgers(bc_up), u_exact_burgers(bc_down),
    compute_source_burgers(domain_up), compute_source_burgers(domain_down),
    nonlinear_coeff=1.0, max_iter=20, tol=1e-8, l2_reg=1e-8, verbose=False
)

test_all = torch.cat([test_up, test_down], dim=0)
with torch.no_grad():
    u_pred_all = model1.forward(test_up, test_down)
    u_true_all = u_exact_burgers(test_all)
    error = (torch.norm(u_pred_all - u_true_all) / torch.norm(u_true_all)).item()
    print(f"  L2 error (total): {error:.6e}")

print(f"Burgers: L2={error:.6e}, PDE={pde_loss:.6e}, iters={iters}, time={elapsed:.2f}s")
plot_test_fields(test_all, u_pred_all, u_true_all, case_name='burgers', arch_name='piELM')

# =============================================================================
# Test 2: Nonlinear Poisson
# =============================================================================
print("\n" + "="*80)
print("Test 2: Nonlinear Poisson")
print("="*80)

def u_exact_poisson(points):
    x, y = points[:, 0:1], points[:, 1:2]
    return 0.5 * torch.sin(np.pi * x) * torch.sin(np.pi * y)

def compute_source_poisson(points):
    u = u_exact_poisson(points)
    lap_u = -2 * np.pi**2 * u
    return lap_u + u**3

def nonlinear_cubic(u_up, ux_up, uy_up, u_down, ux_down, uy_down):
    F_up = u_up ** 3
    F_down = u_down ** 3
    dF_du_up = 3 * u_up ** 2
    dF_du_down = 3 * u_down ** 2
    return F_up, F_down, dF_du_up, dF_du_down, torch.zeros_like(u_up), torch.zeros_like(u_down), torch.zeros_like(u_up), torch.zeros_like(u_down)

model2 = piELM_decomp(in_dim=2, out_dim=1, hidden_dim=200,
                      w_scaled=[2.0, 2.0], b_scaled=[1.0, 1.0],
                      device=device, dtype=dtype)

pde_loss2, bc_loss2, if_loss2, elapsed2, iters2 = model2.fit_newton(
    domain_up, domain_down, bc_up, bc_down, interface,
    u_exact_poisson(bc_up), u_exact_poisson(bc_down),
    compute_source_poisson(domain_up), compute_source_poisson(domain_down),
    nonlinear_cubic, max_iter=20, tol=1e-8, l2_reg=1e-8, verbose=False
)

test_all = torch.cat([test_up, test_down], dim=0)
with torch.no_grad():
    u_pred_all = model2.forward(test_up, test_down)
    u_true_all = u_exact_poisson(test_all)
    error2 = (torch.norm(u_pred_all - u_true_all) / torch.norm(u_true_all)).item()
    print(f"  L2 error (total): {error2:.6e}")

print(f"Poisson: L2={error2:.6e}, PDE={pde_loss2:.6e}, iters={iters2}, time={elapsed2:.2f}s")
plot_test_fields(test_all, u_pred_all, u_true_all, case_name='poisson', arch_name='piELM')

# =============================================================================
# Test 3: Polynomial Saddle
# =============================================================================
print("\n" + "="*80)
print("Test 3: Polynomial Saddle")
print("="*80)

def u_exact_saddle(points):
    x, y = points[:, 0:1], points[:, 1:2]
    base = x * (1 - x) * y * (1 - y)
    modulation = 0.5 * (x - 0.3)**2 + 0.4 * (y - 0.7)**2 + 0.6 * (x - 0.7)**2 + 0.3 * (y - 0.3)**2
    return base * modulation

def compute_laplacian(points):
    x, y = points[:, 0:1], points[:, 1:2]
    f = x * (1 - x) * y * (1 - y)
    f_x = (1 - 2*x) * y * (1 - y)
    f_y = x * (1 - x) * (1 - 2*y)
    f_xx = -2 * y * (1 - y)
    f_yy = -2 * x * (1 - x)
    lap_f = f_xx + f_yy
    g = 0.5 * (x - 0.3)**2 + 0.4 * (y - 0.7)**2 + 0.6 * (x - 0.7)**2 + 0.3 * (y - 0.3)**2
    g_x = 1.0 * (x - 0.3) + 1.2 * (x - 0.7)
    g_y = 0.8 * (y - 0.7) + 0.6 * (y - 0.3)
    lap_g = 3.6
    return f * lap_g + g * lap_f + 2 * (f_x * g_x + f_y * g_y)

def compute_source_saddle(points):
    return compute_laplacian(points) + 0.1 * u_exact_saddle(points)**3

model3 = piELM_decomp(in_dim=2, out_dim=1, hidden_dim=200,
                      w_scaled=[2.0, 2.0], b_scaled=[1.0, 1.0],
                      device=device, dtype=dtype)

pde_loss3, bc_loss3, if_loss3, elapsed3, iters3 = model3.fit_newton(
    domain_up, domain_down, bc_up, bc_down, interface,
    u_exact_saddle(bc_up), u_exact_saddle(bc_down),
    compute_source_saddle(domain_up), compute_source_saddle(domain_down),
    nonlinear_cubic, max_iter=20, tol=1e-8, l2_reg=1e-8, verbose=False
)

test_all = torch.cat([test_up, test_down], dim=0)
with torch.no_grad():
    u_pred_all = model3.forward(test_up, test_down)
    u_true_all = u_exact_saddle(test_all)
    error3 = (torch.norm(u_pred_all - u_true_all) / torch.norm(u_true_all)).item()
    print(f"  L2 error (total): {error3:.6e}")

print(f"Saddle:  L2={error3:.6e}, PDE={pde_loss3:.6e}, iters={iters3}, time={elapsed3:.2f}s")
plot_test_fields(test_all, u_pred_all, u_true_all, case_name='saddle', arch_name='piELM')

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Burgers:  L2={error:.6e}")
print(f"Poisson:  L2={error2:.6e}")
print(f"Saddle:   L2={error3:.6e}")
print("="*80)
print("All plots regenerated with Newton-LLSQ!")
print(f"Log saved to: {LOG_PATH}")
sys.stdout = sys.__stdout__
_log_file.close()
