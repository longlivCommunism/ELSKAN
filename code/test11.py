import os
import sys
import atexit
import torch
import numpy as np
import matplotlib.pyplot as plt
from Models import piCKANELM_decomp, piELM_decomp
import time

# All figures and logs are saved here
OUT_DIR = 'nonlinear_results'
os.makedirs(OUT_DIR, exist_ok=True)


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

# Set device
device = torch.device('cuda')  # Use CUDA as requested
dtype = torch.float64

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")


def generate_domain_2regions(nx_up=20, ny_up=20, nx_down=20, ny_down=20, 
                             nx_bc=40, ny_bc=40, nx_ic=40):
   
    # Interior points - Upper domain
    x_up = torch.linspace(0, 1, nx_up, device=device, dtype=dtype)
    y_up = torch.linspace(0.5, 1.0, ny_up, device=device, dtype=dtype)
    X_up, Y_up = torch.meshgrid(x_up, y_up, indexing='ij')
    domain_up = torch.stack([X_up.flatten(), Y_up.flatten()], dim=1)
    
    # Interior points - Lower domain
    x_down = torch.linspace(0, 1, nx_down, device=device, dtype=dtype)
    y_down = torch.linspace(0, 0.5, ny_down, device=device, dtype=dtype)
    X_down, Y_down = torch.meshgrid(x_down, y_down, indexing='ij')
    domain_down = torch.stack([X_down.flatten(), Y_down.flatten()], dim=1)
    
    # Boundary points - Upper domain (3 boundaries: top, left, right)
    x_bc = torch.linspace(0, 1, nx_bc, device=device, dtype=dtype)
    y_bc = torch.linspace(0.5, 1.0, ny_bc, device=device, dtype=dtype)
    
    bc_top = torch.stack([x_bc, torch.ones_like(x_bc)], dim=1)
    bc_left_up = torch.stack([torch.zeros_like(y_bc), y_bc], dim=1)
    bc_right_up = torch.stack([torch.ones_like(y_bc), y_bc], dim=1)
    bc_up = torch.vstack([bc_top, bc_left_up, bc_right_up])
    
    # Boundary points - Lower domain (3 boundaries: bottom, left, right)
    y_bc_down = torch.linspace(0, 0.5, ny_bc, device=device, dtype=dtype)
    
    bc_bottom = torch.stack([x_bc, torch.zeros_like(x_bc)], dim=1)
    bc_left_down = torch.stack([torch.zeros_like(y_bc_down), y_bc_down], dim=1)
    bc_right_down = torch.stack([torch.ones_like(y_bc_down), y_bc_down], dim=1)
    bc_down = torch.vstack([bc_bottom, bc_left_down, bc_right_down])
    
    # Interface points (y = 0.5)
    x_ic = torch.linspace(0, 1, nx_ic, device=device, dtype=dtype)
    interface = torch.stack([x_ic, 0.5 * torch.ones_like(x_ic)], dim=1)
    
    return domain_up, domain_down, bc_up, bc_down, interface


def test_burgers_equation():
   
    print("\n" + "="*80)
    print("Test Case 1: Burgers Equation")
    print("="*80)
    
    # Generate domain
    domain_up, domain_down, bc_up, bc_down, interface = generate_domain_2regions(
        nx_up=25, ny_up=25, nx_down=25, ny_down=25,
        nx_bc=50, ny_bc=50, nx_ic=50
    )
    
    # Initialize model
    model = piCKANELM_decomp(
        in_dim=2, 
        out_dim=1, 
        hidden=50,  # M = 100 * (degree2+1)
        degree_sub_up=3, 
        degree_sub_down=3, 
        degree2=3,
        coeff_scaled_up=1.0,
        coeff_scaled_down=1.0,
        device=device,
        dtype=dtype
    )
    
    # Manufactured solution
    def u_exact(points):
        x, y = points[:, 0:1], points[:, 1:2]
        return 0.1 * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    
    # Compute source term from manufactured solution
    # For u = 0.1*sin(pi*x)*sin(pi*y)
    # Lap(u) = -2*pi^2*u
    # u*grad(u) = 0.1*u*(pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
    def compute_source(points):
        x, y = points[:, 0:1], points[:, 1:2]
        u = u_exact(points)
        lap_u = -2 * np.pi**2 * u
        u_x = 0.1 * np.pi * torch.cos(np.pi * x) * torch.sin(np.pi * y)
        u_y = 0.1 * np.pi * torch.sin(np.pi * x) * torch.cos(np.pi * y)
        convection = u * (u_x + u_y)
        return lap_u + convection
    
    source_up = compute_source(domain_up)
    source_down = compute_source(domain_down)
    
    # Boundary conditions
    Tbc_up = u_exact(bc_up)
    Tbc_down = u_exact(bc_down)
    
    print(f"\nDomain sizes:")
    print(f"  Upper: {domain_up.shape[0]} points")
    print(f"  Lower: {domain_down.shape[0]} points")
    print(f"  Interface: {interface.shape[0]} points")
    print(f"  BC Upper: {bc_up.shape[0]} points")
    print(f"  BC Lower: {bc_down.shape[0]} points")
    
    # Solve with Newton method, recording relative error after each iteration
    print("\n" + "-"*60)
    print("Solving with Newton-LLSQ method...")
    print("-"*60)
    
    rel_err_history = []
    test_up, test_down = generate_test_points()
    error_callback = make_error_callback(model, domain_up, domain_down, u_exact, rel_err_history, test_up, test_down)
    
    pde_loss, bc_loss, inter_loss, elapsed_time, n_iters = model.fit_newton_simple(
        domain_up, domain_down, bc_up, bc_down, interface,
        Tbc_up, Tbc_down, source_up, source_down,
        nonlinear_coeff=1.0,
        max_iter=20,
        tol=1e-8,
        l2_reg=1e-10,
        verbose=True,
        callback=error_callback
    )
    
    print(f"\n" + "-"*60)
    print("Results:")
    print(f"  Converged in {n_iters} iterations")
    print(f"  Training time: {elapsed_time:.4f} seconds")
    print(f"  PDE loss: {pde_loss:.6e}")
    print(f"  BC loss: {bc_loss:.6e}")
    print(f"  Interface loss: {inter_loss:.6e}")
    
    # Compute L2 error
    with torch.no_grad():
        test_all = torch.cat([test_up, test_down], dim=0)
        u_pred_all = model.forward(test_up, test_down)
        u_true_all = u_exact(test_all)
        error_total = (torch.norm(u_pred_all - u_true_all) / torch.norm(u_true_all)).item()

        print(f"  L2 error (total): {error_total:.6e}")
    
    # Relative error decay curve (log scale)
    plot_newton_error_history(rel_err_history, "Burgers Equation", case_name='burgers')
    
    # Visualization
    plot_test_fields(test_all, u_pred_all, u_true_all,
                     case_name='burgers', arch_name='CKAN')
    
    return model


def test_nonlinear_poisson():
    """
    Test Case 2: Nonlinear Poisson equation
    Lap(u) + u^3 = f(x,y)
    
    Manufactured solution: u = sin(pi*x) * sin(pi*y) * 0.5
    """
    print("\n" + "="*80)
    print("Test Case 2: Nonlinear Poisson Equation (u^3 term)")
    print("="*80)
    
    # Generate domain
    domain_up, domain_down, bc_up, bc_down, interface = generate_domain_2regions(
        nx_up=25, ny_up=25, nx_down=25, ny_down=25,
        nx_bc=50, ny_bc=50, nx_ic=50
    )
    
    # Initialize model
    model = piCKANELM_decomp(
        in_dim=2, 
        out_dim=1, 
        hidden=50,
        degree_sub_up=3, 
        degree_sub_down=3, 
        degree2=3,
        coeff_scaled_up=1.0,
        coeff_scaled_down=1.0,
        device=device,
        dtype=dtype
    )
    
    # Manufactured solution
    def u_exact(points):
        x, y = points[:, 0:1], points[:, 1:2]
        return 0.5 * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    
    # Source term: f = Lap(u) + u^3
    def compute_source(points):
        u = u_exact(points)
        lap_u = -2 * np.pi**2 * u
        return lap_u + u**3
    
    source_up = compute_source(domain_up)
    source_down = compute_source(domain_down)
    
    # Boundary conditions
    Tbc_up = u_exact(bc_up)
    Tbc_down = u_exact(bc_down)
    
    # Define nonlinear function for u^3
    def nonlinear_cubic(u_up, ux_up, uy_up, u_down, ux_down, uy_down):
        """F(u) = u^3, dF/du = 3*u^2"""
        F_up = u_up ** 3
        F_down = u_down ** 3
        
        dF_du_up = 3 * u_up ** 2
        dF_du_down = 3 * u_down ** 2
        
        # No gradient dependence
        dF_dux_up = torch.zeros_like(u_up)
        dF_dux_down = torch.zeros_like(u_down)
        dF_duy_up = torch.zeros_like(u_up)
        dF_duy_down = torch.zeros_like(u_down)
        
        return F_up, F_down, dF_du_up, dF_du_down, dF_dux_up, dF_dux_down, dF_duy_up, dF_duy_down
    
    print("\n" + "-"*60)
    print("Solving with Newton-LLSQ method...")
    print("-"*60)
    
    rel_err_history = []
    test_up, test_down = generate_test_points()
    error_callback = make_error_callback(model, domain_up, domain_down, u_exact, rel_err_history, test_up, test_down)

    pde_loss, bc_loss, inter_loss, elapsed_time, n_iters = model.fit_newton(
        domain_up, domain_down, bc_up, bc_down, interface,
        Tbc_up, Tbc_down, source_up, source_down,
        nonlinear_cubic,
        max_iter=20,
        tol=1e-8,
        l2_reg=1e-10,
        verbose=True,
        callback=error_callback
    )
    
    print(f"\n" + "-"*60)
    print("Results:")
    print(f"  Converged in {n_iters} iterations")
    print(f"  Training time: {elapsed_time:.4f} seconds")
    print(f"  PDE loss: {pde_loss:.6e}")
    print(f"  BC loss: {bc_loss:.6e}")
    print(f"  Interface loss: {inter_loss:.6e}")
    
    # Compute L2 error
    with torch.no_grad():
        test_all = torch.cat([test_up, test_down], dim=0)
        u_pred_all = model.forward(test_up, test_down)
        u_true_all = u_exact(test_all)
        error_total = (torch.norm(u_pred_all - u_true_all) / torch.norm(u_true_all)).item()

        print(f"  L2 error (total): {error_total:.6e}")
    
    # Relative error decay curve (log scale)
    plot_newton_error_history(rel_err_history, "Nonlinear Poisson", case_name='poisson')
    
    # Visualization
    plot_test_fields(test_all, u_pred_all, u_true_all,
                     case_name='poisson', arch_name='CKAN')
    
    return model


def test_polynomial_saddle():
    """
    Test Case 3b: Polynomial saddle surface (reaction-diffusion)
    Lap(u) + 0.1*u^3 = f(x,y)

    COMPLETELY NEW STRATEGY: Use polynomial solution, NOT Gaussian!
    u = x*(1-x)*y*(1-y) * [10*(x-0.3)^2 + 8*(y-0.7)^2 + 12*(x-0.7)^2 + 6*(y-0.3)^2]

    This creates a multi-peak surface that is EXACTLY polynomial (degree 6)!
    Shape is COMPLETELY different from sinusoidal solutions.
    """
    print("\n" + "=" * 80)
    print("Test Case 3b: Polynomial Saddle Surface (Reaction-Diffusion)")
    print("=" * 80)

    # Optimal sampling (tuned for stability)
    domain_up, domain_down, bc_up, bc_down, interface = generate_domain_2regions(
        nx_up=25, ny_up=25,      # Keep at 25 (stable)
        nx_down=25, ny_down=25,  
        nx_bc=50, ny_bc=50,      
        nx_ic=50                 
    )

    # OPTIMAL configuration (found by grid search on CUDA)
    model = piCKANELM_decomp(
        in_dim=2, 
        out_dim=1, 
        hidden=50,               # Reduced from 100 -> 80 (KEY!)
        degree_sub_up=3,         # Reduced from 4 -> 3 (KEY!)
        degree_sub_down=3,       
        degree2=3,               
        coeff_scaled_up=1.0,
        coeff_scaled_down=1.0,
        device=device,
        dtype=dtype
    )

    # Manufactured solution: Polynomial saddle with multiple local extrema
    # CRITICAL: Keep coefficients SMALL to avoid numerical explosion!
    def u_exact(points):
        x, y = points[:, 0:1], points[:, 1:2]
        base = x * (1 - x) * y * (1 - y)
        # Ultra-small coefficients for numerical stability: 2->0.5, 1.5->0.4, 2.5->0.6, 1->0.3
        modulation = 0.5 * (x - 0.3) ** 2 + 0.4 * (y - 0.7) ** 2 + 0.6 * (x - 0.7) ** 2 + 0.3 * (y - 0.3) ** 2
        return base * modulation

    # Compute Laplacian analytically
    def compute_laplacian(points):
        x, y = points[:, 0:1], points[:, 1:2]

        # f = x*(1-x)*y*(1-y)
        f = x * (1 - x) * y * (1 - y)
        f_x = (1 - 2 * x) * y * (1 - y)
        f_y = x * (1 - x) * (1 - 2 * y)
        f_xx = -2 * y * (1 - y)
        f_yy = -2 * x * (1 - x)
        lap_f = f_xx + f_yy

        # g = 0.5*(x-0.3)^2 + 0.4*(y-0.7)^2 + 0.6*(x-0.7)^2 + 0.3*(y-0.3)^2
        g = 0.5 * (x - 0.3) ** 2 + 0.4 * (y - 0.7) ** 2 + 0.6 * (x - 0.7) ** 2 + 0.3 * (y - 0.3) ** 2
        g_x = 1.0 * (x - 0.3) + 1.2 * (x - 0.7)
        g_y = 0.8 * (y - 0.7) + 0.6 * (y - 0.3)
        lap_g = 1.0 + 1.2 + 0.8 + 0.6  # g_xx + g_yy = 3.6

        # Lap(u) = f*Lap(g) + g*Lap(f) + 2*(f_x*g_x + f_y*g_y)
        return f * lap_g + g * lap_f + 2 * (f_x * g_x + f_y * g_y)

    # Source term
    def compute_source(points):
        u = u_exact(points)
        lap_u = compute_laplacian(points)
        reaction = 0.1 * u ** 3
        return lap_u + reaction

    source_up = compute_source(domain_up)
    source_down = compute_source(domain_down)

    # Boundary conditions
    Tbc_up = u_exact(bc_up)
    Tbc_down = u_exact(bc_down)

    # Define nonlinear function
    def nonlinear_cubic_weak(u_up, ux_up, uy_up, u_down, ux_down, uy_down):
        F_up = 0.1 * u_up ** 3
        F_down = 0.1 * u_down ** 3

        dF_du_up = 0.3 * u_up ** 2
        dF_du_down = 0.3 * u_down ** 2

        dF_dux_up = torch.zeros_like(u_up)
        dF_dux_down = torch.zeros_like(u_down)
        dF_duy_up = torch.zeros_like(u_up)
        dF_duy_down = torch.zeros_like(u_down)

        return F_up, F_down, dF_du_up, dF_du_down, dF_dux_up, dF_dux_down, dF_duy_up, dF_duy_down

    print(f"\nDomain sizes:")
    print(f"  Upper: {domain_up.shape[0]} points")
    print(f"  Lower: {domain_down.shape[0]} points")
    print(f"  Interface: {interface.shape[0]} points")
    print(f"  BC Upper: {bc_up.shape[0]} points")
    print(f"  BC Lower: {bc_down.shape[0]} points")

    print("\n" + "-" * 60)
    print("Solving with Newton-LLSQ method...")
    print("-" * 60)

    rel_err_history = []
    test_up, test_down = generate_test_points()
    error_callback = make_error_callback(model, domain_up, domain_down, u_exact, rel_err_history, test_up, test_down)

    pde_loss, bc_loss, inter_loss, elapsed_time, n_iters = model.fit_newton(
        domain_up, domain_down, bc_up, bc_down, interface,
        Tbc_up, Tbc_down, source_up, source_down,
        nonlinear_cubic_weak,
        max_iter=20,             # Keep at 20 (more iterations cause divergence!)
        tol=1e-8,                
        l2_reg=1e-9,             # Stronger regularization: 1e-10 -> 1e-9 (KEY!)
        verbose=True,
        callback=error_callback
    )

    print(f"\n" + "-" * 60)
    print("Results:")
    print(f"  Converged in {n_iters} iterations")
    print(f"  Training time: {elapsed_time:.4f} seconds")
    print(f"  PDE loss: {pde_loss:.6e}")
    print(f"  BC loss: {bc_loss:.6e}")
    print(f"  Interface loss: {inter_loss:.6e}")

    # Compute L2 error
    with torch.no_grad():
        test_all = torch.cat([test_up, test_down], dim=0)
        u_pred_all = model.forward(test_up, test_down)
        u_true_all = u_exact(test_all)
        error_total = (torch.norm(u_pred_all - u_true_all) / torch.norm(u_true_all)).item()

        print(f"  L2 error (total): {error_total:.6e}")

    # Visualization
    # Relative error decay curve (log scale)
    plot_newton_error_history(rel_err_history, "Polynomial Saddle", case_name='saddle')
    
    plot_test_fields(test_all, u_pred_all, u_true_all,
                     case_name='saddle', arch_name='CKAN')

    return model


def generate_test_points(n_up=2000, n_down=2000, seed=12345):
    """
    Independent random test points (uniform, fixed seed), disjoint from the
    regular-grid collocation points used for training.
    Upper region: [0,1] x [0.5,1]; Lower region: [0,1] x [0,0.5].
    """
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


def make_error_callback(model, domain_up, domain_down, u_exact_func, error_list,
                        test_up=None, test_down=None):
    """
    Create a callback that records relative L2 error after each Newton iteration.
    Error is evaluated on held-out test points (not the training collocation points).
    The callback time is excluded from training time by fit_newton / fit_newton_simple.
    """
    if test_up is None or test_down is None:
        test_up, test_down = generate_test_points()
    test_points_all = torch.vstack([test_up, test_down])
    u_exact_all = u_exact_func(test_points_all)
    u_norm = torch.norm(u_exact_all).item()

    def callback(iter_num, beta_flat):
        with torch.no_grad():
            u_pred_all = model.forward(test_up, test_down)
            rel_error = torch.norm(u_pred_all - u_exact_all).item() / u_norm
            error_list.append(rel_error)
        # Stop when relative error <= 1e-8
        return rel_error <= 1e-8

    return callback


def plot_newton_error_history(error_history, title, case_name='case'):
    """Plot relative L2 error versus Newton iteration on a log scale."""
    if len(error_history) == 0:
        print("Warning: no error history to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    iters = np.arange(1, len(error_history) + 1)
    ax.semilogy(iters, error_history, 'o-', linewidth=2, markersize=6, color='navy',
                label='Relative L2 error')
    ax.set_xlabel('Newton iteration', fontsize=14)
    ax.set_ylabel('Relative L2 error (log scale)', fontsize=14)
    ax.set_title(f'{title}: error decay per Newton iteration', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()

    filename = os.path.join(OUT_DIR, f'newton_error_history_{case_name}.jpg')
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Error history figure saved: {filename}")
    plt.close(fig)


def plot_test_fields(test_all, u_pred_all, u_true_all, case_name='case', arch_name='CKAN'):
    """Plot on the held-out test points used for the L2 error: turbo prediction + hot log-scale absolute error.
    test_all: (N, 2) test coordinates, u_pred_all / u_true_all: (N, 1) predicted / exact values."""
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


if __name__ == "__main__":
    # Log all console output to nonlinear_results/
    LOG_PATH = os.path.join(OUT_DIR, 'test_newton_solver_log.txt')
    _log_file = open(LOG_PATH, 'w', encoding='utf-8')
    atexit.register(_log_file.close)
    _orig_stdout = sys.stdout
    sys.stdout = Tee(_orig_stdout, _log_file)

    # Set random seeds for reproducibility (CPU + GPU)
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)  # For multi-GPU
    
    # Additional settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # CRITICAL: Force deterministic algorithms (may slow down on CUDA)
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass  # Older PyTorch versions don't have this
    
    print("\n" + "="*80)
    print("Newton-LLSQ Nonlinear Solver Test Suite")
    print("Based on locELM paper (CMAME 2021)")
    print(f"Random seed: {SEED} (for reproducibility)")
    print(f"Device: {device}")
    print("="*80)
  
    # Test 1: Burgers equation
    model1 = test_burgers_equation()
    
    # Test 2: Nonlinear Poisson
    model2 = test_nonlinear_poisson()
    
    # Test 3: Polynomial Saddle (COMPLETELY DIFFERENT SHAPE!)
    model3 = test_polynomial_saddle()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print(f"Log saved to: {LOG_PATH}")
    print("="*80)
    sys.stdout = _orig_stdout
    _log_file.close()

