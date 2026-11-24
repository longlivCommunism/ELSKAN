import torch
import numpy as np
import matplotlib.pyplot as plt
from efficentKAN import efficentKAN
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

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


def compute_derivatives(model, points):
    """Compute u, du/dx, du/dy, d²u/dx², d²u/dy² using autograd"""
    points.requires_grad_(True)
    u = model(points)
    
    # First derivatives
    grads = torch.autograd.grad(u, points, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, points, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, points, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    
    return u, u_x, u_y, u_xx, u_yy


def train_two_stage(model, domain_up, domain_down, bc_up, bc_down, interface,
                    Tbc_up, Tbc_down, pde_loss_fn, u_exact_func,
                    adamw_epochs=10, verbose=True):
    """
    Single-stage training with Adam only (L-BFGS removed).

    Records relative L2 error and PDE loss every 10 epochs (evaluation time excluded),
    and plots both on log scale.
    """
    import time
    start_time = time.time()
    
    # Collect all collocation points
    X_pde = torch.vstack([domain_up, domain_down])
    X_bc = torch.vstack([bc_up, bc_down])
    T_bc = torch.vstack([Tbc_up, Tbc_down])
    X_if = interface
    
    # For error tracking (time not counted)
    error_history = []
    loss_history = []
    eval_time = 0.0
    # Independent held-out test points (same as Newton evaluation)
    test_up, test_down = generate_test_points()
    X_test = torch.vstack([test_up, test_down])
    u_exact_test = u_exact_func(X_test)
    u_norm = torch.norm(u_exact_test).item()
    
    def record_error_and_loss(epoch, stage_name):
        """Record relative error on test points and PDE loss (execution time excluded)"""
        nonlocal eval_time
        t_eval = time.time()
        
        # Relative L2 error on independent test points (no grad needed)
        with torch.no_grad():
            u_pred = model(X_test)
            rel_error = torch.norm(u_pred - u_exact_test).item() / u_norm
            error_history.append((epoch, rel_error, stage_name))
        
        # PDE loss (requires grad)
        loss_pde_val = pde_loss_fn(X_pde).detach().item()
        loss_history.append((epoch, loss_pde_val, stage_name))
        
        eval_time += time.time() - t_eval
        return rel_error
    
    # ========== Stage 1: AdamW training ==========
    if verbose:
        print("\n" + "="*60)
        print("Adam Training")
        print("="*60)
    
    optimizer_adam = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam, mode='min', factor=0.5, patience=500
    )
    
    for epoch in range(adamw_epochs):
        optimizer_adam.zero_grad()
        
        # PDE loss
        loss_pde = pde_loss_fn(X_pde)
        
        # BC loss
        u_bc = model(X_bc)
        loss_bc = torch.mean((u_bc - T_bc) ** 2)
        
        # Interface continuity loss
        u_if_up = model(X_if)
        X_if_down = X_if.clone()
        X_if_down[:, 1] = X_if[:, 1] - 1e-6  # Slightly below interface
        u_if_down = model(X_if_down)
        loss_if = torch.mean((u_if_up - u_if_down) ** 2)
        
        # Total loss (weighted)
        loss = loss_pde + 50.0 * loss_bc + 10.0 * loss_if
        
        loss.backward()
        optimizer_adam.step()
        scheduler.step(loss.detach())
        
        if epoch % 10 == 0 or epoch == adamw_epochs - 1:
            rel_err_now = record_error_and_loss(epoch, "AdamW")
            if rel_err_now <= 1e-8:
                if verbose:
                    print(f"Early stop at epoch {epoch}: rel_err={rel_err_now:.3e} <= 1e-8")
                break
        
        if verbose and (epoch % 1000 == 0 or epoch == adamw_epochs - 1):
            print(f"Epoch {epoch:5d}: Loss={loss.item():.6e}, "
                  f"PDE={loss_pde.item():.6e}, BC={loss_bc.item():.6e}, IF={loss_if.item():.6e}")
    
    
    # Final evaluation
    model.eval()

    # PDE loss: needs autograd, so it must stay outside no_grad
    loss_pde_final = pde_loss_fn(X_pde.clone().detach()).item()

    # BC / interface losses: no gradient needed
    with torch.no_grad():
        u_bc = model(X_bc)
        loss_bc_final = torch.mean((u_bc - T_bc) ** 2).item()
        u_if_up = model(X_if)
        X_if_down = X_if.clone()
        X_if_down[:, 1] = X_if[:, 1] - 1e-6
        u_if_down = model(X_if_down)
        loss_if_final = torch.mean((u_if_up - u_if_down) ** 2).item()
    model.train()
    
    elapsed_time = time.time() - start_time - eval_time
    
    # Plot PDE loss and relative error curves (log scale)
    plot_training_history(loss_history, error_history, adamw_epochs)
    
    if verbose:
        print("\n" + "="*60)
        print(f"Training completed in {elapsed_time:.2f}s (eval time excluded: {eval_time:.2f}s)")
        print(f"Final - PDE: {loss_pde_final:.6e}, BC: {loss_bc_final:.6e}, IF: {loss_if_final:.6e}")
        print("="*60)
    
    return loss_pde_final, loss_bc_final, loss_if_final, elapsed_time


def plot_training_history(loss_history, error_history, adamw_epochs):
    """
    Plot PDE loss and relative L2 error over both stages on log scale.
    loss_history / error_history: lists of (epoch, value, stage_name)
    """
    if len(loss_history) == 0 or len(error_history) == 0:
        print("Warning: no training history to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    styles = {"AdamW": ('o-', 'steelblue')}

    for ax, history, ylabel, title in [
        (ax1, loss_history, 'PDE loss (log scale)', 'PDE Loss'),
        (ax2, error_history, 'Relative L2 error (log scale)', 'Relative L2 Error'),
    ]:
        for stage, (marker, color) in styles.items():
            pts = [(e, v) for e, v, s in history if s == stage]
            if pts:
                xs, ys = zip(*pts)
                ax.semilogy(xs, ys, marker, color=color, linewidth=2, markersize=4, label=stage)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)

    plt.tight_layout()
    filename = 'two_stage_training_history.jpg'
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Training history figure saved: {filename}")
    plt.close(fig)


def plot_results_kan(model, u_exact_func, title, filename, case_name='case', arch_name='KAN'):
    import matplotlib.colors as mcolors
    
    import matplotlib.tri as mtri
    
    # Same independent test points as the L2 evaluation (matches Newton plots)
    test_up, test_down = generate_test_points()
    test_all = torch.cat([test_up, test_down], dim=0)
    
    with torch.no_grad():
        u_pred = model(test_all)
        u_exact = u_exact_func(test_all)
    
    U_pred = u_pred.reshape(-1).cpu().numpy()
    U_exact = u_exact.reshape(-1).cpu().numpy()
    U_error = np.abs(U_pred - U_exact)
    X = test_all[:, 0].cpu().numpy()
    Y = test_all[:, 1].cpu().numpy()
    tri = mtri.Triangulation(X, Y)
    
    fig_pred = plt.figure(figsize=(6, 4.8), dpi=120)
    ax_pred = fig_pred.add_subplot(111)
    ax_pred.set_xlabel('x', fontsize=18)
    ax_pred.set_ylabel('y', fontsize=18)
    contour_turbo = ax_pred.tricontourf(tri, U_pred, cmap='turbo', levels=100)
    ax_pred.axhline(y=0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    fig_pred.colorbar(contour_turbo, ax=ax_pred)
    plt.tight_layout()
    pred_filename = f'nonlinear_{case_name}_{arch_name}_prediction.jpg'
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
    error_filename = f'nonlinear_{case_name}_{arch_name}_error.jpg'
    fig_log.savefig(error_filename, dpi=150, bbox_inches='tight')
    print(f"Error figure saved: {error_filename}")
    plt.close(fig_log)


def plot_log_error_kan(model, u_exact_func, title, filename, case_name='case', arch_name='KAN'):
    print(f"Note: plot_log_error_kan is deprecated. Use plot_results_kan instead.")
    pass


# =============================================================================
# Test 1: Burgers Equation
# =============================================================================
def test_burgers_kan():
    print("\n" + "="*80)
    print("Test 1: Burgers Equation (EfficientKAN)")
    print("="*80)
    
    domain_up, domain_down, bc_up, bc_down, interface = generate_domain_2regions(
        nx_up=25, ny_up=25, nx_down=25, ny_down=25,
        nx_bc=50, ny_bc=50, nx_ic=50
    )
    
    # Model
    model = efficentKAN(
        layers_hidden=[2, 64, 64, 1],
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_range=[-1, 1]
    ).to(device).to(dtype)
    
    # Exact solution
    def u_exact(points):
        x, y = points[:, 0:1], points[:, 1:2]
        return 0.1 * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    
    # PDE loss function
    def pde_loss_fn(points):
        u, u_x, u_y, u_xx, u_yy = compute_derivatives(model, points)
        lap_u = u_xx + u_yy
        
        # Burgers: Lap(u) + u * (u_x + u_y) = f
        x, y = points[:, 0:1], points[:, 1:2]
        u_true = u_exact(points)
        lap_true = -2 * np.pi**2 * u_true
        u_x_true = 0.1 * np.pi * torch.cos(np.pi * x) * torch.sin(np.pi * y)
        u_y_true = 0.1 * np.pi * torch.sin(np.pi * x) * torch.cos(np.pi * y)
        f = lap_true + u_true * (u_x_true + u_y_true)
        
        residual = lap_u + u * (u_x + u_y) - f
        return torch.mean(residual ** 2)
    
    # Train
    pde_loss, bc_loss, if_loss, elapsed = train_two_stage(
        model, domain_up, domain_down, bc_up, bc_down, interface,
        u_exact(bc_up), u_exact(bc_down), pde_loss_fn, u_exact,
        adamw_epochs=10000, verbose=True
    )
    
    # Evaluate
    test_up, test_down = generate_test_points()
    test_all = torch.cat([test_up, test_down], dim=0)
    with torch.no_grad():
        u_pred = model(test_all)
        u_true = u_exact(test_all)
        l2_error = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
    
    print(f"\nBurgers: L2={l2_error:.6e}, PDE={pde_loss:.6e}, Time={elapsed:.2f}s")
    
    plot_results_kan(model, u_exact, "Burgers Equation (EfficientKAN)",
                    "newton_test_kan_burgers_equation.png",
                    case_name='burgers', arch_name='KAN')
    
    return model, l2_error


# =============================================================================
# Test 2: Nonlinear Poisson
# =============================================================================
def test_nonlinear_poisson_kan():
    print("\n" + "="*80)
    print("Test 2: Nonlinear Poisson (EfficientKAN)")
    print("="*80)
    
    domain_up, domain_down, bc_up, bc_down, interface = generate_domain_2regions(
        nx_up=25, ny_up=25, nx_down=25, ny_down=25,
        nx_bc=50, ny_bc=50, nx_ic=50
    )
    
    # Model
    model = efficentKAN(
        layers_hidden=[2, 64, 64, 1],
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_range=[-1, 1]
    ).to(device).to(dtype)
    
    # Exact solution
    def u_exact(points):
        x, y = points[:, 0:1], points[:, 1:2]
        return 0.5 * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    
    # PDE loss function
    def pde_loss_fn(points):
        u, u_x, u_y, u_xx, u_yy = compute_derivatives(model, points)
        lap_u = u_xx + u_yy
        
        # Nonlinear Poisson: Lap(u) + u^3 = f
        u_true = u_exact(points)
        lap_true = -2 * np.pi**2 * u_true
        f = lap_true + u_true**3
        
        residual = lap_u + u**3 - f
        return torch.mean(residual ** 2)
    
    # Train
    pde_loss, bc_loss, if_loss, elapsed = train_two_stage(
        model, domain_up, domain_down, bc_up, bc_down, interface,
        u_exact(bc_up), u_exact(bc_down), pde_loss_fn, u_exact,
        adamw_epochs=10000, verbose=True
    )
    
    # Evaluate
    test_up, test_down = generate_test_points()
    test_all = torch.cat([test_up, test_down], dim=0)
    with torch.no_grad():
        u_pred = model(test_all)
        u_true = u_exact(test_all)
        l2_error = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
    
    print(f"\nPoisson: L2={l2_error:.6e}, PDE={pde_loss:.6e}, Time={elapsed:.2f}s")
    
    plot_results_kan(model, u_exact, "Nonlinear Poisson (EfficientKAN)",
                    "newton_test_kan_nonlinear_poisson.png",
                    case_name='poisson', arch_name='KAN')
    
    return model, l2_error


# =============================================================================
# Test 3: Polynomial Saddle
# =============================================================================
def test_polynomial_saddle_kan():
    print("\n" + "="*80)
    print("Test 3: Polynomial Saddle (EfficientKAN)")
    print("="*80)
    
    domain_up, domain_down, bc_up, bc_down, interface = generate_domain_2regions(
        nx_up=25, ny_up=25, nx_down=25, ny_down=25,
        nx_bc=50, ny_bc=50, nx_ic=50
    )
    
    # Model
    model = efficentKAN(
        layers_hidden=[2, 64, 64, 1],
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_range=[-1, 1]
    ).to(device).to(dtype)
    
    # Exact solution
    def u_exact(points):
        x, y = points[:, 0:1], points[:, 1:2]
        base = x * (1 - x) * y * (1 - y)
        modulation = 0.5 * (x - 0.3)**2 + 0.4 * (y - 0.7)**2 + 0.6 * (x - 0.7)**2 + 0.3 * (y - 0.3)**2
        return base * modulation
    
    def compute_laplacian_exact(points):
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
    
    # PDE loss function
    def pde_loss_fn(points):
        u, u_x, u_y, u_xx, u_yy = compute_derivatives(model, points)
        lap_u = u_xx + u_yy
        
        # Polynomial Saddle: Lap(u) + 0.1*u^3 = f
        u_true = u_exact(points)
        lap_true = compute_laplacian_exact(points)
        f = lap_true + 0.1 * u_true**3
        
        residual = lap_u + 0.1 * u**3 - f
        return torch.mean(residual ** 2)
    
    # Train
    pde_loss, bc_loss, if_loss, elapsed = train_two_stage(
        model, domain_up, domain_down, bc_up, bc_down, interface,
        u_exact(bc_up), u_exact(bc_down), pde_loss_fn, u_exact,
        adamw_epochs=10000, verbose=True
    )
    
    # Evaluate
    test_up, test_down = generate_test_points()
    test_all = torch.cat([test_up, test_down], dim=0)
    with torch.no_grad():
        u_pred = model(test_all)
        u_true = u_exact(test_all)
        l2_error = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
    
    print(f"\nSaddle: L2={l2_error:.6e}, PDE={pde_loss:.6e}, Time={elapsed:.2f}s")
    
    plot_results_kan(model, u_exact, "Polynomial Saddle (EfficientKAN)",
                    "newton_test_kan_polynomial_saddle.png",
                    case_name='saddle', arch_name='KAN')
    
    return model, l2_error


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    print("="*80)
    print("EfficientKAN Training Tests (Adam only)")
    print("Adam (10000 epochs)")
    print("="*80)
    
    # Run all tests with error handling
    results = {}
    
    try:
        print("\n>>> Running Test 1: Burgers Equation...")
        model1, err1 = test_burgers_kan()
        results['Burgers'] = err1
    except Exception as e:
        print(f"❌ Burgers test failed: {e}")
        results['Burgers'] = None
    
    try:
        print("\n>>> Running Test 2: Nonlinear Poisson...")
        model2, err2 = test_nonlinear_poisson_kan()
        results['Poisson'] = err2
    except Exception as e:
        print(f"❌ Poisson test failed: {e}")
        results['Poisson'] = None

    try:
        print("\n>>> Running Test 3: Polynomial Saddle...")
        model3, err3 = test_polynomial_saddle_kan()
        results['Saddle'] = err3
    except Exception as e:
        print(f"❌ Saddle test failed: {e}")
        results['Saddle'] = None
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, err in results.items():
        if err is not None:
            print(f"{name:10s}: L2={err:.6e}")
        else:
            print(f"{name:10s}: FAILED")
    print("="*80)

