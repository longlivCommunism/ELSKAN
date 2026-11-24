import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# from pyDOE2 import lhs


def plot(X, pred_vals=None, analy_vals=None, error=None, loss=None, epoch=None, plot_type='1d', title='', title_pred="",
         title_analy="", title_error="",
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
		# 绘画预测图
		plt.figure(num=1, figsize=(6, 4.8))
		plt.xlabel('t', fontsize=18)
		plt.ylabel('x', fontsize=18)
		pred_vals = pred_vals.squeeze()
		if value_range:
			v_min, v_max = value_range
			levels = np.linspace(v_min, v_max, 26)
			cs = plt.tricontourf(y_coord, x_coord, pred_vals, levels=levels, cmap='rainbow', alpha=1, vmin=v_min,
			                     vmax=v_max, extend='both')
			cbar = plt.colorbar(cs, ticks=[v_min, v_max])
			cbar.ax.set_ylim(v_min, v_max)
		else:
			plt.tricontourf(y_coord, x_coord, pred_vals, levels=25, cmap='rainbow', alpha=1)
			plt.colorbar()

		# 绘画真实图
		plt.figure(num=2, figsize=(6, 4.8))
		plt.xlabel('t', fontsize=18)
		plt.ylabel('x', fontsize=18)
		analy_vals = analy_vals.squeeze()
		if value_range:
			cs = plt.tricontourf(y_coord, x_coord, analy_vals, levels=levels, alpha=1, cmap='rainbow', vmin=v_min,
			                     vmax=v_max, extend='both')
			cbar = plt.colorbar(cs, ticks=[v_min, v_max])
			cbar.ax.set_ylim(v_min, v_max)
		else:
			plt.tricontourf(y_coord, x_coord, analy_vals, levels=25, alpha=1, cmap='rainbow')
			plt.colorbar()

		# 绘画误差图
		plt.figure(num=3, figsize=(6, 4.8))
		plt.xlabel('t', fontsize=18)
		plt.ylabel('x', fontsize=18)
		error = error.squeeze()
		if error_range:
			e_min, e_max = error_range
			levels = np.linspace(e_min, e_max, 26)
			cs = plt.tricontourf(y_coord, x_coord, error, levels=levels, alpha=1, cmap='rainbow', vmin=e_min,
			                     vmax=e_max, extend='max')
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


def plot_for_crack(X, pred_vals=None, analy_vals=None, error=None, loss=None, epoch=None, plot_type='1d', title='',
                   title_pred="", title_analy="", title_error="",
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
		# 绘画预测图
		plt.figure(num=1, figsize=(6, 4.8))
		plt.xlabel('x', fontsize=18)
		plt.ylabel('y', fontsize=18)
		pred_vals = pred_vals.squeeze()
		if value_range:
			v_min, v_max = value_range
			levels = np.linspace(v_min, v_max, 26)
			cs = plt.tricontourf(x_coord, y_coord, pred_vals, levels=levels, cmap='rainbow', alpha=1, vmin=v_min,
			                     vmax=v_max, extend='both')
			cbar = plt.colorbar(cs, ticks=[v_min, v_max])
			cbar.ax.set_ylim(v_min, v_max)
		else:
			plt.tricontourf(x_coord, y_coord, pred_vals, levels=25, cmap='rainbow', alpha=1)
			plt.colorbar()

		# 绘画真实图
		plt.figure(num=2, figsize=(6, 4.8))
		plt.xlabel('x', fontsize=18)
		plt.ylabel('y', fontsize=18)
		analy_vals = analy_vals.squeeze()
		if value_range:
			cs = plt.tricontourf(y_coord, x_coord, analy_vals, levels=levels, alpha=1, cmap='rainbow', vmin=v_min,
			                     vmax=v_max, extend='both')
			cbar = plt.colorbar(cs, ticks=[v_min, v_max])
			cbar.ax.set_ylim(v_min, v_max)
		else:
			plt.tricontourf(x_coord, y_coord, analy_vals, levels=25, alpha=1, cmap='rainbow')
			plt.colorbar()

		# 绘画误差图
		plt.figure(num=3, figsize=(6, 4.8))
		plt.xlabel('x', fontsize=18)
		plt.ylabel('y', fontsize=18)
		error = error.squeeze()
		if error_range:
			e_min, e_max = error_range
			levels = np.linspace(e_min, e_max, 26)
			cs = plt.tricontourf(y_coord, x_coord, error, levels=levels, alpha=1, cmap='rainbow', vmin=e_min,
			                     vmax=e_max, extend='max')
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
	"""
	Plot error map with logarithmic colorbar scale for fair comparison across methods

	Parameters:
	-----------
	X : array-like, shape (n, 2) or (n, 3)
		Coordinates [x, y] or [x, y, z]
	pred_vals : array-like, optional
		Predicted values (not used in log plot, kept for API consistency)
	analy_vals : array-like, optional
		Analytical values (not used in log plot, kept for API consistency)
	error : array-like, shape (n,)
		Absolute error values to plot
	title : str
		Main title (not used, kept for consistency)
	title_error : str, default="Error (Log Scale)"
		Title for the error plot
	value_range : tuple of (float, float), optional
		(e_min, e_max) for error range. Fixed range for colorbar.
		Must be positive values for logarithmic scale.
	"""
	import matplotlib.colors as mcolors

	# Extract coordinates
	if X.shape[1] == 2:
		x_coord, y_coord = X[:, 0], X[:, 1]
	elif X.shape[1] == 3:
		x_coord, y_coord, z_coord = X[:, 0], X[:, 1], X[:, 2]
	else:
		raise ValueError("X must have 2 or 3 columns for 2D plotting")

	# Plot error with logarithmic colorbar
	plt.figure(figsize=(6, 4.8))
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
	levels = np.logspace(np.log10(e_min), np.log10(e_max), 11)

	# Plot with logarithmic scale and fixed colorbar range
	cs = plt.tricontourf(x_coord, y_coord, error_clipped, levels=levels,
	                     cmap='rainbow', norm=norm, alpha=1)

	# Create colorbar with all tick labels visible
	cbar = plt.colorbar(cs, ticks=levels, format='%.0e')
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
		# 获得边界条件的矩阵
		H_bc_h = F.tanh(Xbc[:, 1][:, None] @ self.weight_h + self.bias_h)
		H_bc_l = F.tanh(Xbc[:, 0][:, None] @ self.weight_l + self.bias_l)
		H_bc = H_bc_h * H_bc_l
		# 获得初始条件的矩阵
		H_ic_h = F.tanh(Xic[:, 1][:, None] @ self.weight_h + self.bias_h)
		H_ic_l = F.tanh(Xic[:, 0][:, None] @ self.weight_l + self.bias_l)
		H_ic = H_ic_h * H_ic_l

		# 获得初始条件的一阶导数
		Ht_ic = (1. - H_ic_h ** 2) * H_ic_l * self.weight_h

		# 获取计算域的具体导数矩阵
		H_f_h = F.tanh(Xf[:, 1][:, None] @ self.weight_h + self.bias_h)
		H_f_l = F.tanh(Xf[:, 0][:, None] @ self.weight_l + self.bias_l)
		# 一阶导数
		Hy = (1. - H_f_h ** 2) * H_f_l * self.weight_h
		Hx = (1. - H_f_l ** 2) * H_f_h * self.weight_l
		gradH = Hx + Hy
		# 二阶导数
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
		# 获得边界条件的矩阵
		H_bc_h = F.tanh(Xbc[:, 0][:, None] @ self.weight_h + self.bias_h)
		H_bc_l = F.tanh(Xbc[:, 1][:, None] @ self.weight_l + self.bias_l)
		H_bc = H_bc_h * H_bc_l
		# 获得初始条件的矩阵
		H_ic_h = F.tanh(Xic[:, 0][:, None] @ self.weight_h + self.bias_h)
		H_ic_l = F.tanh(Xic[:, 1][:, None] @ self.weight_l + self.bias_l)
		H_ic = H_ic_h * H_ic_l

		# 获得初始条件的一阶导数
		Ht_ic = (1. - H_ic_h ** 2) * H_ic_l * self.weight_h

		# 获取计算域的具体导数矩阵
		H_f_h = F.tanh(Xf[:, 0][:, None] @ self.weight_h + self.bias_h)
		H_f_l = F.tanh(Xf[:, 1][:, None] @ self.weight_l + self.bias_l)
		# 一阶导数
		Hx = (1. - H_f_h ** 2) * H_f_l * self.weight_h
		Hy = (1. - H_f_l ** 2) * H_f_h * self.weight_l
		gradH = Hx + Hy
		# 二阶导数
		Hxx = -2. * H_f_h * (1. - H_f_h ** 2) * H_f_l * self.weight_h ** 2
		Hyy = -2. * H_f_l * (1. - H_f_l ** 2) * H_f_h * self.weight_l ** 2
		lapH = Hxx + Hyy

		return Hx, Hy, gradH, Hxx, Hyy, lapH, H_bc, H_ic


class sub_piELM_fixed(nn.Module):
	def __init__(self, in_dim, hidden_dim, w_scaled, b_scaled, device=torch.device('cpu'), dtype=torch.float32):
		super(sub_piELM_fixed, self).__init__()

		self.in_dim = in_dim
		self.hidden_dim = hidden_dim
		self.device = device
		self.dtype = dtype

		weight = np.random.uniform(-w_scaled, w_scaled, size=(in_dim, hidden_dim))
		bias = np.random.uniform(-b_scaled, b_scaled, size=(1, hidden_dim))

		self.w = nn.Parameter(torch.from_numpy(weight).to(device=device, dtype=dtype), requires_grad=False)
		self.b = nn.Parameter(torch.from_numpy(bias).to(device=device, dtype=dtype), requires_grad=False)

	def forward(self, x):
		return torch.tanh(x @ self.w + self.b)

	def generate_mat(self, Xf, Xbc, Xif):
		"""
		Generate feature matrices for ELM.
		For ELM: Z = tanh(X @ w + b), so:
		- dZ/dx = (1 - tanh^2(H)) * w[0]
		- dZ/dy = (1 - tanh^2(H)) * w[1]
		- d²Z/dx² = -2*tanh(H)*(1-tanh²(H))*w[0]²
		- d²Z/dy² = -2*tanh(H)*(1-tanh²(H))*w[1]²
		"""
		# Hidden layer output
		H_bc = self.forward(Xbc)
		Z_if = self.forward(Xif)
		Zf = self.forward(Xf)

		# Pre-compute common terms
		H_internal = Xf @ self.w + self.b  # (N, M)
		tanh_H = torch.tanh(H_internal)
		sech_sq = 1.0 - tanh_H ** 2  # = (1 - tanh²(H)), same as (1 - Z²)

		# First derivatives: dZ/dx_i = sech²(H) * w[i]
		Hx = sech_sq * self.w[0, :]  # (N, M) element-wise * (M,)
		Hy = sech_sq * self.w[1, :]  # (N, M) element-wise * (M,)
		gradH = Hx + Hy  # gradient

		# Second derivatives: d²Z/dx² = -2*tanh(H)*sech²(H)*w²
		Hxx = -2.0 * tanh_H * sech_sq * (self.w[0, :] ** 2)
		Hyy = -2.0 * tanh_H * sech_sq * (self.w[1, :] ** 2)
		LapH = Hxx + Hyy  # Laplacian

		# Interface derivatives
		H_if = Xif @ self.w + self.b
		tanh_H_if = torch.tanh(H_if)
		sech_sq_if = 1.0 - tanh_H_if ** 2
		Hx_if = sech_sq_if * self.w[0, :]
		Hy_if = sech_sq_if * self.w[1, :]
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

		self.up_model = sub_piELM_fixed(in_dim=in_dim, hidden_dim=hidden_dim, w_scaled=w_scaled[0],
		                                b_scaled=b_scaled[0],
		                                device=device, dtype=dtype)
		self.down_model = sub_piELM_fixed(in_dim=in_dim, hidden_dim=hidden_dim, w_scaled=w_scaled[0],
		                                  b_scaled=b_scaled[0],
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
		_, _, _, _, _, _, LapH_d, Z_if_d, gradH_if_d, H_bc_d = self.down_model.generate_mat(Xf_down, Xbc_down,
		                                                                                    Xinterface)

		row1 = torch.hstack(
			[LapH_u, torch.zeros(LapH_u.shape[0], LapH_d.shape[1], device=self.device, dtype=self.dtype)])
		row2 = torch.hstack(
			[torch.zeros(LapH_d.shape[0], LapH_u.shape[1], device=self.device, dtype=self.dtype), LapH_d])
		row3 = torch.hstack(
			[H_bc_u, torch.zeros(H_bc_u.shape[0], H_bc_d.shape[1], device=self.device, dtype=self.dtype)])
		row4 = torch.hstack(
			[torch.zeros(H_bc_d.shape[0], H_bc_u.shape[1], device=self.device, dtype=self.dtype), H_bc_d])
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

	def fit_newton(self, Xf_up, Xf_down, Xbc_up, Xbc_down, Xinterface, Tbc_u, Tbc_d,
	               source_up, source_down, nonlinear_func,
	               max_iter=20, tol=1e-6, l2_reg=1e-7, verbose=True):
		"""
		Newton-LLSQ method for piELM (following locELM paper CMAME 2021)

		Key idea: Newton iteration + Linear Least Squares
		- At each Newton step, solve for INCREMENT field v (not the full solution u)
		- v satisfies: L*v + J(u^k)*v = f - [L*u^k + F(u^k)]
		- Use accumulation vector to store sum of all increments

		Reference: Dong & Li, CMAME 387 (2021) 114129, Appendix A
		"""
		import time
		start_time = time.time()

		# Pre-compute matrices (fixed throughout Newton iteration)
		Z_up, Hx_up, Hy_up, _, _, _, LapH_up, Z_if_up, gradH_if_up, H_bc_up = \
			self.up_model.generate_mat(Xf_up, Xbc_up, Xinterface)
		Z_down, Hx_down, Hy_down, _, _, _, LapH_down, Z_if_down, gradH_if_down, H_bc_down = \
			self.down_model.generate_mat(Xf_down, Xbc_down, Xinterface)

		M = self.hidden_dim

		# Initialize: u^(0) = 0 (start from zero)
		# Use accumulation vector to store beta (follows paper's approach)
		beta_accumulated = torch.zeros(2 * M, 1, device=self.device, dtype=self.dtype)

		if verbose:
			print(f"Newton-LLSQ: Starting from zero initialization")

		# Newton iteration
		for iter_count in range(max_iter):
			# ============ Step 1: Compute current solution u^(k) ============
			u_up = Z_up @ beta_accumulated[:M, :]
			u_down = Z_down @ beta_accumulated[M:, :]

			ux_up = Hx_up @ beta_accumulated[:M, :]
			ux_down = Hx_down @ beta_accumulated[M:, :]
			uy_up = Hy_up @ beta_accumulated[:M, :]
			uy_down = Hy_down @ beta_accumulated[M:, :]

			# ============ Step 2: Compute nonlinear terms at u^(k) ============
			F_up, F_down, dF_du_up, dF_du_down, dF_dux_up, dF_dux_down, dF_duy_up, dF_duy_down = \
				nonlinear_func(u_up, ux_up, uy_up, u_down, ux_down, uy_down)

			# ============ Step 3: Compute RHS for increment equation ============
			# RHS = f - [L*u^(k) + F(u^(k))]
			RHS_up = source_up - (LapH_up @ beta_accumulated[:M, :] + F_up)
			RHS_down = source_down - (LapH_down @ beta_accumulated[M:, :] + F_down)

			# Boundary condition for increment: v = g - u^(k)
			RHS_bc_up = Tbc_u - (H_bc_up @ beta_accumulated[:M, :])
			RHS_bc_down = Tbc_d - (H_bc_down @ beta_accumulated[M:, :])

			# Interface conditions for increment: v_up - v_down = -u^(k)_up + u^(k)_down
			RHS_if_u = -(Z_if_up @ beta_accumulated[:M, :] - Z_if_down @ beta_accumulated[M:, :])
			RHS_if_g = -(gradH_if_up @ beta_accumulated[:M, :] - gradH_if_down @ beta_accumulated[M:, :])

			# Check convergence before solving for increment
			residual_norm = torch.norm(torch.vstack([RHS_up, RHS_down])).item()

			if iter_count == 0:
				initial_residual = max(residual_norm, 1e-10)

			if verbose:
				print(f"Newton iter {iter_count + 1}: ||R|| = {residual_norm:.6e}")

			if residual_norm < tol * initial_residual:
				if verbose:
					print(f"  Converged!")
				break

			# ============ Step 4: Build Jacobian for increment v ============
			# Linearized equation: L*v + J*v = RHS
			# where J = dF/du * Z + dF/dux * Hx + dF/duy * Hy
			J_up = LapH_up + dF_du_up * Z_up + dF_dux_up * Hx_up + dF_duy_up * Hy_up
			J_down = LapH_down + dF_du_down * Z_down + dF_dux_down * Hx_down + dF_duy_down * Hy_down

			# Construct full Jacobian matrix
			A_row1 = torch.hstack(
				[J_up, torch.zeros(J_up.shape[0], J_down.shape[1], device=self.device, dtype=self.dtype)])
			A_row2 = torch.hstack(
				[torch.zeros(J_down.shape[0], J_up.shape[1], device=self.device, dtype=self.dtype), J_down])
			A_row3 = torch.hstack(
				[H_bc_up, torch.zeros(H_bc_up.shape[0], H_bc_down.shape[1], device=self.device, dtype=self.dtype)])
			A_row4 = torch.hstack(
				[torch.zeros(H_bc_down.shape[0], H_bc_up.shape[1], device=self.device, dtype=self.dtype), H_bc_down])
			A_row5 = torch.hstack([Z_if_up, -Z_if_down])
			A_row6 = torch.hstack([gradH_if_up, -gradH_if_down])

			A = torch.vstack([
				A_row1, A_row2,
				A_row3 * np.sqrt(50), A_row4 * np.sqrt(50),
				A_row5 * np.sqrt(10), A_row6 * np.sqrt(10)
			])

			b = torch.vstack([
				RHS_up, RHS_down,
				RHS_bc_up * np.sqrt(50), RHS_bc_down * np.sqrt(50),
				RHS_if_u * np.sqrt(10), RHS_if_g * np.sqrt(10)
			])

			# ============ Step 5: Solve linear system for increment v ============
			I_reg = torch.eye(A.shape[1], device=self.device, dtype=self.dtype) * l2_reg

			try:
				# Solve: (A^T A + lambda*I) * delta_beta = A^T * b
				delta_beta = torch.linalg.solve(A.T @ A + I_reg, A.T @ b)
			except:
				delta_beta = torch.linalg.pinv(A.T @ A + I_reg) @ (A.T @ b)

			# ============ Step 6: Update accumulated beta ============
			# u^(k+1) = u^(k) + v  =>  beta_accumulated += delta_beta
			beta_accumulated = beta_accumulated + delta_beta

			# Check if increment is small enough
			if torch.norm(delta_beta) < 1e-12:
				if verbose:
					print(f"  Increment too small, converged!")
				break

		# Update model's beta with accumulated result
		self.beta.data = beta_accumulated.reshape(-1, self.out_dim)

		# Compute final losses
		with torch.no_grad():
			u_up_final = Z_up @ beta_accumulated[:M, :]
			u_down_final = Z_down @ beta_accumulated[M:, :]

			F_up_final, F_down_final, _, _, _, _, _, _ = \
				nonlinear_func(u_up_final, Hx_up @ beta_accumulated[:M, :], Hy_up @ beta_accumulated[:M, :],
				               u_down_final, Hx_down @ beta_accumulated[M:, :], Hy_down @ beta_accumulated[M:, :])

			pde_loss = (torch.mean((LapH_up @ beta_accumulated[:M, :] + F_up_final - source_up) ** 2) +
			            torch.mean((LapH_down @ beta_accumulated[M:, :] + F_down_final - source_down) ** 2)).item()
			bc_loss = (torch.mean((H_bc_up @ beta_accumulated[:M, :] - Tbc_u) ** 2) +
			           torch.mean((H_bc_down @ beta_accumulated[M:, :] - Tbc_d) ** 2)).item()
			if_loss = (torch.mean((Z_if_up @ beta_accumulated[:M, :] - Z_if_down @ beta_accumulated[M:, :]) ** 2) +
			           torch.mean((gradH_if_up @ beta_accumulated[:M, :] - gradH_if_down @ beta_accumulated[
				           M:, :]) ** 2)).item()

		elapsed_time = time.time() - start_time

		return pde_loss, bc_loss, if_loss, elapsed_time, iter_count + 1

	def fit_newton_simple(self, Xf_up, Xf_down, Xbc_up, Xbc_down, Xinterface, Tbc_u, Tbc_d,
	                      source_up, source_down, nonlinear_coeff=1.0,
	                      max_iter=20, tol=1e-6, l2_reg=1e-7, verbose=True):
		"""
		Simplified Newton solver for common nonlinear PDEs like Burgers equation
		(Less optimized version for comparison)
		"""

		def nonlinear_burgers(u_up, ux_up, uy_up, u_down, ux_down, uy_down):
			c = nonlinear_coeff
			F_up = c * u_up * (ux_up + uy_up)
			F_down = c * u_down * (ux_down + uy_down)
			dF_du_up = c * (ux_up + uy_up)
			dF_du_down = c * (ux_down + uy_down)
			dF_dux_up = c * u_up
			dF_dux_down = c * u_down
			dF_duy_up = c * u_up
			dF_duy_down = c * u_down
			return F_up, F_down, dF_du_up, dF_du_down, dF_dux_up, dF_dux_down, dF_duy_up, dF_duy_down

		return self.fit_newton(Xf_up, Xf_down, Xbc_up, Xbc_down, Xinterface,
		                       Tbc_u, Tbc_d, source_up, source_down,
		                       nonlinear_burgers, max_iter, tol, l2_reg, verbose)


class piCKANELM_SOV2(nn.Module):
	def __init__(self, in_dim, hidden_nodes, out_dim, degree1_high, degree1_low, degree1_high_need, degree2,
	             coeff_scaled_high=1.,
	             coeff_scaled_low=1., device=torch.device('cpu'), dtype=torch.float32):
		super(piCKANELM_SOV2, self).__init__()
		self.in_dim = in_dim
		self.hidden_nodes = hidden_nodes
		self.out_dim = out_dim
		self.degree1_high = degree1_high
		self.degree1_low = degree1_low
		self.degree1_high_need = degree1_high_need
		self.degree2 = degree2

		self.cheb_coeff_high = nn.Parameter(torch.empty(degree1_high_need, hidden_nodes).to(device=device, dtype=dtype),
		                                    requires_grad=False)
		self.cheb_coeff_low = nn.Parameter(torch.empty(degree1_low + 1, hidden_nodes).to(device=device, dtype=dtype),
		                                   requires_grad=False)
		self.beta = nn.Parameter(torch.ones(hidden_nodes * (degree2 + 1), out_dim, device=device, dtype=dtype),
		                         requires_grad=False)

		nn.init.normal_(self.cheb_coeff_high, mean=0.0, std=coeff_scaled_high / (in_dim * degree1_high_need))
		nn.init.normal_(self.cheb_coeff_low, mean=0.0, std=coeff_scaled_low / (in_dim * (degree1_low + 1)))

	# 阶数衰减，对冲导数爆炸
	# degrees = torch.arange(degree1_high - degree1_high_need + 1, degree1_high + 1, device=device, dtype=dtype)
	# self.cheb_coeff_high.data = self.cheb_coeff_high.data / (degrees[:, None] / degrees[0])

	def forward(self, X):
		H1_x = self.chebyshev(X[:, 0][:, None], 1, self.degree1_low).reshape(X.shape[0], -1) @ self.cheb_coeff_low
		H1_t = self.chebyshev(X[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(X.shape[0],
		                                                                                              -1) @ self.cheb_coeff_high
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
		cheby_poly = self.chebyshev(x, dim, degree_max)  # 首先是不用cut的

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
		cheby_poly2[:, :, 0], cheby_poly2[:, :, 1] = cheby_poly2[:, :, 0].clone() * 0., cheby_poly2[
			:, :, 1].clone() * 0.
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
		# 获取边界和初始条件的矩阵
		bcmid_x = self.chebyshev(Xbc[:, 0][:, None], 1, self.degree1_low).reshape(Xbc.shape[0],
		                                                                          -1) @ self.cheb_coeff_low
		bcmid_t = self.chebyshev(Xbc[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xbc.shape[0],
		                                                                                                   -1) @ self.cheb_coeff_high
		H_bc_x = self.chebyshev(F.tanh(bcmid_x), self.hidden_nodes, self.degree2)
		H_bc_t = self.chebyshev(F.tanh(bcmid_t), self.hidden_nodes, self.degree2)
		H_bc = (H_bc_x * H_bc_t).reshape(Xbc.shape[0], -1)

		icmid_x = self.chebyshev(Xic[:, 0][:, None], 1, self.degree1_low).reshape(Xic.shape[0],
		                                                                          -1) @ self.cheb_coeff_low
		icmid_t = self.chebyshev(Xic[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xic.shape[0],
		                                                                                                   -1) @ self.cheb_coeff_high
		H_ic_x = self.chebyshev(F.tanh(icmid_x), self.hidden_nodes, self.degree2)
		H_ic_t = self.chebyshev(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
		H_ic = (H_ic_x * H_ic_t).reshape(Xic.shape[0], -1)

		# 获取初始条件的时间域一阶导
		dZdy_ic = self.chebyshev1(Xic[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(
			Xic.shape[0], -1) @ self.cheb_coeff_high
		tanh1_y_ic = 1 - F.tanh(icmid_t) ** 2
		dHxdZ_ic = self.chebyshev1(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
		Ht_ic = dHxdZ_ic * dZdy_ic[:, :, None] * tanh1_y_ic[:, :, None] * H_ic_x
		Ht_ic = Ht_ic.clone().reshape(Xic.shape[0], -1)

		# 获取内部点的矩阵
		x, t = Xtf[:, 0].clone()[:, None], Xtf[:, 1].clone()[:, None]
		Z_x = self.chebyshev(x, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_low
		Z_t = self.chebyshev(t, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0],
		                                                                              -1) @ self.cheb_coeff_high
		H_x = self.chebyshev(F.tanh(Z_x), self.hidden_nodes, self.degree2)
		H_t = self.chebyshev(F.tanh(Z_t), self.hidden_nodes, self.degree2)
		H = H_x * H_t

		# 获取导数
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

		Hxx = (ddHxdZZ * (tanh1_x * dZdx)[:, :, None] ** 2 + dHxdZ * (tanh2_x * dZdx ** 2)[:, :, None] + dHxdZ *
		       (tanh1_x * ddZdxx)[:, :, None]) * H_t
		Hyy = (ddHydZZ * (tanh1_y * dZdy)[:, :, None] ** 2 + dHydZ * (tanh2_y * dZdy ** 2)[:, :, None] + dHydZ *
		       (tanh1_y * ddZdyy)[:, :, None]) * H_x
		Hxx = Hxx.clone().reshape(Xtf.shape[0], -1)
		Hyy = Hyy.clone().reshape(Xtf.shape[0], -1)
		LapH = Hxx + Hyy

		return H, Hx, Hy, gradH, Hxx, Hyy, LapH, H_bc, H_ic, Ht_ic


class piCKANELM_SOV2_4Heat(nn.Module):
	def __init__(self, in_dim, hidden_nodes, out_dim, degree1_high, degree1_low, degree1_high_need, degree2,
	             coeff_scaled_high=1.,
	             coeff_scaled_low=1., device=torch.device('cpu'), dtype=torch.float32):
		super(piCKANELM_SOV2_4Heat, self).__init__()
		self.in_dim = in_dim
		self.hidden_nodes = hidden_nodes
		self.out_dim = out_dim
		self.degree1_high = degree1_high
		self.degree1_low = degree1_low
		self.degree1_high_need = degree1_high_need
		self.degree2 = degree2

		self.cheb_coeff_high = nn.Parameter(torch.empty(degree1_high_need, hidden_nodes).to(device=device, dtype=dtype),
		                                    requires_grad=False)
		self.cheb_coeff_low = nn.Parameter(torch.empty(degree1_low + 1, hidden_nodes).to(device=device, dtype=dtype),
		                                   requires_grad=False)
		self.beta = nn.Parameter(torch.ones(hidden_nodes * (degree2 + 1), out_dim, device=device, dtype=dtype),
		                         requires_grad=False)

		nn.init.normal_(self.cheb_coeff_high, mean=0.0, std=coeff_scaled_high / (in_dim * degree1_high_need))
		nn.init.normal_(self.cheb_coeff_low, mean=0.0, std=coeff_scaled_low / (in_dim * (degree1_low + 1)))

		# 阶数衰减，对冲导数爆炸
		degrees = torch.arange(degree1_high - degree1_high_need + 1, degree1_high + 1, device=device, dtype=dtype)
		self.cheb_coeff_high.data = self.cheb_coeff_high.data / (degrees[:, None] / degrees[0])

	def forward(self, X):
		H1_t = self.chebyshev(X[:, 1][:, None], 1, self.degree1_low).reshape(X.shape[0], -1) @ self.cheb_coeff_low
		H1_x = self.chebyshev(X[:, 0][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(X.shape[0],
		                                                                                              -1) @ self.cheb_coeff_high
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
		cheby_poly = self.chebyshev(x, dim, degree_max)  # 首先是不用cut的

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
		cheby_poly2[:, :, 0], cheby_poly2[:, :, 1] = cheby_poly2[:, :, 0].clone() * 0., cheby_poly2[
			:, :, 1].clone() * 0.
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
		# 获取边界和初始条件的矩阵
		bcmid_t = self.chebyshev(Xbc[:, 1][:, None], 1, self.degree1_low).reshape(Xbc.shape[0],
		                                                                          -1) @ self.cheb_coeff_low
		bcmid_x = self.chebyshev(Xbc[:, 0][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xbc.shape[0],
		                                                                                                   -1) @ self.cheb_coeff_high
		H_bc_x = self.chebyshev(F.tanh(bcmid_x), self.hidden_nodes, self.degree2)
		H_bc_t = self.chebyshev(F.tanh(bcmid_t), self.hidden_nodes, self.degree2)
		H_bc = (H_bc_x * H_bc_t).reshape(Xbc.shape[0], -1)

		icmid_t = self.chebyshev(Xic[:, 1][:, None], 1, self.degree1_low).reshape(Xic.shape[0],
		                                                                          -1) @ self.cheb_coeff_low
		icmid_x = self.chebyshev(Xic[:, 0][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xic.shape[0],
		                                                                                                   -1) @ self.cheb_coeff_high
		H_ic_x = self.chebyshev(F.tanh(icmid_x), self.hidden_nodes, self.degree2)
		H_ic_t = self.chebyshev(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
		H_ic = (H_ic_x * H_ic_t).reshape(Xic.shape[0], -1)

		# 获取初始条件的时间域一阶导
		dZdt_ic = self.chebyshev1(
			Xic[:, 1][:, None], 1, self.degree1_low
		).reshape(Xic.shape[0], -1) @ self.cheb_coeff_low
		tanh1_t_ic = 1 - F.tanh(icmid_t) ** 2
		dHtdZ_ic = self.chebyshev1(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
		Ht_ic = dHtdZ_ic * dZdt_ic[:, :, None] * tanh1_t_ic[:, :, None] * H_ic_x
		Ht_ic = Ht_ic.clone().reshape(Xic.shape[0], -1)

		# 获取内部点的矩阵
		x, t = Xtf[:, 0].clone()[:, None], Xtf[:, 1].clone()[:, None]
		Z_t = self.chebyshev(t, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_low
		Z_x = self.chebyshev(x, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0],
		                                                                              -1) @ self.cheb_coeff_high
		H_x = self.chebyshev(F.tanh(Z_x), self.hidden_nodes, self.degree2)
		H_t = self.chebyshev(F.tanh(Z_t), self.hidden_nodes, self.degree2)
		H = H_x * H_t

		# 获取导数
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

		Hxx = (ddHxdZZ * (tanh1_x * dZdx)[:, :, None] ** 2 + dHxdZ * (tanh2_x * dZdx ** 2)[:, :, None] + dHxdZ *
		       (tanh1_x * ddZdxx)[:, :, None]) * H_t
		Hyy = (ddHydZZ * (tanh1_y * dZdy)[:, :, None] ** 2 + dHydZ * (tanh2_y * dZdy ** 2)[:, :, None] + dHydZ *
		       (tanh1_y * ddZdyy)[:, :, None]) * H_x
		Hxx = Hxx.clone().reshape(Xtf.shape[0], -1)
		Hyy = Hyy.clone().reshape(Xtf.shape[0], -1)
		LapH = Hxx + Hyy

		return H, Hx, Hy, gradH, Hxx, Hyy, LapH, H_bc, H_ic


class sub_piCKAN_fixed(nn.Module):
	def __init__(self, in_dim, hidden_nodes, out_dim=1, degree1=3, degree2=3, coeff_scaled=1.,
	             device=torch.device('cpu'), dtype=torch.float32):
		super(sub_piCKAN_fixed, self).__init__()
		self.in_dim = in_dim
		self.hidden_nodes = hidden_nodes
		self.out_dim = out_dim
		self.degree1 = degree1
		self.degree2 = degree2

		self.cheby_coeff = nn.Parameter(
			torch.empty(in_dim * (degree1 + 1), hidden_nodes).to(device=device, dtype=dtype), requires_grad=False)
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
		cheby_poly1 = torch.ones(x.shape[0], dim, degree + 1).to(device=x.device, dtype=x.dtype)
		cheby_poly1[:, :, 0] = cheby_poly1[:, :, 0].clone() * 0.
		cheby_poly = self.chebyshev(x, dim, degree)

		if degree >= 2:
			cheby_poly1[:, :, 2] = 4 * x
		if degree >= 3:
			cheby_poly1[:, :, 3] = 12 * x ** 2 - 3.

		for i in range(4, degree + 1):
			cheby_poly1[:, :, i] = 2 * cheby_poly[:, :, i - 1].clone() + 2 * x * cheby_poly1[:, :, i - 1].clone() - \
			                       cheby_poly1[:, :, i - 2].clone()

		return cheby_poly1

	def chebyshev2(self, x, dim, degree):
		cheby_poly2 = torch.ones(x.shape[0], dim, degree + 1, dtype=x.dtype, device=x.device)
		cheby_poly2[:, :, 0], cheby_poly2[:, :, 1] = cheby_poly2[:, :, 0].clone() * 0., cheby_poly2[
			:, :, 1].clone() * 0.
		cheby_poly1 = self.chebyshev1(x, dim, degree)

		if degree >= 2:
			cheby_poly2[:, :, 2] = 4.
		if degree >= 3:
			cheby_poly2[:, :, 3] = 24 * x

		for i in range(4, degree + 1):
			cheby_poly2[:, :, i] = 4 * cheby_poly1[:, :, i - 1].clone() + 2 * x * cheby_poly2[:, :, i - 1].clone() - \
			                       cheby_poly2[:, :, i - 2].clone()

		return cheby_poly2

	def generate_H(self, Xf, Xbc, Xic, n=(0, 1)):
		# 获取边界和初始条件的矩阵
		H_bc = self.forward(Xbc)
		H_ic = self.forward(Xic)

		# 单变量的系数
		w_x = self.cheby_coeff[:self.degree1 + 1, :]
		w_y = self.cheby_coeff[self.degree1 + 1:, :]

		# 获得界面连续性梯度算子问题
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

		# 获取内部点的矩阵
		x, y = Xf[:, 0].clone()[:, None], Xf[:, 1].clone()[:, None]
		Z = self.chebyshev(Xf, self.in_dim, self.degree1).reshape(Xf.shape[0], -1) @ self.cheby_coeff
		H = self.chebyshev(F.tanh(Z), self.hidden_nodes, self.degree2).reshape(Xf.shape[0], -1)

		# 获取导数
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

		Hxx = ddHdZZ * (tanh1 * dZdx)[:, :, None] ** 2 + dHdZ * (tanh2 * dZdx ** 2)[:, :, None] + dHdZ * \
		      (tanh1 * ddZdxx)[:, :, None]
		Hyy = ddHdZZ * (tanh1 * dZdy)[:, :, None] ** 2 + dHdZ * (tanh2 * dZdy ** 2)[:, :, None] + dHdZ * \
		      (tanh1 * ddZdyy)[:, :, None]
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

		self.ckan_up = sub_piCKAN_fixed(in_dim, hidden, out_dim, degree_sub_up, degree2, coeff_scaled=coeff_scaled_up,
		                                device=device, dtype=dtype)
		self.ckan_down = sub_piCKAN_fixed(in_dim, hidden, out_dim, degree_sub_down, degree2,
		                                  coeff_scaled=coeff_scaled_down, device=device, dtype=dtype)

		self.beta = nn.Parameter(torch.ones(2 * hidden * (degree2 + 1), device=device, dtype=dtype),
		                         requires_grad=False)

	def forward(self, x_up, x_down):
		H_up = self.ckan_up(x_up)
		H_down = self.ckan_down(x_down)
		output_up = H_up @ self.beta[:self.hidden * (self.degree2 + 1)]
		output_down = H_down @ self.beta[self.hidden * (self.degree2 + 1):]
		# Return as column vectors for compatibility with plotting
		return torch.vstack([output_up.reshape(-1, 1), output_down.reshape(-1, 1)])

	def fit(self, up, down, bc_up, bc_down, interface, Tbc_up, Tbc_down, l2_reg=1e-8):
		start_time = time.time()

		# Get device and dtype from input
		device = up.device
		dtype = up.dtype

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
		b = torch.vstack(
			[rhs_1, rhs_2, np.sqrt(50) * Tbc_up, np.sqrt(50) * Tbc_down, np.sqrt(10) * rhs_5, np.sqrt(10) * rhs_6])
		# beta = torch.linalg.pinv(A) @ b
		I_reg = torch.eye(A.shape[1], device=device, dtype=dtype) * l2_reg
		beta = torch.linalg.solve(A.T @ A + I_reg, A.T @ b)
		self.beta.data = beta
		elapsed_time = time.time() - start_time

		pde_loss = (torch.mean((row1 @ beta) ** 2) + torch.mean((row2 @ beta) ** 2)).detach().cpu().numpy()
		bc_loss = (torch.mean(((row3 @ beta) - Tbc_up) ** 2) + torch.mean(
			((row4 @ beta) - Tbc_down) ** 2)).detach().cpu().numpy()
		inter_loss = torch.mean((row5 @ beta) ** 2).detach().cpu().numpy() + torch.mean(
			(row6 @ beta) ** 2).detach().cpu().numpy()

		return pde_loss, bc_loss, inter_loss, elapsed_time

	def fit_newton(self, up, down, bc_up, bc_down, interface, Tbc_up, Tbc_down,
	               source_up, source_down, nonlinear_func,
	               max_iter=20, tol=1e-6, l2_reg=1e-8, verbose=True, callback=None):
		"""
		Newton-LLSQ method for nonlinear PDE solving

		Parameters:
		-----------
		up, down : torch.Tensor
			Interior collocation points for upper and lower domains
		bc_up, bc_down : torch.Tensor
			Boundary collocation points
		interface : torch.Tensor
			Interface collocation points
		Tbc_up, Tbc_down : torch.Tensor
			Boundary condition values
		source_up, source_down : torch.Tensor
			Source terms f(x,y) for the PDE
		nonlinear_func : callable
			Function to compute nonlinear terms F(u, u_x, u_y)
			Should return (F_up, F_down, dF_du_up, dF_du_down, dF_dux_up, dF_dux_down, dF_duy_up, dF_duy_down)
		max_iter : int
			Maximum Newton iterations
		tol : float
			Convergence tolerance
		l2_reg : float
			L2 regularization parameter
		verbose : bool
			Print iteration information

		Returns:
		--------
		pde_loss, bc_loss, inter_loss, elapsed_time, newton_iters
		"""
		import time
		start_time = time.time()

		# Time spent in the user callback (e.g. error evaluation) is excluded from training time
		callback_time = 0.0

		# Initialize beta to zero (or use previous solution if available)
		beta = torch.zeros_like(self.beta.data).reshape(-1, 1)  # Make it (2*M, 1)

		# Get device and dtype
		device = beta.device
		dtype = beta.dtype

		# Pre-compute matrices (these don't change during Newton iteration)
		Mat_up = self.ckan_up.generate_H(up, bc_up, interface)
		Mat_down = self.ckan_down.generate_H(down, bc_down, interface)

		H_up, Hx_up, Hy_up, _, _, _, LapH_up, gradH_ic_up, H_bc_up, H_ic_up = Mat_up
		H_down, Hx_down, Hy_down, _, _, _, LapH_down, gradH_ic_down, H_bc_down, H_ic_down = Mat_down

		# Split beta indices
		M = self.hidden * (self.degree2 + 1)

		# Newton iteration
		for iter_count in range(max_iter):
			# ==================== Step 1: Compute current solution ====================
			# u = H @ beta
			u_up = H_up @ beta[:M, :]
			u_down = H_down @ beta[M:, :]

			# u_x = Hx @ beta, u_y = Hy @ beta
			ux_up = Hx_up @ beta[:M, :]
			ux_down = Hx_down @ beta[M:, :]
			uy_up = Hy_up @ beta[:M, :]
			uy_down = Hy_down @ beta[M:, :]

			# ==================== Step 2: Compute nonlinear terms ====================
			# F(u, u_x, u_y) and its derivatives
			F_up, F_down, dF_du_up, dF_du_down, dF_dux_up, dF_dux_down, dF_duy_up, dF_duy_down = \
				nonlinear_func(u_up, ux_up, uy_up, u_down, ux_down, uy_down)

			# ==================== Step 3: Compute residual ====================
			# R = Lap(u) + F(u, u_x, u_y) - f
			R_up = LapH_up @ beta[:M, :] + F_up - source_up
			R_down = LapH_down @ beta[M:, :] + F_down - source_down

			# Boundary residuals
			R_bc_up = H_bc_up @ beta[:M, :] - Tbc_up
			R_bc_down = H_bc_down @ beta[M:, :] - Tbc_down

			# Interface residuals (C0 and C1 continuity)
			R_ic_c0 = H_ic_up @ beta[:M, :] - H_ic_down @ beta[M:, :]
			R_ic_c1 = gradH_ic_up @ beta[:M, :] - gradH_ic_down @ beta[M:, :]

			# ==================== Step 4: Compute Jacobian ====================
			# For the linearized increment equation: J @ delta_beta = -R
			# J_pde = Lap(H) + dF/du * H + dF/dux * Hx + dF/duy * Hy

			# For upper domain
			dF_du_diag_up = torch.diag_embed(dF_du_up.squeeze())
			dF_dux_diag_up = torch.diag_embed(dF_dux_up.squeeze())
			dF_duy_diag_up = torch.diag_embed(dF_duy_up.squeeze())

			J_up_up = LapH_up + dF_du_diag_up @ H_up + dF_dux_diag_up @ Hx_up + dF_duy_diag_up @ Hy_up
			J_up_down = torch.zeros(J_up_up.shape[0], LapH_down.shape[1], device=device, dtype=dtype)

			# For lower domain
			dF_du_diag_down = torch.diag_embed(dF_du_down.squeeze())
			dF_dux_diag_down = torch.diag_embed(dF_dux_down.squeeze())
			dF_duy_diag_down = torch.diag_embed(dF_duy_down.squeeze())

			J_down_up = torch.zeros(LapH_down.shape[0], J_up_up.shape[1], device=device, dtype=dtype)
			J_down_down = LapH_down + dF_du_diag_down @ H_down + dF_dux_diag_down @ Hx_down + dF_duy_diag_down @ Hy_down

			# Boundary Jacobian (linear, same as original)
			J_bc_up = torch.hstack(
				[H_bc_up, torch.zeros(H_bc_up.shape[0], H_bc_down.shape[1], device=device, dtype=dtype)])
			J_bc_down = torch.hstack(
				[torch.zeros(H_bc_down.shape[0], H_bc_up.shape[1], device=device, dtype=dtype), H_bc_down])

			# Interface Jacobian (linear, same as original)
			J_ic_c0 = torch.hstack([H_ic_up, -H_ic_down])
			J_ic_c1 = torch.hstack([gradH_ic_up, -gradH_ic_down])

			# ==================== Step 5: Assemble linear system ====================
			# Apply weights to different constraint types
			weight_bc = np.sqrt(50)
			weight_ic = np.sqrt(10)

			J_row1 = torch.hstack([J_up_up, J_up_down])
			J_row2 = torch.hstack([J_down_up, J_down_down])

			J = torch.vstack([
				J_row1,
				J_row2,
				weight_bc * J_bc_up,
				weight_bc * J_bc_down,
				weight_ic * J_ic_c0,
				weight_ic * J_ic_c1
			])

			R_all = torch.vstack([
				R_up,
				R_down,
				weight_bc * R_bc_up,
				weight_bc * R_bc_down,
				weight_ic * R_ic_c0,
				weight_ic * R_ic_c1
			])

			# ==================== Step 6: Solve for increment ====================
			# J @ delta_beta = -R  =>  delta_beta = -(J^T J + reg)^{-1} J^T R
			I_reg = torch.eye(J.shape[1], device=device, dtype=dtype) * l2_reg

			# Use more stable solver with increased regularization if needed
			try:
				delta_beta = -torch.linalg.solve(J.T @ J + I_reg, J.T @ R_all)
			except RuntimeError:
				# If singular, increase regularization
				I_reg = torch.eye(J.shape[1], device=device, dtype=dtype) * (l2_reg * 100)
				delta_beta = -torch.linalg.solve(J.T @ J + I_reg, J.T @ R_all)

			# ==================== Step 7: Update solution with line search ====================
			# Simple backtracking line search for stability
			alpha = 1.0
			beta_new = beta + alpha * delta_beta
			residual_norm = torch.norm(R_all).item()

			# Try smaller steps if residual increases too much
			for ls_iter in range(5):
				# Compute new residual with beta_new
				u_up_new = H_up @ beta_new[:M, :]
				u_down_new = H_down @ beta_new[M:, :]
				ux_up_new = Hx_up @ beta_new[:M, :]
				ux_down_new = Hx_down @ beta_new[M:, :]
				uy_up_new = Hy_up @ beta_new[:M, :]
				uy_down_new = Hy_down @ beta_new[M:, :]

				F_up_new, F_down_new, _, _, _, _, _, _ = \
					nonlinear_func(u_up_new, ux_up_new, uy_up_new, u_down_new, ux_down_new, uy_down_new)

				R_up_new = LapH_up @ beta_new[:M, :] + F_up_new - source_up
				R_down_new = LapH_down @ beta_new[M:, :] + F_down_new - source_down
				R_bc_up_new = H_bc_up @ beta_new[:M, :] - Tbc_up
				R_bc_down_new = H_bc_down @ beta_new[M:, :] - Tbc_down
				R_ic_c0_new = H_ic_up @ beta_new[:M, :] - H_ic_down @ beta_new[M:, :]
				R_ic_c1_new = gradH_ic_up @ beta_new[:M, :] - gradH_ic_down @ beta_new[M:, :]

				R_all_new = torch.vstack([R_up_new, R_down_new,
				                          weight_bc * R_bc_up_new, weight_bc * R_bc_down_new,
				                          weight_ic * R_ic_c0_new, weight_ic * R_ic_c1_new])
				residual_norm_new = torch.norm(R_all_new).item()

				# Accept if residual decreased or is small enough
				if residual_norm_new < residual_norm * 1.5 or residual_norm_new < tol * 10:
					beta = beta_new
					residual_norm = residual_norm_new
					break
				else:
					# Reduce step size
					alpha *= 0.5
					beta_new = beta + alpha * delta_beta
			else:
				# If line search failed, use smaller step anyway
				beta = beta + 0.1 * delta_beta

			# ==================== Step 8: Check convergence ====================
			delta_norm = torch.norm(delta_beta).item()
			# residual_norm already computed above

			stop_requested = False
			# Per-iteration callback (e.g. relative error recording); its time is excluded
			if callback is not None:
				t_cb = time.time()
				# Sync self.beta so that model.forward inside the callback sees the current iterate
				self.beta.data = beta.reshape(-1)
				stop_requested = bool(callback(iter_count + 1, beta.reshape(-1)))
				callback_time += time.time() - t_cb

			if verbose:
				print(f"Newton iter {iter_count + 1}: ||delta_beta|| = {delta_norm:.6e}, ||R|| = {residual_norm:.6e}")

			if stop_requested:
				break
			if delta_norm < tol:
				if verbose:
					print(f"Newton converged in {iter_count + 1} iterations")
				break

		# ==================== Step 9: Set final solution ====================
		self.beta.data = beta.reshape(-1)  # Convert back to 1D
		elapsed_time = time.time() - start_time - callback_time

		# ==================== Step 10: Compute final losses ====================
		# Recompute with final beta
		u_up_final = H_up @ beta[:M, :]
		u_down_final = H_down @ beta[M:, :]
		ux_up_final = Hx_up @ beta[:M, :]
		ux_down_final = Hx_down @ beta[M:, :]
		uy_up_final = Hy_up @ beta[:M, :]
		uy_down_final = Hy_down @ beta[M:, :]

		F_up_final, F_down_final, _, _, _, _, _, _ = \
			nonlinear_func(u_up_final, ux_up_final, uy_up_final, u_down_final, ux_down_final, uy_down_final)

		pde_loss_up = torch.mean((LapH_up @ beta[:M, :] + F_up_final - source_up) ** 2)
		pde_loss_down = torch.mean((LapH_down @ beta[M:, :] + F_down_final - source_down) ** 2)
		pde_loss = (pde_loss_up + pde_loss_down).detach().cpu().numpy()

		bc_loss_up = torch.mean((H_bc_up @ beta[:M, :] - Tbc_up) ** 2)
		bc_loss_down = torch.mean((H_bc_down @ beta[M:, :] - Tbc_down) ** 2)
		bc_loss = (bc_loss_up + bc_loss_down).detach().cpu().numpy()

		inter_loss_c0 = torch.mean((H_ic_up @ beta[:M, :] - H_ic_down @ beta[M:, :]) ** 2)
		inter_loss_c1 = torch.mean((gradH_ic_up @ beta[:M, :] - gradH_ic_down @ beta[M:, :]) ** 2)
		inter_loss = (inter_loss_c0 + inter_loss_c1).detach().cpu().numpy()

		return pde_loss, bc_loss, inter_loss, elapsed_time, iter_count + 1

	def fit_newton_simple(self, up, down, bc_up, bc_down, interface, Tbc_up, Tbc_down,
	                      source_up, source_down, nonlinear_coeff=1.0,
	                      max_iter=20, tol=1e-6, l2_reg=1e-8, verbose=True, callback=None):
		"""
		Simplified Newton solver for common nonlinear PDEs like Burgers equation

		Solves: Lap(u) + c*u*grad(u) = f(x,y)

		Parameters:
		-----------
		nonlinear_coeff : float
			Coefficient for nonlinear term (e.g., c in Burgers equation)

		Example usage for Burgers equation:

		"""

		def nonlinear_burgers(u_up, ux_up, uy_up, u_down, ux_down, uy_down):
			"""
			Nonlinear term for Burgers-type equation: F(u) = c * u * (u_x + u_y)
			Returns: F and its derivatives dF/du, dF/dux, dF/duy
			"""
			c = nonlinear_coeff

			# Nonlinear term: F = c * u * (u_x + u_y)
			F_up = c * u_up * (ux_up + uy_up)
			F_down = c * u_down * (ux_down + uy_down)

			# Derivatives:
			# dF/du = c * (u_x + u_y)
			dF_du_up = c * (ux_up + uy_up)
			dF_du_down = c * (ux_down + uy_down)

			# dF/du_x = c * u
			dF_dux_up = c * u_up
			dF_dux_down = c * u_down

			# dF/du_y = c * u
			dF_duy_up = c * u_up
			dF_duy_down = c * u_down

			return F_up, F_down, dF_du_up, dF_du_down, dF_dux_up, dF_dux_down, dF_duy_up, dF_duy_down

		return self.fit_newton(up, down, bc_up, bc_down, interface,
		                       Tbc_up, Tbc_down, source_up, source_down,
		                       nonlinear_burgers, max_iter, tol, l2_reg, verbose, callback)


# ==================== Example usage functions ====================

def example_nonlinear_poisson(model, domain_up, domain_down, bc_up, bc_down, interface):
	"""
	Example: Nonlinear Poisson equation
	Lap(u) + sin(u) = f(x,y)

	This demonstrates how to define custom nonlinear functions
	"""

	def nonlinear_sin(u_up, ux_up, uy_up, u_down, ux_down, uy_down):
		"""F(u) = sin(u), dF/du = cos(u), dF/dux = dF/duy = 0"""
		F_up = torch.sin(u_up)
		F_down = torch.sin(u_down)

		dF_du_up = torch.cos(u_up)
		dF_du_down = torch.cos(u_down)

		# No dependence on gradients
		dF_dux_up = torch.zeros_like(u_up)
		dF_dux_down = torch.zeros_like(u_down)
		dF_duy_up = torch.zeros_like(u_up)
		dF_duy_down = torch.zeros_like(u_down)

		return F_up, F_down, dF_du_up, dF_du_down, dF_dux_up, dF_dux_down, dF_duy_up, dF_duy_down

	# Define source terms (manufactured solution)
	x_up, y_up = domain_up[:, 0:1], domain_up[:, 1:2]
	x_down, y_down = domain_down[:, 0:1], domain_down[:, 1:2]

	# Example: u_exact = sin(pi*x)*sin(pi*y)
	# Then: Lap(u) = -2*pi^2*sin(pi*x)*sin(pi*y)
	# f = Lap(u_exact) + sin(u_exact)
	u_exact_up = torch.sin(np.pi * x_up) * torch.sin(np.pi * y_up)
	u_exact_down = torch.sin(np.pi * x_down) * torch.sin(np.pi * y_down)

	source_up = -2 * np.pi ** 2 * u_exact_up + torch.sin(u_exact_up)
	source_down = -2 * np.pi ** 2 * u_exact_down + torch.sin(u_exact_down)

	# Boundary conditions
	x_bc_up, y_bc_up = bc_up[:, 0:1], bc_up[:, 1:2]
	x_bc_down, y_bc_down = bc_down[:, 0:1], bc_down[:, 1:2]

	Tbc_up = torch.sin(np.pi * x_bc_up) * torch.sin(np.pi * y_bc_up)
	Tbc_down = torch.sin(np.pi * x_bc_down) * torch.sin(np.pi * y_bc_down)

	# Solve
	pde_loss, bc_loss, inter_loss, elapsed_time, n_iters = model.fit_newton(
		domain_up, domain_down, bc_up, bc_down, interface,
		Tbc_up, Tbc_down, source_up, source_down,
		nonlinear_sin, max_iter=20, tol=1e-6, verbose=True
	)

	return pde_loss, bc_loss, inter_loss, elapsed_time, n_iters


def example_drift_diffusion_carrier(model, domain_up, domain_down, bc_up, bc_down, interface,
                                    potential_up, potential_down, mobility=1.0, diffusion=1.0):
	"""
	Example: Drift-diffusion equation for carrier transport
	Lap(n) - mobility * div(n * grad(phi)) = G - R

	This is relevant for your semiconductor drift-diffusion problems
	"""

	def nonlinear_drift_diffusion(n_up, nx_up, ny_up, n_down, nx_down, ny_down):
		"""
		Nonlinear term: F(n) = -mu * div(n * grad(phi))
							 = -mu * (n_x * phi_x + n * phi_xx + n_y * phi_y + n * phi_yy)
							 ≈ -mu * (n_x * phi_x + n_y * phi_y)  (if phi is given)
		"""
		mu = mobility

		# Compute potential gradients (assuming potential is known)
		# In real application, you would compute grad(phi) from the solution
		phi_x_up = torch.autograd.grad(potential_up.sum(), domain_up, create_graph=True)[0][:, 0:1]
		phi_y_up = torch.autograd.grad(potential_up.sum(), domain_up, create_graph=True)[0][:, 1:2]
		phi_x_down = torch.autograd.grad(potential_down.sum(), domain_down, create_graph=True)[0][:, 0:1]
		phi_y_down = torch.autograd.grad(potential_down.sum(), domain_down, create_graph=True)[0][:, 1:2]

		# Drift term: -mu * (n_x * phi_x + n_y * phi_y)
		F_up = -mu * (nx_up * phi_x_up + ny_up * phi_y_up)
		F_down = -mu * (nx_down * phi_x_down + ny_down * phi_y_down)

		# Derivatives
		dF_dn_up = torch.zeros_like(n_up)
		dF_dn_down = torch.zeros_like(n_down)

		dF_dnx_up = -mu * phi_x_up
		dF_dnx_down = -mu * phi_x_down

		dF_dny_up = -mu * phi_y_up
		dF_dny_down = -mu * phi_y_down

		return F_up, F_down, dF_dn_up, dF_dn_down, dF_dnx_up, dF_dnx_down, dF_dny_up, dF_dny_down

	# Define source terms (generation - recombination)
	source_up = torch.zeros(domain_up.shape[0], 1, device=domain_up.device, dtype=domain_up.dtype)
	source_down = torch.zeros(domain_down.shape[0], 1, device=domain_down.device, dtype=domain_down.dtype)

	# Boundary conditions (example: n=0 at boundaries)
	Tbc_up = torch.zeros(bc_up.shape[0], 1, device=bc_up.device, dtype=bc_up.dtype)
	Tbc_down = torch.zeros(bc_down.shape[0], 1, device=bc_down.device, dtype=bc_down.dtype)

	# Solve
	pde_loss, bc_loss, inter_loss, elapsed_time, n_iters = model.fit_newton(
		domain_up, domain_down, bc_up, bc_down, interface,
		Tbc_up, Tbc_down, source_up, source_down,
		nonlinear_drift_diffusion, max_iter=20, tol=1e-6, verbose=True
	)

	return pde_loss, bc_loss, inter_loss, elapsed_time, n_iters


class piELMHKAN(nn.Module):
	def __init__(self, in_dim, hidden_nodes, out_dim, num_grid1=5, num_grid2=5, knot_grid1=[-1., 1.],
	             knot_grid2=[-1, 1.], coeff1_sacle=1.5,
	             sigmoid_scale=1., device=torch.device('cpu'), dtype=torch.float32):
		super(piELMHKAN, self).__init__()
		self.in_dim = in_dim
		self.hidden_nodes = hidden_nodes
		self.out_dim = out_dim
		self.num_grid1 = num_grid1
		self.num_grid2 = num_grid2

		# 两种基函数的网格
		self.grid1 = torch.linspace(knot_grid1[0], knot_grid1[1], num_grid1 + 1, device=device, dtype=dtype)[
			None, :].expand(in_dim, num_grid1 + 1)
		self.grid2 = torch.linspace(knot_grid2[0], knot_grid2[1], num_grid2, device=device, dtype=dtype)[
			None, :].expand(hidden_nodes, num_grid2)
		# 两层的控制系数
		ctrl = np.random.uniform(-coeff1_sacle, coeff1_sacle, size=(in_dim, hidden_nodes, num_grid1)).transpose(0, 2,
		                                                                                                        1).reshape(
			in_dim * num_grid1, hidden_nodes)
		self.ctrl = nn.Parameter(torch.from_numpy(ctrl).to(device=device, dtype=dtype), requires_grad=False)
		self.beta = nn.Parameter(torch.zeros(hidden_nodes * num_grid2, out_dim, device=device, dtype=dtype),
		                         requires_grad=False)

		# 第一层基函数
		self.act1 = lambda x, grid: torch.sigmoid(sigmoid_scale * (x[:, :, None] - grid[None, :, :]))
		self.sigmoid_scale = sigmoid_scale
		# 第二层基函数
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
		"""一阶导DiffSigmoid"""
		sgm_1 = self.sigmoid_scale * (self.act1(x, grid) - self.act1(x, grid) ** 2)
		bumps = sgm_1[:, :, 1:] - sgm_1[:, :, :-1]
		return bumps

	def sig_2(self, x, grid):
		"""二阶导DiffSigmoid"""
		sgm_1 = self.sigmoid_scale * (self.act1(x, grid) - self.act1(x, grid) ** 2)
		sgm_2 = self.sigmoid_scale * (sgm_1 - 2 * self.act1(x, grid) * sgm_1)
		bumps = sgm_2[:, :, 1:] - sgm_2[:, :, :-1]
		return bumps

	def phi_1(self, x):
		"""一阶导GaussianRBF"""
		return self.act2(x, self.grid2) * -(2 / self.dist ** 2) * (x[:, :, None] - self.grid2[None, :, :])

	def phi_2(self, x):
		"""二阶导GaussianRBF"""
		part1 = self.act2(x, self.grid2) * (-(2 / self.dist ** 2) * (x[:, :, None] - self.grid2[None, :, :])) ** 2
		part2 = self.act2(x, self.grid2) * -(2 / self.dist ** 2)
		return part1 + part2

	def generate_H(self, Xf, Xbc, Xic=None):
		# 获得边界条件的特征矩阵
		H_bc = self.act2(self.sig(Xbc, self.grid1).reshape(Xbc.shape[0], -1) @ self.ctrl, self.grid2).reshape(
			Xbc.shape[0], -1)
		if Xic is not None:
			H_ic = self.act2(self.sig(Xic, self.grid1).reshape(Xic.shape[0], -1) @ self.ctrl, self.grid2).reshape(
				Xic.shape[0], -1)
		else:
			H_ic = None

		# 获取内部点的特征矩阵与pde矩阵
		# 数值矩阵以及单独的变量
		Z = self.sig(Xf, self.grid1).reshape(Xf.shape[0], -1) @ self.ctrl
		H = self.act2(Z, self.grid2).reshape(Xf.shape[0], -1)
		w_x = self.ctrl[:self.num_grid1, :].T  # [hidden_nodes, num_grid1]
		w_y = self.ctrl[self.num_grid1:, :].T  # [hidden_nodes, num_grid2]
		x = Xf[:, 0][:, None]
		y = Xf[:, 1][:, None]
		grid1_x = self.grid1[0, :][None, :]  # [1, num_grid1]
		grid1_y = self.grid1[1, :][None, :]  # [1, num_grid1]
		# 内部函数一阶导与二阶导
		sig_1_x = self.sig_1(x, grid1_x).permute(0, 2, 1).squeeze()
		sig_1_y = self.sig_1(y, grid1_y).permute(0, 2, 1).squeeze()
		sig_2_x = self.sig_2(x, grid1_x).permute(0, 2, 1).squeeze()
		sig_2_y = self.sig_2(y, grid1_y).permute(0, 2, 1).squeeze()
		dZdx = sig_1_x @ w_x.T
		dZdy = sig_1_y @ w_y.T
		dZZdxx = sig_2_x @ w_x.T
		dZZdyy = sig_2_y @ w_y.T
		# 一阶导
		# print(dZdx.shape, dZZdxx.shape)
		Hx = (self.phi_1(Z) * dZdx[:, :, None]).reshape(x.shape[0], -1)
		Hy = (self.phi_1(Z) * dZdy[:, :, None]).reshape(y.shape[0], -1)
		gradHf = Hx + Hy
		# 二阶导
		Hxx = (self.phi_2(Z) * dZdx[:, :, None] ** 2 + self.phi_1(Z) * dZZdxx[:, :, None]).reshape(Xf.shape[0], -1)
		Hyy = (self.phi_2(Z) * dZdy[:, :, None] ** 2 + self.phi_1(Z) * dZZdyy[:, :, None]).reshape(Xf.shape[0], -1)
		LapHf = (Hxx + Hyy).reshape(Xf.shape[0], -1)

		return H, Hx, Hy, gradHf, Hxx, Hyy, LapHf, H_bc, H_ic


# =============== ablation 的架构 ==============#

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
		cheby_poly1 = torch.ones(x.shape[0], dim, degree + 1).to(device=x.device, dtype=x.dtype)
		cheby_poly1[:, :, 0] = cheby_poly1[:, :, 0].clone() * 0.
		cheby_poly = self.chebyshev(x, dim, degree)

		if degree >= 2:
			cheby_poly1[:, :, 2] = 4 * x
		if degree >= 3:
			cheby_poly1[:, :, 3] = 12 * x ** 2 - 3.

		for i in range(4, degree + 1):
			cheby_poly1[:, :, i] = 2 * cheby_poly[:, :, i - 1].clone() + 2 * x * cheby_poly1[:, :, i - 1].clone() - \
			                       cheby_poly1[:, :, i - 2].clone()

		return cheby_poly1

	def chebyshev2(self, x, dim, degree):
		cheby_poly2 = torch.ones(x.shape[0], dim, degree + 1, dtype=x.dtype, device=x.device)
		cheby_poly2[:, :, 0], cheby_poly2[:, :, 1] = cheby_poly2[:, :, 0].clone() * 0., cheby_poly2[
			:, :, 1].clone() * 0.
		cheby_poly1 = self.chebyshev1(x, dim, degree)

		if degree >= 2:
			cheby_poly2[:, :, 2] = 4.
		if degree >= 3:
			cheby_poly2[:, :, 3] = 24 * x

		for i in range(4, degree + 1):
			cheby_poly2[:, :, i] = 4 * cheby_poly1[:, :, i - 1].clone() + 2 * x * cheby_poly2[:, :, i - 1].clone() - \
			                       cheby_poly2[:, :, i - 2].clone()

		return cheby_poly2

	def generate_H(self, Xf, Xbc, Xic, n=(0, 1)):
		# 统一单分支隐变量 Z(x, y)，并生成不含输出层 beta 的特征矩阵
		Z_bc = self.chebyshev(Xbc, self.in_dim, self.degree1).reshape(Xbc.shape[0], -1) @ self.cheby_coeff
		Z_ic = self.chebyshev(Xic, self.in_dim, self.degree1).reshape(Xic.shape[0], -1) @ self.cheby_coeff
		H_bc = self.chebyshev(F.tanh(Z_bc), self.hidden_nodes, self.degree2).reshape(Xbc.shape[0], -1)
		H_ic = self.chebyshev(F.tanh(Z_ic), self.hidden_nodes, self.degree2).reshape(Xic.shape[0], -1)

		# 单变量的系数
		w_x = self.cheby_coeff[:self.degree1 + 1, :]
		w_y = self.cheby_coeff[self.degree1 + 1:, :]

		# 获得界面连续性梯度算子问题
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

		# 获取内部点的矩阵
		x, y = Xf[:, 0].clone()[:, None], Xf[:, 1].clone()[:, None]
		Z = self.chebyshev(Xf, self.in_dim, self.degree1).reshape(Xf.shape[0], -1) @ self.cheby_coeff
		H = self.chebyshev(F.tanh(Z), self.hidden_nodes, self.degree2).reshape(Xf.shape[0], -1)

		# 获取导数
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

		Hxx = ddHdZZ * (tanh1 * dZdx)[:, :, None] ** 2 + dHdZ * (tanh2 * dZdx ** 2)[:, :, None] + dHdZ * \
		      (tanh1 * ddZdxx)[:, :, None]
		Hyy = ddHdZZ * (tanh1 * dZdy)[:, :, None] ** 2 + dHdZ * (tanh2 * dZdy ** 2)[:, :, None] + dHdZ * \
		      (tanh1 * ddZdyy)[:, :, None]
		Hxx = Hxx.clone().reshape(Xf.shape[0], -1)
		Hyy = Hyy.clone().reshape(Xf.shape[0], -1)
		LapH = Hxx + Hyy

		return H, Hx, Hy, gradH, Hxx, Hyy, LapH, gradH_ic, H_bc, H_ic


class piCKANELM_SOV2_add_fusion(nn.Module):
	def __init__(self, in_dim, hidden_nodes, out_dim, degree1_high, degree1_low, degree1_high_need, degree2,
	             coeff_scaled_high=1.,
	             coeff_scaled_low=1., device=torch.device('cpu'), dtype=torch.float32):
		super(piCKANELM_SOV2_add_fusion, self).__init__()
		self.in_dim = in_dim
		self.hidden_nodes = hidden_nodes
		self.out_dim = out_dim
		self.degree1_high = degree1_high
		self.degree1_low = degree1_low
		self.degree1_high_need = degree1_high_need
		self.degree2 = degree2

		self.cheb_coeff_high = nn.Parameter(torch.empty(degree1_high_need, hidden_nodes).to(device=device, dtype=dtype),
		                                    requires_grad=False)
		self.cheb_coeff_low = nn.Parameter(torch.empty(degree1_low + 1, hidden_nodes).to(device=device, dtype=dtype),
		                                   requires_grad=False)
		self.beta = nn.Parameter(torch.ones(hidden_nodes * (degree2 + 1), out_dim, device=device, dtype=dtype),
		                         requires_grad=False)

		nn.init.normal_(self.cheb_coeff_high, mean=0.0, std=coeff_scaled_high / (in_dim * degree1_high_need))
		nn.init.normal_(self.cheb_coeff_low, mean=0.0, std=coeff_scaled_low / (in_dim * (degree1_low + 1)))

		# 阶数衰减，对冲导数爆炸
		degrees = torch.arange(degree1_high - degree1_high_need + 1, degree1_high + 1, device=device, dtype=dtype)
		self.cheb_coeff_high.data = self.cheb_coeff_high.data / (degrees[:, None] / degrees[0])

	def forward(self, X):
		H1_x = self.chebyshev(X[:, 0][:, None], 1, self.degree1_low).reshape(X.shape[0], -1) @ self.cheb_coeff_low
		H1_t = self.chebyshev(X[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(X.shape[0],
		                                                                                              -1) @ self.cheb_coeff_high
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
		cheby_poly = self.chebyshev(x, dim, degree_max)  # 首先是不用cut的

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
		cheby_poly2[:, :, 0], cheby_poly2[:, :, 1] = cheby_poly2[:, :, 0].clone() * 0., cheby_poly2[
			:, :, 1].clone() * 0.
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
		# 获取边界和初始条件的矩阵
		bcmid_x = self.chebyshev(Xbc[:, 0][:, None], 1, self.degree1_low).reshape(Xbc.shape[0],
		                                                                          -1) @ self.cheb_coeff_low
		bcmid_t = self.chebyshev(Xbc[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xbc.shape[0],
		                                                                                                   -1) @ self.cheb_coeff_high
		H_bc_x = self.chebyshev(F.tanh(bcmid_x), self.hidden_nodes, self.degree2)
		H_bc_t = self.chebyshev(F.tanh(bcmid_t), self.hidden_nodes, self.degree2)
		H_bc = (H_bc_x + H_bc_t).reshape(Xbc.shape[0], -1)

		icmid_x = self.chebyshev(Xic[:, 0][:, None], 1, self.degree1_low).reshape(Xic.shape[0],
		                                                                          -1) @ self.cheb_coeff_low
		icmid_t = self.chebyshev(Xic[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(Xic.shape[0],
		                                                                                                   -1) @ self.cheb_coeff_high
		H_ic_x = self.chebyshev(F.tanh(icmid_x), self.hidden_nodes, self.degree2)
		H_ic_t = self.chebyshev(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
		H_ic = (H_ic_x + H_ic_t).reshape(Xic.shape[0], -1)

		# 获取初始条件的时间域一阶导
		dZdy_ic = self.chebyshev1(Xic[:, 1][:, None], 1, self.degree1_high, self.degree1_high_need).reshape(
			Xic.shape[0], -1) @ self.cheb_coeff_high
		tanh1_y_ic = 1 - F.tanh(icmid_t) ** 2
		dHxdZ_ic = self.chebyshev1(F.tanh(icmid_t), self.hidden_nodes, self.degree2)
		Ht_ic = dHxdZ_ic * dZdy_ic[:, :, None] * tanh1_y_ic[:, :, None]
		Ht_ic = Ht_ic.clone().reshape(Xic.shape[0], -1)

		# add fusion: each branch depends on only its own coordinate

		# 获取内部点的矩阵
		x, t = Xtf[:, 0].clone()[:, None], Xtf[:, 1].clone()[:, None]
		Z_x = self.chebyshev(x, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_low
		Z_t = self.chebyshev(t, 1, self.degree1_high, self.degree1_high_need).reshape(Xtf.shape[0],
		                                                                              -1) @ self.cheb_coeff_high
		H_x = self.chebyshev(F.tanh(Z_x), self.hidden_nodes, self.degree2)
		H_t = self.chebyshev(F.tanh(Z_t), self.hidden_nodes, self.degree2)
		H = H_x + H_t

		# 获取导数（add fusion：无交叉乘积项）
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

		Hxx = (ddHxdZZ * (tanh1_x * dZdx)[:, :, None] ** 2 + dHxdZ * (tanh2_x * dZdx ** 2)[:, :, None] + dHxdZ *
		       (tanh1_x * ddZdxx)[:, :, None])
		Hyy = (ddHydZZ * (tanh1_y * dZdy)[:, :, None] ** 2 + dHydZ * (tanh2_y * dZdy ** 2)[:, :, None] + dHydZ *
		       (tanh1_y * ddZdyy)[:, :, None])
		Hxx = Hxx.clone().reshape(Xtf.shape[0], -1)
		Hyy = Hyy.clone().reshape(Xtf.shape[0], -1)
		LapH = Hxx + Hyy

		return H, Hx, Hy, gradH, Hxx, Hyy, LapH, H_bc, H_ic, Ht_ic


class piCKANELM_SOV(nn.Module):
	def __init__(self, in_dim, hidden_nodes, out_dim, degree1_high, degree1_low, degree2, coeff_scaled_high=1.,
	             coeff_scaled_low=1.,
	             device=torch.device('cpu'), dtype=torch.float32):
		super(piCKANELM_SOV, self).__init__()
		self.in_dim = in_dim
		self.hidden_nodes = hidden_nodes
		self.out_dim = out_dim
		self.degree1_high = degree1_high
		self.degree1_low = degree1_low
		self.degree2 = degree2

		self.cheb_coeff_high = nn.Parameter(torch.empty(degree1_high + 1, hidden_nodes).to(device=device, dtype=dtype),
		                                    requires_grad=False)
		self.cheb_coeff_low = nn.Parameter(torch.empty(degree1_low + 1, hidden_nodes).to(device=device, dtype=dtype),
		                                   requires_grad=False)
		self.beta = nn.Parameter(torch.ones(hidden_nodes * (degree2 + 1), out_dim, device=device, dtype=dtype),
		                         requires_grad=False)

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
		cheby_poly1 = torch.ones(x.shape[0], dim, degree + 1).to(device=x.device, dtype=x.dtype)
		cheby_poly1[:, :, 0] = cheby_poly1[:, :, 0].clone() * 0.
		cheby_poly = self.chebyshev(x, dim, degree)

		if degree >= 2:
			cheby_poly1[:, :, 2] = 4 * x
		if degree >= 3:
			cheby_poly1[:, :, 3] = 12 * x ** 2 - 3.

		for i in range(4, degree + 1):
			cheby_poly1[:, :, i] = 2 * cheby_poly[:, :, i - 1].clone() + 2 * x * cheby_poly1[:, :, i - 1].clone() - \
			                       cheby_poly1[:, :, i - 2].clone()

		return cheby_poly1

	def chebyshev2(self, x, dim, degree):
		cheby_poly2 = torch.ones(x.shape[0], dim, degree + 1, dtype=x.dtype, device=x.device)
		cheby_poly2[:, :, 0], cheby_poly2[:, :, 1] = cheby_poly2[:, :, 0].clone() * 0., cheby_poly2[
			:, :, 1].clone() * 0.
		cheby_poly1 = self.chebyshev1(x, dim, degree)

		if degree >= 2:
			cheby_poly2[:, :, 2] = 4.
		if degree >= 3:
			cheby_poly2[:, :, 3] = 24 * x

		for i in range(4, degree + 1):
			cheby_poly2[:, :, i] = 4 * cheby_poly1[:, :, i - 1].clone() + 2 * x * cheby_poly2[:, :, i - 1].clone() - \
			                       cheby_poly2[:, :, i - 2].clone()

		return cheby_poly2

	def generate_H(self, Xtf, Xbc, Xic):
		# 获取边界和初始条件的矩阵
		bcmid_x = self.chebyshev(Xbc[:, 0][:, None], 1, self.degree1_high).reshape(Xbc.shape[0],
		                                                                           -1) @ self.cheb_coeff_high
		bcmid_t = self.chebyshev(Xbc[:, 1][:, None], 1, self.degree1_low).reshape(Xbc.shape[0],
		                                                                          -1) @ self.cheb_coeff_low
		H_bc_x = self.chebyshev(F.tanh(bcmid_x), self.hidden_nodes, self.degree2).reshape(Xbc.shape[0], -1)
		H_bc_t = self.chebyshev(F.tanh(bcmid_t), self.hidden_nodes, self.degree2).reshape(Xbc.shape[0], -1)
		H_bc = H_bc_x * H_bc_t

		icmid_x = self.chebyshev(Xic[:, 0][:, None], 1, self.degree1_high).reshape(Xic.shape[0],
		                                                                           -1) @ self.cheb_coeff_high
		icmid_t = self.chebyshev(Xic[:, 1][:, None], 1, self.degree1_low).reshape(Xic.shape[0],
		                                                                          -1) @ self.cheb_coeff_low
		H_ic_x = self.chebyshev(F.tanh(icmid_x), self.hidden_nodes, self.degree2).reshape(Xic.shape[0], -1)
		H_ic_t = self.chebyshev(F.tanh(icmid_t), self.hidden_nodes, self.degree2).reshape(Xic.shape[0], -1)
		H_ic = H_ic_x * H_ic_t

		# 获取内部点的矩阵
		x, t = Xtf[:, 0].clone()[:, None], Xtf[:, 1].clone()[:, None]
		Z_x = self.chebyshev(x, 1, self.degree1_high).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_high
		Z_t = self.chebyshev(t, 1, self.degree1_low).reshape(Xtf.shape[0], -1) @ self.cheb_coeff_low
		H_x = self.chebyshev(F.tanh(Z_x), self.hidden_nodes, self.degree2)
		H_t = self.chebyshev(F.tanh(Z_t), self.hidden_nodes, self.degree2)
		H = H_x * H_t

		# 获取导数
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

		Hxx = (ddHxdZZ * (tanh1_x * dZdx)[:, :, None] ** 2 + dHxdZ * (tanh2_x * dZdx ** 2)[:, :, None] + dHxdZ *
		       (tanh1_x * ddZdxx)[:, :, None]) * H_t
		Hyy = (ddHydZZ * (tanh1_y * dZdy)[:, :, None] ** 2 + dHydZ * (tanh2_y * dZdy ** 2)[:, :, None] + dHydZ *
		       (tanh1_y * ddZdyy)[:, :, None]) * H_x
		Hxx = Hxx.clone().reshape(Xtf.shape[0], -1)
		Hyy = Hyy.clone().reshape(Xtf.shape[0], -1)
		LapH = Hxx + Hyy

		return H, Hx, Hy, gradH, Hxx, Hyy, LapH, H_bc, H_ic


if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dtype = torch.float64
	torch.set_default_dtype(dtype)
