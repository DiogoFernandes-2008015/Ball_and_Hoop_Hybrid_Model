# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:12:38 2025

@author: IDRZBOOK002
"""

import numpy as np
import scipy as sc
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat
from tqdm import tqdm # For the progress bar
import os
import scipy.io
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, LinearInterpolation
try:
    print(f"JAX is running on: {jax.devices()[0].platform.upper()}")
except IndexError:
    print("No JAX devices found.")

jax.config.update("jax_enable_x64", True)


DATA = loadmat('pythondatatrain.mat')
u = DATA['u']
y = DATA['y']/10
time = DATA['t']


fig, axs = plt.subplots(2, 1, sharex=True) # sharex makes sense for time series

# Plot 1: Input u
axs[0].plot(time, u, color='b') # Added color for clarity
axs[0].set_title('Input Signal (u) vs. Time')
axs[0].set_ylabel('u (Input)')
axs[0].grid(True)

# Plot 2: Output y and Reference yref
axs[1].plot(time, y, 'k', label='y (Output)')
axs[1].set_title('Output (y) vs. Time')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Value')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout() # Adjusts subplot params for a tight layout
plt.show()

time = time.reshape(-1)
u = u.reshape(-1)
y = y.reshape(-1)
x_train = u
# Signal generation parameters
# N = 2048  # Number of samples (power of 2 is efficient for FFT)
N = time.shape[0]
Ts = time[1]-time[0]
fs = 1/Ts
T = time[-1]  # Total time in seconds

print(N, fs, T, Ts)

n_shots = 163 # 8150 / 163 = 50 data points per shot.
n_timesteps_per_shot = N // n_shots

# Reshape data into batches for multiple shooting
t_shots = jnp.array(time.reshape(n_shots, n_timesteps_per_shot))
y_data = jnp.array(y.reshape(n_shots, n_timesteps_per_shot))

# Create a differentiable interpolation object for the input signal
u_interpolation = LinearInterpolation(ts=time, ys=u)

# The JAX-compatible model evaluates the interpolated input u at any time t
def dc_motor_model_jax(t, w, args):
    theta1, theta3,fc,fs,vs, u_interp = args
    u = u_interp.evaluate(t)
    return theta1 * w -fc*jnp.sign(w)-(fs-fc)*jnp.exp(-jnp.abs(w/vs)**2) +  theta3 * u

term = ODETerm(dc_motor_model_jax)
solver = Dopri5()

def create_loss_and_constraint_fns(t_shots, y_data, u_interp):
    """A factory to create the objective and constraint functions."""
    @jax.jit
    def objective_jax(decision_vars):
        theta1, theta3, fc, fs, vs = decision_vars[:5]#####
        w_initial_shots = decision_vars[5:]#####

        def simulate_shot(t_shot, w0):
            saveat = SaveAt(ts=t_shot)
            args = (theta1, theta3, fc, fs, vs, u_interp)########
            sol = diffeqsolve(term, solver, t0=t_shot[0], t1=t_shot[-1], dt0=Ts, y0=w0, saveat=saveat, args=args)
            return sol.ys.flatten()

        w_pred = jax.vmap(simulate_shot)(t_shots, w_initial_shots)
        return jnp.sum((w_pred - y_data)**2)

    @jax.jit
    def continuity_constraints_jax(decision_vars):
        theta1, theta3, fc, fs, vs = decision_vars[:5]#####
        w_initial_shots = decision_vars[5:]########33

        def get_end_state(t_shot, w0):
            args = (theta1, theta3, fc, fs, vs, u_interp)#########
            sol = diffeqsolve(term, solver, t0=t_shot[0], t1=t_shot[-1], dt0=Ts, y0=w0, args=args)
            return sol.ys[-1]

        w_end_of_shots = jax.vmap(get_end_state)(t_shots[:-1], w_initial_shots[:-1])
        # The continuity constraints are the differences between the end state of a shot
        # and the initial state of the next shot.
        return (w_end_of_shots - w_initial_shots[1:]).flatten()

    return objective_jax, continuity_constraints_jax

# Create the specific functions for our data
objective_jax, continuity_constraints_jax = create_loss_and_constraint_fns(t_shots, y_data, u_interpolation)

# Create JIT-compiled gradient and Jacobian functions
objective_grad_func = jax.jit(jax.value_and_grad(objective_jax))
# We use jacrev (reverse-mode AD) as it's compatible with the adjoint-based diffrax solver
# Reshape the output of jacrev to match the expected shape for the constraints Jacobian
constraints_jac_func = jax.jit(jax.jacrev(continuity_constraints_jax))

# Wrapper functions to interface between SciPy (NumPy) and JAX
def obj_for_scipy(dv_np):
    val, grad = objective_grad_func(jnp.array(dv_np))
    return np.array(val), np.array(grad)

def cons_for_scipy(dv_np):
    return np.array(continuity_constraints_jax(jnp.array(dv_np)))

def cons_jac_for_scipy(dv_np):
    jac_jax = constraints_jac_func(jnp.array(dv_np))
    # SciPy expects the Jacobian to be a 2D array where rows correspond to constraints
    # and columns correspond to decision variables.
    # jax.jacrev might return a nested structure depending on the output of the function.
    # We need to flatten it to the expected 2D shape.
    # Use ravel_pytree to flatten the nested Jacobian structure
    flat_jac, _ = jax.flatten_util.ravel_pytree(jac_jax)
    # Reshape the flattened Jacobian to the expected 2D shape (num_constraints, num_decision_vars)
    num_constraints = len(cons_for_scipy(dv_np))
    num_decision_vars = len(dv_np)
    return np.array(flat_jac).reshape(num_constraints, num_decision_vars)


# Set up the optimization problem
initial_guess_np = np.concatenate([[-0.6292, -0.1, 1,1,1], np.zeros(n_shots)])#####
cons = ({'type': 'eq', 'fun': cons_for_scipy, 'jac': cons_jac_for_scipy})

# Run the optimization with a progress bar
max_iterations = 1000
with tqdm(total=max_iterations, desc="Optimizing Parameters") as pbar:
    def callback(xk):
        pbar.update(1)

    print("--- Running Optimization with Automatic Differentiation ---")
    result = minimize(obj_for_scipy,
                      initial_guess_np,
                      method='SLSQP',
                      jac=True, # Tells SciPy that our objective function returns value and gradient
                      constraints=cons,
                      options={'maxiter': max_iterations, 'disp': False}, # Set disp=False for cleaner output with tqdm
                      callback=callback,
                      tol = 10**-8)

print("\nOptimization finished with status:", result.message)

# Extract and display results
theta1_est, theta3_est, fc_est,fs_est,vs_est = result.x[:5]############

#print("\n--- Identification Results ---")
#print(f"Estimated parameters: theta1 = {theta1_est:.4f}, theta2 = {theta2_est:.4f}, theta3 = {theta3_est:.4f}")########

# Simulate the final model prediction
final_args = (theta1_est, theta3_est, fc_est,fs_est,vs_est, u_interpolation)########
final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=y[0], saveat=SaveAt(ts=jnp.array(time)), args=final_args, max_steps=16384)
# final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=y[0], saveat=SaveAt(ts=jnp.array(time)), args=final_args)
yhat = final_sol.ys.flatten()
y_hat_train = yhat


# Plot final results
plt.figure(figsize=(12, 7))
plt.plot(time, y, 'k', label='True state', alpha=0.4)
plt.plot(time, yhat, 'b--', label='Identified Model Prediction', linewidth=2)
plt.xlabel('Time (s)')
plt.title('Model Identification Result with Multisine Input')
plt.legend()
plt.grid(True)
plt.show()

DATA = loadmat('pythondataval.mat')
u = DATA['u']
y = DATA['y']/10
time = DATA['t']
time = time.reshape(-1)
u = u.reshape(-1)
y = y.reshape(-1)
x_valid= u

# Signal generation parameters
# N = 2048  # Number of samples (power of 2 is efficient for FFT)
N = time.shape[0]
Ts = time[1]-time[0]
fs = 1/Ts
T = time[-1]  # Total time in seconds

print(N, fs, T, Ts)

n_shots = 43 # 8150 / 163 = 50 data points per shot.
n_timesteps_per_shot = N // n_shots

# Reshape data into batches for multiple shooting
t_shots = jnp.array(time.reshape(n_shots, n_timesteps_per_shot))
y_data = jnp.array(y.reshape(n_shots, n_timesteps_per_shot))

# Create a differentiable interpolation object for the input signal
u_interpolation = LinearInterpolation(ts=time, ys=u)
# Simulate the final model prediction
final_args = (theta1_est, theta3_est, fc_est,fs_est,vs_est, u_interpolation)#########
final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=y[0], saveat=SaveAt(ts=jnp.array(time)), args=final_args, max_steps=16384)
# final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=y[0], saveat=SaveAt(ts=jnp.array(time)), args=final_args)
yhat = final_sol.ys.flatten()

# Plot final results
plt.figure(figsize=(12, 7))
plt.plot(time, y, 'k', label='True state', alpha=0.4)
plt.plot(time, yhat, 'b--', label='Identified Model Prediction', linewidth=2)
plt.xlabel('Time (s)')
plt.title('Model Identification Result with Multisine Input')
plt.legend()
plt.grid(True)
plt.show()
y_hat_val = yhat
results = {'x_valid':x_valid,'x_train':x_train,'y_hat_train':y_hat_train,'y_hat_val':y_hat_val}


sc.io.savemat('resultados_fric_str_poly.mat',results)