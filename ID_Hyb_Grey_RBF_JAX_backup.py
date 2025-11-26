import numpy as np
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
from jax import random, grad, jit, vmap
from jax.flatten_util import ravel_pytree
    
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

if time is not None:
    time = time.flatten()
    u = u.flatten()
    y = y.flatten()

    N = time.shape[0]
    Ts = time[1] - time[0]
    fs = 1 / Ts
    T = time[-1]
    print(f"N={N}, fs={fs}, T={T}, Ts={Ts}")

# --- RBF Neural Network Helper Functions (from JAX ANN example) ---


def initialize_rbf_network_parameters(key: random.PRNGKey, num_rbf_neurons: int, input_dimension: int) -> list[tuple[jnp.ndarray, float, float]]:
    rbf_parameters = []
    for i in range(num_rbf_neurons):
        # Split the key for each neuron's parameter initialization
        key, subkey_center, subkey_spread, subkey_weight = random.split(key, 4)

        # Initialize centers randomly within a typical range (e.g., -1 to 1)
        center = random.uniform(subkey_center, shape=(input_dimension,), minval=-1.0, maxval=1.0)

        # Initialize spreads (widths) to small positive values.
        # Using random.uniform generates values between 0 and 1. Adding a small constant
        # prevents zero or near-zero spreads which can lead to division by zero or very sharp functions.
        spread = random.uniform(subkey_spread, shape=())*.9 + 0.1  # Example: spread between 0.1 and 0.6

        # Initialize weights randomly (e.g., between -1 and 1)
        weight = random.uniform(subkey_weight, shape=(), minval=0, maxval=1.0)

        rbf_parameters.append((center, float(spread), float(weight))) # Convert JAX scalars to Python floats

    return rbf_parameters

def calculate_rbf_network_output(
    rbf_parameters: list[tuple[jnp.ndarray, float, float]],
    net_input: jnp.ndarray,
    bias: float):
    if not isinstance(net_input, jnp.ndarray):
        net_input = jnp.array(net_input) # Ensure input is a JAX array for consistent operations

    network_output = 0.0

    for center, spread, weight in rbf_parameters:
        # Calculate the Euclidean distance between the input and the neuron's center
        distance = jnp.linalg.norm(net_input - center)

        # Calculate the activation of the RBF neuron using a Gaussian function
        # The formula is exp(-(distance^2) / (2 * spread^2))
        # A small epsilon is added to the denominator to prevent division by zero if spread is extremely small
        activation = jnp.exp(-(distance**2) / (2 * (spread**2 + 1e-9))) # Added 1e-9 for numerical stability

        # Multiply the activation by the neuron's weight
        weighted_activation = activation * weight

        # Sum up the weighted activations
        network_output += weighted_activation

    # Add the bias term to the total sum
    network_output += bias

    return jnp.array(network_output)

# --- New Hybrid ODE Model ---

def hybrid_dc_motor_model_jax(t, w, args):
    """The ODE function combining physical terms and a neural network."""
    theta1, theta3, params_nn, bias, u_interp = args
    u = u_interp.evaluate(t)

    # The NN models the nonlinear dynamics (e.g., friction)
    # It takes velocity (w) and input (u) as inputs
    nn_input = jnp.array([w])
    nn_output = calculate_rbf_network_output(params_nn, nn_input, bias)

    # The new dynamic equation: dw/dt = linear_terms + nn_output
    return theta1 * w + theta3 * u + nn_output

term = ODETerm(hybrid_dc_motor_model_jax)

# --- Setup for Multiple Shooting ---
n_shots = 163
# n_shots = 43 # fixed
n_timesteps_per_shot = N // n_shots

t_shots = jnp.array(time.reshape(n_shots, n_timesteps_per_shot))
y_data = jnp.array(y.reshape(n_shots, n_timesteps_per_shot))
u_interpolation = LinearInterpolation(ts=time, ys=u)

# --- NN and Optimization Configuration ---
nn_neurons = 100
dim_input = 1
solver = Dopri5()

# --- Create Initial Guess and Parameter Structures ---
key = random.key(0)
initial_theta1 = -0.6
initial_theta3 = 7.0
initial_bias = random.normal(key)
initial_params_nn = initialize_rbf_network_parameters(key,nn_neurons, dim_input)
initial_w_shots = np.zeros(n_shots)

# Store the structure of the NN parameters for later unflattening
flat_initial_nn_params, params_nn_struct = ravel_pytree(initial_params_nn)
len_nn_params = len(flat_initial_nn_params)

# Create the full, flattened initial guess vector for the optimizer
initial_guess_np = np.concatenate([
    np.array([initial_theta1, initial_theta3,initial_bias]),
    np.array(flat_initial_nn_params),
    initial_w_shots
])

# --- JIT-compiled Objective and Constraint Functions ---
# **CORRECTION**: The factory pattern is removed to avoid JIT closure issues.
# The unflattening logic is now explicitly inside each jitted function.

@jit
def objective_jax_nn(decision_vars):
    # Manually unflatten parameters inside the jitted function
    theta1 = decision_vars[0]
    theta3 = decision_vars[1]
    bias = decision_vars[2]
    params_nn = params_nn_struct(decision_vars[3:len_nn_params+3])

    w_initial_shots = decision_vars[len_nn_params+3:]

    def simulate_shot(t_shot, w0):
        saveat = SaveAt(ts=t_shot)
        args = (theta1, theta3, params_nn, bias, u_interpolation)
        sol = diffeqsolve(term, solver, t0=t_shot[0], t1=t_shot[-1], dt0=Ts, y0=w0, saveat=saveat, args=args)
        return sol.ys.flatten()

    w_pred = vmap(simulate_shot)(t_shots, w_initial_shots)
    return jnp.sum((w_pred - y_data)**2)

@jit
def continuity_constraints_jax_nn(decision_vars):
    # Manually unflatten parameters inside the jitted function
    theta1 = decision_vars[0]
    theta3 = decision_vars[1]
    bias = decision_vars[2]
    params_nn = params_nn_struct(decision_vars[3:len_nn_params+3])

    w_initial_shots = decision_vars[len_nn_params+3:]

    def get_end_state(t_shot, w0):
        args = (theta1, theta3, params_nn,bias, u_interpolation)
        sol = diffeqsolve(term, solver, t0=t_shot[0], t1=t_shot[-1], dt0=Ts, y0=w0, args=args)
        return sol.ys[-1]

    w_end_of_shots = vmap(get_end_state)(t_shots[:-1], w_initial_shots[:-1])
    return w_end_of_shots - w_initial_shots[1:]

# --- Create JIT-compiled Gradient and Jacobian Functions ---
objective_grad_func_nn = jit(jax.value_and_grad(objective_jax_nn))
constraints_jac_func_nn = jit(jax.jacrev(continuity_constraints_jax_nn))


# --- Wrapper Functions for SciPy Optimizer ---

def obj_for_scipy(dv_np):
    val, grad = objective_grad_func_nn(jnp.array(dv_np))
    return np.array(val), np.array(grad)

def cons_for_scipy(dv_np):
    return np.array(continuity_constraints_jax_nn(jnp.array(dv_np)))

def cons_jac_for_scipy(dv_np):
    jac_jax = constraints_jac_func_nn(jnp.array(dv_np))
    return np.array(jac_jax) # SciPy can handle the jacobian structure directly

# --- Run Optimization ---

cons = ({'type': 'eq', 'fun': cons_for_scipy, 'jac': cons_jac_for_scipy})
max_iterations = 10 # Increased iterations for the more complex model

with tqdm(total=max_iterations, desc="Optimizing Hybrid Model") as pbar:
    def callback(xk):
        pbar.update(1)

    print("\n--- Running Optimization with Neural Network ---")
    result = minimize(
        obj_for_scipy,
        initial_guess_np,
        method='SLSQP',
        jac=True,
        constraints=cons,
        options={'maxiter': max_iterations, 'disp': False},
        callback=callback
    )

print("\nOptimization finished with status:", result.message)

# --- Extract and Display Results ---

# Unflatten the final optimized parameters
theta1_est = result.x[0]
theta3_est = result.x[1]
bias_est = result.x[2]
params_nn_est = params_nn_struct(result.x[3:len_nn_params+3])

# Use the first identified shot state as the initial state for the full simulation
w0_est = result.x[len_nn_params+3]

print("\n--- Identification Results ---")
print(f"Estimated physical parameters: theta1 = {theta1_est:.4f}, theta3 = {theta3_est:.4f}")

# --- Time-Domain Validation Plot ---
final_args = (theta1_est, theta3_est, params_nn_est,bias_est, u_interpolation)
final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=w0_est,
                        saveat=SaveAt(ts=jnp.array(time)), args=final_args, max_steps=16384)
yhat = final_sol.ys.flatten()
y_hat_train = yhat
x_train = u
plt.figure(figsize=(12, 7))
plt.plot(time, y, 'k', label='Measured Data (y)', alpha=0.6)
plt.plot(time, yhat, 'r--', label='Hybrid Model Prediction (y_hat)', linewidth=2)
plt.plot(time, y - yhat, 'b-', label='Error', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (w)')
plt.title('Time-Domain Validation of the Hybrid Model')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(y,yhat, 'ko')
plt.xlabel('Measured (y)')
plt.ylabel('Predicted (yhat)')
plt.grid(True)
plt.show()

DATA = loadmat('pythondataval.mat')
u = DATA['u']
y = DATA['y']/10
time = DATA['t']
time = time.reshape(-1)
u = u.reshape(-1)
y = y.reshape(-1)

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
# --- Time-Domain Validation Plot ---
final_args = (theta1_est, theta3_est, params_nn_est,bias_est, u_interpolation)
final_sol = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=Ts, y0=w0_est,
                        saveat=SaveAt(ts=jnp.array(time)), args=final_args, max_steps=16384)
yhat = final_sol.ys.flatten()

plt.figure(figsize=(12, 7))
plt.plot(time, y, 'k', label='Measured Data (y)', alpha=0.6)
plt.plot(time, yhat, 'r--', label='Hybrid Model Prediction (y_hat)', linewidth=2)
plt.plot(time, y - yhat, 'b-', label='Error', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (w)')
plt.title('Time-Domain Validation of the Hybrid Model')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(y,yhat, 'ko')
plt.xlabel('Measured (y)')
plt.ylabel('Predicted (yhat)')
plt.grid(True)
plt.show()
y_hat_val = yhat 
x_valid = u
results = {'x_valid':x_valid,'x_train':x_train,'y_hat_train':y_hat_train,'y_hat_val':y_hat_val}


sc.io.savemat('resultados_hib_rbf_insider.mat',results)