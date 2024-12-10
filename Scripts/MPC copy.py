import numpy as np
import matplotlib.pyplot as plt
import do_mpc
import casadi
from sympy import symbols, N
from Lagrangian_V4_MatrixForm import compute_lagrangian_matrices
from scipy.linalg import solve_continuous_are
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

# Desired reference state
x_ref = np.array([0.5, 0.4, 0.8, 0.0, 0.0, 0.0])

# --- System Parameters ---
m1_val, m2_val, k_val = 10.0, 1.0, 10.0
l_val, r_val = 10.0, 2.0
I1_val, I2_val = 1 / 12 * m1_val * l_val**2, m2_val * r_val**2 / 2
c1_val, c2_val, c3_val = 0.5, 0.5, 0.5  # Friction coefficients
t_span = (0, 20)
dt = 0.5
num_steps = int((t_span[1] - t_span[0]) / dt)

# routing = np.array([
#         [1,-1, -1, 1,  1, -1],
#         [0, 0,  1, 1, -1, -1],
#         [0, 0,0,  0,  -1,  1]
# ])
routing = np.array([
        [1, -1,  1, -1],
        [0,  1, -1, -1],
        [0,  0, -1,  1]
])

# --- Compute Symbolic Matrices ---
M, C, K = compute_lagrangian_matrices()
substitutions = {
    symbols('m1'): m1_val,
    symbols('m2'): m2_val,
    symbols('k'): k_val,
    symbols('l'): l_val,
    symbols('r'): r_val,
    symbols('I1'): I1_val,
    symbols('I2'): I2_val,
    symbols('c1'): c1_val,
    symbols('c2'): c2_val,
    symbols('c3'): c3_val,
}
M_eval = M.subs(substitutions).applyfunc(N)
C_eval = C.subs(substitutions).applyfunc(N)
K_eval = K.subs(substitutions).applyfunc(N)

M_np = np.array(M_eval.tolist(), dtype=float)
C_np = np.array(C_eval.tolist(), dtype=float)
K_np = np.array(K_eval.tolist(), dtype=float)

n = M_np.shape[0]

# --- Continuous-time State-Space ---
A_top = np.hstack([np.zeros((n, n)), np.eye(n)])
A_bottom = np.hstack([-np.linalg.solve(M_np, K_np), -np.linalg.solve(M_np, C_np)])
A_c = np.vstack([A_top, A_bottom])

B_c = np.vstack([
    np.zeros((n, routing.shape[1])),
    np.linalg.inv(M_np) @ (r_val * routing)
])

nx = A_c.shape[0]
nu = B_c.shape[1]


# --- do-mpc Model (Continuous) ---
model_type = 'continuous'
model = do_mpc.model.Model(model_type)

# Define state & input variables
x_var = model.set_variable(var_type='_x', var_name='x', shape=(nx,1))
u_var = model.set_variable(var_type='_u', var_name='u', shape=(nu,1))

# dx/dt = A_c x + B_c u
x_dot = A_c @ x_var + B_c @ u_var
model.set_rhs('x', x_dot)

model.setup()

# --- MPC Controller ---
mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 20,       # Prediction horizon
    't_step': dt,          # Sampling time for the MPC
    'n_robust': 1,
    'store_full_solution': True,
    'nlpsol_opts': {'ipopt.linear_solver': 'mumps', 'ipopt.print_level':0, 'print_time':0}
}
mpc.set_param(**setup_mpc)

# Cost function weights
Q = np.diag([1, 1, 1, 0.0, 0.0, 0.0]) * 1000
R = np.eye(nu)*0.0001
P_terminal = solve_continuous_are(A_c, B_c, Q, R)
print(P_terminal)

# Cost function
mterm = (x_var - x_ref).T @ Q @ (x_var - x_ref) *200 # Terminal cost
lterm = (x_var - x_ref).T @ Q @ (x_var - x_ref) + u_var.T @ R @ u_var  # Stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)

# Input constraints: u >= 0
mpc.bounds['lower','_u','u'] = np.zeros((nu,1))
mpc.bounds['upper','_u','u'] = np.ones((nu,1))*300
# mpc.bounds['lower', '_x', 'x'] = np.array([0, 0, 0, -np.inf, -np.inf, -np.inf])  # Lower bounds
# mpc.bounds['upper', '_x', 'x'] = np.array([1/(2*np.pi), 1/(2*np.pi), 1/(2*np.pi), np.inf, np.inf, np.inf])  # Upper bounds
# Define terminal equality constraint as CasADi expression
# terminal_constraint_expr = x_var - x_ref  # Assuming x_ref is a CasADi symbolic variable or compatible array

# # Add the constraint
# mpc.set_nl_cons('terminal_constraint', terminal_constraint_expr)

mpc.setup()

# --- Simulator ---
simulator = do_mpc.simulator.Simulator(model)
params_simulator = {
    't_step': dt,
    'integration_tool': 'cvodes'  # try 'cvodes' or 'rk' if 'idas' does not work
}
simulator.set_param(**params_simulator)
simulator.setup()

# Initial conditions
x0 = np.zeros((nx,1))
mpc.x0 = x0
simulator.x0 = x0
mpc.set_initial_guess()

# Run Closed-Loop Simulation
x_history = [x0.squeeze()]
u_history = []

current_state = x0
for k in range(num_steps):
    # Get optimal control from MPC
    u0 = mpc.make_step(current_state)
    # Simulate one step ahead with the simulator
    next_state = simulator.make_step(u0)
    x_history.append(next_state.squeeze())
    u_history.append(u0.squeeze())
    current_state = next_state

x_history = np.array(x_history)
u_history = np.array(u_history)


import numpy as np
from scipy.signal import ss2tf, tf2zpk
import matplotlib.pyplot as plt

# Define the plant's continuous-time state-space matrices
A_plant = A_c  # System dynamics from your code
B_plant = B_c
C_plant = np.eye(A_plant.shape[0])  # Assuming full-state output
D_plant = np.zeros((A_plant.shape[0], B_plant.shape[1]))

# Define the MPC gain matrix
# For simplicity, we approximate the MPC as a state-feedback gain `K` computed from P_terminal
K_mpc = np.linalg.inv(R) @ B_plant.T @ P_terminal  # Feedback gain approximation

# Calculate the closed-loop system dynamics
A_cl = A_plant - B_plant @ K_mpc  # Closed-loop state matrix
B_cl = B_plant  # Input matrix remains the same
C_cl = C_plant  # Output matrix remains the same
D_cl = D_plant  # Feedthrough matrix remains the same

# Convert closed-loop system to transfer function
numerator, denominator = ss2tf(A_c, B_c, C_cl, D_cl, input=0)

# Compute poles and zeros of the closed-loop system
zeros, poles, _ = tf2zpk(numerator[0], denominator)

# Plot the pole-zero plot
plt.figure()
plt.scatter(np.real(poles), np.imag(poles), marker='x', label='Poles', color='blue')
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', label='Zeros', color='red')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.title("Pole-Zero Plot After MPC")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.show()
