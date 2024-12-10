'''
LQR Controller
'''
import numpy as np
from sympy import symbols, N
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.interpolate import interp1d
from Lagrangian_V4_MatrixForm import compute_lagrangian_matrices

# Define the desired reference state
x_ref = np.array([0.5, 0.4, 0.8, 0.0, 0.0, 0.0])  # Reference generalized coordinates and velocities

def check_controllability(A, B):
    """Check if the system is controllable."""
    n = A.shape[0]
    Q_c = B  # Start with B
    for i in range(1, n):
        Q_c = np.hstack([Q_c, np.linalg.matrix_power(A, i) @ B])  # Append A^i B
    rank = np.linalg.matrix_rank(Q_c)
    return rank == n, Q_c

# Compute matrices
M, C, K = compute_lagrangian_matrices()

# Substitute numerical values
m1_val, m2_val, k_val = 10.0, 1.0, 10.0
l_val, r_val = 10.0, 2.0
I1_val, I2_val = 1 / 12 * m1_val * l_val**2, m2_val * r_val**2 / 2
c1_val, c2_val, c3_val = 0.5, 0.5, 0.5  # Friction coefficients

dt = 0.1  # Time step
t_span = (0, 10)  # Time span
num_steps = int((t_span[1] - t_span[0]) / dt)
# routing = np.array([
#         [1,-1, -1, 1,  1, -1],
#         [0, 0,  1, 1, -1, -1],
#         [0, 0,0,  0,  -1,  1]
# ])

routing = np.array([
        [1, -1,  1, -1],
        [0,  1, -1, -1],
        [0,0,  -1,  1]
])

# routing = np.array([
#         [-1,1],
#         [1,-1],
#         [0,1]
# ])

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

# Substitute and convert entries to floats
M_eval = M.subs(substitutions).applyfunc(N)
C_eval = C.subs(substitutions).applyfunc(N)
K_eval = K.subs(substitutions).applyfunc(N)

# Convert to NumPy arrays
M_np = np.array(M_eval.tolist(), dtype=np.float32)
C_np = np.array(C_eval.tolist(), dtype=np.float32)
K_np = np.array(K_eval.tolist(), dtype=np.float32)

# Define the state-space dynamics
def state_space_dynamics(t, x, A, B, u):
    u_t = u(t)  # Input as a function of time
    dxdt = A @ x + B @ u_t
    return dxdt

# Compute state-space matrices
def compute_state_space(M, K, C):
    """Compute the A, B, C, D matrices for the state-space representation."""
    n = M.shape[0]  # Number of degrees of freedom
    
    # State-space matrices
    A_top = np.hstack([np.zeros((n, n)), np.eye(n)])  # Top part of A
    A_bottom = np.hstack([-np.linalg.solve(M, K), -np.linalg.solve(M, C)])  # Bottom part of A
    A = np.vstack([A_top, A_bottom])  # Combine top and bottom parts

    # Corrected B matrix
    B = np.vstack([
        np.zeros((n, routing.shape[1])),  # Zero padding for velocity states
        np.linalg.inv(M) @ (r_val * routing)  # Matrix multiplication
    ])
    
    C = np.eye(2 * n)  # Output matrix (identity for full-state feedback)
    D = np.zeros((2 * n, routing.shape[1]))  # Direct feedthrough matrix

    return A, B, C, D

# Compute A, B, C, D matrices
A, B, C, D = compute_state_space(M_np, K_np, C_np)
print(np.linalg.eig(A))
# # Check controllability
# is_controllable, controllability_matrix = check_controllability(A, B)

# # Output results
# print("Controllability Matrix (Q_c):")
# print(controllability_matrix)
# print("\nIs the system controllable?", is_controllable)

# Define LQR controller function
def lqr(A, B, Q, R):
    """Solve the continuous-time LQR controller."""
    P = solve_continuous_are(A, B, Q, R)  # Solve Riccati equation
    K = np.linalg.inv(R) @ (B.T @ P)  # Compute gain matrix
    return K, P

# Define Q and R matrices
cared_states_matrix = np.zeros((6, 6))
cared_states_matrix[:3, :3] = np.eye(3)

Q = cared_states_matrix * 100  # State cost matrix
# R = np.eye(B.shape[1]) * 0.00001  # Control effort cost matrix
R = np.eye(B.shape[1]) * 0.00001  # Control effort cost matrix

# Compute LQR gain matrix
K_lqr, P = lqr(A, B, Q, R)

print("LQR Gain Matrix (K):")
print(K_lqr)


# Update the state-space dynamics with LQR control and reference tracking
def state_space_dynamics_with_reference(t, x, A, B, K, x_ref, u_log):
    # Compute control input with reference tracking
    u_t = -K @ (x - x_ref)
    dxdt = A @ x + B @ u_t
    u_log.append(u_t)
    return dxdt

# Initial conditions (all states start at zero)
x0 = np.zeros(A.shape[0])

u_log = []

# Solve the closed-loop system with LQR and reference point
solution_with_reference = solve_ivp(
    fun=lambda t, x: state_space_dynamics_with_reference(t, x, A, B, K_lqr, x_ref, u_log),
    t_span=t_span,
    y0=x0,
    t_eval=np.linspace(t_span[0], t_span[1], num_steps)
)

u_values = np.array(u_log)

# Extract states from the solution
states_with_reference = solution_with_reference.y.T

# plot pole-zero
from scipy.signal import ss2tf, tf2zpk

input_index = 0
A_cl = A - B @ K_lqr


# Convert the closed-loop system to transfer function
numerator_cl, denominator_cl = ss2tf(A_cl, B, C, D, input=input_index)

# Compute poles and zeros for the closed-loop system
zeros_cl, poles_cl, gain_cl = tf2zpk(numerator_cl[0], denominator_cl)

# Plot the pole-zero plot for the closed-loop system
plt.figure()
plt.scatter(np.real(poles_cl), np.imag(poles_cl), marker='x', label='Poles', color='blue')
plt.scatter(np.real(zeros_cl), np.imag(zeros_cl), marker='o', label='Zeros', color='red')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.title("Pole-Zero Plot After LQR")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.xlim(-3, 0)  # Fix the x-axis range
plt.ylim(-3.1, 3.1)  # Fix the x-axis range
plt.show()
