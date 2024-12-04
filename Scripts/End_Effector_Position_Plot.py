import jax
import jax.numpy as jnp
import numpy as np
from sympy import symbols, N
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from Lagrangian_V3_MatrixForm import compute_lagrangian_matrices
import numpy.linalg as la

def check_controllability(A, B):
    """Check if the system is controllable."""
    n = A.shape[0]
    Q_c = B  # Start with B
    for i in range(1, n):
        Q_c = jnp.hstack([Q_c, jnp.linalg.matrix_power(A, i) @ B])  # Append A^i B
    rank = la.matrix_rank(Q_c)
    return rank == n, Q_c

# Compute matrices
M, C, K = compute_lagrangian_matrices()

# Substitute numerical values
m1_val, m2_val, k_val, F_val = 1.0, 1.0, 2.0, 20.0
l_val, r_val = 10.0, 2.0
I1_val, I2_val = 1 / 12 * m1_val * l_val**2, m2_val * r_val**2 / 2
dt = 0.1  # Time step
t_span = (0, 30)  # Time span
num_steps = int((t_span[1] - t_span[0]) / dt)

substitutions = {
    symbols('m1'): m1_val,
    symbols('m2'): m2_val,
    symbols('k'): k_val,
    symbols('l'): l_val,
    symbols('r'): r_val,
    symbols('I1'): I1_val,
    symbols('I2'): I2_val
}

# Substitute and convert entries to floats
M_eval = M.subs(substitutions).applyfunc(N)
C_eval = C.subs(substitutions).applyfunc(N)
K_eval = K.subs(substitutions).applyfunc(N)

# Convert to JAX arrays
M_jnp = jnp.array(M_eval.tolist(), dtype=jnp.float32)
C_jnp = jnp.array(C_eval.tolist(), dtype=jnp.float32)
K_jnp = jnp.array(K_eval.tolist(), dtype=jnp.float32)

# External Forces (F)
def external_forces(q, q_dot):
    return jnp.array([
        +F_val * r_val + k_val * r_val * q[0],
        +F_val * r_val + k_val * r_val * q[1],
        -F_val * r_val + k_val * r_val * q[2]
    ])

# Define the state-space dynamics
def state_space_dynamics(t, x, A, B, u):
    u_t = u(t)  # Input as a function of time
    dxdt = A @ x + B @ u_t
    return dxdt

# Define a time-varying input (e.g., step input)
def input_function(t):
    return jnp.array([1.0, 0.0, 0.0])  # Constant input for demonstration

# Compute state-space matrices
def compute_state_space(M, K):
    """Compute the A, B, C, D matrices for the state-space representation."""
    n = M.shape[0]  # Number of degrees of freedom
    
    # State-space matrices
    A_top = jnp.hstack([jnp.zeros((n, n)), jnp.eye(n)])  # Top part of A
    A_bottom = jnp.hstack([-jnp.linalg.solve(M, K), jnp.zeros((n, n))])  # Bottom part of A
    A = jnp.vstack([A_top, A_bottom])  # Combine top and bottom parts

    B = jnp.vstack([jnp.zeros((n, n)), jnp.linalg.inv(M)])  # Input matrix
    C = jnp.eye(2 * n)  # Output matrix (identity for full-state feedback)
    D = jnp.zeros((2 * n, n))  # Direct feedthrough matrix

    return A, B, C, D

# Compute A, B, C, D matrices
A, B, C, D = compute_state_space(M_jnp, K_jnp)

# Check controllability
is_controllable, controllability_matrix = check_controllability(A, B)

# Output results
print("Controllability Matrix (Q_c):")
print(controllability_matrix)
print("\nIs the system controllable?", is_controllable)

# Initial conditions (all states start at zero)
x0 = jnp.zeros(A.shape[0])

# Time span for the simulation
time_span = (0, 30)
time_points = jnp.linspace(time_span[0], time_span[1], num_steps)

# Solve the state-space equations
solution = solve_ivp(
    fun=lambda t, x: state_space_dynamics(t, x, A, B, input_function),
    t_span=time_span,
    y0=x0,
    t_eval=time_points
)

# Extract states from the solution
states = solution.y.T  # Transpose to get time along rows

# Define the function to compute the end-effector position
def end_effector_position(q1, q2, q3, l_val):
    # Compute position using forward kinematics
    x = l_val * (jnp.cos(q1) + jnp.cos(q1 + q2) + jnp.cos(q1 + q2 + q3))
    y = l_val * (jnp.sin(q1) + jnp.sin(q1 + q2) + jnp.sin(q1 + q2 + q3))
    return x, y

# Compute the end-effector position over time
end_effector_x = []
end_effector_y = []
for state in states:  # Iterate over states from the solution
    q1, q2, q3 = state[:3]  # Extract the generalized coordinates
    x, y = end_effector_position(q1, q2, q3, l_val)
    end_effector_x.append(x)
    end_effector_y.append(y)

# Convert to NumPy arrays for plotting
end_effector_x = np.array(end_effector_x)
end_effector_y = np.array(end_effector_y)

# Plot the end-effector trajectory
plt.figure(figsize=(12, 6))
plt.plot(end_effector_x, end_effector_y, label="End-Effector Trajectory")
plt.scatter(end_effector_x[0], end_effector_y[0], color='red', label="Start")
plt.scatter(end_effector_x[-1], end_effector_y[-1], color='green', label="End")
plt.title("End-Effector Trajectory")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid()
plt.axis('equal')  # Ensure equal scaling for X and Y axes
plt.show()

# Plot the end-effector positions over time
plt.figure(figsize=(12, 6))
plt.plot(solution.t, end_effector_x, label="X Position", color="blue")
plt.plot(solution.t, end_effector_y, label="Y Position", color="red")
plt.title("End-Effector Position Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Position (units)")
plt.legend()
plt.grid()
plt.show()
