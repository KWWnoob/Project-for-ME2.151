import jax
import jax.numpy as jnp
import numpy as np
from sympy import symbols, N
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.interpolate import interp1d
from Lagrangian_V4_MatrixForm import compute_lagrangian_matrices

def check_controllability(A, B):
    """Check if the system is controllable."""
    n = A.shape[0]
    Q_c = B  # Start with B
    for i in range(1, n):
        Q_c = jnp.hstack([Q_c, jnp.linalg.matrix_power(A, i) @ B])  # Append A^i B
    rank = np.linalg.matrix_rank(Q_c)
    return rank == n, Q_c

# Compute matrices
M, C, K = compute_lagrangian_matrices()

# Substitute numerical values
m1_val, m2_val, k_val = 1.0, 1.0, 10.0
l_val, r_val = 10.0, 2.0
I1_val, I2_val = 1 / 12 * m1_val * l_val**2, m2_val * r_val**2 / 2
c1_val, c2_val, c3_val = 0.1, 0.1, 0.1  # Friction coefficients

dt = 0.1  # Time step
t_span = (0, 10)  # Time span
num_steps = int((t_span[1] - t_span[0]) / dt)
routing = jnp.array([
        [1, 1, -1, -1],
        [0, -1, 1, -1],
        [0, 0, -1, 1]
])

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

# Convert to JAX arrays
M_jnp = jnp.array(M_eval.tolist(), dtype=jnp.float32)
C_jnp = jnp.array(C_eval.tolist(), dtype=jnp.float32)
K_jnp = jnp.array(K_eval.tolist(), dtype=jnp.float32)

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
    A_top = jnp.hstack([jnp.zeros((n, n)), jnp.eye(n)])  # Top part of A
    A_bottom = jnp.hstack([-jnp.linalg.solve(M, K),-jnp.linalg.solve(M, C)])  # Bottom part of A
    A = jnp.vstack([A_top, A_bottom])  # Combine top and bottom parts

    # Corrected B matrix
    B = jnp.vstack([
        jnp.zeros((n, routing.shape[1])),  # Zero padding for velocity states
        jnp.linalg.inv(M) @ (r_val * routing)  # Matrix multiplication
    ])
    
    C = jnp.eye(2 * n)  # Output matrix (identity for full-state feedback)
    D = jnp.zeros((2 * n, routing.shape[1]))  # Direct feedthrough matrix

    return A, B, C, D

# Compute A, B, C, D matrices
A, B, C, D = compute_state_space(M_jnp, K_jnp, C_jnp)

# Check controllability
is_controllable, controllability_matrix = check_controllability(A, B)

# Output results
print("Controllability Matrix (Q_c):")
print(controllability_matrix)
print("\nIs the system controllable?", is_controllable)

# Define LQR controller function
def lqr(A, B, Q, R):
    """Solve the continuous-time LQR controller."""
    P = solve_continuous_are(A, B, Q, R)  # Solve Riccati equation
    K = np.linalg.inv(R) @ (B.T @ P)  # Compute gain matrix
    return K, P

# Define Q and R matrices
Q = np.eye(A.shape[0])*100 # State cost matrix
R = np.eye(B.shape[1])*0.000001  # Control effort cost matrix

# Compute LQR gain matrix
K_lqr, P = lqr(A, B, Q, R)

print("LQR Gain Matrix (K):")
print(K_lqr)

# Define the desired reference state
# Adjust generalized coordinates or velocities as needed
x_ref = jnp.array([0.5, 0.4, 0.8, 0.0, 0.0, 0.0])  # Reference generalized coordinates and velocities

# Update the state-space dynamics with LQR control and reference tracking
def state_space_dynamics_with_reference(t, x, A, B, K, x_ref, u_log):
    # Compute control input with reference tracking
    u_t = -K @ (x - x_ref)
    dxdt = A @ x + B @ u_t
    u_log.append(u_t)
    return dxdt

# Initial conditions (all states start at zero)
x0 = jnp.zeros(A.shape[0])

u_log = []

# Solve the closed-loop system with LQR and reference point
solution_with_reference = solve_ivp(
    fun=lambda t, x: state_space_dynamics_with_reference(t, x, A, B, K_lqr, x_ref, u_log),
    t_span=t_span,
    y0=x0,
    t_eval=jnp.linspace(t_span[0], t_span[1], num_steps)
)

u_values = np.array(u_log)

# Extract states from the solution
states_with_reference = solution_with_reference.y.T

# Define the function to compute the end-effector position
def end_effector_position(q1, q2, q3, l_val):
    # Compute position using forward kinematics
    x = l_val * (jnp.cos(q1) + jnp.cos(q1 + q2) + jnp.cos(q1 + q2 + q3))
    y = l_val * (jnp.sin(q1) + jnp.sin(q1 + q2) + jnp.sin(q1 + q2 + q3))
    return x, y

# Compute the end-effector position over time with reference tracking
end_effector_x_ref = []
end_effector_y_ref = []
for state in states_with_reference:
    q1, q2, q3 = state[:3]  # Extract generalized coordinates
    x, y = end_effector_position(q1, q2, q3, l_val)
    end_effector_x_ref.append(x)
    end_effector_y_ref.append(y)

# Convert to NumPy arrays for plotting
end_effector_x_ref = np.array(end_effector_x_ref)
end_effector_y_ref = np.array(end_effector_y_ref)

ref = end_effector_position(x_ref[0],x_ref[1],x_ref[2],l_val)

# Plot the controlled end-effector trajectory with reference tracking
plt.figure(figsize=(12, 6))
plt.plot(end_effector_x_ref, end_effector_y_ref, label="Controlled Trajectory (Reference Tracking)", color="green")
plt.scatter(end_effector_x_ref[0], end_effector_y_ref[0], color='red', label="Start (Controlled)")
plt.scatter(end_effector_x_ref[-1], end_effector_y_ref[-1], color='blue', label="End (Controlled)")
plt.scatter([ref[0]], [ref[1]], color='orange', label="Reference Point", marker='x', s=100)
plt.title("Controlled End-Effector Trajectory with Reference Tracking")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid()
plt.axis('equal')
plt.savefig('controlled_trajectory_reference_tracking.png', dpi=300)
plt.close()  # Close the figure after saving

# Plot controlled end-effector positions over time with reference tracking
plt.figure(figsize=(12, 6))
plt.plot(solution_with_reference.t, end_effector_x_ref, label="X Position (Reference Tracking)", color="blue")
plt.plot(solution_with_reference.t, end_effector_y_ref, label="Y Position (Reference Tracking)", color="red")
plt.axhline(y=ref[0], color='orange', linestyle='--', label="X Reference", linewidth=1)
plt.axhline(y=ref[1], color='purple', linestyle='--', label="Y Reference", linewidth=1)
plt.title("End-Effector Position Over Time with Reference Tracking")
plt.xlabel("Time (s)")
plt.ylabel("Position (units)")
plt.legend()
plt.grid()
plt.savefig('end_effector_position_over_time_reference_tracking.png', dpi=300)
plt.close()  # Close the figure after saving

# Plotting the control inputs
plt.figure(figsize=(12, 6))
for i in range(u_values.shape[1]):  # Plot each component of u
    plt.plot(u_values[:, i], label=f"u[{i}]")

plt.title("Control Input Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Input (units)")
plt.legend(title="Control Components")
plt.grid(True)
plt.savefig('control_input_over_time.png', dpi=300)
plt.close()  # Close the figure after saving

'''
show_animation
'''

times = solution_with_reference.t
q1 = [states[0] for states in solution_with_reference.y.T]
q2 = [states[1] for states in solution_with_reference.y.T]
q3 = [states[2] for states in solution_with_reference.y.T]

# Create finer time points for smoother animation
smooth_time_points = np.linspace(t_span[0], t_span[1], 5 * num_steps)  # 5x frames
interp_q1 = interp1d(times, q1, kind='cubic')  # Cubic interpolation
interp_q2 = interp1d(times, q2, kind='cubic')
interp_q3 = interp1d(times, q3, kind='cubic')

# Interpolated values
smooth_q1 = interp_q1(smooth_time_points)
smooth_q2 = interp_q2(smooth_time_points)
smooth_q3 = interp_q3(smooth_time_points)

# Animation Setup
fig, ax = plt.subplots()
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_aspect('equal')
ax.set_title("Lagrangian System Animation")

# Add the reference point to the plot
reference_point = ax.scatter(ref[0], ref[1], color='orange', label="Reference Point", marker='x', s=100)

# Create objects for disks and rods
rod1, = ax.plot([], [], 'b-', lw=2)
rod2, = ax.plot([], [], 'g-', lw=2)
rod3, = ax.plot([], [], 'r-', lw=2)
disk1 = plt.Circle((0, 0), r_val, color='b', fill=True)
disk2 = plt.Circle((l_val, 0), r_val, color='g', fill=True)
disk3 = plt.Circle((2 * l_val, 0), r_val, color='r', fill=True)
ax.add_patch(disk1)
ax.add_patch(disk2)
ax.add_patch(disk3)

# Initialize animation
def init():
    rod1.set_data([], [])
    rod2.set_data([], [])
    rod3.set_data([], [])
    return rod1, rod2, rod3, disk1, disk2, disk3

# Update animation frame
def update(frame):
    x1, y1 = 0, 0  # Fixed base
    x2, y2 = x1 + l_val * np.cos(smooth_q1[frame]), y1 + l_val * np.sin(smooth_q1[frame])
    x3, y3 = x2 + l_val * np.cos(smooth_q1[frame]+smooth_q2[frame]), y2 + l_val * np.sin(smooth_q1[frame]+smooth_q2[frame])
    x4, y4 = x3 + l_val * np.cos(smooth_q1[frame]+smooth_q2[frame]+smooth_q3[frame]), y3 + l_val * np.sin(smooth_q1[frame]+smooth_q2[frame]+smooth_q3[frame])

    # Update rods
    rod1.set_data([x1, x2], [y1, y2])
    rod2.set_data([x2, x3], [y2, y3])
    rod3.set_data([x3, x4], [y3, y4])

    # Update disks
    disk1.center = (x1, y1)
    disk2.center = (x2, y2)
    disk3.center = (x3, y3)

    return rod1, rod2, rod3, disk1, disk2, disk3

# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=len(smooth_time_points),
    init_func=init,
    blit=True,
    interval=1000 * dt / 5  # Adjust playback speed
)
# Save the animation as a video
ani.save('animation.gif', writer='pillow', fps=24)

# Show animation
plt.show()
