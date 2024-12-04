import jax
import jax.numpy as jnp
import numpy as np
from sympy import symbols, N
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

from Lagrangian_V3_MatrixForm import compute_lagrangian_matrices

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

# Equations of Motion
def equations_of_motion(y, t):
    q = y[:3]        # Generalized coordinates
    q_dot = y[3:]    # Velocities

    Force = external_forces(q, q_dot)

    # Compute accelerations
    q_ddot = jnp.linalg.solve(M_jnp, Force - K_jnp @ q)  # q_ddot = M^-1 * (F - Kq)
    return jnp.concatenate((q_dot, q_ddot)) 

# Integrate using Euler's method
def integrate(y0, dt, num_steps):
    times = jnp.linspace(t_span[0], t_span[1], num_steps)

    def step(y, t):
        dydt = equations_of_motion(y, t)
        return y + dt * dydt, y

    _, trajectory = jax.lax.scan(step, y0, times)
    return times, trajectory

# Initial conditions
y0 = jnp.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])  # Slightly perturbed initial state

# Perform integration
times, trajectory = integrate(y0, dt, num_steps)
trajectory_np = np.array(trajectory)  # Convert to NumPy array for compatibility
q1, q2, q3 = trajectory_np[:, 0], trajectory_np[:, 1], trajectory_np[:, 2]

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
    x3, y3 = x2 + l_val * np.cos(smooth_q2[frame]), y2 + l_val * np.sin(smooth_q2[frame])
    x4, y4 = x3 + l_val * np.cos(smooth_q3[frame]), y3 + l_val * np.sin(smooth_q3[frame])

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

# Show animation
plt.show()
