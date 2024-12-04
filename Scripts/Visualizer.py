import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

# Parameters
m1, m2, k, F = 1.0, 1.0, 10.0, 0  # Masses, spring constant, external force
l, r = 10, 2                   # Length of rod and radius of disk
I1, I2 = 1 / 12 * m1 * l**2, m2 * r**2 / 2  # Moments of inertia
dt = 0.1  # Time step
t_span = (0, 30)  # Time span
num_steps = int((t_span[1] - t_span[0]) / dt)

# Mass Matrix (M)
def mass_matrix():
    lr_squared = (l + r)**2
    M = jnp.array([
        [I1 + I2 + 5.0 * m1 * lr_squared + 8.75 * m2 * lr_squared, 
         4.0 * m1 * lr_squared + 8.5 * m2 * lr_squared, 
         6.25 * m2 * lr_squared],
        [4.0 * m1 * lr_squared + 8.5 * m2 * lr_squared, 
         I1 + I2 + 4.0 * m1 * lr_squared + 8.5 * m2 * lr_squared, 
         6.25 * m2 * lr_squared],
        [6.25 * m2 * lr_squared, 
         6.25 * m2 * lr_squared, 
         I1 + I2 + 6.25 * m2 * lr_squared]
    ])
    return M

# Stiffness Matrix (K)
def stiffness_matrix():
    K = jnp.array([
        [k, 0, 0],
        [0, k, 0],
        [0, 0, k]
    ])
    return K

# Controller Gains
K_p = jnp.array([50.0, 50.0, 50.0])  # Proportional gains
K_d = jnp.array([10.0, 10.0, 10.0])  # Derivative gains

# Desired states
q_desired = jnp.array([0.0, 0.0, 0.0])  # Target angles

# External Forces (F) with Controller
def external_forces(q, q_dot):
    # PD Controller
    control_force = -K_p * (q - q_desired) - K_d * q_dot
    
    # Add control forces to external forces
    return control_force + jnp.array([
        +F * r * 3 + k * r * q[0],
        +F * r * 2 + k * r * q[1],
        -F * r * 2 + k * r * q[2]
    ])

# Equations of Motion with Controller
def equations_of_motion(y, t):
    q = y[:3]        # Generalized coordinates
    q_dot = y[3:]    # Velocities

    M = mass_matrix()
    K = stiffness_matrix()
    Force = external_forces(q, q_dot)

    # Compute accelerations
    q_ddot = jnp.linalg.solve(M, Force - K @ q)  # q_ddot = M^-1 * (F - Kq)
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
q1, q2, q3 = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]  # Angular positions

# Create finer time points for smoother animation
smooth_time_points = jnp.linspace(t_span[0], t_span[1], 5 * num_steps)  # 5x frames
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
ax.set_title("Lagrangian System Animation with PD Controller (Smoothed)")

# Create objects for disks and rods
rod1, = ax.plot([], [], 'b-', lw=2)
rod2, = ax.plot([], [], 'g-', lw=2)
rod3, = ax.plot([], [], 'r-', lw=2)
disk1 = plt.Circle((0, 0), r, color='b', fill=True)
disk2 = plt.Circle((l, 0), r, color='g', fill=True)
disk3 = plt.Circle((2 * l, 0), r, color='r', fill=True)
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
    # Positions of the disks based on linkage lengths and angles
    x1, y1 = 0, 0  # Fixed base
    x2, y2 = x1 + l * jnp.cos(smooth_q1[frame]), y1 + l * jnp.sin(smooth_q1[frame])
    x3, y3 = x2 + l * jnp.cos(smooth_q2[frame]), y2 + l * jnp.sin(smooth_q2[frame])
    x4, y4 = x3 + l * jnp.cos(smooth_q3[frame]), y3 + l * jnp.sin(smooth_q3[frame])

    # Update rods
    rod1.set_data([x1, x2], [y1, y2])
    rod2.set_data([x2, x3], [y2, y3])
    rod3.set_data([x3, x4], [y3, y4])

    # Update disks
    disk1.center = (x1, y1)
    disk2.center = (x2, y2)
    disk3.center = (x3, y3)

    return rod1, rod2, rod3, disk1, disk2, disk3

# Create animation with smoother frames
ani = FuncAnimation(
    fig,
    update,
    frames=len(smooth_time_points),
    init_func=init,
    blit=True,
    interval=1000 * dt / 5  # Adjust playback speed for smoother visuals
)

# Show animation
plt.show()
