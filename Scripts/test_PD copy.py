import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

# Parameters
m1, m2, k, F = 1.0, 1.0, 10.0, 0 # Masses, spring constant, external force
l, r = 5, 2                   # Length of rod and radius of disk
I1, I2 = 1 / 12 * m1 * l**2, m2 * r**2 / 2  # Moments of inertia
dt = 0.1  # Time step
t_span = (0, 300)  # Time span
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
K = 10**7 * [5.8038,   -5.7877,   -0.0160,    7.3336,   -7.4008,    0.0672]

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
y0 = jnp.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])  # Slightly perturbed initial state

# Perform integration
times, trajectory = integrate(y0, dt, num_steps)
q1, q2, q3 = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]  # Angular positions

# Plot angular positions
plt.figure(figsize=(10, 6))

plt.plot(times, q1, label='q1 (Angular Position 1)', linewidth=1.5)
plt.plot(times, q2, label='q2 (Angular Position 2)', linewidth=1.5)
plt.plot(times, q3, label='q3 (Angular Position 3)', linewidth=1.5)

plt.title("Angular Positions Over Time", fontsize=14)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Angular Position (rad)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

plt.show()