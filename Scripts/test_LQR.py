'''
Took forever to converge..
'''
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# Parameters
m1, m2, k, F = 1.0, 1.0, 10.0, 0  # Masses, spring constant, external force
l, r = 5, 2                      # Length of rod and radius of disk
I1, I2 = 1 / 12 * m1 * l**2, m2 * r**2 / 2  # Moments of inertia
dt = 0.1                          # Time step
t_span = (0, 300)                 # Time span
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

# Linearize the system
def linearize_system():
    M = mass_matrix()
    K = stiffness_matrix()

    # Linearized A matrix
    A_upper = jnp.hstack([jnp.zeros((3, 3)), jnp.eye(3)])
    A_lower = jnp.hstack([-jnp.linalg.inv(M) @ K, jnp.zeros((3, 3))])
    A = jnp.vstack([A_upper, A_lower])

    # Linearized B matrix
    B_upper = jnp.zeros((3, 3))
    B_lower = jnp.linalg.inv(M)
    B = jnp.vstack([B_upper, B_lower])

    return A, B

# Compute LQR gains
def lqr_gain(A, B, Q, R):
    # Solve the continuous-time algebraic Riccati equation
    P = solve_continuous_are(A, B, Q, R)
    # Compute the LQR gain matrix
    K = jnp.linalg.inv(R) @ B.T @ P
    return K

# Cost Matrices for LQR
Q = jnp.diag(jnp.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]))  # State cost
R = jnp.diag(jnp.array([1.0, 1.0, 1.0]))                    # Control effort cost

# Linearized system matrices
A, B = linearize_system()

# LQR Gain Matrix
K_lqr = lqr_gain(A, B, Q, R)

# External Forces with LQR Controller
def external_forces_lqr(y):
    q = y[:3]        # Generalized coordinates
    q_dot = y[3:]    # Velocities
    state = jnp.concatenate((q, q_dot))

    # LQR Control Force
    control_force = -K_lqr @ state
    return control_force

# Equations of Motion with LQR Controller
def equations_of_motion(y, t):
    q = y[:3]        # Generalized coordinates
    q_dot = y[3:]    # Velocities

    M = mass_matrix()
    K = stiffness_matrix()
    Force = external_forces_lqr(y)

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

plt.title("Angular Positions with LQR Controller Over Time", fontsize=14)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Angular Position (rad)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

plt.show()
