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

# --- Plot Results ---
def end_effector_position(q1, q2, q3, l_val):
    x = l_val * (np.cos(q1) + np.cos(q1 + q2) + np.cos(q1 + q2 + q3))
    y = l_val * (np.sin(q1) + np.sin(q1 + q2) + np.sin(q1 + q2 + q3))
    return x, y

q1_mpc = x_history[:,0]
q2_mpc = x_history[:,1]
q3_mpc = x_history[:,2]

end_effector_x_mpc = []
end_effector_y_mpc = []
for i in range(x_history.shape[0]):
    xx, yy = end_effector_position(q1_mpc[i], q2_mpc[i], q3_mpc[i], l_val)
    end_effector_x_mpc.append(xx)
    end_effector_y_mpc.append(yy)

end_effector_x_mpc = np.array(end_effector_x_mpc)
end_effector_y_mpc = np.array(end_effector_y_mpc)

ref_xy = end_effector_position(x_ref[0], x_ref[1], x_ref[2], l_val)

time_array = np.linspace(t_span[0], t_span[1], x_history.shape[0])

plt.figure(figsize=(12,6))
plt.plot(end_effector_x_mpc, end_effector_y_mpc, label="MPC Trajectory (do-mpc)")
plt.scatter(end_effector_x_mpc[0], end_effector_y_mpc[0], color='red', label="Start")
plt.scatter(end_effector_x_mpc[-1], end_effector_y_mpc[-1], color='blue', label="End")
plt.scatter([ref_xy[0]], [ref_xy[1]], color='orange', marker='x', s=100, label="Reference")
plt.title("MPC End-Effector Trajectory (do-mpc)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid()
plt.axis('equal')
plt.savefig('MPC_controlled_trajectory_reference_tracking.png', dpi=300)
plt.close()  # Close the figure after saving

# Plotting end-effector position over time
plt.figure(figsize=(12,6))
plt.plot(time_array, end_effector_x_mpc, label="X Position")
plt.plot(time_array, end_effector_y_mpc, label="Y Position")
plt.axhline(y=ref_xy[0], color='orange', linestyle='--', label="X Ref")
plt.axhline(y=ref_xy[1], color='purple', linestyle='--', label="Y Ref")
plt.title("End-Effector Position Over Time (MPC, do-mpc)")
plt.xlabel("Time (s)")
plt.ylabel("Position (units)")
plt.legend()
plt.grid()
plt.savefig('MPC_end_effector_position_over_time_reference_tracking.png', dpi=300)
plt.close()  # Close the figure after saving

# Plot control inputs
plt.figure(figsize=(12,6))
for i in range(u_history.shape[1]):
    plt.plot(time_array[:-1], u_history[:,i], label=f"u[{i}]")

plt.title("Control Inputs Over Time (MPC, do-mpc)")
plt.xlabel("Time (s)")
plt.ylabel("Input (units)")
plt.legend()
plt.grid()
plt.savefig('MPC_control_input_over_time.png', dpi=300)
plt.close

'''
show_animation
'''

time_array = np.linspace(t_span[0], t_span[1], x_history.shape[0])
q1_mpc = x_history[:,0]
q2_mpc = x_history[:,1]
q3_mpc = x_history[:,2]

# Create finer time points for smoother animation
smooth_time_points = np.linspace(t_span[0], t_span[1], 5*num_steps)  # 5x frames
interp_q1 = interp1d(time_array, q1_mpc, kind='cubic')  # Cubic interpolation
interp_q2 = interp1d(time_array, q2_mpc, kind='cubic')
interp_q3 = interp1d(time_array, q3_mpc, kind='cubic')

# Interpolated values
smooth_q1 = interp_q1(smooth_time_points)
smooth_q2 = interp_q2(smooth_time_points)
smooth_q3 = interp_q3(smooth_time_points)

# Animation Setup
fig, ax = plt.subplots()
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_aspect('equal')
ax.set_title("MPC Control_System Animation")

# Add the reference point to the plot
reference_point = ax.scatter(ref_xy[0], ref_xy[1], color='orange', label="Reference Point", marker='x', s=100)

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
    interval=1000 * dt / 15  # Adjust playback speed
)
# Save the animation as a video
ani.save('MPC_Control_Animation.gif', writer='pillow', fps=48)

# Show animation
plt.show()
