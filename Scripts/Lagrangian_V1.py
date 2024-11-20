from sympy import symbols, diff, Function

# Define symbols for time, angular position, and angular velocity
t = symbols('t')
x1, x2, x3 = Function('x1')(t), Function('x2')(t), Function('x3')(t)
v1, v2, v3 = diff(x1, t), diff(x2, t), diff(x3, t)

# Parameters
m1, m2, k = symbols('m1 m2 k')  # Mass of disk (m1), mass of rod (m2), spring constant
l, r = symbols('l r')           # Length of the rod and radius of the disk
I1, I2 = symbols('I1 I2')       # Moments of inertia for disks (I1) and rods (I2)

# Lists for generalized coordinates
angular_pos_list = [x1, x2, x3]
angular_velo_list = [v1, v2, v3]

# Initialize energy terms
T_rot_disks = 0
T_rot_rods = 0
T_trans_rods = 0
T_trans_disks = 0
V_spring = 0

# Rotational energy for disks
for i in range(3):
    T_rot_disks += 1/2 * I1 * angular_velo_list[i]**2

# Rotational energy for rods
for i in range(3):
    T_rot_rods += 1/2 * I2 * angular_velo_list[i]**2

# Translational energy for rods (linear velocity depends on angular velocity)
sum_angle_velo = 0
for i in range(3):
    sum_angle_velo += angular_velo_list[i]
    T_trans_rods += 1/2 * m2 * (sum_angle_velo * (i + 0.5) * (l + r))**2

# Translational energy for disks
sum_angle_velo = 0
for i in range(2):  # Only disks 2 and 3 move in this case
    sum_angle_velo += angular_velo_list[i]
    T_trans_disks += 1/2 * m1 * (sum_angle_velo * (i + 1) * (l + r))**2

# Potential energy of the springs
for i in range(3):
    V_spring += 1/2 * k * angular_pos_list[i]**2 #TODO: double check

# Lagrangian
L = T_rot_disks + T_rot_rods + T_trans_rods + T_trans_disks - V_spring

# Print the Lagrangian
print("Lagrangian (L):", L)

# Derive Euler-Lagrange equations
euler_lagrange_eqs = []
for x, v in zip(angular_pos_list, angular_velo_list):
    dL_dv = diff(L, v)                      # Partial derivative of L with respect to velocity
    dL_dx = diff(L, x)                      # Partial derivative of L with respect to position
    d_dt_dL_dv = diff(dL_dv, t)             # Time derivative of dL/dv
    eq = d_dt_dL_dv - dL_dx                 # Euler-Lagrange equation
    euler_lagrange_eqs.append(eq)

# Print Euler-Lagrange equations
for i, eq in enumerate(euler_lagrange_eqs, 1):
    print(f"Euler-Lagrange Equation {i}: {eq}")
