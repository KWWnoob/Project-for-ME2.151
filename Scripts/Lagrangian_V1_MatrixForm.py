from sympy import symbols, diff, Function, Matrix, simplify

# Define time and generalized coordinates
t = symbols('t')
q1, q2, q3 = Function('q1')(t), Function('q2')(t), Function('q3')(t)  # Generalized coordinates
q = Matrix([q1, q2, q3])  # Generalized coordinate vector
q_dot = q.diff(t)         # Velocity vector
q_ddot = q_dot.diff(t)    # Acceleration vector

# Parameters
m1, m2, k, F = symbols('m1 m2 k F')  # Masses, spring constant, extensionternal force
l, r = symbols('l r')                  # Length of rod and radius of disk
I1, I2 = symbols('I1 I2')              # Moments of inertia for disks (I1) and rods (I2)

# Define energies (TODO: maybe graviational energy)
T_rot_disks = 1/2 * I1 * (q_dot[0]**2 + q_dot[1]**2 + q_dot[2]**2)
T_rot_rods = 1/2 * I2 * (q_dot[0]**2 + q_dot[1]**2 + q_dot[2]**2)

T_trans_rods = sum(
    1/2 * m2 * ((sum(q_dot[:i+1])) * (i + 0.5) * (l + r))**2 for i in range(3)
)
T_trans_disks = sum(
    1/2 * m1 * ((sum(q_dot[:i+1])) * (i + 1) * (l + r))**2 for i in range(2)
)

V_spring = 1/2 * k * (q1**2 + q2**2 + q3**2) #TODO: need more careful considerations

# Total Lagrangian
L = T_rot_disks + T_rot_rods + T_trans_rods + T_trans_disks - V_spring

# Derive Euler-Lagrange equations in symbolic form
euler_lagrange_eqs = Matrix([
    diff(diff(L, q_dot[i]), t) - diff(L, q[i]) for i in range(3)
])

# Define external forces as specified (TODO: diffrent routings + frictions as damping effect)
F_external = Matrix([
    F*r - k * q1 * r,
    F*r - k * q2 * r,
    F*r - k * q3 * r
])

# Split into mass, damping, and stiffness terms
M = euler_lagrange_eqs.jacobian(q_ddot)  # Mass matrix
C = euler_lagrange_eqs.jacobian(q_dot) - M.diff(t)  # Coriolis/Damping terms
K = euler_lagrange_eqs.jacobian(q)  # Stiffness matrix
Force = euler_lagrange_eqs - M * q_ddot - C * q_dot - K * q - F_external  # Subtract external forces

# Simplify the external forces vector
F_simplified = simplify(Force)

# Print results
print("Mass Matrix (M):")
print(M)
print("\nDamping Matrix (C):")
print(C)
print("\nStiffness Matrix (K):")
print(K)
print("\nSimplified External Forces (F):")
print(F_simplified)
