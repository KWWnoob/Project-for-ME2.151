'''
Now with friction

'''

import jax.numpy as jnp
from sympy import symbols, diff, Function, Matrix, simplify, N

def compute_lagrangian_matrices():
    # Define time and generalized coordinates
    t = symbols('t')
    q1, q2, q3 = Function('q1')(t), Function('q2')(t), Function('q3')(t)  # Generalized coordinates
    q = Matrix([q1, q2, q3])  # Generalized coordinate vector
    q_dot = q.diff(t)         # Velocity vector
    q_ddot = q_dot.diff(t)    # Acceleration vector

    # Parameters
    m1, m2, k = symbols('m1 m2 k')  # Masses, spring constant
    l, r = symbols('l r')           # Length of rod and radius of disk
    I1, I2 = symbols('I1 I2')       # Moments of inertia for disks (I1) and rods (I2)
    c1, c2, c3 = symbols('c1 c2 c3')  # Damping coefficients (friction)

    # Define kinetic energy (rotational and translational)
    T_rot_disks = 1/2 * I1 * sum(q_dot[i]**2 for i in range(3))
    T_rot_rods = 1/2 * I2 * sum(q_dot[i]**2 for i in range(3))

    T_trans_rods = sum(
        1/2 * m2 * ((sum(q_dot[:i+1])) * (i + 0.5) * (l + r))**2 for i in range(3)
    )
    T_trans_disks = sum(
        1/2 * m1 * ((sum(q_dot[:i+1])) * (i + 1) * (l + r))**2 for i in range(2)
    )

    # Define potential energy (spring energy)
    V_spring = 1/2 * k * sum(q[i]**2 for i in range(3))

    # Total Lagrangian
    L = T_rot_disks + T_rot_rods + T_trans_rods + T_trans_disks - V_spring

    # Generalized friction forces
    friction_forces = Matrix([-c1 * q_dot[0], -c2 * q_dot[1], -c3 * q_dot[2]])

    # Derive Euler-Lagrange equations
    euler_lagrange_eqs = Matrix([
        diff(diff(L, q_dot[i]), t) - diff(L, q[i]) + friction_forces[i]
        for i in range(3)
    ])

    # Compute Mass (M), Damping (C), and Stiffness (K) matrices
    M = euler_lagrange_eqs.jacobian(q_ddot)  # Mass matrix
    C = euler_lagrange_eqs.jacobian(q_dot) - M.diff(t)  # Damping matrix
    K = euler_lagrange_eqs.jacobian(q)  # Stiffness matrix

    return simplify(M), simplify(C), simplify(K)

if __name__ == '__main__':
    # Call the function
    M, C, K = compute_lagrangian_matrices()
    
    # Substitute numerical values
    m1_val, m2_val, k_val = 1.0, 1.0, 10.0
    l_val, r_val = 10.0, 2.0
    I1_val, I2_val = 1 / 12 * m1_val * l_val**2, m2_val * r_val**2 / 2
    c1_val, c2_val, c3_val = 0.1, 0.1, 0.1  # Friction coefficients

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

    print("Mass Matrix (JAX, M):")
    print(M_jnp)
    print("\nDamping Matrix (JAX, C):")
    print(C_jnp)
    print("\nStiffness Matrix (JAX, K):")
    print(K_jnp)
