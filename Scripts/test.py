from sympy import symbols, diff, Function

# Define symbols for time, position, and velocity
t = symbols('t')
x = Function('x')(t)
v = diff(x, t)
m, k = symbols('m k')  # mass and spring constant

# Define Lagrangian: L = T - V (Kinetic Energy - Potential Energy)
T = (1/2) * m * v**2  # Kinetic Energy
V = (1/2) * k * x**2  # Potential Energy
L = T - V

# Compute the Euler-Lagrange equation
dL_dx = diff(L, x)
dL_dv = diff(L, v)
d_dt_dL_dv = diff(dL_dv, t)

euler_lagrange = d_dt_dL_dv - dL_dx
print(f"Euler-Lagrange equation: {euler_lagrange}")
