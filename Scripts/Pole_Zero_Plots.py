import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import ss2zpk

from scipy.signal import ss2tf, tf2zpk


from End_Effector_Position_Plot import compute_state_space, M_jnp, K_jnp, C_jnp

# Reuse the state-space computation from End_Effector_Position_Plot.py
A, B, C, D = compute_state_space(M_jnp, K_jnp)

# Define functions for controllability and observability
def check_controllability(A, B):
    """Check if the system is controllable."""
    n = A.shape[0]
    Q_c = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    rank = np.linalg.matrix_rank(Q_c)
    return rank == n

def check_observability(A, C):
    """Check if the system is observable."""
    n = A.shape[0]
    Q_o = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])
    rank = np.linalg.matrix_rank(Q_o)
    return rank == n

# Check controllability and observability
controllable = check_controllability(A, B)
observable = check_observability(A, C)

print(f"System is controllable: {controllable}")
print(f"System is observable: {observable}")

input_index = 0
numerator, denominator = ss2tf(A, B, C, D, input=input_index)
zeros, poles, gain = tf2zpk(numerator[0], denominator)

plt.figure()
plt.scatter(np.real(poles), np.imag(poles), marker='x', label='Poles', color='blue')
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', label='Zeros', color='red')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.title("Pole-Zero Plot")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.show()

# Check system stability (all poles must have negative real parts)
stable = all(np.real(poles) < 0)
print(f"System is stable: {stable}")