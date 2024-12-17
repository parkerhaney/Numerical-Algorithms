import numpy as np

# Define the composite Midpoint rule function
def composite_Midpoint(F, a, b, N):
    h = (b - a) / N  # Step size
    midpoints = a + h * (np.arange(N) + 0.5)  # Midpoints of each subinterval
    return h * np.sum(F(midpoints))

# Define the function f(x) = exp(-x^2) / sqrt(pi)
def f(x):
    return np.exp(-x**2) / np.sqrt(np.pi)

# Exact integral value
I_exact = 1

# Values of N to test
N_values = [1, 2, 4, 8, 16, 32]

# Calculate approximations and errors
errors = []
for N in N_values:
    I_approx = composite_Midpoint(f, -5, 5, N)
    error = abs(I_exact - I_approx)
    errors.append(error)
    print(f"N = {N}, Approximation = {I_approx:.11f}, Error = {error:.5e}")

# Expected convergence rate analysis
h_values = [(10 / N) for N in N_values]  # h = (b - a) / N for each N
convergence_rate = [errors[i] / (h_values[i]**2) for i in range(len(N_values))]
print("\nConvergence Rate (Error/h^2):", [f"{rate:.5e}" for rate in convergence_rate])
# We are seeing a faster rate of convergance when this code is ran
