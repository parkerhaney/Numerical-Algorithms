import numpy as np

# Define the function for the ODE
def dydt(t, y):
    return 2 * t * y**2

# Backward Euler method
def backward_euler(f, y0, t_start, t_end, dt):
    n_steps = int((t_end - t_start) / dt) + 1
    t_values = np.linspace(t_start, t_end, n_steps)
    y_values = np.zeros(n_steps)
    y_values[0] = y0

    for i in range(1, n_steps):
        t_next = t_values[i]
        # Solve the implicit equation: y_next = y_current + dt * f(t_next, y_next)
        # Using Newton's method
        y_current = y_values[i - 1]

        # Initial guess for y_next
        y_next = y_current
        for _ in range(100):  # Perform Newton's method iteration
            f_val = y_next - y_current - dt * f(t_next, y_next)
            f_prime = 1 - dt * (4 * t_next * y_next)  # Derivative of the equation
            y_next_new = y_next - f_val / f_prime
            if abs(y_next_new - y_next) < 1e-12:  # Convergence tolerance
                y_next = y_next_new
                break
            y_next = y_next_new

        y_values[i] = y_next

    return t_values, y_values

# Define parameters
a = 0.99
y0 = 1
t_start = 0
t_end = a
k_values = [3, 4, 5]  # Corresponding to step sizes
step_sizes = [a / (10**k) for k in k_values]

# Solve for each step size and store the results
results = {}
for dt in step_sizes:
    t_vals, y_vals = backward_euler(dydt, y0, t_start, t_end, dt)
    results[dt] = (t_vals, y_vals)

# Exact solution for error calculation
def exact_solution(t):
    return 1 / (1 - t**2)

# Calculate the absolute error at t = a for each step size
errors = {}
for dt, (t_vals, y_vals) in results.items():
    approx_y_at_a = y_vals[-1]
    exact_y_at_a = exact_solution(a)
    errors[dt] = abs(approx_y_at_a - exact_y_at_a)

# Present results
import pandas as pd

df_results = pd.DataFrame({
    "Step Size (dt)": step_sizes,
    "Approximation at t = a": [results[dt][1][-1] for dt in step_sizes],
    "Exact Solution at t = a": exact_solution(a),
    "Absolute Error": [errors[dt] for dt in step_sizes],
})

print(df_results)
