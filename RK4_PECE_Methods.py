import numpy as np
import pandas as pd

# Define the function for the ODE
def f(t, y):
    if t <= 0:
        return y + t
    else:
        return y - t

# RK4 Method Implementation
def rk4(f, y0, t_start, t_end, dt):
    n_steps = int((t_end - t_start) / dt) + 1
    t_values = np.linspace(t_start, t_end, n_steps)
    y_values = np.zeros(n_steps)
    y_values[0] = y0

    for i in range(1, n_steps):
        t_current = t_values[i - 1]
        y_current = y_values[i - 1]

        # RK4 stages
        k1 = f(t_current, y_current)
        k2 = f(t_current + dt / 2, y_current + dt * k1 / 2)
        k3 = f(t_current + dt / 2, y_current + dt * k2 / 2)
        k4 = f(t_current + dt, y_current + dt * k3)

        # Update step
        y_next = y_current + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_values[i] = y_next

    return t_values, y_values

# PECE Predictor-Corrector Scheme
def pece(f, y0, t_start, t_end, dt):
    n_steps = int((t_end - t_start) / dt) + 1
    t_values = np.linspace(t_start, t_end, n_steps)
    y_values = np.zeros(n_steps)
    y_values[0] = y0

    # Start with RK4 for the first step to initialize the method
    if n_steps > 1:
        y_values[1] = rk4(f, y0, t_start, t_start + dt, dt)[1][-1]

    # PECE method for remaining steps
    for i in range(1, n_steps - 1):
        t_current = t_values[i]
        t_next = t_values[i + 1]
        y_current = y_values[i]

        # Predictor (Adams-Bashforth two-step)
        if i == 1:
            y_pred = y_current + dt * f(t_current, y_current)
        else:
            y_pred = y_values[i] + (dt / 2) * (3 * f(t_current, y_current) - f(t_values[i - 1], y_values[i - 1]))

        # Corrector (Adams-Moulton one-step)
        y_corr = y_current
        for _ in range(10):  # Perform fixed-point iteration
            y_corr_new = y_values[i] + (dt / 2) * (f(t_current, y_current) + f(t_next, y_pred))
            if abs(y_corr_new - y_corr) < 1e-12:  # Convergence tolerance
                break
            y_corr = y_corr_new

        # Update the solution
        y_values[i + 1] = y_corr

    return t_values, y_values

# Parameters for the problem
y0 = 1  # Initial condition
t_start = -1
t_end = 1
dt = 0.1  # Step size

# Solve using the RK4 method
t_values_rk4, y_values_rk4 = rk4(f, y0, t_start, t_end, dt)

# Solve using the PECE method
t_values_pece, y_values_pece = pece(f, y0, t_start, t_end, dt)

# Compute the exact solution at the same points
exact_y_values = np.array([
    np.exp(t + 1) - t - 1 if t <= 0 else np.exp(t + 1) - 2 * np.exp(t) + t + 1
    for t in t_values_rk4
])

# Calculate the absolute errors
errors_rk4 = np.abs(y_values_rk4 - exact_y_values)
errors_pece = np.abs(y_values_pece - exact_y_values)

# Present the results
df_results = pd.DataFrame({
    "t": t_values_rk4,
    "RK4 Approximation": y_values_rk4,
    "PECE Approximation": y_values_pece,
    "Exact Solution": exact_y_values,
    "RK4 Error": errors_rk4,
    "PECE Error": errors_pece,
})

print(df_results)
