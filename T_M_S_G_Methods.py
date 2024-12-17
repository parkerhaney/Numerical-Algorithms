import numpy as np
from scipy.integrate import quad, fixed_quad

# Define the functions
def f1(x):
    return 3 * x + 2

def f2(x):
    return (x - 1) ** 2

def f3(x):
    return x ** 3 + 2

# Composite Trapezoidal Rule
def trapezoidal_rule(func, a, b, n):
    x = np.linspace(a, b, n+1)
    y = func(x)
    h = (b - a) / n
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

# Midpoint Rule
def midpoint_rule(func, a, b, n):
    h = (b - a) / n
    midpoints = a + h * (np.arange(n) + 0.5)
    return h * np.sum(func(midpoints))

# Simpson's Rule (custom implementation)
def simpsons_rule(func, a, b, n):
    if n % 2 == 1:  # Simpson's rule requires an even number of intervals
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

# Exact values for integrals using scipy's quad
a, b = 1, 3
I1_exact, _ = quad(f1, a, b)
I2_exact, _ = quad(f2, a, b)
I3_exact, _ = quad(f3, a, b)

# Apply trapezoidal and midpoint rules for each function
n = 100  # Number of intervals for approximation
I1_trap = trapezoidal_rule(f1, a, b, n)
I2_trap = trapezoidal_rule(f2, a, b, n)
I3_trap = trapezoidal_rule(f3, a, b, n)

I1_mid = midpoint_rule(f1, a, b, n)
I2_mid = midpoint_rule(f2, a, b, n)
I3_mid = midpoint_rule(f3, a, b, n)

# Calculate errors for part (b)
error_trap = [abs(I1_exact - I1_trap), abs(I2_exact - I2_trap), abs(I3_exact - I3_trap)]
error_mid = [abs(I1_exact - I1_mid), abs(I2_exact - I2_mid), abs(I3_exact - I3_mid)]

# Output results for part (b)
print("Trapezoidal Rule Errors:", [f"{err:.5e}" for err in error_trap])
print("Midpoint Rule Errors:", [f"{err:.5e}" for err in error_mid])

# Apply custom Simpson's rule for each function
I1_simp = simpsons_rule(f1, a, b, n)
I2_simp = simpsons_rule(f2, a, b, n)
I3_simp = simpsons_rule(f3, a, b, n)

# Gaussian Quadrature (fixed_quad for 5 points)
I1_gauss, _ = fixed_quad(f1, a, b, n=5)
I2_gauss, _ = fixed_quad(f2, a, b, n=5)
I3_gauss, _ = fixed_quad(f3, a, b, n=5)

# Calculate errors for part (c)
error_simp = [abs(I1_exact - I1_simp), abs(I2_exact - I2_simp), abs(I3_exact - I3_simp)]
error_gauss = [abs(I1_exact - I1_gauss), abs(I2_exact - I2_gauss), abs(I3_exact - I3_gauss)]

# Output results for part (c)
print("Simpson's Rule Errors:", [f"{err:.5e}" for err in error_simp])
print("Gaussian Quadrature Errors:", [f"{err:.5e}" for err in error_gauss])
