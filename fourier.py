import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.integrals.transforms import fourier_transform

# Define symbols
x = sp.symbols('x')
omega = sp.symbols('omega', real=True)
n = sp.symbols('n', integer=True)

# Get the function from the user
function_str = input("Enter the function f(x), you can use 'pi' for π: ")

# Parse the function string into a sympy expression
f = sp.sympify(function_str)

# Get the interval from the user
a_str = input("Enter the start of the interval (a), you can use 'pi': ")
b_str = input("Enter the end of the interval (b), you can use 'pi': ")

a = sp.sympify(a_str).evalf()
b = sp.sympify(b_str).evalf()

# Compute the period
T = b - a

# Fundamental angular frequency
omega0 = 2 * sp.pi / T

# Ask for number of terms
N = int(input("Enter the number of terms for Fourier series approximation (N): "))

# Compute a0
a0 = (1 / T) * sp.integrate(f, (x, a, b))
print(f"\nComputed a0 (average value over one period): {a0}")

# Initialize lists for an and bn
an_list = []
bn_list = []

print("\nComputing Fourier coefficients:")
for n_val in range(1, N+1):
    # Compute an
    an_expr = (2 / T) * sp.integrate(f * sp.cos(n_val * omega0 * x), (x, a, b))
    an_val = an_expr.evalf()
    an_list.append(an_val)
    
    # Compute bn
    bn_expr = (2 / T) * sp.integrate(f * sp.sin(n_val * omega0 * x), (x, a, b))
    bn_val = bn_expr.evalf()
    bn_list.append(bn_val)
    
    print(f"n = {n_val}: an = {an_val}, bn = {bn_val}")

# Construct the Fourier series approximation
s_approx = a0 / 2
for n_val in range(1, N+1):
    s_approx += an_list[n_val-1] * sp.cos(n_val * omega0 * x) + bn_list[n_val-1] * sp.sin(n_val * omega0 * x)

print("\nFourier series approximation s(x):")
print(s_approx)

# Convert sympy expressions to numerical functions
f_num = sp.lambdify(x, f, 'numpy')
s_num = sp.lambdify(x, s_approx, 'numpy')

# Create x values for plotting
x_vals = np.linspace(float(a - T), float(b + T), 1000)

# Evaluate the functions
f_vals = f_num(x_vals)
s_vals = s_num(x_vals)

# Plot the original function and Fourier series approximation
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f_vals, label='Original function')
plt.plot(x_vals, s_vals, label=f'Fourier series approximation (N={N})', linestyle='--')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Fourier Series Approximation')
plt.grid(True)
plt.show()

# Attempt to compute the Fourier Transform symbolically
try:
    F = fourier_transform(f, x, omega)
    print("\nFourier Transform of the function:")
    sp.pprint(F)
except Exception as e:
    print("\nUnable to compute the Fourier Transform symbolically.")
    print("Error:", e)

# Compute the numerical Fourier Transform
from scipy.fft import fft, fftfreq, fftshift

# Sampling settings
N_fft = 2048  # Increased number of samples for better resolution
dx = (x_vals[-1] - x_vals[0]) / N_fft
x_fft = np.linspace(x_vals[0], x_vals[-1], N_fft)
f_fft = f_num(x_fft)

# Perform FFT
F_vals = fft(f_fft)
freqs = fftfreq(N_fft, d=dx)
F_vals_shifted = fftshift(F_vals)
freqs_shifted = fftshift(freqs)

# Plot the magnitude spectrum of the Fourier Transform
plt.figure(figsize=(10, 6))
plt.plot(freqs_shifted, np.abs(F_vals_shifted))
plt.title('Magnitude Spectrum of the Fourier Transform')
plt.xlabel('Frequency ω')
plt.ylabel('|F(ω)|')
plt.grid(True)
plt.show()
