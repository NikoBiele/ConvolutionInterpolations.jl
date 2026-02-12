"""
Generalized Convolution Kernel Discovery via Symbolic Boundary Value Problems

This script implements a systematic analytical approach for discovering high-order convolution
interpolation kernels by solving symbolic boundary value problems. It generalizes the method
of R.G. Keys (1981) to arbitrary polynomial degrees, enabling discovery of kernels that would
be intractable to derive by hand.

Mathematical Framework
----------------------
The approach constructs piecewise polynomial kernels k(s) defined over M_eqs intervals, each
with polynomial degree p_deg. The kernel must satisfy:

1. Interpolation condition: k(0) = 1, k(i) = 0 for integer i ≠ 0
2. Derivative continuity: Continuous up to d_cont derivatives at interval boundaries
3. Polynomial reproduction: Exact reproduction of polynomials up to a target degree
4. Symmetry: Even symmetry k(-s) = k(s)

The problem is formulated as a system of symbolic equations using SymPy, with additional
constraints determined through exhaustive combinatorial search to achieve optimal frequency
domain characteristics.

Novel Contributions
-------------------
- Automated discovery of kernels up to 13th degree (b13 kernel: 9 equations, degree 13, C6)
- Systematic exploration of boundary condition combinations
- Frequency domain optimization via FFT analysis during search
- Taylor series analysis to ensure desired order of accuracy
- Rational coefficient representation for arbitrary precision evaluation
- Verification framework for published kernels and their Taylor order

Verification Capability
-----------------------
By setting the search parameters (M_eqs, p_deg, d_cont) to match published kernels, this
script can independently derive and verify existing methods:
- Keys' cubic kernel (1981): M_eqs=2, p_deg=3, d_cont=1
- Meijering et al. quintic/septic kernels (1999): p_deg=5/7
- Confirms Taylor series order of accuracy through symbolic expansion

This provides rigorous verification of literature claims and enables direct comparison of
different kernel families.

Discovery Process
-----------------
1. Generate symbolic BVP equations for interpolation and continuity constraints
2. Enumerate combinations of additional boundary conditions
3. For each combination:
   - Solve symbolic system to obtain kernel coefficients
   - Verify polynomial reproduction via Taylor series expansion
   - Evaluate frequency domain response via FFT
   - Accept if derivative at Nyquist frequency exceeds threshold
4. Select kernel with steepest frequency response while maintaining passband flatness

Key Kernels Discovered
----------------------
- b5: 3 equations, degree 5, C3 continuous, reproduces quintics
- b7: 4 equations, degree 7, C4 continuous, reproduces septics (optimal polynomial reproduction)
- b9: 5 equations, degree 9, C5 continuous, 7th order accurate
- b11: 6 equations, degree 11, C6 continuous, 7th order accurate  
- b13: 7 equations, degree 13, C6 continuous, 7th order accurate

All discovered kernels achieve 7th order convergence and are stored with exact rational
coefficients, enabling extended precision evaluation beyond Float64 machine precision.

References
----------
R.G. Keys, "Cubic Convolution Interpolation for Digital Image Processing,"
IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 6,
pp. 1153-1160, December 1981.

E.H.W. Meijering, K.J. Zuiderveld, and M.A. Viergever, "Image Reconstruction by
Convolution with Symmetrical Piecewise nth-Order Polynomial Kernels," IEEE
Transactions on Image Processing, vol. 8, no. 2, pp. 192-201, February 1999.

Author: Nikolaj Maack Bielefeld
License: MIT
"""

#%%
import sympy as sp
from sympy import solve
from sympy import *
import itertools
from itertools import combinations
import numpy as np
import math
import time
import threading

def kernel_func(s, coefs_final, coefficients):
    s_abs = abs(s)
    kernel_val = 0.0
    for i in range(M_eqs):
        if s_abs < i+1:
            for j in range(p_deg+1):
                try:
                    kernel_val += float(coefs_final[coefficients[i, j]]) * s_abs**j
                except:
                    kernel_val += 0.0
            return kernel_val
        else:
            continue
    return kernel_val

def find_closest(numbers, target):
    closest_index = 0
    smallest_difference = abs(numbers[0] - target)
    
    for i, num in enumerate(numbers[1:], 1):
        difference = abs(num - target)
        if difference < smallest_difference:
            closest_index = i
            smallest_difference = difference
    
    return closest_index

def generate_symbolic_bvp_equations(M_eqs, p_deg, d_cont):
    """
    Generate symbolic equations for the boundary value problem.
    
    Parameters:
    M_eqs (int): Number of polynomials (equations)
    p_deg (int): Degree of polynomials
    d_cont (int): Highest derivative continuity
    
    Returns:
    tuple: (equations, coefficients)
    """
    
    # Define symbolic coefficients
    A = sp.Matrix([[sp.Symbol(f'A_{{{i+1},{j}}}') for j in range(p_deg + 1)] for i in range(M_eqs)])
    
    equations = []
    
    def polynomial_term(i, j, x, n):
        """Generate term of polynomial and its derivatives"""
        # if j - n < 0:
            # return 0
        coeff = sp.prod([(j-k) for k in range(n)])
        return A[i, j] * coeff * x**(j-n)
    
    # Zeroth derivative (values) - write equation for each end of each polynomial
    for i in range(M_eqs):
        # Left end of the polynomial
        x_left = i
        if i == 0:
            # First boundary condition: u(0) = 1
            equations.append(sum(polynomial_term(i, j, x_left, 0) for j in range(p_deg + 1)) - 1)
            # # extra boundary condition: u(-1) = 0
            # x_left = i - 1
            # equations.append(sum(polynomial_term(i, j, x_left, 0) for j in range(p_deg + 1)) + 1)
        else:
            # Interior point: u(i) = 0
            equations.append(sum(polynomial_term(i, j, x_left, 0) for j in range(p_deg + 1)))
            
        # Right end of the polynomial
        # if i == M_eqs-1: # extra addition (from paper)
        #     pass
        # else:
        x_right = i + 1
        equations.append(sum(polynomial_term(i, j, x_right, 0) for j in range(p_deg + 1)))
    
    # Higher derivatives - loop over nodes
    for n in range(1, d_cont + 1):
        for k in range(M_eqs + 1):
            x = k
            if k == 0:
                # if n == 1:
                if n % 2 == 0:
                    pass
                    # new boundary condition: du(-1) = -du(+1)
                    # equations.append(sum(polynomial_term(0, j, -1, n) for j in range(n, p_deg + 1)) + sum(polynomial_term(0, j, 1, n) for j in range(n, p_deg + 1))) # if p_deg-j-n >= 0))
                else:
                    # Left boundary condition: n-th derivative = 0
                    equations.append(sum(polynomial_term(0, j, x, n) for j in range(n, p_deg + 1))) # if p_deg-j-n >= 0))
            elif k == M_eqs:
                # Right boundary condition: n-th derivative = 0
                equations.append(sum(polynomial_term(M_eqs-1, j, x, n) for j in range(n, p_deg + 1))) # if p_deg-j-n >= 0))
            else:
                # Continuity of n-th derivative between equations
                left_eq = sum(polynomial_term(k-1, j, x, n) for j in range(n,p_deg + 1)) # if p_deg-j-n >= 0)
                right_eq = sum(polynomial_term(k, j, x, n) for j in range(n,p_deg + 1)) # if p_deg-j-n >= 0)
                equations.append(left_eq - right_eq)

    return sp.Matrix(equations), A


class TimeoutException(Exception):
    pass

def solve_with_timeout(A, b, timeout=15):
    result = []
    exception = []

    def solve_function():
        try:
            coefs = sp.solve(A, b)
            result.append(coefs)
        except Exception as e:
            exception.append(e)

    thread = threading.Thread(target=solve_function)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("Linear solve timed out")
        return None
    elif exception:
        print(f"An error occurred: {exception[0]}")
        return None
    else:
        return result[0]
    
#%%
extra_eq = 0 # number of additional equations
M_eqs = 2 + extra_eq # number of polynomials            # 2, 3, 3, 4, 4, 5, 5, 6, 6,
# Example usage                 accuracy # 3, 4, 5, 5, 5, 5, 
p_deg = 2*(M_eqs-extra_eq)-1  # degree of polynomials           # 3, 3, 5, 5, 7, 7, 9, 9, 11,
d_cont = p_deg-2 #-2 # highest derivative continuity  # 1, 1, 2, 2, 3, 3, 4, 4, 5,
max_taylor = 7 #M_eqs #+1 # maximum number of terms in the Taylor expansion
solver_timeout = 20*60 # seconds till time out
highest_fixed_coef = p_deg # d_cont # p_deg #
skip_first_equation = False # True
print(f"Number of equations: {M_eqs}")
print(f"Degree of polynomials: {p_deg}")
print(f"Highest derivative continuity: {d_cont}")
print(f"Maximum number of terms in the Taylor expansion: {max_taylor}")
print(f"Solver timeout: {solver_timeout} seconds")
print(f"Skip first equation: {skip_first_equation}")
print(f"Highes degree of fixed coefficient: {highest_fixed_coef}")

#%%
# Further combinations which work (relaxed derivatives)
# RESULTS WITH 'highest_fixed_coef'=p_deg
# 2 equations
# M_eqs=2, p_deg=3, d_cont=1, taylor=2, derivative at x=0.5: -2.321, result from paper
# M_eqs=2, p_deg=5, d_cont=1, taylor=2, derivative at x=0.5: -2.431, very good, slightly better than paper
# M_eqs=2, p_deg=7, d_cont=1, taylor=2, derivative at x=0.5: -2.524, very good, even better
# M_eqs=2, p_deg=9, d_cont=1, taylor=2, derivative at x=0.5: -2.601, very good, slightly < 1 at x=0
# M_eqs=2, p_deg=11, d_cont=1, taylor=2, derivative at x=0.5: -2.664, very good, slightly < 1 at x=0
# M_eqs=2, p_deg=5, d_cont=2, taylor=2, derivative at x=0.5: -2.427, approx 0.05 ripple in stopband
# M_eqs=2, p_deg=7, d_cont=2, taylor=2, derivative at x=0.5: -2.521, approx 0.05 ripple in stopband

# 3 equations
# M_eqs=3, p_deg=5, d_cont=2, taylor=4, derivative at x=0.5: -3.136, 5th order accurate! small passband ripple
# M_eqs=3, p_deg=5, d_cont=1, taylor=2, derivative at x=0.5: -4.751, ripple in passband, ripples in stopband
# M_eqs=3, p_deg=7, d_cont=1, taylor=2, derivative at x=0.5: -4.855, ripple in passband, ripples in stopband
# M_eqs=3, p_deg=9, d_cont=1, taylor=2, derivative at x=0.5: -4.818, ripple in passband, ripples in stopband
# M_eqs=3, p_deg=11, d_cont=1, taylor=2, derivative at x=0.5: -4.818, same solution as above

# 4 equations (d_cont=1 or d_cont=2)
# M_eqs=4, p_deg=7, d_cont=1, taylor=2, derivative at x=0.5: -4.701, very good! only minor ripples
# M_eqs=4, p_deg=7, d_cont=2, taylor=2, derivative at x=0.5: -3.904, not=1 at x=0, minor ripples in stopband
# M_eqs=4, p_deg=9, d_cont=1, taylor=2, derivative at x=0.5: -4.818, same solution as for 3 equations
# M_eqs=4, p_deg=9, d_cont=2, taylor=2, derivative at x=0.5: nothing satisfying constraints
# M_eqs=4, p_deg=11, d_cont=1, taylor=2, derivative at x=0.5: -4.818, same solution as for 3 equations

# generate the basic equations
equations_raw, coefficients = generate_symbolic_bvp_equations(M_eqs, p_deg, d_cont)

print(f"Number of raw equations: {len(equations_raw)}")
print("\nEquations:")
for i, eq in enumerate(equations_raw):
    print(f"Equation {i+1}: {eq} = 0")

print("\nCoefficients:")
sp.pprint(coefficients[:, ::-1])

# # Mapping to paper notation
# paper_notation = {
#     'A_{1,3}': 'A1', 'A_{1,2}': 'B1', 'A_{1,1}': 'C1', 'A_{1,0}': 'D1',
#     'A_{2,3}': 'A2', 'A_{2,2}': 'B2', 'A_{2,1}': 'C2', 'A_{2,0}': 'D2',
#     'A_{3,3}': 'A3', 'A_{3,2}': 'B3', 'A_{3,1}': 'C3', 'A_{3,0}': 'D3',
# }

# print("\nEquations in paper notation:")
# for i, eq in enumerate(equations_raw):
#     paper_eq = eq
#     for gen, paper in paper_notation.items():
#         paper_eq = paper_eq.subs(gen, paper)
#     print(f"Equation {i+1}: {paper_eq} = 0")

#%%
# test different boundary conditions
# until we find some that work
# Calculate the number of coefficients and existing equations
num_coefficients = M_eqs * (p_deg + 1)
num_existing_eqs = len(equations_raw)

# Calculate how many additional equations are needed
num_additional_eqs = num_coefficients - num_existing_eqs
lowest_fixed_coef = math.ceil((M_eqs*p_deg - num_additional_eqs)/M_eqs)
print(f"Number of additional equations needed: {num_additional_eqs}")

# Find coefficients set to specific values
set_coeffs = {coefficients[0,j] for j in range(d_cont+1)}
print(set_coeffs)

#%%
if num_additional_eqs == 0:
    fft_grads = {0: 100} # arbitrary large number
    zeros_vals = {0:'N/A'}
    coefs_all = {0: solve(equations_raw)}
else:
    # Generate additional equations for the lowest power coefficients

    # Generate all possible indices of combinations for the additional equations
    rows, cols = M_eqs, highest_fixed_coef
    all_indices = [(i, j) for i in range(1 if skip_first_equation else 0, rows) for j in range(lowest_fixed_coef,highest_fixed_coef+1)]
    combinations = list(itertools.combinations(all_indices, num_additional_eqs))
    print(f"Number of combinations: {len(combinations)}")
    print(combinations)
    equations = sp.zeros(M_eqs*(p_deg+1), 1)
    for i in range((p_deg+1)*(M_eqs)-num_additional_eqs):
        equations[i] = equations_raw[i]
    # sp.pprint(equations)

    # loop through the combinations
    fft_grads = dict()
    coefs_all = dict()
    zeros_vals = dict()
    den_large_global = 1000 # initialise large denominator
    best_derivative = 0.0
    combi_count = 0
    for combination in combinations: #[-1:0:-1]:
        print(f"\nCombination {combi_count+1}/{len(combinations)}: {combination}")
        combi_count += 1

        break_flag = False
        for index in combination:
            i, j = index
            if coefficients[i,j] in set_coeffs:
                print(f"Skipping {coefficients[i,j]}, it is already fixed.")
                break_flag = True
                break
        if break_flag:
            continue # skip the rest, go to the next combination
        
        # set the coefficients
        k = 0
        unknowns_range = num_additional_eqs
        a_k = sp.symbols([f'a_{i}' for i in range(unknowns_range)])
        for index in combination:
            i, j = index
            # if coefficients[i,j] not in set_coeffs:
            equations[k+M_eqs*(p_deg+1)-num_additional_eqs] = coefficients[i,j] - a_k[k]
            k += 1
            if k == num_additional_eqs:
                break

        # Solve the equations
        sol = sp.solve(equations, coefficients)
        
        ## now set up the kernel equations
        new_eqs = sp.zeros(M_eqs, 1)
        s = sp.symbols('s')
        for i in range(M_eqs):
            for j in range(p_deg+1):
                try:
                    new_eqs[i] += sol[coefficients[i, j]] * s**j
                except:
                    print(f"Empty solution for coefficient[{i},{j}], setting this to zero.")
                    pass

        # evaluate the kernel
        g_neg = sp.zeros(M_eqs,1)
        # negative direction
        for i in range(M_eqs): # go from s and backwards
            g_neg[i] = new_eqs[i].subs(s, s+i).simplify()

        # positive direction
        g_pos = sp.zeros(M_eqs,1)
        for i in range(M_eqs):
            g_pos[i] = sum([(-1)**j * new_eqs[i].coeff(s, j)*s**j for j in range(p_deg+1)]).subs(s, s-i-1).simplify()

        coef_neg = sp.Matrix([sp.Symbol(f'coef_{{{i}}}') for i in range(M_eqs)])
        coef_pos = sp.Matrix([sp.Symbol(f'coef_{{{i}}}') for i in range(M_eqs)])

        g_neg = [coef_neg[i] * g_neg[i] for i in range(M_eqs)]
        g_pos = [coef_pos[i] * g_pos[i] for i in range(M_eqs)]

        fp = sp.symbols(f'fp:{max_taylor+1}') # derivatives (index zero is fx)
        h = sp.symbols('h') # step

        taylor_step = sp.zeros(max_taylor, 1)
        print(f"taylor_step: {taylor_step}")
        for i in range(max_taylor):
            taylor_step[i] = fp[i]*h**i/factorial(i)
        
        # taylor_step[0] = fp[0]
        # taylor_step[1] = (h)*fp[1]
        # taylor_step[2] = (h)**2*fp[2]/2
        # taylor_step[3] = (h)**3*fp[3]/6
        # taylor_step[4] = (h)**4*fp[4]/24
        # taylor_step[5] = (h)**5*fp[5]/120

        for i in range(M_eqs):
            g_neg[i] = g_neg[i].subs(coef_neg[i], sum(taylor_step[0:max_taylor]).subs({h:-i*h}))
            g_pos[i] = g_pos[i].subs(coef_pos[i], sum(taylor_step[0:max_taylor]).subs({h:(i+1)*h}))

        f_taylor = sum(taylor_step[0:max_taylor]).subs({h:s*h})

        g = sum( g_neg[i] + g_pos[i] for i in range(M_eqs) )

        subtract = f_taylor - g
        subtract = sp.collect(sp.expand(subtract),s)

        if subtract.coeff(s, 0) != 0:
            print(f"Skipping combination {combination} due to non-zero constant term.")
            continue # skip the rest, go to the next combination

        # solve the equation
        eqs_sol = sp.zeros(p_deg, 1)
        unknowns_range = num_additional_eqs if num_additional_eqs < p_deg else p_deg
        for i in range(unknowns_range):
            eqs_sol[i] = subtract.coeff(s, i+1)
        solve_for = sp.symbols([f'a_{i}' for i in range(unknowns_range)])
        print(f"solving equations {eqs_sol}")
        print(f"solving for {solve_for}")

        # Attempt to solve with a timeout
        start_time = time.time()
        coefs = solve_with_timeout(eqs_sol, solve_for, solver_timeout)
        # coefs = sp.solve(eqs_sol, solve_for)
        end_time = time.time()

        if coefs is not None:
            print(f"Solved iteration {combi_count} in {end_time - start_time:.2f} seconds")
            # print(coefs)
            if coefs == [] or nan in coefs:
                print(f"Empty solution for {p_deg-j} for combination {combination}. Skipping equations.")
                continue
        else:
            print(f"Moving to next iteration due to timeout or error.")
            continue

        set_to_zero = {i:0 for i in a_k}
        break_flag = false
        for min_taylor in range(max_taylor,0,-1):
            taylor_count = 0
            deriv_zeros = {fp[i]:0 for i in range(min_taylor, max_taylor+1)}
            coefs_final = {key:sol[key].subs(coefs).subs(set_to_zero).subs(deriv_zeros) for key in sol}
            for index in coefs_final.keys():
                expr = coefs_final[index]
                symbols = expr.free_symbols
                if not symbols:
                    taylor_count += 1
                    if taylor_count == M_eqs*(p_deg+1):
                        print(f"Eliminated symbols by setting Taylor terms >= {min_taylor} to zero.")
                        break_flag = True # all good, no symbols
                        min_taylor_save = min_taylor
                        break
                    else:
                        pass
                else:
                    print(f'Free symbols for {index}: {symbols} for Taylor term >= {min_taylor} set to zero.')
                    break # some symbols still in expression, go to next
            if break_flag:
                break

        print(f"The non-simplified coeffs are:")
        print(coefs)
        print(f"The simplified coefficients are:")
        print(coefs_final)

        # do fft
        tstart = -5
        tend = 5
        steps = 1000 #int((tend-tstart)/tstep)
        ttot = tend - tstart
        tstep = ttot/steps
        t = np.linspace(tstart, tend, steps, endpoint=False)
        fft_result = np.fft.fft([kernel_func(s, coefs_final, coefficients) for s in t])
        freq = np.fft.fftfreq(len(t), tstep)
        # shift
        fft_result_shifted = np.fft.fftshift(fft_result)
        freq_shifted = np.fft.fftshift(freq)
        fft_result_normalized = np.real(fft_result_shifted)/(np.max(np.abs(fft_result_shifted)))
        fft_result_normalized_abs = np.abs(fft_result_shifted)/(np.max(np.abs(fft_result_shifted)))
        # calculate derivative at 0.5
        index_zero = find_closest(freq_shifted, 0.0)
        index_near_zero = find_closest(freq_shifted, -0.25)
        index_near_near_zero = find_closest(freq_shifted, -0.25/2)
        index_half = find_closest(freq_shifted, -0.5)
        zero_val = fft_result_normalized_abs[index_zero]
        near_zero_val = fft_result_normalized_abs[index_near_zero]
        near_near_zero_val = fft_result_normalized_abs[index_near_near_zero]
        derivative = (fft_result_normalized[index_half+1]-fft_result_normalized[index_half-1])/(freq_shifted[index_half+1]-freq_shifted[index_half-1])
        print(f"Current derivative: {derivative:.3f}, best derivative: {best_derivative:.3f}")

        if derivative > best_derivative and math.isclose(zero_val, 1.0, rel_tol=0.05) and \
                    math.isclose(near_near_zero_val, 1.0, rel_tol=0.075) and \
                    math.isclose(near_zero_val, 1.0, rel_tol=0.1): #\
                        #  and (abs(den_large_local) < den_large_global):
            print(f"New best combination at {combination}!")
            abs_derivative = np.abs(fft_result_normalized[index_half+1]-fft_result_normalized[index_half-1])/np.abs(freq_shifted[index_half+1]-freq_shifted[index_half-1])
            print(f"Derivative approximately: {abs_derivative:.3f}")
            best_derivative = derivative
            # den_large_global = abs(den_large_local)
            fft_grads[combi_count-1] = derivative
            zeros_vals[combi_count-1] = zero_val
            coefs_all[combi_count-1] = coefs_final
            # break
        
#%%
# print(den_large_global)
large_grad = 0.0
for (key, grad) in fft_grads.items():
    if grad > large_grad:
        large_grad = grad
        fft_grad_best = fft_grads[key]
        coefs_best = coefs_all[key]
        key_best = key

# print(f"Best derivative: {fft_grad_best} at {key_best}")
print(f'Final coefs: {coefs_best}')
# print(fft_grad_best)

# # %%
# if num_additional_eqs == 0:
#     coefs_final = coefs_all
# else:
#     coefs_final = coefs_all[key_best]
# s = sp.symbols('s')
# kernel = sp.zeros(M_eqs, 1)
# for i in range(M_eqs):
#     for j in range(p_deg+1):
#         try:
#             kernel[i] += coefs_final[coefficients[i, j]] * s**j
#         except:
#             kernel[i] += 0.0
# sp.pprint(kernel)
# %%

import numpy as np
r = np.linspace(-4, 4, 10_000)

quint = [kernel_func(s, coefs_best, coefficients) for s in r]

import matplotlib.pyplot as plt
plt.plot(r, quint)
plt.show()
# # %%
# from scipy.fft import fft, fftfreq
# import numpy as np
# # Number of sample points
# N = 600
# # sample spacing
# x_max = 1.0
# T = x_max / N
# x = np.linspace(-x_max, x_max, 2*N+1)
# y = [kernel_func(s, coefs_final, coefficients) for s in x]
# yf = fft(y)
# xf = fftfreq(N, T)[:N//2]
# import matplotlib.pyplot as plt
# plt.plot(xf, np.abs(yf[0:N//2])/np.max(np.abs(yf[0:N//2])))
# plt.xlim(0, 2.0)
# plt.grid()
# plt.show()
#%%

import matplotlib.pyplot as plt
fs = 10_000 # sampling frequency
tmax = 100.0
t = np.arange(-tmax, tmax, 1/fs)
f = [kernel_func(s, coefs_best, coefficients) for s in t]
k_shifted = np.fft.fftshift(np.fft.fftfreq(len(t), 1/fs)) #np.arange(0,N)
fft_shifted = np.fft.fftshift(np.fft.fft(f))
fft_result_normalized = np.abs(fft_shifted)/np.max(np.abs(fft_shifted))
index_half = find_closest(k_shifted, -0.5)
derivative = -np.abs(fft_result_normalized[index_half+1]-fft_result_normalized[index_half-1])/(k_shifted[index_half+1]-k_shifted[index_half-1])
plt.plot(k_shifted, fft_result_normalized)
plt.title(label=f"Derivative at x=0.5: {derivative:.3f}")
plt.xlim(0, 5)

def ideal(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] <= 0.5 and x[i] >= -0.5:
            y[i] = 1.0
        else:
            y[i] = 0.0
    return y
freq_ideal = np.linspace(-5.0, 5.0, 1000)
ideal_signal = ideal(freq_ideal)
plt.plot(freq_ideal, ideal_signal) # freq_shifted <= 0.5
plt.yscale('log')
plt.ylim(1e-7, 2)
plt.xlim(-5,5)
plt.show()

#%%

# ### boundary conditions
# M_eqs = 3
s = sp.symbols('s')
kernel = sp.zeros(M_eqs, 1)
for i in range(M_eqs):
    for j in range(p_deg+1):
        try:
            kernel[i] += coefs_final[coefficients[i, j]] * s**j
        except:
            print(f"Empty solution for coefficient[{i},{j}], setting this to zero.")
            kernel[i] += 0.0
            pass
sp.pprint(kernel)

#%%
# evaluate the kernel
g_neg = sp.zeros(M_eqs,1)
# negative direction
for i in range(M_eqs): # go from s and backwards
    g_neg[i] = kernel[i].subs(s, s+i).simplify()

# # positive direction
g_pos = sp.zeros(M_eqs,1)
for i in range(M_eqs):
    g_pos[i] = sum([(-1)**j * kernel[i].coeff(s, j)*s**j for j in range(p_deg+1)]).subs(s, s-i-1).simplify()
# #%%

coef_neg = sp.Matrix([sp.Symbol(f'coef_{{{-i}}}') for i in range(M_eqs)])
coef_pos = sp.Matrix([sp.Symbol(f'coef_{{{1+i}}}') for i in range(M_eqs)])
print(f"coef_neg: {coef_neg}")
print(f"coef_neg1: {coef_neg[0]}")

g_neg = [coef_neg[i] * g_neg[i] for i in range(M_eqs)]
g_pos = [coef_pos[i] * g_pos[i] for i in range(M_eqs)]

#%%
# max_taylor_eff = 5
# taylor_step = sp.zeros(max_taylor_eff, 1)
# print(f"taylor_step: {taylor_step}")
# for i in range(max_taylor_eff):
#     taylor_step[i] = fp[i]*h**i/factorial(i)
# print(f"taylor_step: {taylor_step}")

# for i in range(M_eqs):
#     g_neg[i] = g_neg[i].subs(coef_neg[i], sum(taylor_step[0:max_taylor_eff]).subs({h:-i*h}))
#     g_pos[i] = g_pos[i].subs(coef_pos[i], sum(taylor_step[0:max_taylor_eff]).subs({h:(i+1)*h}))

# f_taylor = sum(taylor_step[0:max_taylor_eff]).subs({h:s*h})

g = sum( g_neg[i] + g_pos[i] for i in range(M_eqs) )
g = g.expand().collect(s)
sp.pprint(g)

#%%
eqs = sp.zeros(p_deg,1)
for i in range(p_deg):
    eqs[i] = g.coeff(s, i+1)
sp.pprint(eqs)
#%%
remove_poly_terms_above = 3
sp.pprint(eqs[remove_poly_terms_above:])
#%%
sol = solve(eqs[remove_poly_terms_above:], coef_neg[1:])
sp.pprint(sol)
#%%
for i in range(1,len(sol)+1):
    print(f"coef_neg[{i}] =", sol[coef_neg[i]])
# print("coef_neg[2] =", sol[coef_neg[2]])
# print("coef_neg[3] =", sol[coef_neg[3]])
#%%
####################################################
#%%
def find_closest(numbers, target):
    closest_index = 0
    smallest_difference = abs(numbers[0] - target)
    
    for i, num in enumerate(numbers[1:], 1):
        difference = abs(num - target)
        if difference < smallest_difference:
            closest_index = i
            smallest_difference = difference
    
    return closest_index
derivative = np.diff(fft_result_normalized)/np.diff(freq_shifted)
print(derivative[find_closest(freq_shifted, 0.5)])
plt.plot(freq_shifted[2:], derivative[1:])
# plt.xlim([0.0, 2.0])
plt.ylim([min(derivative), 1.0])
plt.show()
print(find_closest(freq_shifted, 0.5))
# %%
index_half = find_closest(freq_shifted, 0.5)
print(f"index_half: {index_half}")
derivative = -abs(fft_result_normalized[index_half+1]-fft_result_normalized[index_half-1])/((freq_shifted[index_half+1]-freq_shifted[index_half-1]))
print(derivative)
# %%
freq_shifted[index_half+1]-freq_shifted[index_half-1]
# %%

# %%