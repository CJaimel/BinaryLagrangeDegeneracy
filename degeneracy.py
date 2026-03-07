import numpy as np
import numpy.polynomial.polynomial as poly

import scipy.constants as const
from scipy import optimize

import pycbc.psd, pycbc.filter, pycbc.noise
from pycbc.types import TimeSeries

import pyswarms as ps

#for the waveforms of the quasi-circular binary system and quasi-circular Lagrange triple
import waveforms

plt.rcParams['text.usetex'] = True

#define the solar mass in [kg] and parsec in [meters]
sol = 1.989e30
pc = 3.086e16

def findQuadDegeneracy_plus(a, m1_2B, m2_2B, r_2B, i_2B_deg, M_3B, beta_1, i_3B_deg, atol = 1e-23):
    #initialize binary
    binary = waveforms.CircularBinary(a, m1_2B, m2_2B, r_2B, i_2B_deg, 0)
    w_2B = np.sqrt(const.G*binary.M/(binary.l**3))
    i_2B = np.radians(i_2B_deg)

    #free parameters of the Lagrange triple
    M_3B *= sol
    i_3B = np.radians(i_3B_deg)

    #parameters to solve
    r_3B = ((M_3B/binary.M_c)**(5/3))*(binary.r*beta_1*np.abs(3*beta_1 - 1)*(1 + (np.cos(i_3B))**2))/(1 + (np.cos(i_2B))**2)
    b = (const.G*M_3B/(w_2B**2))**(1/3)

    #find the appropriate integration constant Phi_c
    phi_3B = 0
    shift = np.linspace(0, 2*np.pi, 1000)
    for phi in shift:
        temp = [b/pc, M_3B*beta_1/sol, M_3B*beta_1/sol, M_3B*(1 - 2*beta_1)/sol, r_3B/pc, i_3B_deg, phi]
        lagrange = waveforms.CircularLagrangeTriple(*temp)
        check = np.abs(binary.h_quad(t_max = 1, radRxn = True, pol = "plus")[:4] - lagrange.h_quad(t_max = 1, radRxn = True, pol = "plus")[:4])
            
        if all(np.isclose(check, 0, atol = atol)):
            phi_3B = phi

    #return b and r_3B in [pc]
    return b/pc, r_3B/pc, phi_3B

def findQuadDegeneracy_cross(a, m1_2B, m2_2B, r_2B, i_2B_deg, M_3B, beta_1, i_3B_deg, atol = 1e-23):
    #initialize binary
    binary = waveforms.CircularBinary(a, m1_2B, m2_2B, r_2B, i_2B_deg, 0)
    w_2B = np.sqrt(const.G*binary.M/(binary.l**3))
    i_2B = np.radians(i_2B_deg)

    #free parameters of the Lagrange triple
    M_3B *= sol
    i_3B = np.radians(i_3B_deg)

    #parameters to solve
    r_3B = ((M_3B/binary.M_c)**(5/3))*(binary.r*beta_1*np.abs(3*beta_1 - 1)*np.cos(i_3B))/np.cos(i_2B)
    b = (const.G*M_3B/(w_2B**2))**(1/3)

    #find the appropriate integration constant Phi_c
    phi_3B = 0
    shift = np.linspace(0, 2*np.pi, 1000)
    for phi in shift:
        temp = [b/pc, M_3B*beta_1/sol, M_3B*beta_1/sol, M_3B*(1 - 2*beta_1)/sol, r_3B/pc, i_3B_deg, phi]
        lagrange = waveforms.CircularLagrangeTriple(*temp)
        check = np.abs(binary.h_quad(t_max = 1, radRxn = True, pol = "cross")[:4] - lagrange.h_quad(t_max = 1, radRxn = True, pol = "cross")[:4])
            
        if all(np.isclose(check, 0, atol = atol)):
            phi_3B = phi

    #return b and r_3B in [pc]
    return b/pc, r_3B/pc, phi_3B

def F(b1, b2):
    num = 3*((b1 + b2 - 1)**2)*((2*b1**2 + (b1 + b2)*(2*b2 - 1))**2) + (b1 - 3*b1**2 + 2*b1**3 + b2*(b2*(3 - 2*b2) - 1))**2
    denom = 27*(b1**2)*(b2**2)*((b1 + b2 - 1)**2) + ((b1 - b2)**2)*((2*b1 + b2 - 1)**2)*((b1 + 2*b2 - 1)**2)

    return np.sqrt(num/denom)

def iota(b1, b2, ratio_w_3w):
    return np.degrees(np.arcsin(np.sqrt( 2 + 1/((1/4) - (9/2)*ratio_w_3w/F(b1, b2)))))

def iota_arccos(b1, b2, ratio_w_3w):
    return np.degrees(np.arccos(np.sqrt( 4/((27/2)*ratio_w_3w/F(b1, b2) - 1) - 9) ))

def find_M_3B(a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B, b1):
    binary = waveforms.CircularBinary(a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B)
    
    #quantities obtainable from the known binary system
    amp_quadPlus_2B = binary.h_quad(t_max = 100, ampOnly = True, pol = "plus")
    amp_octcqPlus_2B_w = binary.h_octcq(t_max = 100, ampOnly = True, pol = "plus")[0]
    amp_octcqPlus_2B_3w = binary.h_octcq(t_max = 100, ampOnly = True, pol = "plus")[1]
    w_2B = np.sqrt(const.G*(m1_2B*sol + m2_2B*sol)/((a*pc)**3))

    #parameters obtainable from the equivalent Lagrange triple
    b3 = 1 - 2*b1
    i_3B = np.radians(iota(b1, b1, amp_octcqPlus_2B_w/amp_octcqPlus_2B_3w))

    term1 = (np.sqrt(3)*b3/8)*(amp_quadPlus_2B/amp_octcqPlus_2B_w)*(np.sin(i_3B)*(5 + np.cos(i_3B)**2)/(1 + np.cos(i_3B)**2))*((w_2B/const.c)**(1/3))
    term2 = (9*np.sqrt(27)/8)*(b1*b3/np.abs(b1 - b3))*(amp_quadPlus_2B/amp_octcqPlus_2B_3w)*np.sin(i_3B)*((w_2B/const.c)**(1/3))

    return ((const.c**2)/const.G) * ((1/2)*(term1**(-3)) + (1/2)*(term2**(-3)))

def find_r(a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B, b1):
    t = np.linspace(0, 1000, 10000)
    binary = waveforms.CircularBinary(a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B)
    
    #quantities obtainable from the known binary system
    amp_quadPlus_2B = binary.h_quad(t_max = 100, ampOnly = True, pol = "plus")
    amp_octcqPlus_2B_w = binary.h_octcq(t_max = 100, ampOnly = True, pol = "plus")[0]
    amp_octcqPlus_2B_3w = binary.h_octcq(t_max = 100, ampOnly = True, pol = "plus")[1]
    w_2B = np.sqrt(const.G*(m1_2B*sol + m2_2B*sol)/((a*pc)**3))

    #parameters obtainable from the equivalent Lagrange triple
    i_3B = np.radians(iota(b1, b1, amp_octcqPlus_2B_w/amp_octcqPlus_2B_3w))
    b3 = 1 - 2*b1
    M_3B = find_M_3B(a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B, b1)

    term1 = 2*((const.G*M_3B/(const.c**2))**(5/3))*((w_2B/const.c)**(2/3))*b1*np.abs(b1 - b3)*(1 + np.cos(i_3B)**2)
    term2 = (np.sqrt(3)/4)*((const.G*M_3B/(const.c**2))**2)*(w_2B/const.c)*b1*b3*np.abs(b1 - b3)*np.sin(i_3B)*(5 + np.cos(i_3B)**2)
    term3 = (9*np.sqrt(27)/4)*((const.G*M_3B/(const.c**2))**2)*(w_2B/const.c)*(b1**2)*b3*np.sin(i_3B)*(1 + np.cos(i_3B)**2)
    
    return (term1 + term2 + term3)/(amp_quadPlus_2B + amp_octcqPlus_2B_w + amp_octcqPlus_2B_3w)

def findOctDegeneracy_plus(params_binary, beta_1, filename = "", radRxn = False, suppressPrint = False, atol = 1e-23):
    #initialization
    binary = waveforms.CircularBinary(*params_binary)
    
    #generate the data
    ratio_w_3w = binary.h_octcq(t_max = 10, radRxn = False, ampOnly = True, pol = "plus")[0]/binary.h_octcq(t_max = 10, radRxn = False, ampOnly = True, pol = "plus")[1]
    
    b1 = np.linspace(0, 1, 500000)
    b2 = np.linspace(0, 1, 500000)
    i_3B = iota(b1, b2, ratio_w_3w)
    i_3B[b1 > 0.5] = np.nan
    
    #plot M_3B and r_3B for the given binary system
    #find the values of b1 where i_3B is defined (i.e., the values of b1 that are part of the domain of i_3B)
    b1_filtered = b1[np.where(~np.isnan(i_3B))]

    if suppressPrint == False:
        print("Min beta_1 for true degeneracy = {}".format(b1_filtered[0]))
        print("Max beta_1 for true degeneracy = {}".format(b1_filtered[-1]))

    if b1_filtered[0] < beta_1 < b1_filtered[-1]:
        #calculate the parameters obtainable from the given binary system      
        w_2B = np.sqrt(const.G*(params_binary[1]*sol + params_binary[2]*sol)/((params_binary[0]*pc)**3))

        #find the parameters of the equivalent Lagrange triple
        M_3B = find_M_3B(*params_binary, beta_1)
        r_3B = find_r(*params_binary, beta_1)
        b = (const.G*M_3B/(w_2B**2))**(1/3)
        m1_3B = M_3B*beta_1
        m3_3B = M_3B - 2*m1_3B
        i_3B_deg = iota(beta_1, beta_1, ratio_w_3w)

        delta_t = 1/5000
        t_max = 3
        phi_3B = 0
        shift = np.linspace(0, 2*np.pi, 1000)

        for phi in shift:
            temp = [b/pc, m1_3B/sol, m1_3B/sol, m3_3B/sol, r_3B/pc, i_3B_deg, phi]
            lagrange = waveforms.CircularLagrangeTriple(*temp)
            check = np.abs(binary.h_octcq(t_max, delta_t, radRxn = True, pol = "plus")[:4] - lagrange.h_octcq(t_max, delta_t, radRxn = True, pol = "plus")[:4])
            
            if all(np.isclose(check, 0, atol = atol)):
                phi_3B = phi

        #compile the parameters of the triple into one list
        return [b/pc, m1_3B/sol, m1_3B/sol, m3_3B/sol, r_3B/pc, i_3B_deg, phi_3B]
    else:
        print("Mass ratio not in the region for true degeneracy.")


def octDegeneracy_cross_equations(init, params_binary, beta_1):
    #initial conditions must be in SI units (distances in [m], masses in [kg], angles in [rad])
    r_3B, M_3B, i_3B = init
    b1 = beta_1
    binary = waveforms.CircularBinary(*params_binary)
    w_2B = np.sqrt(const.G*binary.M/(binary.l**3))

    eq1 = binary.h_quad(1, ampOnly = True, pol = "cross") - 4*((const.G*M_3B/(const.c**2))**(5/3))*((w_2B/const.c)**(2/3))*b1*np.abs(3*b1 - 1)*np.cos(i_3B)/r_3B
    eq2 = binary.h_octcq(1, ampOnly = True, pol = "cross")[0] - (np.sqrt(27)/4)*((const.G*M_3B/(const.c**2))**2)*(w_2B/const.c)*b1*(1 - 2*b1)*np.abs(3*b1 - 1)*np.sin(2*i_3B)/r_3B
    eq3 = binary.h_octcq(1, ampOnly = True, pol = "cross")[1] - (9*np.sqrt(27)/4)*((const.G*M_3B/(const.c**2))**2)*(w_2B/const.c)*(b1**2)*(1 - 2*b1)*np.sin(2*i_3B)/r_3B
    
    return [eq1, eq2, eq3]

def findOctDegeneracy_cross(init, a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B, beta_1, method = "lm", root_tol = 1e-25, atol = 1e-23):
    binary = waveforms.CircularBinary(*[a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B])
    w_2B = np.sqrt(const.G*binary.M/(binary.l**3))
    
    #input initial conditions for root finding must be in pc and solar masses
    #r_3B in [m], M_3B in [kg], and i_3B in [rad]
    solved_params_3B = optimize.root(octDegeneracy_cross_equations, [init[0]*pc, init[1]*sol, np.radians(init[2])],
                            method = method, tol = root_tol, args = ([a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B], beta_1))
    print(solved_params_3B.message)
    r_3B, M_3B, i_3B = solved_params_3B.x
    b = (const.G*M_3B/(w_2B**2))**(1/3)
    m1_3B = M_3B*beta_1
    m3_3B = M_3B - 2*m1_3B
    i_3B_deg = np.degrees(i_3B)

    delta_t = 1/5000
    t_max = 3
    phi_3B = 0
    shift = np.linspace(0, 2*np.pi, 1000)
    
    for phi in shift:
        temp = [b/pc, m1_3B/sol, m1_3B/sol, m3_3B/sol, r_3B/pc, i_3B_deg, phi]
        lagrange = waveforms.CircularLagrangeTriple(*temp)
        check = np.abs(binary.h_octcq(t_max, delta_t, radRxn = True, pol = "cross")[:4] - lagrange.h_octcq(t_max, delta_t, radRxn = True, pol = "cross")[:4])
                
        if all(np.isclose(check, 0, atol = atol)):
            phi_3B = phi
    
    return [b/pc, m1_3B/sol, m1_3B/sol, m3_3B/sol, r_3B/pc, i_3B_deg, phi_3B]

def octDegeneracy_cross_PSO(x, params_binary, beta_1):
    #r_3B, M_3B, i_3B = x
    a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B = params_binary
    b1 = beta_1
    binary = waveforms.CircularBinary(*params_binary)
    w_2B = np.sqrt(const.G*binary.M/(binary.l**3))
    
    eq1 = binary.h_quad(1, ampOnly = True, pol = "cross") - 4*((const.G*x[:, 1]/(const.c**2))**(5/3))*((w_2B/const.c)**(2/3))*b1*np.abs(3*b1 - 1)*np.cos(x[:, 2])/x[:, 0]
    eq2 = binary.h_octcq(1, ampOnly = True, pol = "cross")[0] - (np.sqrt(27)/4)*((const.G*x[:, 1]/(const.c**2))**2)*(w_2B/const.c)*b1*(1 - 2*b1)*np.abs(3*b1 - 1)*np.sin(2*x[:, 2])/x[:, 0]
    eq3 = binary.h_octcq(1, ampOnly = True, pol = "cross")[1] - (9*np.sqrt(27)/4)*((const.G*x[:, 1]/(const.c**2))**2)*(w_2B/const.c)*(b1**2)*(1 - 2*b1)*np.sin(2*x[:, 2])/x[:, 0]
    
    return (eq1**2 + eq2**2 + eq3**2)

def findOctDegeneracy_cross_PSO(a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B, beta_1, pso_options = {'c1': 2, 'c2': 2, 'w': 1}, n_particles = 20, iters = 1000, atol = 1e-23):
    b1 = beta_1
    binary = waveforms.CircularBinary(*[a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B])
    w_2B = np.sqrt(const.G*binary.M/(binary.l**3))
    
    #input initial conditions for root finding must be in pc and solar masses
    #r_3B in [m], M_3B in [kg], and i_3B in [rad]
    options = pso_options
    min_bound = np.array([1e6*pc, (1e-2)*sol, np.radians(0)])
    max_bound = np.array([1e7*pc, 100*sol, np.pi/2])
    bounds = (min_bound, max_bound)
    
    optimizer = ps.single.GlobalBestPSO(n_particles = n_particles, dimensions = 3, options = options, bounds = bounds)
    cost, pos = optimizer.optimize(octDegeneracy_cross_PSO, iters = iters, params_binary = [a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B], beta_1 = b1)
    print("Minimum error found: {}".format(cost))
    
    r_3B, M_3B, i_3B = pos
    b = (const.G*M_3B/(w_2B**2))**(1/3)
    m1_3B = M_3B*beta_1
    m3_3B = M_3B - 2*m1_3B
    i_3B_deg = np.degrees(i_3B)

    delta_t = 1/5000
    t_max = 3
    phi_3B = 0
    shift = np.linspace(0, 2*np.pi, 1000)
    
    for phi in shift:
        temp = [b/pc, m1_3B/sol, m1_3B/sol, m3_3B/sol, r_3B/pc, i_3B_deg, phi]
        lagrange = waveforms.CircularLagrangeTriple(*temp)
        check = np.abs(binary.h_octcq(t_max, delta_t, radRxn = True, pol = "cross")[:4] - lagrange.h_octcq(t_max, delta_t, radRxn = True, pol = "cross")[:4])
                
        if all(np.isclose(check, 0, atol = atol)):
            phi_3B = phi
    
    return [b/pc, m1_3B/sol, m1_3B/sol, m3_3B/sol, r_3B/pc, i_3B_deg, phi_3B]