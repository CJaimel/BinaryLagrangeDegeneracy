import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from decimal import Decimal

#for converting the waveforms to a PyCBC Time Series data type
from pycbc.types import TimeSeries

#define the solar mass in [kg] and parsec in [meters]
sol = 1.989e30
pc = 3.086e16

class CircularBinary:
    r"""CircularBinary contains the waveforms for a quasi-circular binary system up to the 0.5-PN approximation.
    Upon initialization, the user must supply the following parameters of the binary:

    l = initial separation distance in [pc],
    m1, m2 = masses of the individual masses in [solar mass],
    r = distance to the binary system in [pc],
    i_deg = orbital inclination angle in [deg],
    phi = integration constant corresponding to the value of the phase at coalescence [rad]
    """
    
    def __init__(self, l, m1, m2, r, i_deg, phi):
        #convert distances to [m] and masses to [kg]
        l *= pc
        r *= pc
        m1 *= sol
        m2 *= sol

        #parameters of the binary
        self.l = l
        self.m1 = m1
        self.m2 = m2
        self.r = r
        self.i_deg = i_deg
        self.phi = phi

        #total mass and chirp mass in [kg]
        self.M = m1 + m2
        self.M_c = self.M**(-1/5) * (m1*m2)**(3/5)

        #coalescence time in [s]
        self.t_c = ((const.c**5)/(const.G**3))*(5/256)*(l**4)/((self.M_c**(5/3))*(self.M**(4/3)))

    #time evolution of the separation distance
    def a(self, t):
        return l*(1 - t/self.t_c)**(1/4)

    #time evolution of the orbital frequency
    def omega(self, t):
        return (5**(3/8))/8 * ((const.G*self.M_c/(const.c**3))**(-5/8)) * (self.t_c - t)**(-3/8)

    #time evolution of the phase
    def phase(self, t):
        return -(5**(-5/8)) * ((const.G*self.M_c/(const.c**3))**(-5/8)) * (self.t_c - t)**(5/8)

    #quadrupole waveform
    def h_quad(self, t_max, delta_t = 1/1000, radRxn = False, ampOnly = False, pol = "both", **args):
        dt = delta_t
        t = np.arange(0, t_max, dt)
        t_ret = t - self.r/const.c
        
        i = np.radians(self.i_deg)
    
        if radRxn == True:
            w = self.omega(t)
            PHI = self.phase(t)
        else:
            w = np.sqrt(const.G*self.M/(self.l**3))
            PHI = w*t_ret

        amp_quadPlus2B = (4/self.r)*((const.G*self.M_c/(const.c**2))**(5/3))*((w/const.c)**(2/3))*((1 + (np.cos(i))**2)/2)
        amp_quadCross2B = (4/self.r)*((const.G*self.M_c/(const.c**2))**(5/3))*((w/const.c)**(2/3))*np.cos(i)

        h_quadPlus2B = amp_quadPlus2B*np.cos(2*PHI + 2*self.phi)
        h_quadCross2B = amp_quadCross2B*np.sin(2*PHI + 2*self.phi)

        #following the convention of PyCBC waveforms, we offset our waveform so that
        #t = 0 corresponds to the coalescence time
        offset = -len(t)*delta_t
        h_quadPlus2B = TimeSeries(h_quadPlus2B, delta_t = delta_t, epoch = offset)
        h_quadPlus2B = TimeSeries(h_quadPlus2B, delta_t = delta_t, epoch = offset)
        
        if ampOnly == True:
            if pol == "plus":
                return amp_quadPlus2B
            elif pol == "cross":
                return amp_quadCross2B
            else:
                return np.array([amp_quadPlus2B, amp_quadCross2B])
        else:
            if pol == "plus":
                return h_quadPlus2B
            elif pol == "cross":
                return h_quadCross2B
            else:
                return h_quadPlus2B, h_quadCross2B

    #octupole waveform
    def h_octcq(self, t_max, delta_t = 1/1000, radRxn = False, ampOnly = False, pol = "both", **args):
        dt = delta_t
        t = np.arange(0, t_max, dt)
        t_ret = t - self.r/const.c
        
        i = np.radians(self.i_deg)
        mu = self.m1*self.m2/self.M
    
        if radRxn == True:
            w = self.omega(t)
            PHI = self.phase(t)
        else:
            w = np.sqrt(const.G*self.M/(self.l**3))
            PHI = w*t_ret
        
        scalePlus2B = ((const.G**2)/(const.c**5))*mu*(self.m2 - self.m1)*w*np.sin(i)/(4*self.r)
        scaleCross2B = 3*((const.G**2)/(const.c**5))*mu*(self.m2 - self.m1)*w*np.sin(2*i)/(4*self.r)
        
        h_octcqPlus_omega2B = 5 + np.cos(i)**2
        h_octcqPlus_3omega2B = 9*(1 + np.cos(i)**2)
        h_octcqCross_omega2B = 1
        h_octcqCross_3omega2B = 3

        h_octcqPlus2B = scalePlus2B*(h_octcqPlus_omega2B*np.cos(PHI + self.phi) - h_octcqPlus_3omega2B*np.cos(3*PHI + 3*self.phi))
        h_octcqCross2B = scaleCross2B*(h_octcqCross_omega2B*np.sin(PHI + self.phi) - h_octcqCross_3omega2B*np.sin(3*PHI + 3*self.phi))

        #following the convention of PyCBC waveforms, we offset our waveform so that
        #t = 0 corresponds to the coalescence time
        offset = -len(t)*delta_t
        h_octcqPlus2B = TimeSeries(h_octcqPlus2B, delta_t = delta_t, epoch = offset)
        h_octcqCross2B = TimeSeries(h_octcqCross2B, delta_t = delta_t, epoch = offset)
        
        if ampOnly == True:
            if pol == "plus":
                return np.abs(np.array([scalePlus2B*h_octcqPlus_omega2B, scalePlus2B*h_octcqPlus_3omega2B]))
            elif pol == "cross":
                return np.abs(np.array([scaleCross2B*h_octcqCross_omega2B, scaleCross2B*h_octcqCross_3omega2B]))
            else:
                return np.abs(np.array([h_octcqPlus_omega2B, h_octcqPlus_3omega2B, h_octcqCross_omega2B, h_octcqCross_3omega2B]))
        else:
            if pol == "plus":
                return h_octcqPlus2B
            elif pol == "cross":
                return h_octcqCross2B
            else:
                return h_octcqPlus2B, h_octcqPlus2B

class CircularLagrangeTriple:
    r"""
    CircularLagrangeTriple contains the waveforms for a quasi-circular Lagrange three-body system up to the 0.5-PN approximation.
    Upon initialization, the user must supply the following parameters of the binary:

    l = initial separation distance in [pc],
    m1, m2, m3 = masses of the individual masses in [solar mass],
    r = distance to the binary system in [pc],
    i_deg = orbital inclination angle in [deg],
    phi = integration constant corresponding to the value of the phase at coalescence [rad]
    """
    
    def __init__(self, l, m1, m2, m3, r, i_deg, phi):
        #convert distances to [m] and masses to [kg]
        l *= pc
        r *= pc
        m1 *= sol
        m2 *= sol
        m3 *= sol

        #parameters of the binary
        self.l = l
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.r = r
        self.i_deg = i_deg
        self.phi = phi

        #total mass and chirp mass in [kg]
        self.M = m1 + m2 + m3
        self.M_c = (self.M**(-1/5)) * (0.5 * ((self.m1**2)*(self.m2 - self.m3)**2 + (self.m2**2)*(self.m3 - self.m1)**2 + (self.m3**2)*(self.m1 - self.m2)**2)/(self.m1*self.m2 + self.m1*self.m3 + self.m2*self.m3))**(3/5)

        #coalescence time in [s]
        self.t_c = ((const.c**5)/(const.G**3))*(5/256)*(l**4)/((self.M_c**(5/3))*(self.M**(4/3)))

        #phase shifts in the waveforms
        self.Psi_quad = np.arctan(np.sqrt(3)*self.m3*(self.m2 - self.m1)/(2*self.m1*self.m2 - self.m3*(self.m1 + self.m2)))
        self.Psi_omega = np.arctan((self.m1 - self.m2)*((self.m3 - self.m1)*(self.m3 - self.m2) - 3*self.m1*self.m2)/(np.sqrt(3)*self.m3*(self.m1*(self.m1 - self.m3) + self.m2*(self.m2 - self.m3))))
        self.Psi_3omega = np.arctan((self.m3 - self.m1)*(self.m1 - self.m2)*(self.m2 - self.m3)/(3*np.sqrt(3)*self.m1*self.m2*self.m3))
    
    #time evolution of the separation distance
    def a(self, t):
        return l*(1 - t/self.t_c)**(1/4)

    #time evolution of the orbital frequency
    def omega(self, t):
        return (5**(3/8))/8 * ((const.G*self.M_c/(const.c**3))**(-5/8)) * (self.t_c - t)**(-3/8)

    #time evolution of the phase
    def phase(self, t):
        return -(5**(-5/8)) * ((const.G*self.M_c/(const.c**3))**(-5/8)) * (self.t_c - t)**(5/8)

    def h_quad(self, t_max, delta_t = 1/1000, radRxn = False, ampOnly = False, pol = "both", **args):
        dt = delta_t
        t = np.arange(0, t_max, dt)
        t_ret = t - self.r/const.c
        
        i = np.radians(self.i_deg)

        if radRxn == True:
            w = self.omega(t)
            PHI = self.phase(t)
        else:
            w = np.sqrt(const.G*self.M/(self.l**3))
            PHI = w*t_ret
        
        amp_quadPlus3B = (4/self.r)*((const.G*self.M/(const.c**2))**(5/6))*((const.G*self.M_c/(const.c**2))**(5/6))*((w/const.c)**(2/3))*((1 + (np.cos(i))**2)/2)*(np.sqrt(self.m1*self.m2 + self.m1*self.m3 + self.m2*self.m3)/self.M)
        amp_quadCross3B = (4/self.r)*((const.G*self.M/(const.c**2))**(5/6))*((const.G*self.M_c/(const.c**2))**(5/6))*((w/const.c)**(2/3))*np.cos(i)*(np.sqrt(self.m1*self.m2 + self.m1*self.m3 + self.m2*self.m3)/self.M)
    
        h_quadPlus3B = amp_quadPlus3B*np.cos(2*PHI - self.Psi_quad + 2*self.phi)
        h_quadCross3B = amp_quadCross3B*np.sin(2*PHI - self.Psi_quad + 2*self.phi)

        #following the convention of PyCBC waveforms, we offset our waveform so that
        #t = 0 corresponds to the coalescence time
        offset = -len(t)*delta_t
        h_quadPlus3B = TimeSeries(h_quadPlus3B, delta_t = delta_t, epoch = offset)
        h_quadPlus3B = TimeSeries(h_quadPlus3B, delta_t = delta_t, epoch = offset)
        
        if ampOnly == True:
            if pol == "plus":
                return amp_quadPlus3B
            elif pol == "cross":
                return amp_quadCross3B
            else:
                return np.array([amp_quadPlus3B, amp_quadCross3B])
        else:
            if pol == "plus":
                return h_quadPlus3B
            elif pol == "cross":
                return h_quadCross3B
            else:
                return h_quadPlus3B, h_quadCross3B
    
    def h_octcq(self, t_max, delta_t = 1/1000, radRxn = False, ampOnly = False, pol = "both", **args):
        dt = delta_t
        t = np.arange(0, t_max, dt)
        t_ret = t - self.r/const.c
        
        i = np.radians(self.i_deg)
        
        if radRxn == True:
            w = self.omega(t)
            PHI = self.phase(t)
        else:
            w = np.sqrt(const.G*self.M/(self.l**3))
            PHI = w*t_ret
        
        scalePlus3B = ((const.G*self.M/(const.c**2))**2)*(w/const.c)*np.sin(i)/(4*self.r)
        amp_octcqPlus_omega3B = (5 + np.cos(i)**2)*np.sqrt(((np.sqrt(3)/2)*self.m3*(self.m1*(self.m1 - self.m3) + self.m2*(self.m2 - self.m3))/(self.M**3))**2 - (0.5*(self.m1 - self.m2)*((self.m3 - self.m1)*(self.m3 - self.m2) - 3*self.m1*self.m2)/(self.M**3))**2)
        amp_octcqPlus_3omega3B = 9*(1 + np.cos(i)**2)*np.sqrt((3*np.sqrt(3)*self.m1*self.m2*self.m3/(self.M**3))**2 + ((self.m3 - self.m1)*(self.m1 - self.m2)*(self.m2 - self.m3)/(self.M**3))**2)
    
        scaleCross3B = ((const.G*self.M/(const.c**2))**2)*(w/const.c)*np.sin(2*i)/(4*self.r)
        amp_octcqCross_omega3B = 3*np.sqrt( ((np.sqrt(3)/2)*self.m3*(self.m1*(self.m3 - self.m1) + self.m2*(self.m3 - self.m2))/(self.M**3))**2 + (0.5*(self.m1 - self.m2)*((self.m3 - self.m1)*(self.m3 - self.m2) - 3*self.m1*self.m2)/(self.M**3))**2 )
        amp_octcqCross_3omega3B = 9*np.sqrt( (3*np.sqrt(3)*self.m1*self.m2*self.m3/(self.M**3))**2 + ((self.m3 - self.m1)*(self.m1 - self.m2)*(self.m2 - self.m3)/(self.M**3))**2 )
        
        h_octcqPlus3B = -scalePlus3B*(amp_octcqPlus_3omega3B*np.cos(3*PHI - self.Psi_3omega + 3*self.phi) - amp_octcqPlus_omega3B*np.cos(PHI - self.Psi_omega + self.phi))
        h_octcqCross3B = -scaleCross3B*(amp_octcqCross_3omega3B*np.sin(3*PHI - self.Psi_3omega + 3*self.phi) - amp_octcqCross_omega3B*np.sin(PHI - self.Psi_omega + self.phi))

        #following the convention of PyCBC waveforms, we offset our waveform so that
        #t = 0 corresponds to the coalescence time
        offset = -len(t)*delta_t
        h_octcqPlus3B = TimeSeries(h_octcqPlus3B, delta_t = delta_t, epoch = offset)
        h_octcqCross3B = TimeSeries(h_octcqCross3B, delta_t = delta_t, epoch = offset)
        
        if ampOnly == True:
            if pol == "plus":
                return np.abs(scalePlus3B*amp_quadPlus3B)
            elif pol == "cross":
                return np.abs(scaleCross3B*amp_quadCross3B)
            else:
                return np.array([np.abs(scalePlus3B*amp_quadPlus3B), np.abs(scaleCross3B*amp_quadCross3B)])
        else:
            if pol == "plus":
                return h_octcqPlus3B
            elif pol == "cross":
                return h_octcqCross3B
            else:
                return h_octcqPlus3B, h_octcqCross3B

#combined waveform of the binary
def h_combined2B(a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B, t_max, delta_t = 1/1000, radRxn = True, **args):
    binary = CircularBinary(a, m1_2B, m2_2B, r_2B, i_2B_deg, phi_2B)
    
    h_combinedPlus2B = binary.h_quad(t_max, delta_t = delta_t, radRxn = radRxn)[0] + binary.h_octcq(t_max, delta_t = delta_t, radRxn = radRxn)[0]
    h_combinedCross2B = binary.h_quad(t_max, delta_t = delta_t, radRxn = radRxn)[1] + binary.h_octcq(t_max, delta_t = delta_t, radRxn = radRxn)[1]
    
    return h_combinedPlus2B, h_combinedCross2B

#combined waveform of the Lagrange triple
def h_combined3B(b, m1_3B, m2_3B, m3_3B, r_3B, i_3B_deg, phi_3B, t_max, delta_t = 1/1000, radRxn = True, **args):
    lagrange = CircularLagrangeTriple(b, m1_3B, m2_3B, m3_3B, r_3B, i_3B_deg, phi_3B)
    
    h_combinedPlus3B = lagrange.h_quad(t_max, delta_t = delta_t, radRxn = radRxn)[0] + lagrange.h_octcq(t_max, delta_t = delta_t, radRxn = radRxn)[0]
    h_combinedCross3B = lagrange.h_quad(t_max, delta_t = delta_t, radRxn = radRxn)[1] + lagrange.h_octcq(t_max, delta_t = delta_t, radRxn = radRxn)[1]
    
    return h_combinedPlus3B, h_combinedCross3B

#plot the waveforms in the time domain
def plot_waveform_time(params_2B, params_3B, t_max, delta_t, radRxn = True, pol = "plus", size = "vertical", filename = ""):
    #initialization
    binary = CircularBinary(*params_2B)
    lagrange = CircularLagrangeTriple(*params_3B)

    label_2B = "Binary"
    label_3B = "Lagrange"
    
    if size == "vertical":
        figsize = (7, 5)
    elif size == "horizontal":
        figsize = (10, 6)

    #create figure object and subfigure object
    fig, (a0, a1, a2) = plt.subplots(nrows = 3, ncols = 1, sharex = True, gridspec_kw={'height_ratios': [1, 1, 2]}, figsize = figsize)
    
    if pol:
        a0.plot(lagrange.h_quad(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "plus").sample_times,
                    lagrange.h_quad(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "plus"),
                    ls = "solid", color = "blue")
        a0.plot(binary.h_quad(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "plus").sample_times,
                    binary.h_quad(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "plus"),
                    ls = "dashed", color = "orange")
        a0.set_ylabel("$h^{+}_{\mathrm{quad}}$", fontsize = "x-large")

        a1.plot(lagrange.h_octcq(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "plus").sample_times,
                    lagrange.h_octcq(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "plus"),
                    ls = "solid", color = "blue")
        a1.plot(binary.h_octcq(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "plus").sample_times,
                    binary.h_octcq(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "plus"),
                    ls = "dashed", color = "orange")
        a1.set_ylabel("$h^{+}_{\mathrm{oct+cq}}$", fontsize = "x-large")
    elif pol == "cross":
        a0.plot(lagrange.h_quad(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "cross").sample_times,
                    lagrange.h_quad(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "cross"),
                    ls = "solid", color = "blue")
        a0.plot(binary.h_quad(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "cross").sample_times,
                    binary.h_quad(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "cross"),
                    ls = "dashed", color = "orange")
        a0.set_ylabel("$h^{+}_{\mathrm{quad}}$", fontsize = "x-large")

        a1.plot(lagrange.h_octcq(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "cross").sample_times,
                    lagrange.h_octcq(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "cross"),
                    ls = "solid", color = "blue")
        a1.plot(binary.h_octcq(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "cross").sample_times,
                    binary.h_octcq(t_max = t_max, delta_t = delta_t, radRxn = radRxn, pol = "cross"),
                    ls = "dashed", color = "orange")
        a1.set_ylabel("$h^{+}_{\mathrm{oct+cq}}$", fontsize = "x-large")
        
    if pol:
        a2.plot(h_combined3B(*params_3B, t_max = t_max, delta_t = delta_t, radRxn = radRxn)[0].sample_times,
                 h_combined3B(*params_3B, t_max = t_max, delta_t = delta_t, radRxn = radRxn)[0],
                 ls = "solid", color = "blue", label = label_3B)
        a2.plot(h_combined2B(*params_2B, t_max = t_max, delta_t = delta_t, radRxn = radRxn)[0].sample_times,
                 h_combined2B(*params_2B, t_max = t_max, delta_t = delta_t, radRxn = radRxn)[0],
                 ls = "dashed", color = "orange", label = label_2B)
    elif pol == "cross":
        a2.plot(h_combined3B(*params_3B, t_max = t_max, delta_t = delta_t, radRxn = radRxn)[1].sample_times,
                 h_combined3B(*params_3B, t_max = t_max, delta_t = delta_t, radRxn = radRxn)[1],
                 ls = "solid", color = "blue", label = label_3B)
        a2.plot(h_combined2B(*params_2B, t_max = t_max, delta_t = delta_t, radRxn = radRxn)[1].sample_times,
                 h_combined2B(*params_2B, t_max = t_max, delta_t = delta_t, radRxn = radRxn)[1],
                 ls = "dashed", color = "orange", label = label_2B)
    
    a2.set_xlabel("Time, $t$ [s]", fontsize = "x-large")
    a2.set_ylabel(r"$h^{+}$", fontsize = "x-large")

    if size == "vertical":
        a2.legend(bbox_to_anchor = (0.2, -0.16))
    else:
        a2.legend(bbox_to_anchor = (0.14, -0.16))

    if not filename:
        pass
    else:
        plt.savefig(filename, bbox_inches = "tight")

    print("Binary Parameters:\nl = {}, m1 = {}, m2 = {}, r = {}, i = {}, phi = {}".format(*params_2B))
    print("")
    print("Lagrange Three-Body Parameters:\nl = {}, m1 = {}, m2 = {}, m3 = {}, r = {}, i = {}, phi = {}".format(*params_3B))

    return fig