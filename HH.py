import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy import signal
from scipy.integrate import odeint
import numpy.fft
import scipy.signal

def sigm(x):
    """
    Sigmoid function (defined as Theta(x) in the paper) (value 1/2 at x = 0, asymptote = 1)
    :param x: (float)  - Depending on which variable of the model (x, y or z), it is usually an arithmetic sequence
                         like alpha* w * x_tau -w*y_tau +I (for dy/dt)
    :return: (flat) - 1/(1+np.exp(-x))
    """
    return 1/(1+np.exp(-x))

# Rate coefficients taken from Ermantrout p.23

def alpham(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))


def betam(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def alphah(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def betah(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def alphan(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))


def betan(V):
    return 0.125 * np.exp(-(V + 65) / 80.0)


# These currents are of the order of 1 nA (but we use  uA per cm^2 here)

# Membrane currents in uA per cm^2

def INa(V, m, h, gNa=120.0, ENa=50.0):  # Sodium (Na) current
    """

    :param V: Potential
    :param m: TODO
    :param h: TODO
    :param gNa: Maximum conductances [mS/cm^2]
    :param ENa: Nernst reversal potentials [mV]
    :return:
    """
    return gNa * (m ** 3) * h * (V - ENa)


def IK(V, n, gK=36.0, EK=-77.0):  # Potassium (K) current
    """
    Potassium (K) current
    :param V: Potential
    :param n: TODO
    :param gK: Maximum conductances [mS/cm^2]
    :param EK: Nernst reversal potentials [mV]
    :return: Potassium current
    """
    return gK * (n ** 4) * (V - EK)


def IL(V, gL=0.3, EL = -54.4):
    """
    Leak current
    :param V: Potential
    :param gL: Maximum conductances [mS/cm^2]
    :param EL: Nernst reversal potentials [mV]
    :return: Leak current
    """
    return gL * (V - EL)

# Injection current in uA per cm^2

def IinjHH(t, typeofcurrent):
    # options for type of current : 'piecewise', 'ramp', 'constant200', 'constant1','rectangle', 'multiplerectangles'
    if typeofcurrent == 'piecewise':
        return 10 * (t > 100) - 10 * (t > 200) + 35 * (t > 300) - 35 * (t > 400)
    if typeofcurrent == 'ramp':
        return 10 * t
    if typeofcurrent == 'constant200':
        return 200 + t - t
    if typeofcurrent == 'constant1':
        return 1 + t - t
    if typeofcurrent == 'rectangle':
        return 5 * (t > 5) - 5 * (t > 10)  # 0*(t>80) - 0*(t>90)
    if typeofcurrent == 'inverserectangle':
        return -10 * (t > 5) + 10 * (t > 30)
    elif typeofcurrent == 'multiplerectangles':
        return 2.2 * (t > 6) - 2.2 * (t > 12) + 2.2 * (t > 12.5) - 2.2 * (t > 18.5) + 2.2 * (t > 100) - 2.2 * (
                    t > 106) + 2.2 * (t > 115) - 2.2 * (t > 121)
    elif typeofcurrent == 'stepfunction':
        return


# We integrate with odeint
def HHequations(w, t, typeofcurrent, Cm=1, number_of_neurons=3, A=np.array([[0,1,1],[1,0,1],[1,1,0]]), kvec=None):
    """

    :param w: Initial condition vector. For more than one neuron, muste be of the form [V0_1, V0_2 ... , m0_1, m0_2, ..., h0_1, h0_2..., n0_1, n0_2...]
    :param t: Time serie
    :param typeofcurrent: (string) Refere to IinjHH function
    :param A: Adjacency Matrix (can be weighted)
    :return:
    """
    """
    We define the differential equations of the Hodgson-Huxley model.

    Args:
        w :  vector containing the variables of the problem (V, n, m, h)
        t :  time
    """
    if kvec is None:
        # The degree of each node should ideally be given when calling the function to avoid repeating this operation
        kvec = np.sum(A, axis=1)

    V, m, h, n = w[0:number_of_neurons], w[number_of_neurons:2*number_of_neurons], w[2*number_of_neurons:3*number_of_neurons], w[number_of_neurons*3:number_of_neurons*4]
    #V, m, h, n = w

    nodes_receiving_Iinj = np.random.randint(6, 20, 3) # array([1,0,0])# TODO Stimulated neurons vector (>0 if neuron receives injected current, 0 if not, <0 if negative current)

    nodes_receiving_Iinj = np.array([20, 20, 20*(t>11.585)])

    #dVdt = (nodes_receiving_Iinj*IinjHH(t, typeofcurrent) - INa(V, m, h) - IK(V, n) - IL(V) + np.dot(A, V) - kvec*V) / Cm

    dVdt = (nodes_receiving_Iinj - INa(V, m, h) - IK(V, n) - IL(V) + 0*(np.dot(A, V) - kvec * V)) / Cm
    #dVdt = (nodes_receiving_Iinj - INa(V, m, h) - IK(V, n) - IL(V)) / Cm

    dmdt = alpham(V) * (1.0 - m) - betam(V) * m

    dhdt = alphah(V) * (1.0 - h) - betah(V) * h

    dndt = alphan(V) * (1.0 - n) - betan(V) * n

    return np.concatenate((dVdt, dmdt, dhdt, dndt))
    #return dVdt, dmdt, dhdt, dndt


def figureHH(figurenumber, typeofcurrent, number_of_neurons, t, w0):
    solution = odeint(HHequations, w0, t, args=(typeofcurrent,))

    V = solution[:, 0:number_of_neurons]
    #print(hilb(V))
    #hilbev, hil = hilb(V)
    #print('moyenne r :', np.mean(hil))
    #plt.plot(t, hil)
    #plt.plot(t, hilbev)
    #plt.show()

    phase1 = phase_from_hilbert(V[:,2].flatten(), t)
    phase2 = phase_from_hilbert(V[:,1].flatten(), t)
    #plt.show()
    plt.plot(t, phase1, 'b')
    plt.plot(t, phase2, 'r')


    print('Fuck : ', phaselockingvalue(phase1, phase2))
    plt.show()

    #f = np.fft.fft(V[:,0])
    #freq = np.fft.fftfreq(len(t), d=t[1]-t[0])
    #plt.plot(freq, f)
    #plt.show()
    print(V)

    m = solution[:, number_of_neurons:number_of_neurons*2]
    h = solution[:, number_of_neurons*2:number_of_neurons*3]
    n = solution[:, number_of_neurons*3:number_of_neurons*4]

    # print('Minimum value of the potential :',min(V), 'mV')

    import operator
    # min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
    #max_index = max(enumerate(V), key=operator.itemgetter(1))[0]
    #print('Maximum value of the potential :', max(V), 'mV at t = ', t[max_index])
    ina = INa(V, m, h)
    ik = IK(V, n)
    il = IL(V)

    plt.figure(figsize=(13, 13))
    plt.subplot(4, 1, 1)
    plt.grid(True)
    plt.plot(t, V)
    # plt.axvline(t[max_index], color='k', linestyle='--')
    plt.ylabel('a) \n Potential [mV]', {"fontsize": 12})
    plt.title("Hodgkin-Huxley neuron for N = 100000 ({})\n".format(typeofcurrent), {"fontsize": 16})

    plt.subplot(4, 1, 2)
    plt.plot(t, IinjHH(t, typeofcurrent), 'k')
    plt.grid(True)
    plt.ylabel('b) \n $I_{inj}$ [$\\mu{A}/cm^2$]', {"fontsize": 12})
    # plt.ylim(-1, 31)

    plt.subplot(4, 1, 3)
    plt.plot(t, ina, 'c', label='$I_{Na}$')
    plt.plot(t, ik, 'y', label='$I_{K}$')
    plt.plot(t, il, 'm', label='$I_{L}$')
    plt.plot(t, ina + ik + il, 'r', label='$I_{membrane}$')
    # plt.axvline(t[max_index], color='k', linestyle='--')
    plt.grid(True)
    plt.ylabel('c) \n Current [$\\mu{A}/cm^2$]', {"fontsize": 12})
    plt.legend(loc="lower right")

    plt.subplot(4, 1, 4)
    plt.plot(t, m, 'r', label='m')
    plt.plot(t, h, 'g', label='h')
    plt.plot(t, n, 'b', label='n')
    plt.grid(True)
    plt.ylabel('d) \n Gating Value (State)', {"fontsize": 12})
    plt.xlabel('t [ms]', {"fontsize": 12})
    plt.legend(loc="lower right")

    # plt.savefig("Hodgkin-Huxley neuron for N = 100000 (multiplerectangles)", bbox_inches='tight')

    return plt.show()



def phase_from_hilbert(timeserie, t):
    """
    Computes the phase of the signal at each time by using an hilbert transform
    :param timeserie:
    :return:
    """
    transform = sp.fftpack.hilbert(timeserie)
    phase_vector = np.angle(transform/timeserie)
    return phase_vector

def phaselockingvalue(phasevector1, phasevector2):
    """
    Computes the phase-locking value between two neurons using their respective phase-vector
    :param phasevector1:
    :param phasevector2:
    :return:
    """
    return 1/len(phasevector1)*np.abs(sum(np.exp(1j*(phasevector1-phasevector2))))


def hilb(timeseries):
    #timeseries = timeseries.T
    hilbvec = np.zeros(timeseries.shape)
    thetavec = np.zeros(timeseries.shape)
    for i in range(0, 3):
        transform = sp.fftpack.hilbert(timeseries[:,i].flatten())
        hilbvec[:,i] = transform
        thetavec[:,i] = np.arctan(transform / timeseries[:,i])
    return thetavec, np.absolute(np.sum(np.exp(1j * thetavec), axis=1) / 3)




# Constants

#Cm = 1.0  # Membrane capacitance [uF/cm^2]
#ENa, EK, EL = 50.0, -77.0, -54.4  # Nernst reversal potentials [mV]
#gNa, gK, gL = 120.0, 36.0, 0.3  # Maximum conductances [mS/cm^2]


if __name__ == '__main__':
    # Initial conditions  V0 , m0, h0, n0
    V0 = -65.0
    V1 = -65.0
    V2 = -65.0
    m0 = alpham(V0) / (alpham(V0) + betam(V0))
    h0 = alphah(V0) / (alphah(V0) + betah(V0))
    n0 = alphan(V0) / (alphan(V0) + betan(V0))
    m1 = alpham(V1) / (alpham(V1) + betam(V1))
    h1 = alphah(V1) / (alphah(V1) + betah(V1))
    n1 = alphan(V1) / (alphan(V1) + betan(V1))
    m2 = alpham(V2) / (alpham(V2) + betam(V2))
    h2 = alphah(V2) / (alphah(V2) + betah(V2))
    n2 = alphan(V2) / (alphan(V2) + betan(V2))
    w0 = np.array([V0, V1, V2, m0, m1, m2, h0, h1, h2, n0, n1, n2], float)  # np.array([-65.0, 0.05, 0.6, 0.32], float)
    # w0 = np.array([V0, m0, h0, n0], float)  # np.array([-65.0, 0.05, 0.6, 0.32], float)
    print(w0)

    number_of_steps = 1000000

    # The time to integrate over
    t = np.linspace(0.0, 100.0, number_of_steps)


    figureHH(1, 'constant1', 3, t, w0)
