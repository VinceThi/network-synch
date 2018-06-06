import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from synchro_integration import give_adjacency_matrix
from math import pi
from scipy import signal
from scipy.integrate import odeint
import numpy.fft
import scipy.signal


############################### Rate coefficients taken from Ermantrout p.23  ##########################################

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



################################ Membrane currents in uA per cm^2 ######################################################
#These currents are of the order of 1 nA (but we use  uA per cm^2 by convention)

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



############################### Synaptic current in uA per cm^2 #######################################################
def IsynHH(t, typeofcurrent):
    # options for type of current : 'piecewise', 'ramp', 'constant200', 'constant1','rectangle', 'multiplerectangles'
    if typeofcurrent == 'piecewise':
        return 10 * (t > 100) - 10 * (t > 200) + 35 * (t > 300) - 35 * (t > 400)
    elif typeofcurrent == 'ramp':
        return 10 * t
    elif typeofcurrent == 'constant200':
        return 200 + t - t
    elif typeofcurrent == 'constant1':
        return 1 + t - t
    elif typeofcurrent == 'constant_random':
        return np.random.randint(0, 2)*np.random.randint(10, 50)*(t > np.random.randint(0, 50))
    elif typeofcurrent == 'rectangle':
        return 5 * (t > 5) - 5 * (t > 10)  # 0*(t>80) - 0*(t>90)
    elif typeofcurrent == 'inverserectangle':
        return -10 * (t > 5) + 10 * (t > 30)
    elif typeofcurrent == 'multiplerectangles':
        return 2.2 * (t > 6) - 2.2 * (t > 12) + 2.2 * (t > 12.5) - 2.2 * (t > 18.5) + 2.2 * (t > 100) - 2.2 * (
                    t > 106) + 2.2 * (t > 115) - 2.2 * (t > 121)
    else:
        print("This current is not an option !")

def Isyn_vec(N):
    vecIsyn = np.zeros(N)
    for i in range(0, N):
        if i < sizes[0]:
            vecIsyn[i] = np.random.randint(10, 30)
        else:
            vecIsyn[i] = np.random.randint(30, 50)
        #vecIsyn[i] = np.random.randint(10, 50)

    return vecIsyn


############################### Synaptic coupling functions ############################################################
def sigm(x):
    """
    Sigmoid function (defined as Theta(x) in the paper) (value 1/2 at x = 0, asymptote = 1)
    :param x: (float)  - Depending on which variable of the model (x, y or z), it is usually an arithmetic sequence
                         like alpha* w * x_tau -w*y_tau +I (for dy/dt)
    :return: (flat) - 1/(1+np.exp(-x))
    """
    return 1/(1+np.exp(-x))



###############################  We integrate with odeint ##############################################################
def HHequations(w, t, N, adjacencymatrix, kvec, coupling, Isynvec, Cm=1):
    """
    We define the differential equations of the Hodgkin-Huxley model.
    :param w (array): Initial condition vector. For more than one neuron, muste be of the form [V0_1, V0_2 ... , m0_1, m0_2, ..., h0_1, h0_2..., n0_1, n0_2...]
    :param t (array): Time serie
    :param N (int): Number of neurons (nodes)
    :param adjacencymatrix (array): Adjacency Matrix (can be weighted)
    :param kvec (array): Degree sequence
    :param Isynvec (array): Synaptic current vector

    :return: dXdt array for odeint
    """
    V, m, h, n = w[0:N], w[N:2*N], w[2*N:3*N], w[3*N:4*N]
    #V, m, h, n = w
    #Isyn = np.random.randint(6, 20, 3) # array([1,0,0])# TODO Stimulated neurons vector (>0 if neuron receives injected current, 0 if not, <0 if negative current)
    #Isyn = np.array([20, 20, 20*(t>11.585)])
    #dVdt = (nodes_receiving_Iinj*IinjHH(t, typeofcurrent) - INa(V, m, h) - IK(V, n) - IL(V) + np.dot(A, V) - kvec*V) / Cm
    #dVdt = (nodes_receiving_Iinj - INa(V, m, h) - IK(V, n) - IL(V)) / Cm
    dVdt = (Isynvec - INa(V, m, h) - IK(V, n) - IL(V) + coupling*(np.dot(adjacencymatrix, V) - kvec * V)) / Cm

    dmdt = alpham(V) * (1.0 - m) - betam(V) * m

    dhdt = alphah(V) * (1.0 - h) - betah(V) * h

    dndt = alphan(V) * (1.0 - n) - betan(V) * n

    return np.concatenate((dVdt, dmdt, dhdt, dndt))



############################### Functions that will help to measure the electrical synchrony between neurons ###########
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



############################### Plot functions #########################################################################
def figureHH_with_gating_variables_and_currents(w0, timelist, N, adjacencymatrix, kvec, coupling, Isynvec):

    solution = odeint(HHequations, w0, timelist, args=(N, adjacencymatrix, kvec, coupling, Isynvec,))

    V = solution[:, 0 : N]
    m = solution[:, N : 2*N]
    h = solution[:, 2*N : 3*N]
    n = solution[:, 3*N : 4*N]
    # print(hilb(V))
    # hilbev, hil = hilb(V)
    # print('moyenne r :', np.mean(hil))
    # plt.plot(t, hil)
    # plt.plot(t, hilbev)
    # plt.show()

    # phase1 = phase_from_hilbert(V[:,2].flatten(), t)
    # phase2 = phase_from_hilbert(V[:,1].flatten(), t)
    # plt.show()
    # plt.plot(t, phase1, 'b')
    # plt.plot(t, phase2, 'r')

    # print('Fuck : ', phaselockingvalue(phase1, phase2))
    # plt.show()

    # f = np.fft.fft(V[:,0])
    # freq = np.fft.fftfreq(len(t), d=t[1]-t[0])
    # plt.plot(freq, f)
    # plt.show()
    # print(V)
    # print('Minimum value of the potential :',min(V), 'mV')
    # import operator
    # min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
    # max_index = max(enumerate(V), key=operator.itemgetter(1))[0]
    # print('Maximum value of the potential :', max(V), 'mV at t = ', t[max_index])

    ina = INa(V, m, h)
    ik = IK(V, n)
    il = IL(V)

    plt.figure(figsize=(10, 10))

    plt.rc('font', family='serif')

    plt.subplot(4, 1, 1)
    plt.grid(True)
    plt.plot(timelist, V)
    # plt.axvline(t[max_index], color='k', linestyle='--')
    plt.ylabel('a) \n Potential [mV]', {"fontsize": 12})
    # plt.title("Hodgkin-Huxley neuron for N = 100000 ({})\n".format(typeofcurrent), {"fontsize": 16})

    plt.subplot(4, 1, 2)
    for i in range(0, N):
        plt.plot(timelist, Isynvec[i]*np.ones(len(timelist)))
    plt.grid(True)
    plt.ylabel('b) \n $I_{inj}$ \n [$\\mu{A}/cm^2$]', {"fontsize": 12})
    plt.ylim([9, 51])

    plt.subplot(4, 1, 3)
    plt.plot(timelist, ina, 'c', label='$I_{Na}$')
    plt.plot(timelist, ik, 'y', label='$I_{K}$')
    plt.plot(timelist, il, 'm', label='$I_{L}$')
    plt.plot(timelist, ina + ik + il, 'r', label='$I_{membrane}$')
    # plt.axvline(t[max_index], color='k', linestyle='--')
    plt.grid(True)
    plt.ylabel('c) \n Current \n[$\\mu{A}/cm^2$]', {"fontsize": 12})
    #plt.legend(loc="lower right")

    plt.subplot(4, 1, 4)
    plt.plot(timelist, m, 'r', label='m')
    plt.plot(timelist, h, 'g', label='h')
    plt.plot(timelist, n, 'b', label='n')
    plt.grid(True)
    plt.ylabel('d) \n Gating Variables', {"fontsize": 12})
    plt.xlabel('t [ms]', {"fontsize": 12})
    #plt.legend(loc="lower right")

    plt.tight_layout()

    # plt.savefig("Hodgkin-Huxley neuron for N = 100000 (multiplerectangles)", bbox_inches='tight')

    return plt.show()

def figureHH(w0, timelist, N, adjacencymatrix, kvec, coupling, Isynvec):

    solution = odeint(HHequations, w0, timelist, args=(N, adjacencymatrix, kvec, coupling, Isynvec,))

    V = solution[:, 0 : N]
    m = solution[:, N : 2*N]
    h = solution[:, 2*N : 3*N]
    n = solution[:, 3*N : 4*N]
    # print(hilb(V))
    # hilbev, hil = hilb(V)
    # print('moyenne r :', np.mean(hil))
    # plt.plot(t, hil)
    # plt.plot(t, hilbev)
    # plt.show()

    # phase1 = phase_from_hilbert(V[:,2].flatten(), t)
    # phase2 = phase_from_hilbert(V[:,1].flatten(), t)
    # plt.show()
    # plt.plot(t, phase1, 'b')
    # plt.plot(t, phase2, 'r')

    # print('Fuck : ', phaselockingvalue(phase1, phase2))
    # plt.show()

    # f = np.fft.fft(V[:,0])
    # freq = np.fft.fftfreq(len(t), d=t[1]-t[0])
    # plt.plot(freq, f)
    # plt.show()
    # print(V)
    # print('Minimum value of the potential :',min(V), 'mV')
    # import operator
    # min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
    # max_index = max(enumerate(V), key=operator.itemgetter(1))[0]
    # print('Maximum value of the potential :', max(V), 'mV at t = ', t[max_index])

    ina = INa(V, m, h)
    ik = IK(V, n)
    il = IL(V)

    plt.figure(figsize=(10, 10))
    plt.rc('font', family='serif')
    plt.subplot(211)
    plt.grid(True)
    for i in range(0, N):
        if i < sizes[0]:
            plt.plot(timelist, V[:,i], color='#ff3700')
        else:
            plt.plot(timelist, V[:,i], color='b')
    # plt.axvline(t[max_index], color='k', linestyle='--')
    ylab = plt.ylabel('$V$ \n $[mV]$', fontsize=25, labelpad=40)
    ylab.set_rotation(0)

    # plt.title("Hodgkin-Huxley neuron for N = 100000 ({})\n".format(typeofcurrent), {"fontsize": 16})

    plt.subplot(212)
    for i in range(0, N):
        if i < sizes[0]:
            plt.plot(timelist, Isynvec[i]*np.ones(len(timelist)), color='#ff3700')
        else:
            plt.plot(timelist, Isynvec[i]*np.ones(len(timelist)), color='b')
    plt.grid(True)
    ylab2 = plt.ylabel('$I_{syn}$ \n $[\\mu{A}/cm^2]$', fontsize=25, labelpad=70)
    ylab2.set_rotation(0)
    plt.xlabel('$t [ms]$', fontsize=25)

    plt.ylim([9, 51])

    plt.tight_layout()


    return plt.show()




############################## Experiences #############################################################################
if __name__ == '__main__':

    # Time parameters
    numberoftimepoints = 10000
    timelist = np.linspace(0, 200, numberoftimepoints)
    deltat = timelist[1] - timelist[0]

    # Structural parameters of the SBM
    N = 100
    m = 50
    sizes = [m, N - m]
    n1 = sizes[0]
    n2 = sizes[1]
    f1 = n1 / N
    f2 = n2 / N
    f = f1 / f2
    beta = (n1 * (n1 - 1) + n2 * (n2 - 1)) / (N * (N - 1))  # Asymetry of the blocks. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
    pq = [[0.8, 0.3],
          [0.3, 0.8]]

    adjacencymatrix = give_adjacency_matrix("average", sizes, pq)
    kvec = np.sum(adjacencymatrix, axis=1)

    # Dynamical parameters
    coupling = 0.1
    Isynvec = Isyn_vec(N)

    # Initial conditions
    V0 = -65
    m0 = alpham(V0) / (alpham(V0) + betam(V0))
    h0 = alphah(V0) / (alphah(V0) + betah(V0))
    n0 = alphan(V0) / (alphan(V0) + betan(V0))

    w0 = np.zeros(4*N)
    for i in range(0, 4*N):
        if i < N:
            w0[i] = V0
        elif i >= N and i < 2*N:
            w0[i] = m0
        elif i >= 2*N and i < 3*N:
            w0[i] = h0
        else:
            w0[i] = n0

    figureHH(w0, timelist, N, adjacencymatrix, kvec, coupling, Isynvec)

    #V0 = -65.0
    #V1 = -65.0
    #V2 = -65.0
    #m0 = alpham(V0) / (alpham(V0) + betam(V0))
    #h0 = alphah(V0) / (alphah(V0) + betah(V0))
    #n0 = alphan(V0) / (alphan(V0) + betan(V0))
    #m1 = alpham(V1) / (alpham(V1) + betam(V1))
    #h1 = alphah(V1) / (alphah(V1) + betah(V1))
    #n1 = alphan(V1) / (alphan(V1) + betan(V1))
    #m2 = alpham(V2) / (alpham(V2) + betam(V2))
    #h2 = alphah(V2) / (alphah(V2) + betah(V2))
    #n2 = alphan(V2) / (alphan(V2) + betan(V2))
    #w0 = np.array([V0, V1, V2, m0, m1, m2, h0, h1, h2, n0, n1, n2], float)  # np.array([-65.0, 0.05, 0.6, 0.32], float)
    # w0 = np.array([V0, m0, h0, n0], float)  # np.array([-65.0, 0.05, 0.6, 0.32], float)
    #print(w0)
    #figureHH_with_gating_variables_and_currents(w0, timelist, N, adjacencymatrix, kvec, coupling, Isynvec)


## Constants
#
#Cm = 1.0                             # Membrane capacitance [uF/cm^2]
#
#ENa, EK, EL = 50.0, -77.0, -54.4     # Nernst reversal potentials [mV]
#
#gNa, gK, gL = 120.0, 36.0, 0.3       # Maximum conductances [mS/cm^2]
#
#
## Rate coefficients
#
#def alpham(V):
#    return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))
#
#def betam(V):
#    return 4.0*np.exp(-(V+65.0) / 18.0)
#
#def alphah(V):
#    return 0.07*np.exp(-(V+65.0) / 20.0)
#
#def betah(V):
#    return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))
#
#def alphan(V):
#    return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))
#
#def betan(V):
#    return 0.125*np.exp(-(V+65) / 80.0)
#
#
## These currents are of the order of 1 nA (but we use  uA per cm^2 here)
#
## Membrane currents in uA per cm^2
#
#def INa(V, m, h):                   # Sodium (Na) current
#    return gNa*(m**3)*h*(V - ENa)
#
#def IK(V, n):                       # Potassium (K) current
#    return gK*(n**4)*(V - EK)
#
#def IL(V):                          # Leak current
#    return gL*(V - EL)
#
## Injection current in uA per cm^2
#
#def IinjHH(t, typeofcurrent):
#    # options for type of current : 'piecewise', 'ramp', 'constant200', 'constant1','rectangle', 'multiplerectangles'
#    if typeofcurrent == 'piecewise':
#        return 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)
#    if typeofcurrent == 'ramp':
#         return 10*t
#    if typeofcurrent == 'constant200':
#         return 200 + t - t
#    if typeofcurrent == 'constant1':
#         return 45 + t - t
#    if typeofcurrent == 'rectangle':
#        return 5*(t>5) - 5*(t>10) # 0*(t>80) - 0*(t>90)
#    if typeofcurrent== 'inverserectangle':
#        return -10*(t>5) + 10*(t>30)
#    elif typeofcurrent == 'multiplerectangles':
#        return 2.2*(t>6) - 2.2*(t>12) + 2.2*(t>12.5) - 2.2*(t>18.5) + 2.2*(t>100) - 2.2*(t>106) + 2.2*(t>115) - 2.2*(t>121)
#
#
## We integrate with odeint
#def HHequations(w, t, typeofcurrent):
#    """
#    We define the differential equations of the Hodgson-Huxley model.
#
#    Args:
#        w :  vector containing the variables of the problem (V, n, m, h)
#        t :  time
#    """
#    V, m, h, n = w
#
#    dVdt = (IinjHH(t, typeofcurrent) - INa(V, m, h) - IK(V, n) - IL(V)) / Cm
#
#    dmdt = alpham(V) * (1.0 - m) - betam(V) * m
#
#    dhdt = alphah(V) * (1.0 - h) - betah(V) * h
#
#    dndt = alphan(V) * (1.0 - n) - betan(V) * n
#
#    return dVdt, dmdt, dhdt, dndt
#
#
#
## The time to integrate over
#numberoftimepoints = 100000
#t = np.linspace(0.0, 100.0, numberoftimepoints)
#
#
## Initial conditions  V0 , m0, h0, n0
#V0 = -65.0
#m0 = alpham(V0) / (alpham(V0) + betam(V0))
#h0 = alphah(V0) / (alphah(V0) + betah(V0))
#n0 = alphan(V0) / (alphan(V0) + betan(V0))
#w0 = np.array([V0, m0, h0, n0], float)  # np.array([-65.0, 0.05, 0.6, 0.32], float)
#print(w0)
#
#
#def figureHH(figurenumber, typeofcurrent, t, w0):
#    solution = odeint(HHequations, w0, t, args=(typeofcurrent,))
#    V = solution[:, 0]
#    m = solution[:, 1]
#    h = solution[:, 2]
#    n = solution[:, 3]
#
#    # print('Minimum value of the potential :',min(V), 'mV')
#    #import operator
#    #min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
#    #max_index = max(enumerate(V), key=operator.itemgetter(1))[0]
#    #print('Maximum value of the potential :', max(V), 'mV at t = ', t[max_index])
#    ina = INa(V, m, h)
#    ik = IK(V, n)
#    il = IL(V)
#
#    plt.figure(figsize=(8, 8))
#    plt.rc('font', family='serif')
#    plt.subplot(4, 1, 1)
#    #plt.grid(True)
#    plt.plot(t, V)
#    # plt.axvline(t[max_index], color='k', linestyle='--')
#    plt.ylabel('a) \n Potential \n [mV]', {"fontsize": 12})
#    #plt.title("Hodgkin-Huxley neuron for numberoftimepoints = 100000 ({})\n".format(typeofcurrent), {"fontsize": 16})
#
#    plt.subplot(4,1,2)
#    plt.plot(t, IinjHH(t, typeofcurrent), 'k')
#   # plt.grid(True)
#    plt.ylabel('b) \n $I_{inj}$ \n [$\\mu{A}/cm^2$]',{"fontsize":12})
#    #plt.ylim(-1, 31)
#
#    plt.subplot(4,1,3)
#    plt.plot(t, ina, 'b', label='$I_{Na}$')
#    plt.plot(t, ik, 'orange', label='$I_{K}$')
#    plt.plot(t, il, 'm', label='$I_{L}$')
#    plt.plot(t, ina + ik + il, 'r', label='$I_{membrane}$')
#    #plt.axvline(t[max_index], color='k', linestyle='--')
#    #plt.grid(True)
#    plt.ylabel('c) \n Current \n [$\\mu{A}/cm^2$]',{"fontsize":12})
#    plt.legend(loc="lower right")
#
#    plt.subplot(4,1,4)
#    plt.plot(t, m, 'r', label='m')
#    plt.plot(t, h, 'g', label='h')
#    plt.plot(t, n, 'b', label='n')
#    #plt.grid(True)
#    plt.ylabel('d) \n Gating Value',{"fontsize":12})
#    plt.xlabel('t [ms]',{"fontsize":12})
#    plt.legend(loc="lower right")
#
#    plt.tight_layout()
#
#    #plt.savefig("Hodgkin-Huxley neuron for N = 100000 (multiplerectangles)", bbox_inches='tight')
#
#    return plt.show()
#
#
#figureHH(2, 'constant1', t, w0)


