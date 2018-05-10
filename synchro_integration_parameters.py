import numpy as np
from synchro_integration import give_initial_conditions

### Parameters

# Time parameters
numberoftimepoints = 5000*5  # 5000
timelist = np.linspace(0, 50, numberoftimepoints)       # attention, en diminuant alpha, ça stabilise plus lentement
deltat = timelist[1] - timelist[0]


# Structural parameter of the SBM
N = 256  # 1000
m = 128  # 500  # int(N/2)  # Donc beta = 1/2
sizes = [m, N-m]
#beta = (m ** 2 + (N - m) ** 2) / N ** 2  # THIS IS TRUE IN THE LIMIT OF LARGE COMMUNITIES !!!! Asymetry of the blocs. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
                                          # beta = (n_1(n_1-1)+n_2(n_2 - 1))/n(n-1)
rho_array = np.linspace(0, 1, 10)        # Average density. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
delta_array = np.linspace(0, 0.55, 10)      # p - q. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
adjacency_mat = "average"                 # or "SBM"
nbadjmat = 1                              # There is truly TRUE_NB_OF_SBM  = nbfreq times nbSBM

pq = [[0.64659, 0.35351],
      [0.35351, 0.64659]]

#pq = [[0.6203, 0.3797],
#      [0.3797, 0.6203]]


pq = [[0.62, 0.38],
      [0.38, 0.62]]

"""
#pq = [[0.54, 0.46],
#      [0.46, 0.54]]
#pq = [[0.98, 0.48],
#      [0.48, 0.98]]
#pq = [[5/18, 1/6],    # 5/18 = 0.27777...    et  1/6 = 0.166666666 ...
#     [1/6, 5/18]]
#pq = [[0.5925, 0.4075],
#      [0.4075, 0.5925]]
#from geometrySBM import to_probability_space
#print(to_probability_space(0.73, 0.5, 0.5))
"""


# Dynamical parameters
            ### Ici-bas, IMPORTANT ____________________________________
sig = m   # Éventuellement, rescaler le temps, diviser les fréquences naturelles par sigma, redéfinir une freq nat rescalé , (il faut que les freq nat et le terme de couplage soient du même ordre)
sigma_array = np.zeros(N)
sigma_array[0:m] = sig/sizes[0]
sigma_array[m:N] = sig/sizes[1]

Beta = 0.1
alpha = np.pi/2 - Beta

freq_distr = "Identical"
nbfreq = 1


# Initial conditions
init_cond = "Uniform"
nbCI = 100
thetas0 = give_initial_conditions(init_cond, N)






#thetas0 = np.linspace(0, 2*np.pi, N)  #np.zeros(N)          # Initial conditions
#thetas0[0:m] = np.linspace(0, 2*np.pi, m)       # np.random.uniform(-np.pi/4, 3*np.pi/4, size=(1, m))[0]
#thetas0[m:N] = np.linspace(np.pi, 3*np.pi, m)   # np.random.uniform(-5*np.pi/4, 7*np.pi/4, size=(1, m))[0]


#thetas0 = np.random.uniform(-np.pi, np.pi, size=(1, N))[0]
#nbCI = 500   ## Must be big to get the chimeras near the saddle curve !!!!!
#sequence0 = np.linspace(0, 2*np.pi, 10000)
#j = 0
#if j == 0:   # To avoid repeating the operations to have thetas_0_array
#    thetas_0_array = np.zeros((nbCI, N))
#    for i in range(0, nbCI):
#         thetas_0_array[i] = np.random.choice(sequence0, (1, N))[0]
#    j += 1

