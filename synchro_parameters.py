# @author: Vincent Thibeault
import numpy as np

### Parameters

# Time parameters
numberoftimepoints = 2000  # 5000
timelist = np.linspace(0, 20, numberoftimepoints) # attention, en diminuant alpha, ça stabilise plus lentement
deltat = timelist[1] - timelist[0]


# Structural parameter of the SBM
N = 200#256   # 1000
m = 130#128   # 500  # int(N/2)  # Donc beta = 1/2
sizes = [m, N-m]
n1 = sizes[0]
n2 = sizes[1]
beta = (n1*(n1-1) + n2*(n2-1)) / (N*(N-1))  # Asymetry of the blocs. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
rho_array = np.linspace(0.4, 0.9, 20)        # Average density. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
delta_array = np.linspace(0, 0.1, 20)        # p - q. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
adjacency_mat = "average"                   # or "SBM"
nbadjmat = 1                                # There is truly TRUE_NB_OF_SBM  = nbfreq times nbSBM

#pq = [[0.64, 0.36],
#      [0.36, 0.64]]
#pq = [[0.675, 0.325],
#      [0.325, 0.675]]
#pq = [[0.821237767786, 0.579132504628],
#      [0.579132504628, 0.821237767786 ]]
pq = [[0.9421581592171383 , 0.5737371065855593],
      [0.5737371065855593, 0.9421581592171383]]
"""
#pq = [[0.54, 0.46],
#      [0.46, 0.54]]
#pq = [[0.98, 0.48],    
#      [0.48, 0.98]]    
#pq = [[5/18, 1/6],    # 5/18 = 0.27777...    et  1/6 = 0.166666666 ...
#     [1/6, 5/18]]
#pq = [[0.5925, 0.4075],
#      [0.4075, 0.5925]]
"""
from geometrySBM import to_probability_space
print(to_probability_space(0.7736842105263158, 0.368421052631579, beta))



# Dynamical parameters
sig = N
coupling = sig/N

coupling_array = np.linspace(0, 20, 200)

Beta = 0.1
alpha = np.pi/2 - Beta

freq_distr = "Identical"
nbfreq = 1


# Initial conditions
init_cond = "Uniform"
nbCI = 20
#thetas0 = np.random.uniform(-np.pi, np.pi, size=(1, N))[0]
thetas0 = np.linspace(0, 2*np.pi, N)


w0 = [0.01, 0.01, 0, 0]


#            ### Ici-bas, IMPORTANT ____________________________________
#print("ssig = m   # Éventuellement, rescaler le temps, diviser les fréquences naturelles par sigma, redéfinir une freq nat rescalé , (il faut que les freq nat et le terme de couplage soient du même ordre)
#print("scoupling = np.zeros(N)
#print("scoupling[0:m] = sig/sizes[0]
#print("scoupling[m:N] = sig/sizes[1]


#thetas0 = np.linspace(0, 2*np.pi, N)  #np.zeros(N)          # Initial conditions
#thetas0[0:m] = np.linspace(0, 2*np.pi, m)       # np.random.uniform(-np.pi/4, 3*np.pi/4, size=(1, m))[0]
#thetas0[m:N] = np.linspace(np.pi, 3*np.pi, m)   # np.random.uniform(-5*np.pi/4, 7*np.pi/4, size=(1, m))[0]

#nbCI = 500   ## Must be big to get the chimeras near the saddle curve !!!!!
#sequence0 = np.linspace(0, 2*np.pi, 10000)
#j = 0
#if j == 0:   # To avoid repeating the operations to have thetas_0_array
#    thetas_0_array = np.zeros((nbCI, N))
#    for i in range(0, nbCI):
#         thetas_0_array[i] = np.random.choice(sequence0, (1, N))[0]
#    j += 1

