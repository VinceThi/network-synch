# @author: Vincent Thibeault

import matplotlib.pyplot as plt
import numpy as np
from stochastic_bloc_model import stochastic_block_model
from scipy.integrate import odeint
from scipy import stats
import networkx as nx
from synchro_dynamics import OA_reduce_kuramoto_odeint
from geometrySBM import to_probability_space
import time as timer
plt.style.use('classic')

#from tkinter import *
#from synchro_dynamics import kuramoto_odeint, dot_theta
#import pickle


### TODO list
# Regarder le spectre de la matrice moyenne versus le spectre de la matrice tiré de SBM lorsqu'il y a des chimères
# Faire des graphiques Nb de chimères obtenues (ou de quoi du genre) selon la dispersion des oscillateurs initialement
# Explorer le chaos pour le minimum du paramètre r_sig selon le paramètre delta (=A), le faire pour notre model
# faire avec alpha_s, (p-e reproduire ou imiter les résultats de Bick 2018)

def give_initial_conditions(initial_conditions_str, N):
    if initial_conditions_str == "Uniform":
      thetas0 = np.random.uniform(-np.pi, np.pi, size=(1, N))[0]
      return thetas0
    else:
        print("This distribution is not an option.")


def give_omega_array(freq_distr_str, N):
    if freq_distr_str == "Identical":
        omega = 0
        return omega
    elif freq_distr_str == "Uniform":
        omega = np.random.uniform(-np.pi, np.pi, size=(1, N))[0]
        return omega
    elif freq_distr_str == "Lorentzian":
        ###                       x_0    gamma = D
        omega = stats.cauchy.rvs(loc=0, scale=0.0006, size=N)
        return omega
    elif freq_distr_str == "Gaussian":
        mu, sig = 0, np.pi
        omega = np.random.normal(mu, sig, N)
        return omega
    else:
        print("This distribution is not an option.")


def give_adjacency_matrix(A_string, sizes, pq):

    if A_string == "SBM":
        A = np.array(nx.to_numpy_matrix(stochastic_block_model(sizes, pq, nodelist=None, seed=None, directed=False, selfloops=False,sparse=True)))
        return A 
    elif A_string == "average":
        N = sum(sizes)
        p_mat_up = pq[0][0] * np.ones((sizes[0], sizes[0]))
        p_mat_low = pq[1][1] * np.ones((sizes[1], sizes[1]))
        q_mat_up = pq[0][1] * np.ones((sizes[0], sizes[1]))
        q_mat_low = pq[1][0] * np.ones((sizes[1], sizes[0]))
        A = np.zeros((N, N))
        A[0:sizes[0], 0:sizes[0]] = p_mat_up
        A[sizes[0]:, sizes[0]:] = p_mat_low
        A[0:sizes[0], sizes[0]:] = q_mat_up
        A[sizes[0]:, 0:sizes[0]] = q_mat_low
        #A = np.block([[p_mat_up, q_mat_up], [q_mat_low, p_mat_low]])   # For newer numpy 3.4
        return A
#print(give_adjacency_matrix("average", [3,2], [[1,2],[3,4]]))

def betafun(n1, n2):
    return (n1*(n1-1) + n2*(n2-1)) / ( (n1 + n2)* (n1 + n2 - 1))


def integrate_sync_dynamics_SBM(sync_dynamics, thetas0, coupling, alpha, freq_distr_str, A_string, sizes, pq, numberoffreq, numberofrandmat, timelist, rglob=False, r1r2=False, r12t=True):
    """
    Integration of a dynamics on the stochastic bloc model.
    :param sync_dynamics (function): Kuramoto dynamics for now, the code need to be adptated for any dynamics later . We need togive
                     the paramters (omega, A, sigma, alpha) in integrate_dynamics_SBM
    :param thetas0 (array): Initial conditions. Positions of the
    :param sigma (array): Coupling strength
    :param alpha (float): Phase-lag
    :param freq_distr_str (str): string telling which frequency distribution we want (see the function "give_omega_array")
    :param A_string (str): string telling on which matrix we want to integrate the dynamics (see the function "give_adjacency_matrix")
    :param sizes(array) and pq (array matrix): see the documentation of the function 'stochastic_bloc_model' in the script stochastic bloc model
    :param numberoffreq (int): number of natural frequency vectors on which we will avg
    :param numberofrandmat (int): numberoffreq multiplied by the true number of random matrix on which we want to average
    :param timelist (array): integration time

    :return: r (float): Global order parameter
    :return: rtlist (array): Global order parameter vs time
    :return: r1, r2 (floats): Global in-group synchrony (Two communities network)
    :return: rt1, rt2 (arrays): Global in-group synchrony (Two communities network) vs time

    """

    N = sum(sizes)
    w0ode = thetas0

    #ttot = timer.clock()
    rsublist1, rtsublist1, r1sublist1, r2sublist1, rt1sublist1, rt2sublist1 = [], [], [], [], [], []
    i = 0   # To indicate the number of frequency loop
    for j in range(0, numberoffreq):
        #tfreq = timer.clock()
        omega = give_omega_array(freq_distr_str, N)
        rsublist, rtsublist, r1sublist, r2sublist, rt1sublist, rt2sublist = [], [], [], [], [], []

        for k in range(0, numberofrandmat):
            A = give_adjacency_matrix(A_string, sizes, pq)
            ### We find the solution for a matrix and a natural frequency array
            allsolutionsode = odeint(sync_dynamics, w0ode, timelist, args=(omega, A, coupling, alpha)) # Each column is for one oscillator (there is N columns) and each line is for the time (there is numberoftimepoints lines)

            ### Cette boucle ralentie évidemment le code
            #dot_theta_matrix = np.zeros((len(timelist), N))  # Useful to have dot_theta to plot the solution (we see the oscillations better)
            #for j in range(len(timelist)):
            #    dot_theta_matrix[j, :] = dot_theta(omega, allsolutionsode[j, :], A, sigma, alpha)#, N)
            """
            ### We plot the last solution obtain from the precedent code block
            
            plt.figure(figsize=(15, 9))
            plt.subplot(2, 1, 1)
            plt.grid(True)
            for thetasol in range(N//2, N//2+5):
                # plt.plot(timelist, allsolutions[thetasol])
                plt.plot(timelist, allsolutionsode[:, thetasol])
            plt.ylabel('$\\theta$', {"fontsize": 30})
            # plt.legend()
            # plt.xlim([0, 20])
            # plt.ylim([-100, 100])

            plt.subplot(2, 1, 2)
            plt.grid(True)
            for thetasol in range(N//2, N//2+5):
                plt.plot(timelist, dot_theta_matrix[:, thetasol])
            plt.ylabel('$\dot{\\theta}$', {"fontsize": 30})
            plt.xlabel('$t$', {"fontsize": 30})

            plt.show()
            """
            if r1r2 == True:
                ### Synchronization order parameter for each community r1, r2 (the code is for two communities)
                r1 = 0
                r2 = 0
                tinf = 4*len(timelist)//5
                for t in range(tinf, len(timelist)):
                    r1 += np.absolute(sum(np.exp(1j * allsolutionsode[t, 0:sizes[0]])))/sizes[0]
                    r2 += np.absolute(sum(np.exp(1j * allsolutionsode[t, sizes[0]:N])))/sizes[1]

                r1 = r1 / (len(timelist)-tinf)
                r2 = r2 / (len(timelist)-tinf)
                r1sublist.append(r1)
                r2sublist.append(r2)
            elif rglob == True:
                ### Standard synchronization order parameter
                r = 0
                tinf = 4*len(timelist)//5
                for t in range(tinf, len(timelist)):
                    r += np.absolute(sum(np.exp(1j * allsolutionsode[t, :]))) / N
                r = r / (len(timelist)-tinf)    # r avg on time
                rsublist.append(r)
                rtsublist.append(np.absolute(np.sum(np.exp(1j * allsolutionsode), axis=1) / N))

            elif r12t == True:
                rtsublist.append(np.absolute(np.sum(np.exp(1j * allsolutionsode), axis=1) / N))
                rt1sublist.append(np.absolute(np.sum(np.exp(1j * allsolutionsode[:, 0:sizes[0]]), axis=1) / sizes[0]))
                rt2sublist.append(np.absolute(np.sum(np.exp(1j * allsolutionsode[:, sizes[0]:N]), axis=1) / sizes[1]))


        # Average over the random matrix ensemble SBM
        rsublist1.append(1/numberofrandmat*sum(rsublist))
        rtsublist1.append(1/numberofrandmat*np.array([sum(i) for i in zip(*rtsublist)]))
        r1sublist1.append(1/numberofrandmat*sum(r1sublist))
        r2sublist1.append(1/numberofrandmat*sum(r2sublist))
        rt1sublist1.append(1/numberofrandmat*np.array([sum(i) for i in zip(*rt1sublist)]))
        rt2sublist1.append(1/numberofrandmat*np.array([sum(i) for i in zip(*rt2sublist)]))
        i += 1

    # Avg over natural frequencies and random matrix ensemble
    r = 1/numberoffreq*sum(rsublist1)
    rtlist = 1/numberoffreq*np.array([sum(i) for i in zip(*rtsublist1)])
    r1 = 1/numberoffreq*sum(r1sublist1)
    r2 = 1/numberoffreq*sum(r2sublist1)
    rt1list = 1/numberoffreq*np.array([sum(i) for i in zip(*rt1sublist1)])
    rt2list = 1/numberoffreq*np.array([sum(i) for i in zip(*rt2sublist1)])

    #if r12t == False:
    #    print("r = ", r)
    #    print("r1 = ", r1)
    #    print("r2 = ", r2, "\n")
    #print((timer.clock() - ttot)/60, "minutes process time total")
    
    return r, r1, r2, np.array(rtlist), np.array(rt1list), np.array(rt2list) #, dot_theta_matrix


def integrate_reduced_sync_dynamics_SBM(w0, coupling, alpha, sizes, pq, timelist):
    """
    :param w0 (list 1D): vector containing initial conditions of the variable of the problem (rho, phi) (len(w0) = 2*q = 2 * number of communities)
    :param t (array): time list
    :param omega (array or float): natural frequency
    :param pq (list of lists (matrix)): Affinity matrix (see stochastic_bloc_model.py)
    :param sizes (list): Sizes of the blocks (see stochastic_bloc_model.py)
    :param coupling (float): sigma/N
    :param alpha (float): Phase lag

    :return: allsolutionsode: The first q solutions are the rho solutions and the last q solutions are the phi solutions
             ex: q = 2
              rho1 = allsolutionsode[:,0]
              rho2 = allsolutionsode[:,1]
              phi1 = allsolutionsode[:,2]
              phi2 = allsolutionsode[:,3]
    """
    w0ode = w0
    time = timer.clock()
    omega = give_omega_array("Identical", sum(sizes))

    allsolutionsode = odeint(OA_reduce_kuramoto_odeint, w0ode, timelist, args=(omega, pq, sizes, coupling, alpha))
    rhor_vs_t = np.zeros((len(timelist), len(sizes)))
    phir_vs_t = np.zeros((len(timelist), len(sizes)))
    f = np.array(sizes)/sum(sizes)
    i = 0
    for rhophilist in allsolutionsode:
        rhor_vs_t[i,:] = rhophilist[:len(sizes)]
        phir_vs_t[i,:] = rhophilist[len(sizes):]
        i += 1
    Rt = np.absolute(np.sum(f*rhor_vs_t*np.exp(1j*phir_vs_t), axis=1))
    #print(Rt)
    Rsum = 0
    tinf = 4*len(timelist)//5
    for t in range(tinf, len(timelist)):
        Rsum += Rt[t]
    Rmoyt = Rsum/(len(timelist) - tinf)
    #print(Rmoyt)
    #print(np.shape(rhor_vs_t), np.shape(phir_vs_t))
    #print(rhor_vs_t, "\n", "\n", phir_vs_t)
    #frhoexp = f*rhor_vs_t*np.exp(1j*phir_vs_t)
    #print(frhoexp)
    #print("\n", np.absolute(np.sum(frhoexp, axis=1)))
    #rhoavgtime = list(np.zeros(len(sizes)))
    #phiavgtime = list(np.zeros(len(sizes)))
    #
    #rho = rhoavgtime/(len(timelist) - tinf)
    #R = np.absolute(sum(f*))

    return allsolutionsode, Rt, Rmoyt


def generate_r_map(rho_array, delta_array, sync_dynamics, thetas0, sigma, alpha, A_string, freq_distr_str, sizes, beta, numberoffreq, numberofrandmat, timelist):

    t0 = timer.clock()
    r_map = np.zeros((len(rho_array), len(delta_array)))
    r1_map = np.zeros((len(rho_array), len(delta_array)))
    r2_map = np.zeros((len(rho_array), len(delta_array)))

    i = 0  # On fixe une ligne pour itérer sur les colonnes j
    for delta in delta_array:
        j = 0  # On fixe une colonne j
        for rho in rho_array:
            print(i, j)
            print("delta = {} et rho = {} \n".format(delta, rho))
            probability_space_tuple = to_probability_space(rho, delta, beta)  # IMPORTANT : return (q, p)
            p = probability_space_tuple[1]
            q = probability_space_tuple[0]
            # print(p,q)
            if p > 1 or p < 0 or q > 1 or q < 0:
                j += 1
            else:
                pq = [[p, q],
                      [q, p]]

                # Solutions
                solution = integrate_sync_dynamics_SBM(sync_dynamics, thetas0, sigma, alpha,  A_string, freq_distr_str, sizes, pq, numberoffreq, numberofrandmat, timelist, rglob=False, r1r2=True, r12t=False)
                r_map[i, j] = solution[0]  
                r1_map[i, j] = solution[1] 
                r2_map[i, j] = solution[2] 
                # r_vs_time_array = solution[3]
                #
                # i = 0
                # plt.figure(figsize=(7, 7))
                # for r in r_vs_time_array:
                #    plt.plot(timelist, r, label="$(\\rho, \\Delta, \\beta) = ({},{},{})$".format(round(rho,4), round(delta,4), beta))
                #    i += 1
                #    plt.legend(loc='best', fontsize=20)
                #    plt.ylabel('$Order\:parameter\:r$', fontsize=25)
                #    plt.xlabel('$Time\:t$', fontsize=25)
                # plt.show()

                j += 1
        i += 1

    print(r_map)
    print(r1_map)
    print(r2_map)
    print((timer.clock()-t0) / 60, "minutes to process")
    return r_map, r1_map, r2_map


def sync_phase_transition(sync_dynamics, thetas0, w0, coupling_array, alpha, freq_distr_str, A_string, sizes, pq, numberoffreq, numberofrandmat, timelist):
    total_time = timer.clock()
    rlist = []
    rlistth = []
    for coupling in coupling_array:
        print(coupling)

        solutions = integrate_sync_dynamics_SBM(sync_dynamics, thetas0, coupling, alpha, freq_distr_str, A_string, sizes, pq, numberoffreq, numberofrandmat, timelist, rglob=True, r1r2=False, r12t=False)
        rlist.append(solutions[0])

        solutionsth = integrate_reduced_sync_dynamics_SBM(w0, coupling, alpha, sizes, pq, timelist)
        rlistth.append(solutionsth[2])
        print("\n")
  
    ttot = (timer.clock() - total_time)/60
    print(ttot, "minutes to process")

    return np.array(rlist), np.array(rlistth), ttot


def generate_chimera_map(rho_array, delta_array, beta, sizes, sync_dynamics, coupling, alpha, initial_conditions_str, freq_distr_str, A_string, numberinitialcond, numberoffreq, numberofrandmat, timelist, density_map_bool=True):
    """
    :param rho_array:
    :param delta_array:
    :param sync_dynamics:
    :param coupling:
    :param alpha:
    :param sizes:
    :param initial_conditions_str:
    :param freq_distr_str:
    :param A_string:
    :param numberinitialcond:
    :param numberoffreq:
    :param numberofrandmat:
    :param timelist:
    :return: chimera_map (array): R_r in a (rho, delta)-map
             density_map (array): density = nbofchimera/nbCI in a (rho, delta)-map
    """
    N = sum(sizes)
    t0 = timer.clock()
    chimera_map = np.zeros((len(rho_array), len(delta_array)))
    density_map = np.zeros((len(rho_array), len(delta_array)))    # density = nbofchimera/nbCI

    i = 0

    for delta in delta_array:
        t_one_delta = timer.clock()
        j = 0  # On fixe une colonne j
        for rho in rho_array:
            probability_space_tuple = to_probability_space(rho, delta, beta)  # IMPORTANT : return (q, p)
            p = probability_space_tuple[1]
            q = probability_space_tuple[0]

            print(i, j)
            print("delta = ", delta, ",", "rho = ", ",", delta, "p = ", np.round(p, 3), ",", "q = ", np.round(q, 3))

            if p > 1 or p < 0 or q > 1 or q < 0:
                chimera_map[i, j] = np.nan
                density_map[i, j] = np.nan
                j += 1
            else:
                pq = [[p, q],
                      [q, p]]

                numberofchimera = 0   # Count the number of chimera
                # Solutions
                if p != 0 and q != 0:    # Because there is noperfect synchronization in this case
                    for k in range(0, numberinitialcond):
                        thetas0 = give_initial_conditions(initial_conditions_str, N)
                        solutions = integrate_sync_dynamics_SBM(sync_dynamics, thetas0, coupling, alpha, freq_distr_str, A_string, sizes, pq, numberoffreq, numberofrandmat, timelist, rglob=False, r1r2=True, r12t=False)
                        r1 = solutions[1]
                        r2 = solutions[2]
                        print(r1, r2)
                        if r1 - r2 > 0 and np.abs(r1 - r2) > 0.01:
                            if r1 > 0.93:       #r1 < 0.97 and r2 > 0.97:
                                print("p = ", pq[0][0], "\n", "q = ", pq[0][1], "\n", "delta = ", delta, "\n", "rho = ", delta)
                                print("r1 = ", r1)
                                print("r2 = ", r2)
                                chimera_map[i, j] = r2
                                numberofchimera += 1
                                if density_map_bool == False:
                                    break
                            else:
                                print("You should integrate for a longer time!")
                        elif r1 - r2 < 0 and np.abs(r1 - r2) > 0.01:
                            if r2 > 0.93:     #r1 > 0.97 and r2 < 0.97:
                                print("p = ", pq[0][0], "\n", "q = ", pq[0][1], "\n", "delta = ", delta, "\n", "rho = ", delta)
                                print("r1 = ", r1)
                                print("r2 = ", r2)
                                chimera_map[i, j] = r1
                                numberofchimera += 1
                                if density_map_bool == False:
                                    break
                            else:
                                print("You should integrate for a longer time!")

                        else:
                            chimera_map[i, j] = r1

                density_map[i, j] = numberofchimera/numberinitialcond

                j += 1
        print((timer.clock() - t_one_delta)/60, "minutes to process one complete rho loop \n")
        i += 1
    ttot = (timer.clock()-t0)/60
    print(ttot, "minutes to process")
    return chimera_map, density_map, ttot







############################################### Generate r map #####################################################

"""
### Parameters
from plot_r_map_good import plot_r_map

# Time parameters
numberoftimepoints = 1000
timelist = np.linspace(0, 50, numberoftimepoints)
deltat = timelist[1] - timelist[0]

# Structural parameter of the SBM
N = 20
rho_array = np.linspace(0, 1, 10)        # Average density. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
delta_array = np.linspace(-1, 1, 10)     # p - q. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
m = 10  # int(N/2)  # Donc beta = 1/2
beta = (m ** 2 + (N - m) ** 2) / N ** 2  # Asymetry of the blocs. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
sizes = [m, N-m]

nbSBM = 10      # There is truly TRUE_NB_OF_SBM  = nbfreq times nbSBM
nbfreq = 10
sigma = 0.5
coupling = np.zeros(N)
coupling[0:m] = sigma/sizes[0]
coupling[m:N] = sigma/sizes[1]

print(sigma)

thetas0 = np.linspace(0, 2*np.pi, N)  # Initial conditions  

maps = generate_r_map(rho_array, delta_array, kuramoto_odeint, thetas0, coupling, 0, "Uniform", "SBM", sizes, nbfreq, nbSBM, timelist)
r_map = maps[0]
r1_map = maps[1]
r2_map = maps[2]

# ATTENTION DE NE PAS OVERWRITE DES DATAS
# import pickle
# pickle.dump(r_map, open("ATTENTIONdata/r_map5050_sig05_10SBM_10freq_N20_beta05_0_to_250_5000time_noavgk", "wb"))
# pickle.dump(r1_map, open("ATTENTIONdata/r1maybebest_map5050_sig05_10SBM_10freq_N20_beta05_0_to_250_5000time_noavgk", "wb"))
# pickle.dump(r2_map, open("ATTENTIONdata/r2maybebest_map5050_sig05_10SBM_10freq_N20_beta05_0_to_250_5000time_noavgk", "wb"))
print("N = ", N, ", m = ", m, ", beta = ", beta)
print("#rho = ", len(rho_array), ", #delta = ", len(delta_array))
print("#timepoints = ", numberoftimepoints)
print("#SBM = ", nbSBM)
print("#freq = ", nbfreq)

plot_r_map(r_map, sigma, "$r$")
"""



############################################ Generate a phase transition ###############################################

"""
### Parameters
from plot_r_sigma_good import plot_phase_transition_r_vs_sigma

# Time parameters
numberoftimepoints = 1000
timelist = np.linspace(0, 50, numberoftimepoints)
deltat = timelist[1] - timelist[0]

# Structural parameter of the SBM
N = 20
m = 10  # int(N/2)  # Donc beta = 1/2
pq = [[0.7, 0.2],[0.2, 0.7]]
sizes = [m, N-m]
alpha = 0
numberoffreq = 10
numberofrandmat = 10
numberofsigma = 10

thetas0 = np.linspace(0, 2*np.pi, N)  # Initial conditions  

sig_rlist = sync_phase_transition(kuramoto_odeint, thetas0, alpha, "Uniform", "SBM", sizes, pq, numberoffreq, numberofrandmat, numberofsigma, timelist)
coupling = sig_rlist[0]
rlist = sig_rlist[1]
plot_phase_transition_r_vs_sigma(coupling, rlist, sizes, pq)
"""



######################################## Generate Chimera r1 and r2 vs time ############################################
"""

def plot_r1_r2(timelist, rt1list, rt2list, filename, timestr):
    plt.figure(figsize=(12, 8))
    plt.plot(timelist, rt1list, label="$r_1(t)$")
    plt.plot(timelist, rt2list, label="$r_2(t)$")
    plt.legend(loc='best', fontsize=20)
    plt.ylabel('$Order\:parameters$', fontsize=25)
    plt.xlabel('$Time\:t$', fontsize=25)
    fig = plt.gcf()
    plt.show()
    from tkinter import messagebox
    if messagebox.askyesno("Python", "Would you like to save the plot?"):
        #fig.savefig("Images/chimeras/chimerar1r2_N{}_p{}_q{}_alpha{}_sigma{}_beta{}_{}freqdistr_adjmat{}_thetas0_separateuniform_time{}to{}_{}pts.jpg".format( N, pq[0][0], pq[0][1], np.round(alpha, 2), sigma, beta, freq_distr, adjacency_mat, timelist[0], timelist[-1], len(timelist)))
        fig.savefig("data/{}_{}_r1_r2.jpg".format(filename, timestr))
"""
"""
### Parameters


# Time parameters
numberoftimepoints = 10000  # 5000
timelist = np.linspace(0, 10, numberoftimepoints)
deltat = timelist[1] - timelist[0]


# Structural parameter of the SBM
N = 1000
m = 500  # int(N/2)  # Donc beta = 1/2
sizes = [m, N-m]
beta = (m ** 2 + (N - m) ** 2) / N ** 2  # Asymetry of the blocs. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
adjacency_mat = "average"
pq = [[0.64, 0.36],
      [0.36, 0.64]]
nbSBM = 1   # There is truly TRUE_NB_OF_SBM  = nbfreq times nbSBM


# Dynamical parameters
            ### Ici-bas, IMPORTANT ____________________________________
sigma = m   # Éventuellement, rescaler le temps, diviser les fréquences naturelles par sigma, redéfinir une freq nat rescalé , (il faut que les freq nat et le terme de couplage soient du même ordre)
coupling = np.zeros(N)
coupling[0:m] = sigma/sizes[0]
coupling[m:N] = sigma/sizes[1]

Beta = 0.1
alpha = np.pi/2 - Beta

freq_distr = "Identical"
nbfreq = 1


# Initial conditions
thetas0 = np.zeros(N)#np.linspace(0, 2*np.pi, N)            # Initial conditions
thetas0[0:m] = np.random.uniform(-np.pi/4, 3*np.pi/4, size=(1, m))[0]     # np.linspace(0, 2*np.pi, m)
thetas0[m:N] = np.random.uniform(-5*np.pi/4, 7*np.pi/4, size=(1, m))[0]   # np.linspace(np.pi, 3*np.pi, m)


# Solutions
solutions = integrate_sync_dynamics_SBM(kuramoto_odeint, thetas0, coupling, alpha, freq_distr, adjacency_mat, sizes, pq, nbfreq, nbSBM, timelist)
rtlist = solutions[3]
rt1list = solutions[4]
rt2list = solutions[5]
from tkinter import messagebox
if messagebox.askyesno("Python", "Would you like to save the data?"):
    pickle.dump(rt1list, open("data/pickle_data/chimeras/rt1list_N{}_p{}_q{}_alpha{}_sigma{}_beta{}_{}freqdistr_adjmat{}_thetas0_XXX_time{}to{}_{}pts.p".format(N, pq[0][0], pq[0][1], np.round(alpha, 2), sigma, beta, freq_distr, adjacency_mat, timelist[0], timelist[-1], len(timelist)), "wb"))
    pickle.dump(rt2list, open("data/pickle_data/chimeras/rt2list_N{}_p{}_q{}_alpha{}_sigma{}_beta{}_{}freqdistr_adjmat{}_thetas0_XXX_time{}to{}_{}pts.p".format(N, pq[0][0], pq[0][1], np.round(alpha, 2), sigma, beta, freq_distr, adjacency_mat, timelist[0], timelist[-1], len(timelist)), "wb"))


# Plot
plot_r1_r2(timelist, rt1list, rt2list)
#plt.plot(timelist, rtlist)
#plt.show()

"""








