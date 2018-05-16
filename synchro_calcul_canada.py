import os
import sys
#from synchro_integration import beta_inf, give_omega_array, give_adjacency_matrix, give_initial_conditions, integrate_sync_dynamics_SBM
from scipy.integrate import odeint
from synchro_dynamics import kuramoto_odeint
import numpy as np
from geometrySBM import to_probability_space
import time as timer
import argparse
import json


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
    #elif freq_distr_str == "Lorentzian":
    #    ###                       x_0    gamma = D
    #    omega = stats.cauchy.rvs(loc=0, scale=0.0006, size=N)
    #    return omega
    elif freq_distr_str == "Gaussian":
        mu, sig = 0, np.pi
        omega = np.random.normal(mu, sig, N)
        return omega
    else:
        print("This distribution is not an option.")


def give_adjacency_matrix(A_string, sizes, pq):
    #if A_string == "SBM":
    #    A = np.array(nx.to_numpy_matrix(
    #        stochastic_block_model(sizes, pq, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True)))
    #    return A
    if A_string == "average":
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
        # A = np.block([[p_mat_up, q_mat_up], [q_mat_low, p_mat_low]])   # For newer numpy 3.4
        return A


def betafun(n1, n2):
    return (n1*(n1-1) + n2*(n2-1)) / ( (n1 + n2)* (n1 + n2 - 1))


def integrate_sync_dynamics_SBM(sync_dynamics, thetas0, coupling, alpha, freq_distr_str, A_string, sizes, pq,
                                numberoffreq, numberofrandmat, timelist, rglob=False, r1r2=False, r12t=True):
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

    # ttot = timer.clock()
    rsublist1, rtsublist1, r1sublist1, r2sublist1, rt1sublist1, rt2sublist1 = [], [], [], [], [], []
    i = 0  # To indicate the number of frequency loop
    for j in range(0, numberoffreq):
        # tfreq = timer.clock()
        omega = give_omega_array(freq_distr_str, N)
        rsublist, rtsublist, r1sublist, r2sublist, rt1sublist, rt2sublist = [], [], [], [], [], []

        for k in range(0, numberofrandmat):
            A = give_adjacency_matrix(A_string, sizes, pq)
            ### We find the solution for a matrix and a natural frequency array
            allsolutionsode = odeint(sync_dynamics, w0ode, timelist, args=(omega, A, coupling, alpha))  # Each column is for one oscillator (there is N columns) and each line is for the time (there is numberoftimepoints lines)

            ### Cette boucle ralentie évidemment le code
            # dot_theta_matrix = np.zeros((len(timelist), N))  # Useful to have dot_theta to plot the solution (we see the oscillations better)
            # for j in range(len(timelist)):
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
                tinf = 4 * len(timelist) // 5
                for t in range(tinf, len(timelist)):
                    r1 += np.absolute(sum(np.exp(1j * allsolutionsode[t, 0:sizes[0]]))) / sizes[0]
                    r2 += np.absolute(sum(np.exp(1j * allsolutionsode[t, sizes[0]:N]))) / sizes[1]

                r1 = r1 / (len(timelist) - tinf)
                r2 = r2 / (len(timelist) - tinf)
                r1sublist.append(r1)
                r2sublist.append(r2)
            elif rglob == True:
                ### Standard synchronization order parameter
                r = 0
                tinf = 4 * len(timelist) // 5
                for t in range(tinf, len(timelist)):
                    r += np.absolute(sum(np.exp(1j * allsolutionsode[t, :]))) / N
                r = r / (len(timelist) - tinf)  # r avg on time
                rsublist.append(r)
                rtsublist.append(np.absolute(np.sum(np.exp(1j * allsolutionsode), axis=1) / N))

            elif r12t == True:
                rtsublist.append(np.absolute(np.sum(np.exp(1j * allsolutionsode), axis=1) / N))
                rt1sublist.append(np.absolute(np.sum(np.exp(1j * allsolutionsode[:, 0:sizes[0]]), axis=1) / sizes[0]))
                rt2sublist.append(np.absolute(np.sum(np.exp(1j * allsolutionsode[:, sizes[0]:N]), axis=1) / sizes[1]))

        # Average over the random matrix ensemble SBM
        rsublist1.append(1 / numberofrandmat * sum(rsublist))
        rtsublist1.append(1 / numberofrandmat * np.array([sum(i) for i in zip(*rtsublist)]))
        r1sublist1.append(1 / numberofrandmat * sum(r1sublist))
        r2sublist1.append(1 / numberofrandmat * sum(r2sublist))
        rt1sublist1.append(1 / numberofrandmat * np.array([sum(i) for i in zip(*rt1sublist)]))
        rt2sublist1.append(1 / numberofrandmat * np.array([sum(i) for i in zip(*rt2sublist)]))
        i += 1

    # Avg over natural frequencies and random matrix ensemble
    r = 1 / numberoffreq * sum(rsublist1)
    rtlist = 1 / numberoffreq * np.array([sum(i) for i in zip(*rtsublist1)])
    r1 = 1 / numberoffreq * sum(r1sublist1)
    r2 = 1 / numberoffreq * sum(r2sublist1)
    rt1list = 1 / numberoffreq * np.array([sum(i) for i in zip(*rt1sublist1)])
    rt2list = 1 / numberoffreq * np.array([sum(i) for i in zip(*rt2sublist1)])

    # if r12t == False:
    #    print("r = ", r)
    #    print("r1 = ", r1)
    #    print("r2 = ", r2, "\n")
    # print((timer.clock() - ttot)/60, "minutes process time total")

    return r, r1, r2, np.array(rtlist), np.array(rt1list), np.array(rt2list)  # , dot_theta_matrix


def generate_chimera_map(rho_array, delta_array, alpha, N, n1, sync_dynamics, coupling, initial_conditions_str, freq_distr_str, A_string, numberinitialcond, numberoffreq, numberofrandmat, timelist):
    n2 = N - n1
    sizes = [n1, n2]
    f = n1/n2
    beta = betafun(n1, n2)
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
                r_cumul  = 0
                numberofchimera = 0   # Count the number of chimera
                # Solutions
                if p != 0 and q != 0:    # Because there is noperfect synchronization in this case
                    for k in range(0, numberinitialcond):
                        thetas0 = give_initial_conditions(initial_conditions_str, N)
                        solutions = integrate_sync_dynamics_SBM(sync_dynamics, thetas0, coupling, alpha, freq_distr_str, A_string, sizes, pq, numberoffreq, numberofrandmat, timelist, rglob=False, r1r2=True, r12t=False)
                        r1 = solutions[1]
                        r2 = solutions[2]
                        #print(r1, r2)
                        if r1 - r2 > 0 and np.abs(r1 - r2) > 0.01:
                            if r1 > 0.93:  # r1 < 0.97 and r2 > 0.97:
                                print("p = ", pq[0][0], "\n", "q = ", pq[0][1], "\n", "delta = ", delta, "\n", "rho = ", delta)
                                print("r1 = ", r1)
                                print("r2 = ", r2)
                                r_cumul += r2
                                numberofchimera += 1

                            else:
                                print("You should integrate for a longer time!")
                        elif r1 - r2 < 0 and np.abs(r1 - r2) > 0.01:
                            if r2 > 0.93:  # r1 > 0.97 and r2 < 0.97:
                                print("p = ", pq[0][0], "\n", "q = ", pq[0][1], "\n", "delta = ", delta, "\n", "rho = ", delta)
                                print("r1 = ", r1)
                                print("r2 = ", r2)
                                r_cumul += r1
                                numberofchimera += 1
                                break
                            else:
                                print("You should integrate for a longer time!")

                        else:
                            chimera_map[i, j] = r1

                chimera_map[i, j] = r_cumul/numberinitialcond
                density_map[i, j] = numberofchimera /numberinitialcond

                j += 1
        print((timer.clock() - t_one_delta ) /60, "minutes to process one complete rho loop \n")
        i += 1
    ttot = (timer.clock( ) -t0 ) /60
    print(ttot, "minutes to process")

    #with open('data/chimera_matrix_n1{}_alpha{}.json'.format(n1, alpha), 'w') as outfile:
    #    json.dump(chimera_map.tolist(), outfile)
    #with open('data/density_matrix_n1{}_alpha{}.json'.format(n1, alpha), 'w') as outfile:
    #    json.dump(density_map.tolist(), outfile)


    return chimera_map, density_map, ttot


def main(arguments):
    parser = argparse.ArgumentParser(description='Generate chimera map and density map for given alpha and f')
    parser.add_argument('alpha', help='Phase lag', type=float)
    parser.add_argument('n1', help='Size of the first community', type=int)

    args = parser.parse_args(arguments)

    ##################################################### Constant parameters ############################################
    # Time parameters
    numberoftimepoints = 200  # 5000
    timelist = np.linspace(0, 10, numberoftimepoints)  # attention, en diminuant alpha, ça stabilise plus lentement
    deltat = timelist[1] - timelist[0]

    # Structural parameter of the SBM
    N = 100  # 256   # 1000
    rho_array = np.linspace(0, 1, 5)  # Average density. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
    delta_array = np.linspace(0, 1, 5)  # p - q. See Jean-Gab article on finite size analysis of the detectability limit of the SBM
    adjacency_mat = "average"  # or "SBM"

    # Dynamical parameters
    sig = N
    coupling = sig / N
    freq_distr = "Identical"
    nbfreq = 1
    nbmat = 1

    # Initial conditions
    init_cond = "Uniform"
    nbCI = 200
    alpha = args.alpha
    n1 = args.n1
    chimera_map, density_map, ttot = generate_chimera_map(rho_array, delta_array, alpha, N, n1, kuramoto_odeint, coupling, init_cond, "Identical", "average", nbCI, nbfreq, nbmat, timelist)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


