# @author: Vincent Thibeault

import numpy as np

### Kuramoto dynamics with phase delay
"""
# ATTENTION ATTENTION le couplage n'est pas divisé par le degré moyen !!!!!!!!!!! C'est un test...
### et c'est beaucoup mieux , voir équation 11 du review d'arenas 2008 et l'explication!!!  Pas nécessairement!!!!!, on ne veut pas comparer des réseaux, on aimerait
### un comportement intensif , voir Rodrigues 2016 le review   Voir aussi Abrams 2008, 2017 chimère kuramoto
"""

def dot_theta(omega, theta, adjacencymatrix, coupling, alpha):
    return omega + coupling*(np.cos(theta + alpha) * np.dot(adjacencymatrix, np.sin(theta)) - np.sin(theta+alpha) * np.dot(adjacencymatrix, np.cos(theta)))

def kuramoto_odeint(w, t, omega, adjacencymatrix, coupling, alpha):
    """
    Function used for the integration of the coupled dynamical system with odeint.

    theta: Oscillator position

    Args:
    w: vector containing the variable of the problem (theta)
    adjacencymatrix: Adjacency matrix

    Return:
    dthetadt

    The shape of these matrix is (N, numberoftimepoints).
    """
    theta = w
    dthetadt = dot_theta(omega, theta, adjacencymatrix, coupling, alpha)#, N)
    return dthetadt


### Reduced Kuramoto dynamics with Ott-Antonson ansatz
def dot_rho(rho, phi, pq, sizes, coupling, alpha):
    f = np.array(sizes)/sum(sizes)
    return coupling*(1-rho**2)/2*(np.cos(phi + alpha) * np.dot(np.array(pq), f*rho*np.cos(phi)) + np.sin(phi+alpha) * np.dot(np.array(pq), f*rho*np.sin(phi)))

def dot_phi(rho, phi, omega, pq, sizes, coupling, alpha):
    f = np.array(sizes)/sum(sizes)
    return omega + coupling*(1+rho**2)/(2*rho)*(np.cos(phi + alpha) * np.dot(np.array(pq), f*rho*np.sin(phi)) - np.sin(phi+alpha) * np.dot(np.array(pq), f*rho*np.cos(phi)))

def OA_reduce_kuramoto_odeint(w, t, omega, pq, sizes, coupling, alpha):
    """

    :param w: vector containing the variable of the problem (theta) (len(w) = 2*q = 2 * number of communities)
    :param t: time list
    :param omega: natural frequency
    :param pq (list of lists (matrix)): Affinity matrix (see stochastic_bloc_model.py)
    :param sizes (list): Sizes of the blocks (see stochastic_bloc_model.py)
    :param coupling: sigma/N
    :param alpha: Phase lag
    :return:
    """
    rho = w[0]
    phi = w[1]
    drhodt = dot_rho(rho, phi, pq, sizes, coupling, alpha)
    dphidt = dot_phi(rho, phi, omega, pq, sizes, coupling, alpha)
    return np.concatenate([drhodt, dphidt])





