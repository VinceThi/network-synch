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