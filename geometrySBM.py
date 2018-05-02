#!/usr/bin/env python3
"""Parameter utilities for general modular graphs."""
# -*- coding: utf-8 -*-
# @author: Jean-Gabriel Young <jean.gabriel.young@gmail.com>
import numpy as np


ensemble_types = ['simple_undirected', 'simple_directed',
                  'undirected', 'directed']


def get_m_max(n, ensemble="simple_undirected"):
    """Get maximal edge counts between block pairs, for block sizes n."""
    q = len(n)  # number of blocks
    m_max = np.zeros((q, q))
    for i in range(q):
        if ensemble == "simple_undirected":
            m_max[i, i] = n[i] * (n[i] - 1) / 2
        elif ensemble == "simple_directed":
            m_max[i, i] = n[i] * (n[i] - 1)
        elif ensemble == "undirected":
            m_max[i, i] = n[i] * (n[i] + 1) / 2
        else:  # ensemble == "directed":
            m_max[i, i] = n[i] ** 2
        for j in range(i + 1, q):
            m_max[i, j] = n[i] * n[j]
            m_max[j, i] = n[i] * n[j]
    return m_max.astype(int)


def get_beta(w, n, ensemble='simple_undirected'):
    """Get the value of beta for the indicator matrix W and block sizes n."""
    m_max = get_m_max(n, ensemble)
    if ensemble == "simple_undirected" or ensemble == "undirected":
        normalization = np.sum(np.triu(m_max))
        return np.sum(np.triu(m_max * w / normalization))
    elif ensemble == "simple_directed" or ensemble == "directed":
        normalization = np.sum(m_max)
        return np.sum(m_max * w / normalization)


def get_p(w, p_out, p_in):
    """Construct probability matrix from indicator matrix and densities."""
    return w * (p_in - p_out) + np.ones_like(w) * p_out


def to_probability_space(rho, delta, beta):
    """Map parameters from the density space to the probability space.
    Return
    ------
    (p_out, p_in) : tuple of float
       Internal and external densities.
    """
    return (rho - delta * beta, rho + delta * (1 - beta))


def to_density_space(p_out, p_in, beta):
    """Map parameters from the probability space to the density space.
    Return
    ------
    (rho, delta) : tuple of float
       Density space coordinates.
    """
    return ((beta * p_in + (1 - beta) * p_out), p_in - p_out)


def in_allowed_region(rho, delta, beta):
    """Check whether a (rho,Delta) coordinate is in the allowed region."""
    if delta < 0:  # map to upper region under the rotation symmetry
        rho = 1 - rho
    if rho <= beta:
        return rho / beta >= abs(delta)
    else:
        return -rho / (1 - beta) + 1 / (1 - beta) >= abs(delta)


def get_delta_limits(beta, rho):
    """Get extremal values of Delta for fixed beta and rho."""
    lims = [0, 0]  # list since pairs are immutable
    if rho < (1 - beta):
        lims[0] = -rho / (1 - beta)
    else:
        lims[0] = rho / beta - 1 / beta
    if rho < beta:
        lims[1] = rho / beta
    else:
        lims[1] = -rho / (1 - beta) + 1 / (1 - beta)
    return (lims[0], lims[1])


def get_rho_limits(beta, delta):
    """Get extremal values of rho for fixed beta and delta."""
    if delta < 0:
        # return (-rho / (1 - beta), rho / beta - 1 / beta)
        return (-(1 - beta) * delta, beta * delta + 1)
    else:  # delta >0
        return (beta * delta, 1 - (1 - beta) * delta)


def uniform_cover_generator(beta, rho_spacing=0.05, delta_spacing=0.05, ensemble='simple_undirected'):
    """
    Generate a list of parameters covering the allowed region uniformly.
    Notes
    -----
    Return values in the density space.
    """
    for rho in np.arange(0, 1 + rho_spacing, rho_spacing):
        for Delta in np.arange(-1, 1 + delta_spacing, delta_spacing):
            if in_allowed_region(rho, Delta, beta):
                yield (rho, Delta)


def phase_transition_generator(delta_list, rho, beta):
    """
    Generate (p_out, p_in) pairs for a list of delta values at fixed rho.
    The GMG will undergo a detectability phase transition as delta nears 0.
    """
    for delta in delta_list:
        yield to_probability_space(rho, delta, beta)