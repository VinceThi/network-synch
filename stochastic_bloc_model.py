"""Random graph genetarors."""
# -*- coding: utf-8 -*-
# @author: Jean-Gabriel Young <jean.gabriel.young@gmail.com>
import networkx as nx
import itertools
import random
import math

def stochastic_block_model(sizes, p, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True):
   """Return a stochastic block model graph.
   Parameters
   ----------
   sizes : list of ints
     Sizes of blocks
   p : list of list of floats
     Element (i,j) gives the density of edges going from the nodes
     of group i to nodes of group j.
     p must match the number of groups (len(sizes) == len(p)),
     and it must be symmetric if the graph is undirected.
   nodelist : list, optional
     The block tags are assigned according to the node identifiers
     in nodelist. If nodelist is None, then the ordering is the
     range [0,sum(sizes)-1].
   seed : int, optional,  default=None
     Seed for random number generator.
   directed : boolean optional, default=False
     Whether to create a directed graph or not.
   selfloops : boolean optional, default=False
     Whether to include self-loops or not.
   sparse: boolean optional, default=True
     Use the sparse heuristic to speed up the generator.
   Returns
   -------
   g : NetworkX Graph or DiGraph
     Stochastic block model graph of size sum(sizes)
   Raises
   ------
   NetworkXError
     If probabilities are not in [0,1].
     If the probability matrix is not square (directed case).
     If the probability matrix is not symmetric (undirected case).
     If the sizes list does not match nodelist or the probability matrix.
     If nodelist contains duplicate.
   Examples
   --------
   #>>> from __future__ import division
   #>>> import numpy as np
   #>>> import networkx as nx
   #>>>
   #>>> sizes = [75, 75, 300]
   #>>> probs = [[0.25, 0.05, 0.02],
   #...          [0.05, 0.35, 0.07],
   #...          [0.02, 0.07, 0.40]]
   #>>> g = nx.stochastic_block_model(sizes, probs)
   #>>> len(g)
   #450
   #>>> H = nx.algorithms.blockmodel(g, g.graph['partition'])
   #>>> for v in H.nodes_iter(data=True):
   #...     print v[1]['density']
   #...
   #0.254414414414
   #0.357477477477
   #0.401672240803
   #>>> for v in H.edges_iter(data=True):
   #...     print v[2]['weight'] / (sizes[v[0]] * sizes[v[1]])
   #...
   #0.0483555555556
   #0.0195555555556
   #0.0728888888889
   See Also
   --------
   random_partition_graph
   planted_partition_graph
   gaussian_random_partition_graph
   gnp_random_graph
   bipartite_random_graph
   References
   ----------
   .. [1] Holland, P. W., Laskey, K. B., & Leinhardt, S.,
          "Stochastic blockmodels: First steps",
          Social networks, 5(2), 109-137, 1983.
   """
   # Check if dimensions match
   if len(sizes) != len(p):
       raise nx.NetworkXException("'sizes' and 'p' do not match.")
   # Check for probability symmetry (undirected) and shape (directed)
   for row in p:
       if len(p) != len(row):
           raise nx.NetworkXException("'p' must be a square matrix.")
   if not directed:
       p_transpose = [list(i) for i in zip(*p)]
       for i in zip(p, p_transpose):
           for j in zip(i[0], i[1]):
               if abs(j[0] - j[1]) > 1e-08:
                   raise nx.NetworkXException("'p' must be symmetric.")
   # Check for probability range
   for row in p:
       for prob in row:
           if prob < 0 or prob > 1:
               raise nx.NetworkXException("Entries of 'p' not in [0,1].")
   # Check for nodelist consistency
   if nodelist is not None:
       if len(nodelist) != sum(sizes):
           raise nx.NetworkXException("'nodelist' and 'sizes' do not match.")
       if len(nodelist) != len(set(nodelist)):
           raise nx.NetworkXException("nodelist contains duplicate.")
   else:
       nodelist = range(0, sum(sizes))

   # Setup the graph conditionally to the directed switch.
   block_range = range(len(sizes))
   if directed:
       g = nx.DiGraph()
       block_iter = itertools.product(block_range, block_range)
   else:
       g = nx.Graph()
       block_iter = itertools.combinations_with_replacement(block_range, 2)
   # Split nodelist in a partition (list of sets).
   size_cumsum = [sum(sizes[0:x]) for x in range(0, len(sizes) + 1)]
   g.graph['partition'] = [set(nodelist[size_cumsum[x]:size_cumsum[x + 1]])
                           for x in range(0, len(size_cumsum) - 1)]
   # Setup nodes and graph name
   for block_id, nodes in enumerate(g.graph['partition']):
       for node in nodes:
           g.add_node(node, block=block_id)

   g.name = "stochastic_block_model"

   # Test for edge existence
   if seed is not None:
       random.seed(seed)

   for i, j in block_iter:
       if i == j:
           if directed:
               if selfloops:
                   edges = itertools.product(g.graph['partition'][i],
                                             g.graph['partition'][i])
               else:
                   edges = itertools.permutations(g.graph['partition'][i], 2)
           else:
               edges = itertools.combinations(g.graph['partition'][i], 2)
               if selfloops:
                   edges = itertools.chain(edges, zip(g.graph['partition'][i],
                                                      g.graph['partition'][i]))  # noqa
           for e in edges:
               if random.random() < p[i][j]:
                   g.add_edge(*e)
       else:
           edges = itertools.product(g.graph['partition'][i],
                                     g.graph['partition'][j])
       if sparse:
           consume = lambda it, n: next(itertools.islice(it, n, n), None)
           geo = lambda p: math.floor(math.log(random.random()) / math.log(1 - p))  # noqa
           if p[i][j] == 1:  # Test edges cases p_ij = 0 or 1
               for e in edges:
                   g.add_edge(*e)
           elif p[i][j] > 0:
               while True:
                   try:
                       skip = geo(p[i][j])
                       consume(edges, skip)
                       e = next(edges)
                       g.add_edge(*e)  # __safe
                   except StopIteration:
                       break
       else:
           for e in edges:
               if random.random() < p[i][j]:
                   g.add_edge(*e)  # __safe
   return g

#if __name__ == "__main__":
#   # Options parser.
#   import argparse as ap
#   import sys
#   ensemble_types = ['simple_undirected', 'simple_directed',
#                     'undirected', 'directed']
#   prs = ap.ArgumentParser(description="SBM graph generator.")
#   prs.add_argument('--ensemble_type', '-e', type=str, default='simple_undirected',  # noqa
#                    choices=ensemble_types, metavar="",
#                    help="Type of PPM ensemble. Implemented choices are: [" +
#                         " | ".join(ensemble_types) + "]. "
#                         "Simple ensembles do not contain self-loops.")
#   prs.add_argument('--size_vector', '-n', type=int, nargs='+',
#                    help='Size of each block.')
#   prs.add_argument('--probabilities', '-p', type=float, nargs='+',
#                    help='Connection matrix in row-major order.')
#   prs.add_argument('--use_column_major', '-C', action='store_true',
#                    help='Declare connection matrix column-major order.')
#   prs.add_argument('--write_to_file', '-w', action='store_true',
#                    help='Write to automatically generated file.')
#   prs.add_argument('--base', '-b', type=str, default='',
#                    help='Absolute base path to output. -w flag must be used\
#                          for this option to have any effect.')
#   prs.add_argument('--seed', '-d', type=int,
#                    help='RNG seed.')
#   if len(sys.argv) == 1:
#       prs.print_help()
#       sys.exit(1)
#   args = prs.parse_args()
#
#   # Load probability matrix in row major order
#   num_block = len(args.size_vector)
#   probability_matrix = [[0] * num_block for i in range(num_block)]
#   if not args.use_column_major:
#       for idx, p in enumerate(args.probabilities):
#           probability_matrix[idx // num_block][idx % num_block] = p
#   else:
#       for idx, p in enumerate(args.probabilities):
#           probability_matrix[idx % num_block][idx // num_block] = p
#   # Generate from the proper ensemble
#   if args.ensemble_type == ensemble_types[0]:
#       g = stochastic_block_model(args.size_vector, probability_matrix,
#                                  seed=args.seed)
#   elif args.ensemble_type == ensemble_types[1]:
#       g = stochastic_block_model(args.size_vector, probability_matrix,
#                                  directed=True, seed=args.seed)
#   elif args.ensemble_type == ensemble_types[2]:
#       g = stochastic_block_model(args.size_vector, probability_matrix, selfloop=True, seed=args.seed)
#   else:
#       g = stochastic_block_model(args.size_vector, probability_matrix, directed=True, selfloop=True, seed=args.seed)
#
#   if args.write_to_file:
#       prob_list = [str(x) for x in args.probabilities]
#       size_list = [str(x) for x in args.size_vector]
#       fpath = "probs_" + "_".join(prob_list).replace('.', '-') +\
#               "_sizes_" + "_".join(size_list)
#       with open(args.base + fpath, 'w') as f:
#           for e in g.edges():
#               f.write(str(e[0]) + " " + str(e[1]) + "\n")
#   else:
#       for e in g.edges():
#           print(e[0], e[1])
#
#

#import numpy as np
#
#A = np.array(nx.to_numpy_matrix(stochastic_block_model(sizes, p, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True)))
#
#degree_list = []
#for j in range(0, N):
#    k = 0
#    for i in range(0, N):
#        if A[i,j] > 0:
#            k += 1
#    degree_list.append(k)
#
#avg_degree = 1/N*sum(degree_list)
#
#L = np.diag(degree_list) - A
#
#print(A)








