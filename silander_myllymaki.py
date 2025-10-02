"""
Implementation of the algorithm presented in:
Silander, T. and Myllymaki, P., 2012. A simple approach for finding the globally optimal Bayesian network structure. 
arXiv preprint arXiv:1206.6875.
"""

import pandas as pd
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Tuple
from collections import defaultdict
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BIC
from pygobnilp.gobnilp import read_local_scores

"""Helper functions"""
def ct(W: pd.DataFrame) -> pd.DataFrame:
    """Generates a contingency table from the data W."""
    counts = W.value_counts().reset_index()
    counts.columns = list(W.columns) + ["counts"]
    return counts


def vars(ct: pd.DataFrame) -> list:
    """Returns the set of variables in the contingency table ct"""
    return ct.columns.drop('counts').tolist()


def ct2ct(ct:pd.DataFrame, v:int)-> pd.DataFrame:
    """Produces a contigency table by marginalizing the variable v out of ct"""
    return NotImplementedError


def ct2cft(ct:pd.DataFrame, v:int)->pd.DataFrame:
    """Yields a conditional frequency table"""
    return NotImplementedError


def score(cft):
    """Calculates the local score based on the conditional frequency table"""
    return NotImplementedError


"""MAIN FUNCTIONS"""

#  LS is a mapping from (variable, parent set)-pairs to corresponding local scores
LS: Dict[Tuple[str, frozenset], float] = {}

# Step 1: Compute local scores for all (variable, parent set)-pairs
def get_local_scores(ct:pd.DataFrame, evars:List[int]):
    """The main procedure, GetLocalScores, (Algorithm 1) is called with a contingency table ct 
    and the variables evars to be marginalized from it. Initially, it is called with a contingency 
    table for all the variables and the whole variable set V as evars. The algorithm is simply
    a depth first traversal of smaller and smaller contingency tables."""
    vars_ct = vars(ct)
    for v in vars_ct:
        parents = vars_ct.remove(v)
        LS[v][frozenset(parents)] = score(ct2cft(ct, v))

    # Recursively call get_local_scores
    if len(vars_ct) > 1: # Until only one variable remains
        for v in evars:
            get_local_scores(ct2ct(ct,v), [range(1, v-1)])

def get_local_scores_from_file(file_path: str):
    """Reads local scores from a file and populates the LS dictionary.
    Alternative to computing local scores using data."""
    scores = read_local_scores(file_path)
    return scores

# Step 2: For each variable, find the best parent set and its score

def get_best_parents(V:set, v:str, LS:Dict[Tuple[str, frozenset], float]):
    """Having calculated the local scores, finding the best
    parents for a variable v from a set C can be done recursively. The best parents in C for v are either the whole
    candidate set C itself or the best parents for v from
    one of the smaller candidate sets {C \ {c} | c ∈ C}."""

    # bps/bss indexed by frozenset of parents
    bps: Dict[FrozenSet[Any], FrozenSet[Any]] = {} # A map from a candidate set C to a subset of C that is the best parents for v from C
    bss: Dict[FrozenSet[Any], float] = {} # A map from a candidate set C to the score of the best parents for v from C

    cand = [x for x in V if x != v]

    # Iterate over all subsets of cand 
    # Ex. {}, {0}, {1}, {0,1}, {2}, {0,2}, {1,2}, {0,1,2}
    for r in range(len(cand)+1):
        for cs in combinations(cand, r):
            fr_cs = frozenset(cs)
            bps[fr_cs] = fr_cs
            bss[fr_cs] = LS.get((v, fr_cs), float('-inf')) # set -inf if this value is not in LS
            for c in cs:
                c1 = fr_cs - {c}
                if bss[c1] > bss[fr_cs]:
                    bss[fr_cs] = bss[c1]
                    bps[fr_cs] = bps[c1]

    return bps


def get_best_sinks(V, bps, LS):
    pass

def sinks_2_ord(V, sinks):
    pass

def ord_2_net(V, ord, bps):
    pass


# Initially call algorithm 1 with the ct for all variables and the whole variable set V as evars
# V = list(range(data.shape[1]))
# ct_full = ct(data)
# get_local_scores(ct_full, V)

# We use the local scores read from file instead of computing them from data
LS = get_local_scores_from_file("local_scores/local_scores_alarm_100.jaa")
print(LS)

bps = get_best_parents(range(5), 0, LS)
print("\n\n")
print(bps)

# bps_choice, bps_score = get_best_parents(V, LS)
# sink_star, S = get_best_sinks(V, bps_score)
# order = sinks_2_ord(V, sink_star)

# parents, edges = ord_2_net(V, order, bps_choice)

# print("Optimal order:", order)
# print("Edges:", sorted(edges))
# print("Parents:", {v: sorted(list(U)) for v, U in parents.items()})
# print("Best total score:", S[frozenset(V)])