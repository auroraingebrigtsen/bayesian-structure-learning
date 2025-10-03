"""
Implementation of the algorithm presented in:
Silander, T. and Myllymaki, P., 2012. A simple approach for finding the globally optimal Bayesian network structure. 
arXiv preprint arXiv:1206.6875.
"""

import pandas as pd
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Tuple, FrozenSet, Any, Iterable, Optional
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
LS: Dict[str, Dict[FrozenSet, float]] = {}

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

# Step 2: For each variable, find the best parent set and its score

def get_best_parents(V:List[str], v:str, LS:Dict[str, Dict[FrozenSet[str], float]]):
    """Having calculated the local scores, finding the best
    parents for a variable v from a set C can be done recursively. The best parents in C for v are either the whole
    candidate set C itself or the best parents for v from
    one of the smaller candidate sets {C \ {c} | c ∈ C}."""

    V = tuple(sorted(V))  # ensures lexicographic subset order
    # DP tables
    bps: Dict[FrozenSet[str], FrozenSet[str]] = {} # A map from a candidate set C to a subset of C that is the best parents for v from C
    bss: Dict[FrozenSet[str], float] = {} # A map from a candidate set C to the score of the best parents for v from C

    # To avoid blow-up we only consider parents that appear in LS, if we use no pruning for LS then this is the full set of variables
    possible_parents = LS.get(v, set())
    if not possible_parents:
        return {}

    candidates = set()
    for css, score in possible_parents.items():
        for cs in css:
            candidates.add(cs)

    print("candidates for", v, ":", candidates)

    def score(C: FrozenSet[str]) -> float:
        return LS[v].get(C, float("-inf"))

    # Base case
    bps[frozenset()] = frozenset()
    bss[frozenset()] = score(frozenset())

    # Iterate over all subsets of cand 
    # {}         -> 000
    # {A}        -> 100
    # {B}        -> 010
    # {A,B}      -> 110
    # {C}        -> 001
    # {A,C}      -> 101
    # {B,C}      -> 011
    # {A,B,C}    -> 111
    for r in range(1, len(candidates) + 1): # size of the candidate set
        for cs in combinations(candidates, r): # all candidate sets of size r
            C = frozenset(cs)

            # Option 1: take C itself
            best_set = C
            best_score = score(C)

            # Option 2: best of proper subsets by removing one element
            for c in C:
                c1 = C - {c}
                # c1 already computed because we go size-increasing
                if bss[c1] > best_score:

                    best_score = bss[c1]
                    best_set = bps[c1]

            bps[C] = best_set
            bss[C] = best_score

    return bps


def get_best_sinks(
    V: Iterable[str], 
    bps: Dict[str, Dict[FrozenSet[str], FrozenSet[str]]],
    LS: Dict[str, Dict[FrozenSet[str], float]]):
    """
    Implementes algorithm 3: GetBestSinks
    """

    V = tuple(sorted(V))  # so subsets come in lexicographic order
    sinks: Dict[FrozenSet[str], str] = {}
    scores: Dict[FrozenSet[str], float] = {}

    # Base case
    scores[frozenset()] = 0.0
    sinks[frozenset()]  = -1  # no sink

    # process subsets in size-then-lex order
    for r in range(0, len(V) + 1):
        for w in combinations(V, r):
            W = frozenset(w)

            scores[W] = 0.0
            sinks[W]  = -1  # no sink

            # for all sink ∈ W
            for sink in W:
                # since we did pruning when computing LS, we may have a sink that has no entry in LS or bps
                if sink not in bps or sink not in LS:
                    continue

                upvars  = W - {sink}  # W \ {sink}
               #parents = bps[sink][upvars]  # best parents for 'sink' from 'upvars'
                bps_sink = bps[sink]
                parents = bps_sink.get(upvars, frozenset())
                total = scores[upvars] + LS[sink].get(parents, float('-inf'))
    
                # if total > scores[W] then update
                if total > scores[W]:
                    scores[W] = total
                    sinks[W]  = sink
            


    return sinks

def sinks_2_ord(V: List[str], sinks: Dict[FrozenSet[str], str]) -> Dict[str, int]:
    """Implementation of algoritm 4: Sinks2Ord"""
    order = np.zeros(len(V), dtype=str)
    left = set(V)
    for i in reversed(range(len(V))):
        order[i] = sinks[frozenset(left)]
        left.remove(order[i])
    return order

def ord_2_net(V:List[str], order:np.array, bps:Dict[str, Dict[FrozenSet[str], FrozenSet[str]]]):
    parents = np.zeros((len(V),), dtype=FrozenSet)
    predecs = set() # predecessors in the order
    for i in range(1, len(V) + 1):
        parents[i] = bps[order[i]][frozenset(predecs)]
        predecs.add(order[i])
    return parents


# Initially call algorithm 1 with the ct for all variables and the whole variable set V as evars
#V = data.columns
# ct_full = ct(data)
# get_local_scores(ct_full, V)

# We use the local scores read from file instead of computing them from data
LS = read_local_scores("local_scores/local_scores_alarm_100.jaa")
V = list(LS.keys())

print("V:", V)
print(LS)
print("\n\n")

# For each variable, find the best parent set
bps_all: Dict[str, Dict[FrozenSet[str], FrozenSet[str]]] = {}
for v in V:
    bps_all[v] = get_best_parents(V, v, LS)

# Step 3: Find the best sink for each subset of variables, and the best total score
sinks = get_best_sinks(V, bps_all, LS)

# Step 4: Extract the optimal order from the best sinks
order = sinks_2_ord(V, sinks)
print("Optimal order:", order)

# Step 5: Extract the optimal network from the optimal order
network = ord_2_net(V, order, bps_all)
print("Optimal network (parents for each variable):")
for i in range(len(V)):
    print(f"  {V[i]}: {network[i]}")