"""
Implementation of the algorithm presented in:
Silander, T. and Myllymaki, P., 2012. A simple approach for finding the globally optimal Bayesian network structure. 
arXiv preprint arXiv:1206.6875.
"""

from itertools import combinations
from typing import List, Dict, Tuple, FrozenSet, Any, Iterable, Optional
from pygobnilp.gobnilp import read_local_scores


def get_best_parents(V:List[str], v:str, LS:Dict[str, Dict[FrozenSet[str], float]]):
    """Implementation of algorithm 2: GetBestParents"
    
    V: List of all variable names
    v: The variable for which we want to find the best parents
    LS: Local scores, a map from variable name to scores for parent sets (may be pruned)
    returns: A map from candidate parent sets to the best parents from that set for v
    """

    # DP tables
    bps: Dict[FrozenSet[str], FrozenSet[str]] = {} # A map from a candidate set C to a subset of C that is the best parents for v from C
    bss: Dict[FrozenSet[str], float] = {} # A map from a candidate set C to the score of the best parents for v from C

    # We only need to consider variables that appear in the local scores for v
    supp_v = set()
    for ps in LS[v].keys():  # parent sets that have a defined score for v
        supp_v.update(ps)

    candidates = tuple(sorted(supp_v))  # so subsets come in lexicographic order

    # Base case
    bps[frozenset()] = frozenset()
    bss[frozenset()] = LS[v].get(frozenset(), float("-inf"))

    # Iterate over all candidate sets in lexicographic order 
    for r in range(1, len(candidates) + 1): # size of the candidate set
        for cs in combinations(candidates, r): # all candidate sets of size r
            C = frozenset(cs)

            # Option 1: take C itself
            best_set = C
            best_score = LS[v].get(C, float('-inf'))

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
    V: List of all variable names
    bps: Map from variable name to map from candidate parent sets to best parents from that set
    LS: Local scores, a map from variable name to scores for parent sets (may be pruned)
    returns: A map from variable subsets to their best sink (str)
    """

    V = tuple(sorted(V))  # so subsets come in lexicographic order

    # Since the local scores may be pruned, we need to find the supported sets for each variable
    support: Dict[str, set] = {} # map from variable to the set of variables that appear in its local scores
    for child, cs in bps.items():  # variable, {parent set: best parents}
        possible_parents = set()
        for parents in cs.keys():
            possible_parents.update(parents) # add all variables that appear in any parent set
        support[child] = possible_parents

    sinks = {frozenset(): None}
    scores = {frozenset(): 0.0}

    # process subsets in size-then-lex order
    for r in range(1, len(V) + 1):
        for w in combinations(V, r):
            W = frozenset(w)

            best_sink: Optional[str] = None
            best_score = float('-inf')

            # for all sink ∈ W
            for sink in W:
                upvars  = W - {sink}  # W \ {sink}
                # Only keep parents the child can actually have (per pruned LS)
                upvars_v = frozenset(x for x in upvars if x in support.get(sink, set()))

                parents =  bps[sink].get(upvars_v, frozenset()) 
                total = scores[upvars] + LS[sink].get(parents, float('-inf'))

    
                # if total > scores[W] then update
                if total > best_score:
                    best_score = total
                    best_sink = sink
            
            scores[W] = best_score
            sinks[W] = best_sink
            
    return sinks

def sinks_2_ord(V: List[str], sinks: Dict[FrozenSet[str], str]) -> List[str]:
    """Implementation of algoritm 4: Sinks2Ord
    
    V: List of all variable names
    sinks: Map from variable subsets to their best sink
    returns: List of variable names in an optimal order
    """
    order = [None] * len(V)
    left = set(V)

    # iterate backwards over the order
    for i in reversed(range(len(V))):
        order[i] = sinks[frozenset(left)]  # best sink of the remaining variables
        left.remove(order[i])
    return order

def ord_2_net(V:List[str], order:List[str], bps:Dict[str, Dict[FrozenSet[str], FrozenSet[str]]]):
    """Implementation of algorithm 5: Ord2Net
    
    V: List of all variable names
    order: List of variable names in an optimal order
    bps: Map from variable name to map from candidate parent sets to best parents from that
    returns: List of parent sets, where parents[i] is the parent set for order[i]
    """
    parents: List[Set[str]] = [set() for _ in range(len(V))]  # Each entry is the parents of order[i]
    predecs = set()  # Variables that come before the current variable in the order

    # We iterate over the variables in the order, the first one has no predecessors
    for i, child in enumerate(order):

        # find all parents that appear in candidate sets for the current variable
        supported_parents = set().union(*bps[child].keys())

        # only keep predecessors that can be parents of child
        possible_parents = predecs & supported_parents

        # find the best parents from the possible predecessors
        parents[i] = bps[child][frozenset(possible_parents)] 

        # add current variable to predecessors, this one must come before all next ones
        predecs.add(child) 

    return parents



def get_optimal_network(path:str):
    """Compute the optimal network using the Silander-Myllymaki algorithm."""
    
    # Step 1: Compute local scores for all (variable, parent set)-pairs
    LS = read_local_scores(path)
    V = list(LS.keys())

    print("Variables:", V)

    #  Step 2: For each variable, find the best parent set and its score
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
        print(f"  {order[i]}: {network[i]}")


get_optimal_network("local_scores/local_scores_asia_10000.jaa")