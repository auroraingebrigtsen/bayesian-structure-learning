"""
Implementation of the algorithm presented in:
Parviainen, P. and Koivisto, M., 2013. Finding optimal Bayesian networks using precedence constraints. 
The Journal of Machine Learning Research, 14(1), pp.1387-1415
"""

from itertools import combinations, product, chain
from math import ceil, floor
from typing import List, Dict, Tuple, FrozenSet, Iterable, Set
from pygobnilp.gobnilp import read_local_scores
from collections import defaultdict

Edge = Tuple[str, str]

LS= read_local_scores("local_scores/local_scores_asia_10000.jaa")
V: List[str] = list(LS.keys())
M: Set[str] = set(V) 
n = len(V)

print("Variables:", V)
print("Number of variables:", n)

# BUCKET ORDER SCHEME PARAMETERS
m: int = 8  # size of each bucket order
p: int = 1  # number of disjoint bucket orders
assert p * m <= n
assert m >= 2 and p >= 1

a = ceil(m / 2)  # front bucket size

blocks: List[Set[str]] = [set(V[i*m:(i+1)*m]) for i in range(p)]
free_block = V[p*m:]  # remaining variables 
print("Blocks:", blocks)
print("Free block:", free_block)

# all combinations of each block for fronts
front_choices_per_block: List[List[Set[str]]] = [
    [set(F) for F in combinations(block, a)] for block in blocks
]

def downward_closure(LS: Dict[str, Dict[FrozenSet[str], float]]
                    ) -> Dict[str, Set[FrozenSet[str]]]:
    """
    The local scores may only contain some parent sets for each variable v, because of pruning.
    Algorithm 1 requires that for every allowed parent set Z all its subsets are also 
    considered valid (this property is called downward-closed). 

    This function ensures that: for each variable v, if Z appears in LS[v], then every subset S ⊆ Z is
    also included.
    """
    f: Dict[str, Set[FrozenSet[str]]] = {}
    for v, score_map in LS.items():
        fv: Set[FrozenSet[str]] = set()
        for Z in score_map:  # each allowed parent set
            fv.update(
                map(frozenset,
                    chain.from_iterable(combinations(Z, r) for r in range(len(Z) + 1)))
            )
        f[v] = fv
    return f

F_downclosed = downward_closure(LS) 


def generate_partial_orders(
    blocks: List[Set[str]],
    front_choices_per_block: List[List[Set[str]]]
) -> Iterable[Set[Edge]]:
    """
    Generates all partial orders that you get by the two bucket scheme.
    """
    per_block_pairs: List[List[Tuple[Set[str], Set[str]]]] = []

    # for each block and its possible fronts
    for block, fronts in zip(blocks, front_choices_per_block):
        pairs_for_block: List[Tuple[Set[str], Set[str]]] = []
        
        # for each possible choice of front in this block
        for front in fronts:
            back = block - front # the elements not in the front
            pairs_for_block.append((front, back))
    
    per_block_pairs.append(pairs_for_block)
    for choice in product(*per_block_pairs):  # one (front, back) per block
        edges: Set[Edge] = set()
        for front, back in choice:
            edges.update((u, v) for u in front for v in back)
        yield edges

def predecessors(M: Set[str], P: Set[Edge]) -> Dict[str, Set[str]]:
    """
    Function to compute the predecessors of each element in M.
    A predecessor of v is any u with (u,v) in P.

    Returns a map from each element of M to its set of predecessors.
    """
    pred: Dict[str, Set[str]] = {u: set() for u in M}
    for u, v in P:
        if u == v:
            continue
        pred.setdefault(u, set())
        pred.setdefault(v, set()).add(u)
    return pred

def get_ideals(M: Set[str], pred: Dict[str, Set[str]]) -> List[FrozenSet[str]]:
    ideals: Set[FrozenSet[str]] = set()
    stack = [(frozenset(), frozenset(M))]  # (included, remaining)

    while stack:
        included, remaining = stack.pop()
        ideals.add(included)

        available = {x for x in remaining if pred[x] <= included}
        if not available:
            continue

        x = next(iter(available))
        stack.append((included | {x}, remaining - {x}))  # including x
        stack.append((included, remaining - {x}))  # excluding x

    return sorted(ideals, key=lambda s: (len(s), sorted(s)))

def get_maximal(Y: FrozenSet[str], pred: Dict[str, Set[str]]) -> Set[str]:
    """
    Function to get the maximal elements of Y ⊆ M w.r.t. the partial order defined by pred.
    The maximal elements are those that have no successors in Y.
    """
    Ys = set(Y)
    maxes = set(Ys)
    for y in Ys:
        for u in pred[y]:
            if u in Ys:
                maxes.discard(u)
    return maxes

def get_tail(Y: Set[str], Y_hat: Set[str]) -> List[FrozenSet[str]]:
    """
    Function to get the tail of a set Y.
    The tail is the interval [Y_hat, Y], which is all subsets of Y that contain all maximal elements of Y.
    """
    tail_sets: List[FrozenSet[str]] = []

    remaining = Y - Y_hat

    # for every possible subset of the remaining elements
    for r in range(len(remaining) + 1):
        #  for all combinations of size r
        for subset in combinations(remaining, r):
            # combine Y_hat with this subset , since Y_hat must be included
            new_set = frozenset(Y_hat.union(subset))
            tail_sets.append(new_set)

    return tail_sets

def algorithm1(
    M: Set[str],
    P: Set[Edge],
    LS: Dict[str, Dict[FrozenSet[str], float]],
    F_downclosed: Dict[str, Set[FrozenSet[str]]]
):
    """
    Implementation of Algorithm 1.
    """
    pred = predecessors(M, P)
    ideals = get_ideals(M, pred)

    bss: Dict[str, Dict[FrozenSet[str], float]] = {v: {} for v in M}  # best local score for v at ideal Y
    bps: Dict[str, Dict[FrozenSet[str], FrozenSet[str]]] = {v: {} for v in M}  # chosen parent set for v at ideal Y
    ss: Dict[FrozenSet[str], float] = {}  # best total score over DAGs on Y
    prev: Dict[FrozenSet[str], FrozenSet[str]] = {}  # predecessor ideal that yields ss[Y] 

    empty = frozenset()
    ss[empty] = 0.0
    prev[empty] = empty

    for v in M:
        bss[v][empty] = LS[v].get(empty, float('-inf'))
        bps[v][empty] = empty

    # find the best sinks for each ideal
    for Y in ideals[1:]:  # skip the empty set
        Ymax = get_maximal(Y, pred)  # the possible sinks for Y
        best_score = float('-inf')  # the highest total score we can achieve for this ideal Y after deciding on its sink
        best_choice = None  # the smaller ideal (Y without that best sink)
        # which is the "previous state" in the dynamic program that led to this best_score
        for v in Ymax:
            Y_minus_v = frozenset(set(Y) - {v})  # previous ideal = nodes that can be parents of v
            score = ss[Y_minus_v] + bss[v][Y_minus_v]  # best rest + best local for v seen from Y\{v}
            if score > best_score:
                best_score = score
                best_choice = Y_minus_v
        ss[Y] = best_score  # best score for ideal Y
        prev[Y] = best_choice  # store the predecessor ideal (Y\{v*}) that achieved ss[Y].
        # during backtracking, the chosen sink is v* = (Y - prev[Y]).pop(),
        # and its parents are bps[v*][prev[Y]].

        # find the best parents (local dynamic programming step)
        # For each variable v, find its best parent set within the current ideal 
        tail_sets = get_tail(Ymax, set(Y))  # all subsets of Y that include every sink in Y
        for v in M:
            best_bss = float('-inf')
            best_parents = None

            # 1. check if the best parent set is in the tail sets
            Fv = F_downclosed[v]  # valid parent sets for v
            for Z in tail_sets:  # loop over all candidate parent sets
                if Z in Fv:  # only consider valid parent sets
                    local_score_Z = LS[v].get(Z, float('-inf'))  # local score for v with parents Z
                    if local_score_Z > best_bss:
                        best_bss = local_score_Z
                        best_parents = Z

            # 2. check if the best parent set is inherited from a smaller ideal (increasing the network size doesnt improve the score)
            for u in Ymax:
                Y_minus_u = frozenset(set(Y) - {u})
                inherited_score = bss[v][Y_minus_u]  # best local score for v inherited from a smaller ideal (Y without one of its sinks)
                if inherited_score > best_bss:
                    best_bss = inherited_score
                    best_parents = bps[v][Y_minus_u]

            bss[v][Y] = best_bss
            bps[v][Y] = best_parents if best_parents is not None else empty

    return ss[frozenset(M)], {
        'po': po,
        'ideals': ideals,
        'ss': ss,
        'prev': prev,
        'bss': bss,
        'bps': bps,
    }


# ---------- Loop over partial orders and keep the best ----------
best_score = float('-inf')
best_run = None

for po in generate_partial_orders(blocks, front_choices_per_block):
    score, run = algorithm1(M, po, LS, F_downclosed)
    if score > best_score:
        best_score = score
        best_run = run

print("Best score:", best_score)
print("Best partial order edges (po):", best_run['po'])

def reconstruct_parent_map(
    V: List[str],
    ideals: List[FrozenSet[str]],
    ss: Dict[FrozenSet[str], float],
    prev: Dict[FrozenSet[str], FrozenSet[str]],
    bss: Dict[str, Dict[FrozenSet[str], float]],
    bps: Dict[str, Dict[FrozenSet[str], FrozenSet[str]]],
):
    """
    Finds the parent set for each variable in the optimal network found by iterating Algorithm 1.
    """
    parents: Dict[str, List[str]] = {v: [] for v in V}

    Y = frozenset(V)
    # Work backwards through ss/prev to get the sink order and chosen v’s
    while Y:
        Ym = prev[Y]
        if Ym is None:
            break
        added = set(Y) - set(Ym)
        if not added:
            break
        # pick the variable v that was added last
        v = next(iter(added))
        # The parent set for v is what bps[v] had at Ym (where v was still outside)
        Z = bps[v][Ym]
        parents[v] = Z
        Y = Ym
    return parents

pm = reconstruct_parent_map(
    V,
    best_run['ideals'],
    best_run['ss'],
    best_run['prev'],
    best_run['bss'],
    best_run['bps'],
)

for v in V:
    print(f"{v}: {pm[v]}")