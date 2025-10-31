"""
Implementation of the algorithm presented in:
Parviainen, P. and Koivisto, M., 2013. Finding optimal Bayesian networks using precedence constraints. 
The Journal of Machine Learning Research, 14(1), pp.1387-1415
"""

from itertools import combinations, product
from math import ceil
from typing import List, Dict, Tuple, FrozenSet, Iterable, Set
from pygobnilp.gobnilp import read_local_scores

Edge = Tuple[str, str]

def downwards_closed(LS: Dict[str, Dict[FrozenSet[str], float]]
) -> Dict[str, Dict[FrozenSet[str], float]]:
    """
    The algorithm assumes that the local scores are downwards closed. That means that if a parent set Z has a defined score for v,
    then all subsets of Z also have a defined score for v.
    This function ensures that the local scores satisfy this property by adding -inf scores for missing subsets.
    """
    closed = {}
    for v, scored_parent_sets in LS.items():
        base = dict(scored_parent_sets)
        all_ps = set()
        for ps in scored_parent_sets.keys():
            for r in range(len(ps) + 1):
                for subset in combinations(ps, r):
                    all_ps.add(frozenset(subset))
        all_ps.add(frozenset())
        for ps in all_ps:
            if ps not in base:
                base[ps] = float('-inf')
        closed[v] = base
    return closed

def make_blocks_and_fronts(
    V: List[str], m: int, p: int
) -> Tuple[List[Set[str]], List[List[Set[str]]], List[str]]:
    """
    Create p disjoint blocks of size m from V, the free block (remaining variables),
    and the list of possible front subsets (of size a) for each block.

    Returns (blocks, front_choices_per_block, free_block).
    """
    a = ceil(m / 2)  # front bucket size
    blocks: List[Set[str]] = [set(V[i * m : (i + 1) * m]) for i in range(p)]
    free_block = V[p * m :]
    front_choices_per_block: List[List[Set[str]]] = [
        [set(F) for F in combinations(block, a)] for block in blocks
    ]
    return blocks, front_choices_per_block, free_block

def generate_partial_orders(
    blocks: List[Set[str]],
    front_choices_per_block: List[List[Set[str]]]
) -> Iterable[Set[Edge]]:
    """
    Generates all partial orders that you get by the two bucket scheme.
    """
    for choice in product(*front_choices_per_block):  # one front per block
        edges = set()
        for front, block in zip(choice, blocks):
            back = block - front
            edges |= {(u, v) for u in front for v in back}
        yield edges

def predecessors(M: Set[str], P: Set[Edge]) -> Dict[str, Set[str]]:
    """
    Function to compute the predecessors of each element in M.
    A predecessor of v is any u with (u,v) in P.

    Returns a map from each element of M to its set of predecessors.
    """
    pred = {u: set() for u in M}
    for u, v in P:
        if u != v:
            pred[v].add(u)
    return pred

def get_ideals(M: Set[str], pred: Dict[str, Set[str]]) -> List[FrozenSet[str]]:
    """
    Function to get all ideals of the partial order defined by pred.
    An ideal is a subset Y ⊆ M such that for every y ∈ Y, all predecessors of y are also in Y.
    """
    ideals: List[FrozenSet[str]] = []
    seen: Set[FrozenSet[str]] = set()

    def backtrack(included: FrozenSet[str]):
        if included in seen:   # avoid revisiting the same ideal via different addition orders
            return
        seen.add(included)
        ideals.append(included)

        # elements that can be added next (deterministic order)
        available = [x for x in sorted(M - included) if pred[x] <= included]
        for x in available:
            backtrack(included | frozenset({x}))

    backtrack(frozenset())
    ideals.sort(key=lambda s: (len(s), tuple(sorted(s))))
    return ideals

def get_maximal(Y: FrozenSet[str], pred: Dict[str, Set[str]]) -> Set[str]:
    """
    Function to get the maximal elements of Y ⊆ M w.r.t. the partial order defined by pred.
    The maximal elements are those that have no successors in Y.
    """
    maximal: Set[str] = set()

    for y in Y:
        # Check if y appears as a predecessor of any other node in Y
        has_successor_in_Y = any(y in pred[z] for z in Y if z != y)
        if not has_successor_in_Y:
            maximal.add(y)

    return maximal

def algorithm1(
    M: Set[str],
    P: Set[Edge],
    LS: Dict[str, Dict[FrozenSet[str], float]],
):
    """
    Implementation of Algorithm 1.
    """
    pred = predecessors(M, P)
    ideals = get_ideals(M, pred)

    bss: Dict[str, Dict[FrozenSet[str], float]] = {v: {} for v in M}  # best local score for v when parents must lie inside Y
    bps: Dict[str, Dict[FrozenSet[str], FrozenSet[str]]] = {v: {} for v in M}  # chosen parent set for v at ideal Y
    g_p: Dict[FrozenSet[str], float] = {}  # best total score over DAGs on Y
    prev: Dict[FrozenSet[str], FrozenSet[str]] = {}  # predecessor ideal (Y without the chosen sink)

    empty = frozenset()
    g_p[empty] = 0.0
    prev[empty] = empty

    # base local scores at the empty ideal
    for v in M:
        bss[v][empty] = LS[v].get(empty, float('-inf'))
        bps[v][empty] = empty

    # for each non-empty Y ∈ I(P) 
    for Y in ideals[1:]:  
        Ymax = get_maximal(Y, pred)

        # 3a: choose sink v ∈ Ymax
        best_score = float('-inf')
        best_choice = None
        for v in Ymax:  # the sink must be maximal in Y
            Y_minus_v = Y - {v} 
            score = g_p[Y_minus_v] + bss[v][Y_minus_v]  # best rest + best local for v seen from Y\{v}
            if score > best_score:
                best_score = score
                best_choice = Y_minus_v
        g_p[Y] = best_score  # best score for ideal Y
        prev[Y] = best_choice  #

        # 3b: local DP over tail for each v ∈ Y
        for v in M: 
            best_bss = float('-inf')
            best_parents = empty

            # consider all parent sets Z that lie within the tail of Y, which is the interval [Ŷ, Y], where Ŷ = YMax
            LB = frozenset(Ymax - {v})   #  lower bound, must include all maximal elements except v
            UB = frozenset(Y - {v})   # upper bound, must be a subset of Y (excluding v)

            # Check scored parent sets that fall in [LB, UB]
            for Z, score in LS[v].items():  # all scored parent sets for v
                if LB.issubset(Z) and Z.issubset(UB):  # intersection with the tail interval
                    if score > best_bss:
                        best_bss = score
                        best_parents = Z

            # inherit from smaller parent sets Y\{u}, u ∈ Ymax
            for u in Ymax:
                Y_minus_u = Y - {u}
                cand = bss[v].get(Y_minus_u, float('-inf'))
                if cand > best_bss:
                    best_bss = cand
                    best_parents = bps[v][Y_minus_u]

            bss[v][Y] = best_bss
            bps[v][Y] = best_parents if best_parents != empty else empty

    return g_p[frozenset(M)], {
        'P': P, 'ideals': ideals, 'ss': g_p, 'prev': prev, 'bss': bss, 'bps': bps,
}


def reconstruct_parent_map(
    V: List[str],
    prev: Dict[FrozenSet[str], FrozenSet[str]],
    bps: Dict[str, Dict[FrozenSet[str], FrozenSet[str]]],
):
    """
    Finds the parent set for each variable in the optimal network found by iterating Algorithm 1.
    """
    parents: Dict[str, FrozenSet[str]] = {v: frozenset() for v in V}

    Y = frozenset(V)
    while Y:
        Ym = prev[Y]
        if Ym is None:
            break
        added = set(Y) - set(Ym)
        if not added:
            break
        v = next(iter(added))
        Z = bps[v][Ym]
        parents[v] = Z
        Y = Ym
    return parents


def partial_order_approach(local_scores_path: str, m: int = 3, p: int = 2):
    """ Main function to run the partial order approach for Bayesian network structure learning.
    Implements the two-bucket partial order scheme.

    local_scores_path: Path to the local scores file in JAA format.
    m: Size of each bucket order.
    p: Number of disjoint bucket orders. 
    returns: A parent map representing the optimal Bayesian network structure found.
    """

    LS_raw= read_local_scores(local_scores_path)
    LS = downwards_closed(LS_raw)

    V: List[str] = list(LS.keys())
    M: Set[str] = set(V) 
    n = len(V)

    assert p * m <= n
    assert m >= 2 and p >= 1

    print("Variables:", V)
    print("Number of variables:", n)

    blocks, front_choices_per_block, free_block = make_blocks_and_fronts(V, m, p)
    print("Blocks:", blocks)
    print("Free block:", free_block)

    best_score = float('-inf')
    best_run = None

    for P in generate_partial_orders(blocks, front_choices_per_block):
        score, run = algorithm1(M, P, LS)
        if score > best_score:
            best_score = score
            best_run = run

    pm = reconstruct_parent_map(
        V,
        best_run['prev'],
        best_run['bps'],
    )

    print("\nOptimal Parent Map:")
    for v in V:
        print(f"{v}: {pm[v]}")