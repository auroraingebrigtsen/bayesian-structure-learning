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
m: int = 4  # size of each bucket order
p: int = 2  # number of disjoint bucket orders
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
        # include x
        stack.append((included | {x}, remaining - {x}))
        # exclude x
        stack.append((included, remaining - {x}))

    return sorted(ideals, key=lambda s: (len(s), sorted(s)))

def maximal_in_Y(Y: FrozenSet[str], pred: Dict[str, Set[str]]) -> Set[str]:
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

def get_tail(A: Set[str], B: Set[str]) -> List[FrozenSet[str]]:
    """
    Function to get the tail of a set Y.
    The tail is the interval [Y_hat, Y], which is all subsets of Y that contain all maximal elements of Y.
    """
    diff = B - A
    subsets = chain.from_iterable(combinations(diff, r) for r in range(len(diff) + 1))
    return [frozenset(A | set(s)) for s in subsets]

def algorithm1(
    M: Set[str],
    P: Set[Edge],
    LS: Dict[str, Dict[FrozenSet[str], float]],
    F_downclosed: Dict[str, Set[FrozenSet[str]]]
):
    """
    Implementation of Algorithm 1 from the paper.
    """
    pred = predecessors(M, P)
    ideals = get_ideals(M, pred)

    fhat: Dict[Tuple[str, FrozenSet[str]], float] = {}
    g: Dict[FrozenSet[str], float] = {frozenset(): 0.0}

    g_bp: Dict[FrozenSet[str], Tuple[str, FrozenSet[str]]] = {}
    fhat_bp: Dict[Tuple[str, FrozenSet[str]], Tuple[str, FrozenSet[str]]] = {}

    empty = frozenset()
    for v in M:
        fhat[(v, empty)] = LS[v].get(empty, float('-inf'))

    for Y in ideals[1:]:
        Ymax = maximal_in_Y(Y, pred)

        # Phase 2: g_P(Y)
        best = float('-inf'); best_choice = None
        for v in Ymax:
            Ym = frozenset(set(Y) - {v})
            cand = g[Ym] + fhat[(v, Ym)]
            if cand > best:
                best = cand
                best_choice = (v, Ym)
        g[Y] = best
        g_bp[Y] = best_choice

        # Phase 1: fhat[v, Y] for all v
        A = Ymax
        tail_sets = get_tail(A, set(Y))  # same for all v at this Y
        for v in M:
            best_val = float('-inf'); best_src = None
            Fv = F_downclosed[v]
            # tail ∩ Fv
            for Z in tail_sets:
                if Z in Fv:
                    sc = LS[v].get(Z, float('-inf'))
                    if sc > best_val:
                        best_val = sc; best_src = ('tail', Z)
            # recurrence via removing a maximal u
            for u in A:
                Ym_u = frozenset(set(Y) - {u})
                val2 = fhat[(v, Ym_u)]
                if val2 > best_val:
                    best_val = val2; best_src = ('prev', Ym_u)
            fhat[(v, Y)] = best_val
            fhat_bp[(v, Y)] = best_src

    return g[frozenset(M)], {'g_bp': g_bp, 'fhat_bp': fhat_bp}


# loop over all partial orders, and keep the best
best_score = float('-inf')
best_P = None
best_bp = None

for P in generate_partial_orders(blocks, front_choices_per_block):
    score, bp = algorithm1(M, P, LS, F_downclosed)
    if score > best_score:
        best_score, best_P, best_bp = score, P, bp

print("Best score:", best_score)
print("Best partial order edges:", best_P)
print("Best back pointers:", best_bp)  # optional


def reconstruct_parent_map(M: Iterable[str],
                           g_bp: Dict[FrozenSet[str], Tuple[str, FrozenSet[str]]],
                           fhat_bp: Dict[Tuple[str, FrozenSet[str]], Tuple[str, FrozenSet[str]]]
                          ) -> Dict[str, List[str]]:
    M = list(M)
    parents: Dict[str, List[str]] = {v: [] for v in M}

    Y = frozenset(M)
    while Y:
        v, Ym = g_bp[Y]

        # Walk fhat backpointers until we hit the chosen tail set (the actual parent set)
        state = Ym
        visited = set()
        while True:
            key = (v, state)
            if key not in fhat_bp:
                # No record (e.g., state == ∅ and only base score set) → parents = []
                Z = frozenset()
                break
            tag, info = fhat_bp[key]
            if tag == 'tail':
                Z = info           # this is the chosen parent set for v
                break
            elif tag == 'prev':
                if state in visited:
                    raise RuntimeError("Cycle in fhat_bp")
                visited.add(state)
                state = info       # keep walking
            else:
                raise RuntimeError(f"Unknown fhat tag: {tag}")

        parents[v] = sorted(Z)
        Y = Ym

    return parents

def print_parent_map(M: Iterable[str],
                     bp: Dict[str, Dict]) -> None:
    pm = reconstruct_parent_map(M, bp['g_bp'], bp['fhat_bp'])
    for v in M:   # keep your chosen order
        print(f"{v}: {pm[v]!r}")

print_parent_map(M, best_bp)