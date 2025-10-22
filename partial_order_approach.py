"""
Implementation of the algorithm presented in:
Parviainen, P. and Koivisto, M., 2013. Finding optimal Bayesian networks using precedence constraints. 
The Journal of Machine Learning Research, 14(1), pp.1387-1415
"""

from itertools import combinations, product, chain
from math import ceil, floor
from typing import List, Dict, Tuple, FrozenSet, Any, Iterable, Optional, Set
from pygobnilp.gobnilp import read_local_scores

m: int = 4  # size of each bucket order
p: int = 2  # number of disjoint bucket orders

LS = read_local_scores("local_scores/local_scores_asia_10000.jaa")
V = list(LS.keys())
n = len(V)

print("Variables:", V)
print("Number of variables:", n)

# each bucket order has size m, and there are p such orders, so total variables p * m cannot exceed n
assert p*m <= n
assert m >= 2, "Bucket size must be at least 2"
assert p >= 1, "There must be at least one bucket order"


bucket_size = m // 2 # size of each bucket in the partial order
f = n % bucket_size  # size of the final bucket if not full

### BUCKET ORDER SCHEME ###
a = ceil(m/2)           # front size
b = floor(m/2)          # back size (implicit)

# Partition V into p blocks of size m + a free block
blocks = [V[i*m:(i+1)*m] for i in range(p)]
free_block = V[p*m:] # remaining variables
print("Blocks:", blocks)
print("Free block:", free_block)

# find all possible front choices for each block
front_choices_per_block = [
    [set(F) for F in combinations(block, a)] for block in blocks
]

# iterate over all possible partial orders
all_pos: List[Set[Tuple[str, str]]] = []
for choice_tuple in product(*front_choices_per_block):
    edges: Set[Tuple[str, str]] = set()
    for block, front in zip(blocks, choice_tuple):
        back = set(block) - set(front)
        edges.update((u, v) for u in front for v in back)
    all_pos.append(edges)


def generate_partial_orders(
    blocks: List[Set[str]],
    front_choices_per_block: List[List[Set[str]]]
) -> Iterable[Set[Tuple[str, str]]]:
    """Yield one partial order (edge set) at a time."""
    per_block_pairs: List[List[Tuple[Set[str], Set[str]]]] = [
        [(front, set(block) - set(front)) for front in fronts]
        for block, fronts in zip(blocks, front_choices_per_block)
    ]
    for choice in product(*per_block_pairs):  # one (front, back) per block
        edges: Set[Tuple[str, str]] = set()
        for front, back in choice:
            edges.update((u, v) for u in front for v in back)
        yield edges


# def algorithm1(P:Set[Tuple[str,str]]):
#     ideals = get_ideals(P)

#     g: Dict[set, float] = {}
#     g[frozenset()] = 0.0

#     bps

#     for v in V:

#     bps_all: Dict[str, Dict[FrozenSet[str], FrozenSet[str]]] = {}
#     for v in V:
#         bps_all[v] = get_best_parents(V, v, LS)
#     return bps_all



Score = float
LS = Dict[str, Dict[FrozenSet[str], Score]]  # local scores f_v(Z)
Edge = Tuple[str, str]

# ---------- helpers for a fixed P ----------
def predecessors(U: Set[str], edges: Set[Edge]) -> Dict[str, Set[str]]:
    pred = {u: set() for u in U}
    for u, v in edges:
        pred[v].add(u)
        pred.setdefault(u, set())
    return pred

def ideals_of_P(U: Set[str], pred: Dict[str, Set[str]]) -> List[FrozenSet[str]]:
    # simple backtrack to enumerate all ideals
    U = set(U)
    ideals: List[FrozenSet[str]] = []
    def backtrack(included: Set[str], remaining: Set[str]):
        # minimal/available = nodes whose preds ⊆ included
        avail = {x for x in remaining if pred[x] <= included}
        if not avail:
            ideals.append(frozenset(included))
            return
        x = next(iter(avail))
        # include x
        backtrack(included | {x}, remaining - {x})
        # exclude x -> exclude everything ≥ x
        # compute upward closure quickly by BFS on implicit edges
        # since P is only 2-bucket per block, succ can be precomputed similarly if needed:
        # here fall back to a safe but simple approach:
        backtrack(included, remaining - {x})
    backtrack(set(), U)
    ideals = sorted(set(ideals), key=lambda s: (len(s), sorted(s)))
    return ideals

def maximal_in_Y(Y: FrozenSet[str], pred: Dict[str, Set[str]]) -> Set[str]:
    # u is maximal in Y if no y in Y has u as a predecessor (i.e., there is no y with u < y in Y)
    # Equivalent: ∄ y∈Y\{u} such that u ∈ pred[y]
    Ys = set(Y)
    maxes = set(Ys)
    for y in Ys:
        for u in pred[y]:
            if u in Ys:
                maxes.discard(u)
    return maxes

def subsets_between(A: Set[str], B: Set[str]) -> List[FrozenSet[str]]:
    # all Z with A ⊆ Z ⊆ B  (A,B ⊆ U, A subset B)
    base = list(B - A)
    out = []
    for mask in range(1 << len(base)):
        Z = set(A)
        for i, e in enumerate(base):
            if (mask >> i) & 1:
                Z.add(e)
        out.append(frozenset(Z))
    return out

# ---------- Algorithm 1 for a fixed P ----------
def algo1_for_P(
    U: Set[str],
    edges: Set[Edge],
    LS: LS,                           # local scores f_v(Z)
    F_downclosed: Dict[str, Set[FrozenSet[str]]]  # allowed parent sets (downward-closed)
) -> Tuple[Score, Dict]:  # returns g_P(N) and backpointers
    pred = predecessors(U, edges)
    ideals = ideals_of_P(U, pred)  # sorted by size asc
    ideal_index = {Y: i for i, Y in enumerate(ideals)}

    # storage only on ideals
    fhat: Dict[Tuple[str, FrozenSet[str]], Score] = {}
    g: Dict[FrozenSet[str], Score] = {frozenset(): 0.0}

    # backpointers
    g_bp: Dict[FrozenSet[str], Tuple[str, FrozenSet[str]]] = {}         # Y -> (v, Y\{v})
    fhat_bp: Dict[Tuple[str, FrozenSet[str]], Tuple[str, FrozenSet[str]]] = {}  # (v,Y)-> ('tail'|'prev', Z or Y\{u})

    # init fhat[v, ∅] = f_v(∅)
    empty = frozenset()
    for v in U:
        fhat[(v, empty)] = LS[v].get(empty, float('-inf'))

    for Y in ideals[1:]:  # skip empty
        Ymax = maximal_in_Y(Y, pred)           # Y̌
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
        for v in U:
            # tail candidate
            tail_sets = subsets_between(A, set(Y))
            best_val = float('-inf'); best_src = None
            # Look only among tail ∩ F_v
            Fv = F_downclosed[v]
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

    return g[frozenset(U)], {'g_bp': g_bp, 'fhat_bp': fhat_bp}


best_score = float('-inf')
best_P = None
best_bp = None

for edges in generate_partial_orders(blocks, front_choices_per_block):  # your enumerator
    score, bp = algo1_for_P(U, edges, LS, F_downclosed)
    if score > best_score:
        best_score, best_P, best_bp = score, edges, bp

# best_score = global optimum over the POS
# best_P tells you which partial order attained it
# best_bp lets you reconstruct the DAG

print("Best score:", best_score)
print("Best partial order edges:", best_P)
print("Best back pointers:", best_bp)