"""
Implementation of the algorithm presented in:
Silander, T. and Myllymaki, P., 2012. A simple approach for finding the globally optimal Bayesian network structure. 
arXiv preprint arXiv:1206.6875.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from typing import List


# Load iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Discretize continuous variables into 5 bins
disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df_discrete = df.copy()
df_discrete[df.columns[:-1]] = disc.fit_transform(df[df.columns[:-1]])

# Rename column names to feature index
df_discrete.columns = list(range(1, df_discrete.shape[1])) + ['target']

print(df_discrete.head(20))

"""Helper functions"""
def ct(W: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a contingency table from the data W.
    Returns a DataFrame where the index are value combinations and
    the column 'count' holds frequencies.
    """
    counts = W.value_counts().reset_index()
    counts.columns = list(W.columns) + ["counts"]
    return counts


W = df_discrete.drop("target", axis=1)
full_ct = ct(W)
print(full_ct.head(20))

def vars(ct: pd.DataFrame) -> list:
    """Returns the set of variables in the contingency table ct"""
    return ct.columns.drop('counts').tolist()



variables = vars(full_ct)
print(variables)

def ct2ct(ct:pd.DataFrame, v:int)-> pd.DataFrame:
    """Produces a contigency table by marginalizing the variable v out of ct"""
    keep_cols = [c for c in ct.columns if c not in (v, 'counts')]
    out = ct.groupby(keep_cols, as_index=False, sort=False)["counts"].sum()
    return out

marginalized = ct2ct(full_ct, 1)
print(marginalized.head(5))

def ct2cft(ct:pd.DataFrame, v:int)->pd.DataFrame:
    """Yields a conditional frequency table"""
    parent_cols = [col for col in ct.columns if col not in [v, 'counts']]
    return ct.groupby(parent_cols + [v], as_index=False, sort=False)['counts'].sum()


cft = ct2cft(full_ct, 1)
print(cft.head(10))

def score(cft):
    """Calculates the local score based on the conditional frequency table"""
    pass


"""Main functions"""
LS = np.array()
def get_local_scores(ct:pd.DataFrame, evars:List[int]):
    """The main procedure, GetLocalScores, (Algorithm 1)
    is called with a contingency table ct and the variables
    evars to be marginalized from it. Initially, it is called
    with a contingency table for all the variables and the
    whole variable set V as evars. The algorithm is simply
    a depth first traversal of smaller and smaller contingency tables."""
    vars_ct = vars(ct)
    for v in vars_ct:
        LS[v][vars_ct.remove(v)] = score(ct2cft(ct, v))

    # Recursively call get_local_scores
    if len(vars_ct) > 1:
        for v in evars:
            get_local_scores(ct2ct(ct,v), [range(1, v-1)])

def get_best_parents(V, v, LS):
    pass

def get_best_sinks(V, bps, LS):
    pass

def sinks_2_ord(V, sinks):
    pass

def ord_2_net(V, ord, bps):
    pass


# evars starts as "all variables"; canonical elimination uses string/numeric ordering
# LS = get_local_scores(full_ct, evars=V, alpha=1.0)

# bps_choice, bps_score = get_best_parents(V, LS)
# sink_star, S = get_best_sinks(V, bps_score)
# order = sinks_2_ord(V, sink_star)

# parents, edges = ord_2_net(V, order, bps_choice)

# print("Optimal order:", order)
# print("Edges:", sorted(edges))
# print("Parents:", {v: sorted(list(U)) for v, U in parents.items()})
# print("Best total score:", S[frozenset(V)])