"""Script to write local scores to a file using pygobnilp."""

import os
from pygobnilp.gobnilp import Gobnilp
import networkx as nx

DATA_PATH = "pygobnilp/data/asia_10000.dat"
WRITE_PATH = "local_scores/local_scores_asia_10000.jaa"

os.makedirs(os.path.dirname(WRITE_PATH), exist_ok=True)

g = Gobnilp()
g.learn(
    data_source=DATA_PATH,
    data_type="discrete",
    score="DiscreteBIC",
)


parents = {node: set() for node in g.learned_bn.nodes}

# Fill parent sets from edges
for parent, child in g.learned_bn.edges:
    parents[child].add(parent)

# Print nicely
for node, ps in parents.items():
    print(f"{node}: {sorted(ps)}")