# tests/test_against_gobnilp.py
import sys
from pathlib import Path
import pytest

# --- Make repo root importable
ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(ROOT))

from silander_myllymaki import get_optimal_network

from pygobnilp.gobnilp import Gobnilp
import pygobnilp  # for robust data path

# Robust paths
ASIA_DAT =  "pygobnilp/data/asia_10000.dat"
LOCAL_SCORES =  "local_scores/local_scores_asia_10000.jaa"

@pytest.mark.slow
def test_edges_match_gobnilp_reference():
    """
    Compare the learned edge set to Gobnilp's own output on the same data.
    """

    # --- My implementation ---
    cust_parents = get_optimal_network(path=str(LOCAL_SCORES))

    # --- Gobnilp reference ---
    g = Gobnilp()
    g.learn(
        data_source=str(ASIA_DAT),
        data_type="discrete",
        score="DiscreteBIC",
    )
    gob_parents = {node: set() for node in g.learned_bn.nodes}

    # Fill parent sets from edges
    for parent, child in g.learned_bn.edges:
        gob_parents[child].add(parent)

    # --- Compare ---
    assert cust_parents == gob_parents, "Custom parent sets do not match Gobnilp's parent sets!"
