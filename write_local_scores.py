"""Script to write local scores to a file using pygobnilp."""

import os
from pygobnilp.gobnilp import Gobnilp


def write_local_scores(dat_path: str, write_path: str):
    """Write local scores to a file using pygobnilp."""
    os.makedirs(os.path.dirname(write_path), exist_ok=True)
    g = Gobnilp()
    
    g.learn(
        data_source=dat_path,
        data_type="discrete",
        score="DiscreteBIC",
        end="local scores"  # end after computing local scores
    )
    
    g.write_local_scores(write_path)

write_local_scores("data/insurance_10000.dat", "local_scores/local_scores_insurance_10000.jaa")