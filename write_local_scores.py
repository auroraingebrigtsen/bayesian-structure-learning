"""Script to write local scores to a file using pygobnilp."""

import os
from pygobnilp.gobnilp import Gobnilp

DATA_PATH = "pygobnilp/data/alarm_100.dat"
WRITE_PATH = "local_scores/local_scores_alarm_100.jaa"

os.makedirs(os.path.dirname(WRITE_PATH), exist_ok=True)

g = Gobnilp()
g.learn(
    data_source=DATA_PATH,
    data_type="discrete",
    score="DiscreteBIC",
    palim=10, # parent limit
    end="local scores" # end after computing local scores
)

g.write_local_scores(WRITE_PATH)