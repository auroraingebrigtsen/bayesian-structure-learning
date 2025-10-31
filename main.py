from write_local_scores import write_local_scores
from silander_myllymaki import get_optimal_network
from partial_order_approach import partial_order_approach
import os

DATA_PATH = "pygobnilp/data/asia_10000.dat"
SCORES_PATH = "local_scores/local_scores_asia_10000.jaa"

def main():
    if not os.path.exists(SCORES_PATH):
        write_local_scores(DATA_PATH, SCORES_PATH)

    get_optimal_network(SCORES_PATH)
    partial_order_approach(SCORES_PATH, m=3, p=2)


if __name__ == "__main__":
    main()
