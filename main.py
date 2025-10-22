from write_local_scores import write_local_scores
from silander_myllymaki import get_optimal_network

DATA_PATH = "data/insurance_10000.dat"
SCORES_PATH = "local_scores/local_scores_insurance_10000.jaa"

def main():
    write_local_scores(DATA_PATH, SCORES_PATH)
    get_optimal_network(SCORES_PATH)


if __name__ == "__main__":
    main()
