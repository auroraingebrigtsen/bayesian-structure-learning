# bayesian-structure-learning

Implementations of different BN structure learning algorithms.

## Getting Started
The project requires that you have [`uv`](https://docs.astral.sh/uv/) installed.

Create a virtual environment and install requirements:

```bash
uv sync --all-extras --all-packages --all-groups
source .venv/bin/activate
```


## Project Structure

This project **does not** implement score computation directly — instead we rely on [`pygobnilp`](https://bitbucket.org/jamescussens/pygobnilp/src/master/) to calculate local scores.

### Data and Scores
- We use the example datasets provided by `pygobnilp`, available in the folder: `pygobnilp/data/`


- The algorithms in this project assume that **local scores** for your dataset are already available.  
These must be stored as **Jaakkola local-scores files** inside a folder. See the section [Interpreting Jaakkola local-scores file](#interpreting-jaakkola-local-scores-file-jaa-files) for details.

- To generate these files, run: `write_local_scores.py` pointing it to the dataset you want to process. This will create the `local_scores/` folder (if it does not exist) and write the `.jaa` file for you.

### Implementation
- `silander_myllymaki.py` contains an implementation of the algorithm described in:  

    > **Silander, T. & Myllymäki, P. (2012)**  
    > *A simple approach for finding the globally optimal Bayesian network structure*.  
    > [arXiv:1206.6875](https://arxiv.org/abs/1206.6875)



## Interpreting Jaakkola local-scores file (.jaa files)

A `.jaa` file stores **local scores** for Bayesian network structure learning. It lists, for each variable, the score of each **parent set**. 


1. **First line**: a single integer — the **number of variables** in the dataset.
2. Then **one block per variable**:
   - **Header line**:  
     ```
     <VariableName> <K>
     ```
     where `K` is the number of parent sets listed for this variable. K is the number of candidate parent sets that survive the limits and pruning specified. To edit this, refer to the arguments of Gobnilp in `write_local_scores.py` file.
   - **K lines**, one per parent set:  
     ```
     <score> <pcount> [<Parent1> <Parent2> ...]
     ```
