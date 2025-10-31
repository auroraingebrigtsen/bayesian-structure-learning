# bn-structure-learning

Implementations of different BN structure learning algorithms.

## Getting Started
The project requires that you have [`uv`](https://docs.astral.sh/uv/) installed.

Create a virtual environment: 
```bash
uv venv
```

Install requirements:

```bash
uv sync --all-extras --all-packages --all-groups
source .venv/bin/activate
```

Run a file by the uv command:
```bash
uv run <file_name>
```

## Project Structure

This project **does not** implement score computation directly — instead we rely on [`pygobnilp`](https://bitbucket.org/jamescussens/pygobnilp/src/master/) to calculate local scores.

### Data and Scores
- The algorithms in this project assume that **local scores** for your dataset are already available.  
These must be stored as **Jaakkola local-scores files** inside a folder. See the section [Interpreting Jaakkola local-scores file](#interpreting-jaakkola-local-scores-file-jaa-files) for details.

- To generate these files, run: `write_local_scores.py` pointing it to the dataset you want to process. This will create the `local_scores/` folder (if it does not exist) and write the `.jaa` file for you. Alternatively, this step can be included in the main loop.

- The data itself can either be retrieved from the pygobnilp subrepo, available in the folder: `pygobnilp/data/`. or custom datasets can be placed in the `data/`-folder. 

- If you want to generate your own data from a given bayesian network into  the `data/`-folder, use the  `sample_data.py` script. This assumes a `.bif`-file specifying the network exists in the `networks/`-folder.

### Implementation
- `silander_myllymaki.py` contains an implementation of the algorithm described in:  

    > **Silander, T. & Myllymäki, P. (2012)**  
    > *A simple approach for finding the globally optimal Bayesian network structure*.  
    > [arXiv:1206.6875](https://arxiv.org/abs/1206.6875)

- `partial_order_approach.py` contains an implementation of the algorithm described in:  

    > **Parviainen, P. & Koivisto, M. (2013)**  
    > *Finding optimal Bayesian networks using precedence constraints.*  
    > *Journal of Machine Learning Research*, **14**(1), 1387–1415.  
    > [JMLR Paper](https://www.jmlr.org/papers/v14/parviainen13a.html)

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
