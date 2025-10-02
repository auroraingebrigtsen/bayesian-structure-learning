# bayesian-structure-learning

Implementations of different BN structure learning algorithms.


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