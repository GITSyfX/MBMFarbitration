# Tutorials for computation models of goal-directed versus huabitual control system
## TOC
1. arbitration 
2. hierarchical control 
3. successor representation 

------------------------------------------------------------------------
## Task Environment

The task environment implemented in this repository is based on:

**Lee et al., 2014 -- Two-stage decision task**

In addition to the original task structure, the repository also includes
a modified version of the task environment.

Both implementations can be found in:

    utils/env.py

Available environments include:

-   `two_stage` --- modified version with explicit transition structure
-   `two_stage_2014` --- implementation closer to the original Lee et
    al. (2014) task

These environments define the **state space, action space, transition
dynamics, and reward structure** used by the computational models.

------------------------------------------------------------------------
## Project Status

This project is **currently under active development**.

Some models are still in the validation and testing stage. Before
using them for research or analysis, please make sure to:

-   run preliminary simulations
-   verify that the outputs behave as expected
-   confirm that the implementation matches the intended computational
    formulation


