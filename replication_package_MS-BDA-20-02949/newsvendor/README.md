# Official SOF Newsvendor Subset

This directory contains only the two official Stochastic Optimization Forests
newsvendor files needed by the public replication scripts:

- `nv_tree_utilities.py`
- `tree.py`

The original project directory included additional experiments, data, notebooks,
and generated outputs that are not required for this report. The loader in
`src/experiment_suite.py` imports this subset directly and provides lightweight
stubs for optional `mkl` and `gurobipy` imports.
