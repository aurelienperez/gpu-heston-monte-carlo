# GPU-Accelerated Monte Carlo Simulation of the Heston Model

This repository provides a GPU-accelerated implementation of Monte Carlo methods for pricing European options under the **Heston stochastic volatility model**, developed as part of the Master's course *"Massive Parallel Programming on GPU Devices for Big Data"* from the Probabilités et Finance Master's program at Sorbonne Université.

## Project Overview

The project includes implementations of three discretisation schemes:

- **Euler** scheme 
- **Exact scheme** (Broadie-Kaya)
- **Almost Exact scheme** (Haastrecht-Pelsser)

Each scheme leverages **CUDA** for high-performance parallel computations. Trajectories are simulated independently at thread-level, with efficient shared-memory reductions performed at the block level.
