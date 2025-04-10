# Seminar 3: System Stability Analysis

## Overview

This seminar focuses on analyzing the equilibrium types and stability regions in dynamic systems. We explore different types of equilibrium points, their characteristics, and how to classify them based on eigenvalue analysis.

## Key Topics

### Equilibrium Classification
- Identification of 8 distinct equilibrium types
- Analysis of eigenvalues and eigenvectors
- Classification criteria for different stability regions

### Stability Regions
- Visualization of stability boundaries in parameter space
- Simulation of system behavior within different stability regions
- Analysis of transitions between stability regions

### Trajectory Analysis
- Simulation of system trajectories around equilibrium points
- Visualization techniques for phase portraits
- Time-domain analysis of system response

## Implementation Files

The seminar includes several Python files that work together to analyze system stability:

- `sem_3_ans.ipynb`: Main Jupyter notebook with comprehensive analysis
- `phase.py`: Core module for system dynamics and visualization
- `analyze.py`: Script for analyzing different equilibrium regions
- `check_k.py`: Script for parameter validation
- `plot_stability_regions.py`: Visualization tool for stability regions
- `collage_generator.py`: Creates composite images of different equilibrium types

## Equilibrium Types

The code analyzes 8 different equilibrium types:

1. **Stable Focus (Region 1)**
   - Damped oscillations
   - Eigenvalues: Complex with negative real part
   - Condition: k1 < 1, k2 < 0, k2² < 4(1-k1)

2. **Stable Node (Region 2)**
   - Aperiodic damping
   - Eigenvalues: Real and negative
   - Condition: k1 < 1, k2 < 0, k2² > 4(1-k1)

3. **Critical Damping (Region 3)**
   - Boundary between stable focus and stable node
   - Eigenvalues: Repeated real negative values
   - Condition: k1 < 1, k2 < 0, k2² = 4(1-k1)

4. **Neutral Stability/Center (Region 4)**
   - Undamped oscillations
   - Eigenvalues: Pure imaginary (no real part)
   - Condition: k1 < 1, k2 = 0

5. **Unstable Focus (Region 5)**
   - Growing oscillations
   - Eigenvalues: Complex with positive real part
   - Condition: k1 < 1, k2 > 0

6. **Saddle Point (Region 6)**
   - Unstable equilibrium with stable and unstable manifolds
   - Eigenvalues: Real with opposite signs
   - Condition: k1 > 1

7. **Saddle-Node (Region 7)**
   - Special boundary case
   - Eigenvalues: One zero, one negative
   - Condition: k1 = 1, k2 < 0

8. **Degenerate Case (Region 8)**
   - One eigenvalue is zero
   - Eigenvalues: One zero, one non-negative
   - Condition: k1 = 1, k2 ≥ 0

## System Dynamics

The system is represented in matrix form as:

$$\dot{\mathbf{x}} = A\mathbf{x}$$

where A is the system matrix:

$$A = \begin{pmatrix} 0 & 1 \\ -1 + k_1 & k_2 \end{pmatrix}$$

Parameters k₁ and k₂ determine the system's behavior and stability properties.

## Resources

- Each region has a corresponding visualization file (`region_X_simulation.png`)
- The `equilibrium_types_collage.png` provides a composite view of all stability regions
- The Python implementation allows for interactive exploration of different parameter values 