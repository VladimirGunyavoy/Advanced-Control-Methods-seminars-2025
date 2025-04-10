# Seminar 2: Point Motion Simulation and Linear Transformations

## Overview

This seminar focuses on the fundamentals of point motion simulation and linear transformations. We explore how to model and analyze dynamical systems, particularly those involving circular motion and transformation matrices.

## Key Topics

### Circular Motion of a Point
- Simulation of point trajectories with orthogonal velocity vectors
- Analysis of circular motion parameters (radius, angular velocity, period)
- Implementation of circular trajectories using vector operations

### Linear Transformations
- Rotation matrices and their application to motion trajectories
- Scaling transformations and their effect on system dynamics
- Translation operations in phase space
- Composition of multiple transformations

### System Stability
- Analysis of stability conditions for point motion
- Evaluation of system robustness under perturbations
- Visualization of stable and unstable trajectories

## Implementations

The primary implementation is available in the `sem_2_ans.ipynb` Jupyter notebook, which contains:

- Code for simulating point motion with orthogonal velocity
- Visualizations of trajectories in phase space
- Interactive elements for exploring parameter changes
- Mathematical explanations of the underlying principles

## Mathematics

The core mathematical concepts include:

1. **Orthogonal Velocity Condition**:
   - $\mathbf{v}_i = \mathbf{r}_i^{\perp}$ (rotation by 90°)
   - Implemented using rotation matrix: $\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$

2. **Position Update Equation**:
   - $\mathbf{r}_{i+1} = \mathbf{r}_i + \mathbf{v}_i \cdot dt$

3. **Linear Transformation**:
   - $\mathbf{x}' = A\mathbf{x}$ where $A$ is a transformation matrix

## Resources

- The Jupyter notebook contains all necessary code and explanations
- Additional resources and references are provided within the notebook
- Visualization tools help illustrate key concepts 