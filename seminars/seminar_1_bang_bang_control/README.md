# Seminar 1: Bang-Bang Control

This seminar focuses on time-optimal control systems using bang-bang control, where the control input switches between extreme values to achieve optimal performance.

## Overview

Bang-bang control is a type of optimal control that switches abruptly between extreme states. In control theory, it's often the time-optimal solution for many systems, especially when there are constraints on the control input.

## Contents

- `plot_generator.py`: Python script for generating phase portraits of bang-bang control trajectories
- `examples/`: Directory containing example problems and solutions
- `images/`: Directory containing generated images and phase portraits

## Key Concepts

- **Bang-Bang Principle**: Control inputs are always at their minimum or maximum values
- **Switching Curve**: The curve in phase space where the control switches between extremes
- **Time-Optimal Control**: Achieving the desired state in minimum time

## Visualization

Here's an animation showing the full bang-bang control solution:

![Bang-Bang Control Solution](images/full_solution.gif)

This animation demonstrates how the control input switches between extremes at the optimal switching time, bringing the system to the target state in minimal time.

## Mathematical Foundation

For a second-order system with position `p` and velocity `v`, the bang-bang control aims to bring the system to `(0,0)` in minimum time. The control parameters are calculated as follows:

1. The parameter `k` determines the region in phase space:
   ```
   k = +1 if v²·sign(v) >= -2·a_m·p
   k = -1 otherwise
   ```

2. The switching time (ts) and final time (tf) are calculated based on initial conditions.

## Usage

The scripts in this directory can be used to:
1. Calculate bang-bang control parameters for a given initial state
2. Simulate the system evolution under bang-bang control
3. Visualize trajectories in the phase plane
4. Analyze switching curves and time-optimal paths

## Examples

See the `examples/` directory for practical problems and their solutions, including:
- Speed control with different starting conditions
- Speed control with different start and finish points
- Acceleration control with varying constraints

## References

- Kirk, D. E. (2004). Optimal Control Theory: An Introduction. Dover Publications.
- Athans, M., & Falb, P. L. (2006). Optimal Control: An Introduction to the Theory and Its Applications. Dover Publications. 