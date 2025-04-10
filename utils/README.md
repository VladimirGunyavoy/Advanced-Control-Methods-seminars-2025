# Utility Modules for Advanced Control Methods

This directory contains common utility modules that are used across different seminars in the Advanced Control Methods course. These modules provide reusable functionality for simulation, analysis, and visualization of control systems.

## Available Modules

### `simulation_utils.py`

This module provides common utilities for simulation and analysis that can be shared across different seminars, including:

- **Numerical Integration Methods**: 
  - `euler_integrate()`: First-order Euler integration for ODEs
  - `rk4_integrate()`: Fourth-order Runge-Kutta integration for improved accuracy

- **Visualization Tools**:
  - `plot_phase_portrait()`: Generate phase portraits from trajectories

- **Stability Analysis**:
  - `compute_eigenvalues()`: Calculate eigenvalues of system matrices
  - `classify_equilibrium()`: Classify equilibrium points based on eigenvalues
  - `jacobian()`: Compute Jacobian matrices for nonlinear systems

### `control_system.py`

This module provides base classes for implementing various control systems:

- **ControlSystem (ABC)**:
  - Abstract base class that defines the interface for all control system implementations
  - Includes methods for simulation, visualization, and analysis
  - Provides plotting methods for state variables, control inputs, and phase portraits

- **BangBangControlSystem**:
  - Implementation of bang-bang control for time-optimal control
  - Handles switching logic and parameter calculation
  - Provides templates for system dynamics

## Usage

To use these utilities in a seminar, import the required modules:

```python
# Import simulation utilities
from utils.simulation_utils import rk4_integrate, plot_phase_portrait, classify_equilibrium

# Import control system base classes
from utils.control_system import ControlSystem, BangBangControlSystem

# Example: Create a custom control system by inheriting from the base class
class MyControlSystem(ControlSystem):
    def control_law(self, t, state, *args):
        # Implement control law
        return control_input
        
    def system_dynamics(self, t, state, u, *args):
        # Implement system dynamics
        return state_derivative
```

## Extending the Utilities

To add new utility functions or classes:

1. Choose the appropriate module or create a new one if needed
2. Add your implementation with proper docstrings
3. Update this README with information about the new functionality
4. If creating a new module, ensure it follows the same documentation and code style

## Design Principles

The utility modules follow these design principles:

- **Modularity**: Each function and class has a single responsibility
- **Reusability**: Code can be used across different seminars
- **Documentation**: Comprehensive docstrings explain usage and parameters
- **Type Hinting**: Function parameters and return values are clearly specified
- **Flexibility**: Parameters allow customization for different use cases 