"""
Control System Base Classes

This module provides base classes for implementing various control systems.
These classes serve as a foundation for specific control implementations
in the seminars.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class ControlSystem(ABC):
    """
    Abstract base class for control systems.
    
    This class defines the interface for all control system implementations.
    Specific control methods (bang-bang, PID, etc.) should inherit from this class.
    """
    
    def __init__(self, name="Generic Control System"):
        """
        Initialize the control system.
        
        Parameters:
        -----------
        name : str, optional
            Name of the control system (default: "Generic Control System")
        """
        self.name = name
        self.simulation_results = None
    
    @abstractmethod
    def control_law(self, t, state, *args):
        """
        Calculate the control input based on time and state.
        
        Parameters:
        -----------
        t : float
            Current time
        state : array_like
            Current state vector
        args : tuple
            Additional arguments
            
        Returns:
        --------
        array_like
            Control input vector
        """
        pass
    
    @abstractmethod
    def system_dynamics(self, t, state, u, *args):
        """
        Calculate the state derivatives based on current state and control input.
        
        This is the right-hand side of the ODE system.
        
        Parameters:
        -----------
        t : float
            Current time
        state : array_like
            Current state vector
        u : array_like
            Control input vector
        args : tuple
            Additional arguments
            
        Returns:
        --------
        array_like
            State derivative vector
        """
        pass
    
    def simulate(self, initial_state, t_span, dt=0.01, method='euler', *args):
        """
        Simulate the system from initial state over a time span.
        
        Parameters:
        -----------
        initial_state : array_like
            Initial state vector
        t_span : tuple
            (t_start, t_end) tuple defining simulation timespan
        dt : float, optional
            Time step for simulation (default: 0.01)
        method : str, optional
            Integration method: 'euler' or 'rk4' (default: 'euler')
        args : tuple
            Additional arguments for the system dynamics
            
        Returns:
        --------
        tuple
            (t_array, state_array, control_array) containing the simulation results
        """
        # Prepare time array
        t_start, t_end = t_span
        t_array = np.arange(t_start, t_end + dt, dt)
        n_steps = len(t_array)
        
        # Convert initial state to numpy array
        initial_state = np.asarray(initial_state)
        state_dim = len(initial_state)
        
        # Initialize arrays
        state_array = np.zeros((n_steps, state_dim))
        state_array[0] = initial_state
        
        # Determine control dimensionality by calling the control law once
        u0 = self.control_law(t_start, initial_state, *args)
        control_dim = len(np.atleast_1d(u0))
        control_array = np.zeros((n_steps, control_dim))
        control_array[0] = u0
        
        # Simulation loop
        for i in range(1, n_steps):
            t_current = t_array[i-1]
            state_current = state_array[i-1]
            
            # Calculate control input
            u = self.control_law(t_current, state_current, *args)
            control_array[i-1] = u
            
            # Update state based on integration method
            if method.lower() == 'euler':
                # Simple Euler integration
                state_derivative = self.system_dynamics(t_current, state_current, u, *args)
                state_array[i] = state_current + dt * state_derivative
            elif method.lower() == 'rk4':
                # Runge-Kutta 4th order
                k1 = self.system_dynamics(t_current, state_current, u, *args)
                
                # Calculate k2
                t_mid1 = t_current + dt/2
                state_mid1 = state_current + dt * k1 / 2
                u_mid1 = self.control_law(t_mid1, state_mid1, *args)
                k2 = self.system_dynamics(t_mid1, state_mid1, u_mid1, *args)
                
                # Calculate k3
                t_mid2 = t_current + dt/2
                state_mid2 = state_current + dt * k2 / 2
                u_mid2 = self.control_law(t_mid2, state_mid2, *args)
                k3 = self.system_dynamics(t_mid2, state_mid2, u_mid2, *args)
                
                # Calculate k4
                t_next = t_current + dt
                state_next = state_current + dt * k3
                u_next = self.control_law(t_next, state_next, *args)
                k4 = self.system_dynamics(t_next, state_next, u_next, *args)
                
                # Update state using weighted average
                state_array[i] = state_current + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            else:
                raise ValueError(f"Unknown integration method: {method}")
        
        # Calculate final control input
        control_array[-1] = self.control_law(t_array[-1], state_array[-1], *args)
        
        # Store results
        self.simulation_results = (t_array, state_array, control_array)
        
        return self.simulation_results
    
    def plot_state_vs_time(self, state_indices=None, state_labels=None, title=None, save_path=None):
        """
        Plot the state variables against time.
        
        Parameters:
        -----------
        state_indices : list, optional
            Indices of state variables to plot (default: all)
        state_labels : list, optional
            Labels for state variables (default: "State i")
        title : str, optional
            Plot title (default: "{system_name} State vs Time")
        save_path : str, optional
            Path to save the figure (default: None)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available. Run simulate() first.")
        
        t_array, state_array, _ = self.simulation_results
        
        # Set default state indices if none provided
        if state_indices is None:
            state_indices = range(state_array.shape[1])
        
        # Set default state labels if none provided
        if state_labels is None:
            state_labels = [f"State {i}" for i in state_indices]
        
        # Set default title if none provided
        if title is None:
            title = f"{self.name} State vs Time"
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each state variable
        for i, idx in enumerate(state_indices):
            ax.plot(t_array, state_array[:, idx], label=state_labels[i], linewidth=2)
        
        # Add labels and title
        ax.set_xlabel("Time", fontweight='bold')
        ax.set_ylabel("State Value", fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save if path provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_control_vs_time(self, control_indices=None, control_labels=None, title=None, save_path=None):
        """
        Plot the control inputs against time.
        
        Parameters:
        -----------
        control_indices : list, optional
            Indices of control inputs to plot (default: all)
        control_labels : list, optional
            Labels for control inputs (default: "Control i")
        title : str, optional
            Plot title (default: "{system_name} Control vs Time")
        save_path : str, optional
            Path to save the figure (default: None)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available. Run simulate() first.")
        
        t_array, _, control_array = self.simulation_results
        
        # Set default control indices if none provided
        if control_indices is None:
            control_indices = range(control_array.shape[1])
        
        # Set default control labels if none provided
        if control_labels is None:
            control_labels = [f"Control {i}" for i in control_indices]
        
        # Set default title if none provided
        if title is None:
            title = f"{self.name} Control vs Time"
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each control input
        for i, idx in enumerate(control_indices):
            ax.plot(t_array, control_array[:, idx], label=control_labels[i], linewidth=2)
        
        # Add labels and title
        ax.set_xlabel("Time", fontweight='bold')
        ax.set_ylabel("Control Value", fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save if path provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_phase_portrait(self, x_idx=0, y_idx=1, title=None, x_label=None, y_label=None, save_path=None):
        """
        Plot the phase portrait of the system.
        
        Parameters:
        -----------
        x_idx : int, optional
            Index of state variable for x-axis (default: 0)
        y_idx : int, optional
            Index of state variable for y-axis (default: 1)
        title : str, optional
            Plot title (default: "{system_name} Phase Portrait")
        x_label : str, optional
            Label for x-axis (default: "State {x_idx}")
        y_label : str, optional
            Label for y-axis (default: "State {y_idx}")
        save_path : str, optional
            Path to save the figure (default: None)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available. Run simulate() first.")
        
        _, state_array, _ = self.simulation_results
        
        # Set default labels if none provided
        if x_label is None:
            x_label = f"State {x_idx}"
        if y_label is None:
            y_label = f"State {y_idx}"
        
        # Set default title if none provided
        if title is None:
            title = f"{self.name} Phase Portrait"
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the phase trajectory
        ax.plot(state_array[:, x_idx], state_array[:, y_idx], linewidth=2)
        
        # Mark the initial and final points
        ax.scatter(state_array[0, x_idx], state_array[0, y_idx], color='blue', s=100, 
                  marker='o', label='Initial State', zorder=10)
        ax.scatter(state_array[-1, x_idx], state_array[-1, y_idx], color='red', s=100, 
                  marker='s', label='Final State', zorder=10)
        
        # Add labels and title
        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add zero axes if they're within the plot range
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if x_min <= 0 <= x_max:
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        if y_min <= 0 <= y_max:
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        
        # Save if path provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class BangBangControlSystem(ControlSystem):
    """
    Implementation of a bang-bang control system.
    
    Bang-bang control applies maximum control effort in one direction,
    then switches to maximum control effort in the opposite direction
    at a specific switching time.
    """
    
    def __init__(self, max_control=1.0, name="Bang-Bang Control System"):
        """
        Initialize the bang-bang control system.
        
        Parameters:
        -----------
        max_control : float, optional
            Maximum control magnitude (default: 1.0)
        name : str, optional
            Name of the control system (default: "Bang-Bang Control System")
        """
        super().__init__(name)
        self.max_control = max_control
        self.switching_time = None
        self.final_time = None
    
    def calculate_parameters(self, initial_state):
        """
        Calculate bang-bang control parameters for the given initial state.
        
        This method must be implemented by subclasses to calculate:
        - The switching time (when to reverse control)
        - The final time (when target is reached)
        - Any other parameters needed for the specific system
        
        Parameters:
        -----------
        initial_state : array_like
            Initial state vector
            
        Returns:
        --------
        dict
            Dictionary of control parameters
        """
        raise NotImplementedError("Subclasses must implement calculate_parameters()")
    
    def control_law(self, t, state, *args):
        """
        Calculate bang-bang control input at time t.
        
        Parameters:
        -----------
        t : float
            Current time
        state : array_like
            Current state vector
        args : tuple
            Additional arguments
            
        Returns:
        --------
        float
            Control input (acceleration)
        """
        # This is a template implementation that should be overridden
        if self.switching_time is None:
            params = self.calculate_parameters(state)
            self.switching_time = params.get('switching_time', 0)
            self.final_time = params.get('final_time', 0)
        
        # Simple bang-bang control law:
        if t < self.switching_time:
            return self.max_control
        elif t < self.final_time:
            return -self.max_control
        else:
            return 0.0
    
    def system_dynamics(self, t, state, u, *args):
        """
        Define the system dynamics.
        
        Parameters:
        -----------
        t : float
            Current time
        state : array_like
            Current state vector
        u : float
            Control input
        args : tuple
            Additional arguments
            
        Returns:
        --------
        array_like
            State derivative vector
        """
        # This is a template implementation that should be overridden
        # For a double integrator: ẍ = u
        # State: [x, ẋ]
        # Dynamics: [ẋ, u]
        return np.array([state[1], u]) 