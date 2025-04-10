"""
Simulation Utilities

This module provides common utilities for simulation and analysis
that can be shared across different seminars.

The utilities include:
- Numerical integration methods
- Phase portrait generation
- Stability analysis helpers
- Data visualization tools
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global plotting style
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)

def euler_integrate(func, y0, t_span, dt=0.01, args=()):
    """
    Perform Euler integration for an ODE.
    
    Parameters:
    -----------
    func : callable
        The right-hand side of the ODE system dy/dt = f(t, y, *args)
    y0 : array_like
        Initial state vector
    t_span : tuple
        Tuple of (t0, tf) giving the initial and final times
    dt : float, optional
        Time step (default: 0.01)
    args : tuple, optional
        Additional arguments to pass to function
        
    Returns:
    --------
    tuple
        (t, y) where t is the time points and y is the solution array
    """
    t0, tf = t_span
    t = np.arange(t0, tf + dt, dt)
    n = len(t)
    y0 = np.asarray(y0)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(n-1):
        y[i+1] = y[i] + dt * func(t[i], y[i], *args)
        
    return t, y

def rk4_integrate(func, y0, t_span, dt=0.01, args=()):
    """
    Perform 4th-order Runge-Kutta integration for an ODE.
    
    Parameters:
    -----------
    func : callable
        The right-hand side of the ODE system dy/dt = f(t, y, *args)
    y0 : array_like
        Initial state vector
    t_span : tuple
        Tuple of (t0, tf) giving the initial and final times
    dt : float, optional
        Time step (default: 0.01)
    args : tuple, optional
        Additional arguments to pass to function
        
    Returns:
    --------
    tuple
        (t, y) where t is the time points and y is the solution array
    """
    t0, tf = t_span
    t = np.arange(t0, tf + dt, dt)
    n = len(t)
    y0 = np.asarray(y0)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(n-1):
        k1 = func(t[i], y[i], *args)
        k2 = func(t[i] + dt/2, y[i] + dt*k1/2, *args)
        k3 = func(t[i] + dt/2, y[i] + dt*k2/2, *args)
        k4 = func(t[i] + dt, y[i] + dt*k3, *args)
        y[i+1] = y[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
    return t, y

def plot_phase_portrait(trajectories, x_idx=0, y_idx=1, 
                       title="Phase Portrait", x_label="x", y_label="y",
                       eq_point=None, show_legend=True, save_path=None):
    """
    Create a phase portrait from multiple trajectories.
    
    Parameters:
    -----------
    trajectories : list
        List of (t, y) tuples where y is a solution array with state vectors
    x_idx : int, optional
        Index of the state variable to plot on x-axis (default: 0)
    y_idx : int, optional
        Index of the state variable to plot on y-axis (default: 1)
    title : str, optional
        Title of the plot (default: "Phase Portrait")
    x_label : str, optional
        Label for x-axis (default: "x")
    y_label : str, optional
        Label for y-axis (default: "y")
    eq_point : array_like, optional
        Equilibrium point to mark on the plot (default: None)
    show_legend : bool, optional
        Whether to show the legend (default: True)
    save_path : str, optional
        Path to save the figure (default: None)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get color palette
    palette = sns.color_palette("husl", len(trajectories))
    
    # Plot each trajectory
    for i, (t, y) in enumerate(trajectories):
        ax.plot(y[:, x_idx], y[:, y_idx], color=palette[i], 
               label=f'Trajectory {i+1}', linewidth=1.5, alpha=0.8)
        
        # Mark the initial point
        ax.scatter(y[0, x_idx], y[0, y_idx], color=palette[i], s=80, 
                  marker='o', edgecolor='white', linewidth=1, zorder=10)
        
        # Mark the final point
        ax.scatter(y[-1, x_idx], y[-1, y_idx], color=palette[i], s=60, 
                  marker='s', edgecolor='white', linewidth=1, zorder=10)
    
    # Mark equilibrium point if provided
    if eq_point is not None:
        ax.scatter(eq_point[x_idx], eq_point[y_idx], color='red', s=150, 
                  marker='*', label='Equilibrium', edgecolor='white', 
                  linewidth=1.5, zorder=11)
    
    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend if requested
    if show_legend:
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Save if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def compute_eigenvalues(system_matrix):
    """
    Compute eigenvalues of a system matrix.
    
    For a linear system ẋ = Ax, this gives stability information.
    
    Parameters:
    -----------
    system_matrix : array_like
        The system matrix A in the linear system ẋ = Ax
        
    Returns:
    --------
    ndarray
        Array of eigenvalues
    """
    return np.linalg.eigvals(system_matrix)

def classify_equilibrium(eigenvalues):
    """
    Classify the type of equilibrium point based on eigenvalues.
    
    Parameters:
    -----------
    eigenvalues : array_like
        The eigenvalues of the linearized system
        
    Returns:
    --------
    tuple
        (stability, type) where:
        - stability is 'stable', 'unstable', or 'neutral'
        - type is 'node', 'focus', 'saddle', 'center', or 'degenerate'
    """
    # Check for zero eigenvalues (degenerate case)
    if np.any(np.abs(eigenvalues) < 1e-10):
        return 'neutral', 'degenerate'
    
    # Get real and imaginary parts
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    # Check if any complex eigenvalues
    has_complex = np.any(np.abs(imag_parts) > 1e-10)
    
    # Determine stability
    if np.all(real_parts < 0):
        stability = 'stable'
    elif np.all(real_parts > 0):
        stability = 'unstable'
    elif np.any(real_parts < 0) and np.any(real_parts > 0):
        return 'unstable', 'saddle'
    else:  # Some real parts are exactly zero
        stability = 'neutral'
    
    # Determine type
    if has_complex:
        if stability == 'neutral':
            eq_type = 'center'
        else:
            eq_type = 'focus'
    else:
        eq_type = 'node'
    
    return stability, eq_type

def jacobian(func, x, epsilon=1e-6, *args):
    """
    Compute the Jacobian matrix of a vector function numerically.
    
    Parameters:
    -----------
    func : callable
        The vector function f(x, *args)
    x : array_like
        The point at which to compute the Jacobian
    epsilon : float, optional
        Step size for finite difference (default: 1e-6)
    args : tuple, optional
        Additional arguments to pass to the function
        
    Returns:
    --------
    ndarray
        The Jacobian matrix
    """
    x = np.asarray(x)
    n = len(x)
    f0 = func(x, *args)
    m = len(f0)
    J = np.zeros((m, n))
    
    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += epsilon
        J[:, i] = (func(x_perturbed, *args) - f0) / epsilon
        
    return J 