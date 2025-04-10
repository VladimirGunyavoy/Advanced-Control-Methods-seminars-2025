"""
Bang-Bang Control Simulation and Visualization

This module provides tools for simulating and visualizing bang-bang control systems.
Bang-bang control is a time-optimal control strategy that applies maximum control
effort in one direction, then switches to maximum control effort in the opposite
direction at a specific switching time.

The module implements the necessary functions to calculate control parameters,
simulate system dynamics, and visualize trajectories in phase space.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set Seaborn style and context
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)

def calculate_control_params(p0, v0, a_m):
    """
    Calculate the bang-bang control parameters for reaching the origin (0,0).
    
    This function determines the optimal control strategy parameters for
    a double integrator system using bang-bang control. It calculates:
    - The region parameter k10 (determines control direction)
    - The sign parameter g10
    - The switching time ts10 (when to reverse control)
    - The final time tf10 (when target is reached)
    - The control values a111 and a222 for the two control phases
    
    Parameters:
    -----------
    p0 : float
        Initial position
    v0 : float
        Initial velocity
    a_m : float
        Maximum allowed acceleration magnitude
        
    Returns:
    --------
    tuple
        (k10, g10, ts10, tf10, a111, a222)
        
        k10 : int
            Region parameter determining control direction
        g10 : int
            Sign parameter
        ts10 : float
            Switching time (when control reverses direction)
        tf10 : float
            Final time (when target is reached)
        a111 : float
            First phase control value
        a222 : float
            Second phase control value
    """
    # Calculate k10 as per the bang-bang principle
    if v0**2 * np.sign(v0) >= -2 * a_m * p0:
        k10 = 1
    else:
        k10 = -1

    # Calculate g10 - determines the sign based on position
    if p0 == 0:
        if v0 < 0:  # This condition is from the formula, always false in practice
            g10 = -1
        else:
            g10 = 1
    else:
        g10 = 1

    # Normalized time parameter
    t0 = v0/a_m

    # Calculate switching time ts10 
    # (when control switches from maximum to minimum or vice versa)
    sqrt_arg = g10 * k10 * p0 / a_m + 0.5 * t0**2
    ts10 = np.sqrt(abs(sqrt_arg)) + k10 * t0  # Using abs to ensure we get a real number

    # Calculate final time tf10 (when system reaches origin)
    tf10 = 2 * ts10 - k10 * t0

    # Calculate accelerations for the two phases
    a111 = -k10 * a_m  # First phase control
    a222 = -a111       # Second phase control (opposite of first)

    return k10, g10, ts10, tf10, a111, a222

def control_function(t, p0, v0, a_m):
    """
    Compute the control input (acceleration) at a given time t.
    
    This function implements the bang-bang control law, which switches between
    maximum acceleration in opposite directions at the switching time.
    
    Parameters:
    -----------
    t : float
        Current time
    p0 : float
        Initial position
    v0 : float
        Initial velocity
    a_m : float
        Maximum allowed acceleration magnitude
        
    Returns:
    --------
    float
        The acceleration to apply at time t
    """
    # Get control parameters
    k10, g10, ts10, tf10, a111, a222 = calculate_control_params(p0, v0, a_m)

    # Special case for (0,0) initial condition - no control needed
    if p0 == 0.0 and v0 == 0.0:
        return 0.0

    # Bang-bang control law:
    # - Apply a111 from time 0 to ts10
    # - Switch to a222 from time ts10 to tf10
    # - Apply zero control after tf10 (target reached)
    if 0 <= t < ts10:
        return a111
    elif ts10 <= t < tf10:
        return a222
    else:
        return 0.0  # No acceleration after final time (target reached)

def simulate(p0, v0, a_m, dt=0.01):
    """
    Simulate the system dynamics under bang-bang control.
    
    This function performs a time-domain simulation of the system's response
    to bang-bang control, calculating position, velocity, and acceleration
    at each time step using Euler integration.
    
    Parameters:
    -----------
    p0 : float
        Initial position
    v0 : float
        Initial velocity
    a_m : float
        Maximum allowed acceleration magnitude
    dt : float
        Time step for simulation (default: 0.01)
        
    Returns:
    --------
    tuple
        (t_array, p_array, v_array, a_array, ts10, tf10)
        
        t_array : ndarray
            Time points
        p_array : ndarray
            Position at each time point
        v_array : ndarray
            Velocity at each time point
        a_array : ndarray
            Acceleration (control input) at each time point
        ts10 : float
            Switching time
        tf10 : float
            Final time
    """
    # Get control parameters
    k10, g10, ts10, tf10, a111, a222 = calculate_control_params(p0, v0, a_m)

    # Ensure we simulate slightly beyond the final time
    t_max = max(tf10 * 1.2, 0.1)  # Minimum simulation time of 0.1

    # Initialize arrays for state variables
    t_array = np.arange(0, t_max + dt, dt)
    p_array = np.zeros_like(t_array)
    v_array = np.zeros_like(t_array)
    a_array = np.zeros_like(t_array)

    # Set initial conditions
    p_array[0] = p0
    v_array[0] = v0

    # Simulation loop - Euler integration
    for i in range(1, len(t_array)):
        t = t_array[i-1]
        p = p_array[i-1]
        v = v_array[i-1]

        # Get control input for current time
        a = control_function(t, p0, v0, a_m)
        a_array[i-1] = a

        # Update state using Euler integration
        p_array[i] = p + v * dt       # Position update
        v_array[i] = v + a * dt       # Velocity update

    # Calculate the final acceleration
    a_array[-1] = control_function(t_array[-1], p0, v0, a_m)

    return t_array, p_array, v_array, a_array, ts10, tf10

def plot_phase_portrait(simulation_results, save_path="images/phase_portrait.png"):
    """
    Create a phase portrait showing multiple trajectories in position-velocity space.
    
    This function visualizes the system's phase space behavior under bang-bang control,
    highlighting initial points, switching points, and final points for multiple
    trajectories with different initial conditions.
    
    Parameters:
    -----------
    simulation_results : list
        List of tuples (t_array, p_array, v_array, a_array, ts10, tf10, p0, v0)
        containing simulation results for different initial conditions
    save_path : str, optional
        Path where the generated phase portrait should be saved
        (default: "images/phase_portrait.png")
        
    Returns:
    --------
    None
        The function saves the phase portrait to the specified path
    """
    # Set up the figure with seaborn style
    plt.figure(figsize=(12, 10))

    # Get a nice color palette from seaborn for multiple trajectories
    palette = sns.color_palette("husl", len(simulation_results))

    # Plot each trajectory with a different color from the palette
    for i, (t_array, p_array, v_array, a_array, ts10, tf10, p0, v0) in enumerate(simulation_results):
        # Plot the trajectory
        plt.plot(p_array, v_array, color=palette[i], label=f'({p0:.1f}, {v0:.1f})',
                linewidth=2, alpha=0.8)

        # Mark the initial point
        plt.scatter(p0, v0, color=palette[i], s=100, marker='o', 
                   edgecolor='white', linewidth=1.5, zorder=10)

        # Mark the switching point
        switch_idx = np.argmin(np.abs(t_array - ts10))
        plt.scatter(p_array[switch_idx], v_array[switch_idx], color=palette[i], s=60,
                   marker='s', edgecolor='white', linewidth=1, zorder=10)

        # Mark the final point
        final_idx = np.argmin(np.abs(t_array - tf10))
        plt.scatter(p_array[final_idx], v_array[final_idx], color=palette[i], s=80,
                   marker='x', linewidth=2, zorder=10)

    # Plot the target point (0,0)
    plt.scatter(0, 0, color='red', s=200, marker='*', label='Target', 
               edgecolor='white', linewidth=1.5, zorder=11)

    # Add grid, axes, and labels
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    plt.xlabel('Position', fontweight='bold')
    plt.ylabel('Velocity', fontweight='bold')
    plt.title('Phase Portrait: Bang-Bang Control Trajectories', 
             fontsize=16, fontweight='bold')

    # Add legend for trajectories
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, 
              frameon=True, fancybox=True, shadow=True)

    # Equal aspect ratio for proper visualization
    plt.axis('equal')

    # Add background grid and style
    sns.despine(left=False, bottom=False, right=False, top=False)

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_multiple_simulations(a_m=1.0, dt=0.01):
    """
    Run multiple simulations with different initial conditions.
    
    This function simulates the system from various initial points in the
    phase space and collects the results for visualization.
    
    Parameters:
    -----------
    a_m : float, optional
        Maximum allowed acceleration magnitude (default: 1.0)
    dt : float, optional
        Time step for simulation (default: 0.01)
        
    Returns:
    --------
    list
        List of simulation results for different initial conditions, where
        each element is a tuple (t_array, p_array, v_array, a_array, ts10, tf10, p0, v0)
    """
    # Define a range of initial conditions
    initial_conditions = [
        (1.0, 0.0),    # Right of origin, zero velocity
        (-1.0, 0.0),   # Left of origin, zero velocity
        (0.0, 1.0),    # At origin, upward velocity
        (0.0, -1.0),   # At origin, downward velocity
        (1.0, 1.0),    # First quadrant
        (-1.0, 1.0),   # Second quadrant
        (-1.0, -1.0),  # Third quadrant
        (1.0, -1.0),   # Fourth quadrant
        (2.0, 0.5),    # Custom point
        (-1.5, -0.8)   # Custom point
    ]
    
    # Run simulations for each initial condition
    simulation_results = []
    for p0, v0 in initial_conditions:
        # Run simulation
        t_array, p_array, v_array, a_array, ts10, tf10 = simulate(p0, v0, a_m, dt)
        
        # Store results
        simulation_results.append((t_array, p_array, v_array, a_array, ts10, tf10, p0, v0))
    
    return simulation_results

def save_trajectories(simulation_results, filename="examples/random_trajectories.pkl"):
    """
    Save simulation results to a pickle file for later use.
    
    Parameters:
    -----------
    simulation_results : list
        List of simulation results
    filename : str, optional
        Path to save the pickle file (default: "examples/random_trajectories.pkl")
        
    Returns:
    --------
    None
    """
    with open(filename, 'wb') as f:
        pickle.dump(simulation_results, f)
    print(f"Saved trajectories to {filename}")

def main():
    """
    Main function to run the simulation and generate visualizations.
    
    This function orchestrates the entire process:
    1. Run simulations with different initial conditions
    2. Generate and save phase portrait
    3. Save trajectories for later use
    
    Returns:
    --------
    None
    """
    # Set simulation parameters
    a_m = 1.0  # Maximum acceleration
    dt = 0.01  # Time step
    
    # Run simulations
    print("Running simulations...")
    simulation_results = run_multiple_simulations(a_m, dt)
    
    # Generate and save phase portrait
    print("Generating phase portrait...")
    plot_phase_portrait(simulation_results)
    
    # Save trajectories for later use
    print("Saving trajectories...")
    save_trajectories(simulation_results)
    
    print("Done!")

# Run the main function if script is executed directly
if __name__ == "__main__":
    main() 