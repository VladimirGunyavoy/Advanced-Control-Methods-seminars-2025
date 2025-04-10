import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style and context
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)

def calculate_control_params(p0, v0, a_m):
    """
    Calculate the bang-bang control parameters for reaching (0,0).

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
    Control parameters k10, g10, ts10, tf10, a111, a222
    """
    # Calculate k10 as per the given formula
    if v0**2 * np.sign(v0) >= -2 * a_m * p0:
        k10 = 1
    else:
        k10 = -1

    # Calculate g10
    if p0 == 0:
        if v0 < 0:  # This condition is from the formula, always false in practice
            g10 = -1
        else:
            g10 = 1
    else:
        g10 = 1

    t0 = v0/a_m

    # Calculate switching time ts10
    sqrt_arg = g10 * k10 * p0 / a_m + 0.5 * t0**2
    ts10 = np.sqrt(abs(sqrt_arg)) + k10 * t0 # Using abs to ensure we get a real number

    # Calculate final time tf10
    tf10 = 2 * ts10 - k10 * t0

    # Calculate accelerations for the two phases
    a111 = -k10 * a_m
    a222 = -a111

    return k10, g10, ts10, tf10, a111, a222

def control_function(t, p0, v0, a_m):
    """
    Compute the control input (acceleration) at time t.

    Parameters:
    -----------
    t : float
        Current time
    p0 : float
        Initial position
    v0 : float
        Initial velocity
    a_m : float
        Maximum allowed acceleration

    Returns:
    --------
    float : The acceleration to apply at time t
    """
    k10, g10, ts10, tf10, a111, a222 = calculate_control_params(p0, v0, a_m)

    # Special case for (0,0) initial condition
    if p0 == 0.0 and v0 == 0.0:
        return 0.0

    # a11111(t) = {0 ≤ t < ts10: a111, ts10 ≤ t < tf10: a222, 0}
    if 0 <= t < ts10:
        return a111
    elif ts10 <= t < tf10:
        return a222
    else:
        return 0  # No acceleration after final time

def simulate(p0, v0, a_m, dt=0.01):
    """
    Simulate the system from t=0 to t=tf10.

    Parameters:
    -----------
    p0 : float
        Initial position
    v0 : float
        Initial velocity
    a_m : float
        Maximum allowed acceleration
    dt : float
        Time step for simulation

    Returns:
    --------
    t_array, p_array, v_array, a_array, ts10, tf10
    """
    k10, g10, ts10, tf10, a111, a222 = calculate_control_params(p0, v0, a_m)

    # Ensure we simulate at least until tf10, with a minimum simulation time
    t_max = max(tf10 * 1.2, 0.1)  # Use at least 0.1 to handle the case when tf10 = 0

    # Initialize arrays
    t_array = np.arange(0, t_max + dt, dt)  # Add dt to ensure we include at least one point
    p_array = np.zeros_like(t_array)
    v_array = np.zeros_like(t_array)
    a_array = np.zeros_like(t_array)

    # Initial conditions
    p_array[0] = p0
    v_array[0] = v0

    # Simulation loop
    for i in range(1, len(t_array)):
        t = t_array[i-1]
        p = p_array[i-1]
        v = v_array[i-1]

        # Get control input
        a = control_function(t, p0, v0, a_m)
        a_array[i-1] = a

        # Update state using simple Euler integration
        p_array[i] = p + v * dt
        v_array[i] = v + a * dt

    # Calculate the final acceleration
    a_array[-1] = control_function(t_array[-1], p0, v0, a_m)

    return t_array, p_array, v_array, a_array, ts10, tf10

def plot_phase_portrait(simulation_results):
    """
    Create a phase portrait showing all trajectories in position-velocity space.

    Parameters:
    -----------
    simulation_results : list
        List of tuples (t_array, p_array, v_array, a_array, ts10, tf10, p0, v0)
        containing simulation results for different initial conditions
    """
    # Set up the figure with seaborn style
    plt.figure(figsize=(12, 10))

    # Get a nice color palette from seaborn
    palette = sns.color_palette("husl", len(simulation_results))

    # Plot each trajectory with a different color from the palette
    for i, (t_array, p_array, v_array, a_array, ts10, tf10, p0, v0) in enumerate(simulation_results):
        # Plot the trajectory with seaborn styling - changed to lineplot from line to avoid fill
        plt.plot(p_array, v_array, color=palette[i], label=f'({p0:.1f}, {v0:.1f})',
                linewidth=2, alpha=0.8)

        # Mark the initial point
        plt.scatter(p0, v0, color=palette[i], s=100, marker='o', edgecolor='white', linewidth=1.5, zorder=10)

        # Mark the switching point
        switch_idx = np.argmin(np.abs(t_array - ts10))
        plt.scatter(p_array[switch_idx], v_array[switch_idx], color=palette[i], s=60,
                    marker='s', edgecolor='white', linewidth=1, zorder=10)

        # Mark the final point
        final_idx = np.argmin(np.abs(t_array - tf10))
        plt.scatter(p_array[final_idx], v_array[final_idx], color=palette[i], s=80,
                    marker='x', linewidth=2, zorder=10)

    # Plot the target point (0,0)
    plt.scatter(0, 0, color='red', s=200, marker='*', label='Target', edgecolor='white', linewidth=1.5, zorder=11)

    # Add grid, legend, and labels
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    plt.xlabel('Position', fontweight='bold')
    plt.ylabel('Velocity', fontweight='bold')
    plt.title('Phase Portrait: Bang-Bang Control Trajectories', fontsize=16, fontweight='bold')

    # Removed legend from second graph
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=True,
    #           fancybox=True, shadow=True)

    # Equal aspect ratio
    plt.axis('equal')

    # Add background grid and style
    sns.despine(left=False, bottom=False, right=False, top=False)

    plt.tight_layout()
    plt.savefig('phase_portrait.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_multiple_simulations(a_m=1.0, dt=0.01):
    """
    Run simulations from 9 different starting points for individual plots
    and 20 random starting points for the phase portrait.

    Parameters:
    -----------
    a_m : float
        Maximum allowed acceleration
    dt : float
        Time step for simulation
    """
    # Define 9 specific initial conditions for individual plots
    individual_conditions = [
        (-1, 1.0),      # p0 = -1, v0 = 1
        (0.0, 1.0),     # p0 = 0, v0 = 1
        (1.0, 1.0),     # p0 = 1, v0 = 1
        (1.0, 0.0),     # p0 = 1, v0 = 0
        (0.0, -1.0),    # p0 = 0, v0 = -1
        (0.5, -0.5),    # p0 = 0.5, v0 = -0.5
        (-1.0, 0.0),    # p0 = -1, v0 = 0
        (-1.0, -1.0),   # p0 = -1, v0 = -1
        (-0.5, -0.5),   # p0 = -0.5, v0 = -0.5
    ]

    # Create a figure with 3x3 layout for individual simulations
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    axs = axs.flatten()  # Flatten the 2D array to make indexing easier

    # Set a nice color palette for the plots
    colors = sns.color_palette("Set1", 3)

    # Store simulation results for the phase portrait
    simulation_results = []

    # First, simulate and plot the 9 predefined initial conditions
    for i, (p0, v0) in enumerate(individual_conditions):
        # Print control parameters
        k10, g10, ts10, tf10, a111, a222 = calculate_control_params(p0, v0, a_m)
        print(f"Initial conditions: p0 = {p0}, v0 = {v0}")
        print(f"  k10 = {k10}, g10 = {g10}")
        print(f"  ts10 = {ts10:.4f}, tf10 = {tf10:.4f}")
        print(f"  a111 = {a111:.4f}, a222 = {a222:.4f}")
        print("----------------------------")

        # Simulate
        t_array, p_array, v_array, a_array, ts10, tf10 = simulate(p0, v0, a_m, dt)

        # Store results for phase portrait
        simulation_results.append((t_array, p_array, v_array, a_array, ts10, tf10, p0, v0))

        # Plot all on the same subplot with seaborn styling
        axs[i].plot(t_array, p_array, color=colors[0], linewidth=2, label='Position')
        axs[i].plot(t_array, v_array, color=colors[1], linewidth=2, label='Velocity')
        axs[i].plot(t_array, a_array, color=colors[2], linewidth=2, label='Control')

        # Add reference lines
        axs[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axs[i].axvline(x=ts10, color='purple', linestyle='--', label='Switch', alpha=0.7)
        axs[i].axvline(x=tf10, color='orange', linestyle='--', label='Final', alpha=0.7)

        # Set titles and labels with seaborn styling
        axs[i].set_title(f'Simulation (p0={p0}, v0={v0})', fontsize=12, fontweight='bold')
        axs[i].set_xlabel('Time', fontweight='bold')
        axs[i].set_ylabel('Value', fontweight='bold')

        # Added legend to all plots instead of just the first one
        axs[i].legend(loc='best', frameon=True, fancybox=True)

    # Add an overall title
    fig.suptitle('Bang-Bang Control Simulations', fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for the suptitle
    plt.savefig('bang_bang_simulations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Now generate 20 random initial conditions for the phase portrait
    np.random.seed(42)  # For reproducibility

    # Generate random initial conditions within a reasonable range
    random_conditions = []
    for _ in range(200):
        p0 = np.random.uniform(-1.5, 1.5)
        v0 = np.random.uniform(-1.5, 1.5)
        random_conditions.append((p0, v0))

    print("\nSimulating 20 random initial conditions for phase portrait...")

    # Simulate the random initial conditions
    random_results = []
    for p0, v0 in random_conditions:
        t_array, p_array, v_array, a_array, ts10, tf10 = simulate(p0, v0, a_m, dt)
        random_results.append((t_array, p_array, v_array, a_array, ts10, tf10, p0, v0))

    # Save the random trajectories to a file
    import pickle
    with open('random_trajectories.pkl', 'wb') as f:
        pickle.dump(random_results, f)

    print("Random trajectories saved to 'random_trajectories.pkl'")

    # Plot the phase portrait with all random initial conditions
    plot_phase_portrait(random_results)

if __name__ == "__main__":
    a_m = 1.2  # Maximum allowed acceleration
    run_multiple_simulations(a_m)