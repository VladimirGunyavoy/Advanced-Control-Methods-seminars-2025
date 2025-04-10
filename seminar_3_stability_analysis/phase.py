import numpy as np
import matplotlib.pyplot as plt
import os

def equilibrium_type(k1, k2):
    # Stable focus region (damped oscillations)
    if k1 < 1 and k2 < 0 and k2**2 < 4*(1-k1):
        return 1
    # Stable node region (aperiodic damping)
    elif k1 < 1 and k2 < 0 and k2**2 > 4*(1-k1):
        return 2
    # Critical damping line
    elif k1 < 1 and k2 < 0 and abs(k2**2 - 4*(1-k1)) < 0.01:
        return 3
    # Neutral stability line (center)
    elif k1 < 1 and abs(k2) < 0.01:
        return 4
    # Unstable focus region
    elif k1 < 1 and k2 > 0:
        return 5
    # Saddle point region
    elif k1 > 1:
        return 6
    # Special boundary (saddle-node) - for k1=1, k2<0
    elif abs(k1 - 1) < 0.01 and k2 < 0:
        return 7
    # Degenerate case (one eigenvalue zero) - for k1=1, k2>=0
    elif abs(k1 - 1) < 0.01 and k2 >= 0:
        return 8
    else:
        return 0

def get_k_values(region_type):
    """
    Returns example values of k1 and k2 for the specified region type
    """
    if region_type == 1:  # Stable focus
        return 0.5, -0.5
    elif region_type == 2:  # Stable node
        return 0.5, -2.5
    elif region_type == 3:  # Critical damping
        k1 = 0.0
        k2 = -2.0
        return k1, k2
    elif region_type == 4:  # Neutral stability (center)
        return 0.5, 0.0
    elif region_type == 5:  # Unstable focus
        return 0.5, 0.5
    elif region_type == 6:  # Saddle point
        return 1.1, -0.3
    elif region_type == 7:  # Saddle-node
        return 1.0, -0.5
    elif region_type == 8:  # Degenerate case (one eigenvalue zero)
        return 1.0, 0.0
    else:
        raise ValueError("Unknown region type")

def simulate_system(region_type, num_points=100, num_steps=200, dt=0.05):
    """
    Simulates a dynamic system in matrix form for the selected region.
    
    Parameters:
        region_type (int): Region number (1-8)
        num_points (int): Number of initial points
        num_steps (int): Number of simulation steps
        dt (float): Time step
        
    Returns:
        dict: Dictionary with trajectories and system parameters
    """
    # Get k1 and k2 values for the specified region
    k1, k2 = get_k_values(region_type)
    
    # Create system matrix in the form x' = Ax
    A = np.array([
        [0, 1],
        [-1 +  k1,  k2]
    ])
    
    # Generate 10 initial points around the equilibrium position (0, 0)
    # Use a small radius for initial points
    radius = 0.2
    np.random.seed(42)  # for reproducibility
    
    initial_conditions = []
    for _ in range(num_points):
        # Generate random points in a circle of radius 'radius'
        r = radius * np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        x0 = r * np.cos(theta)
        y0 = r * np.sin(theta)
        initial_conditions.append(np.array([x0, y0]))
    
    # Simulate the system for each initial condition
    trajectories = []
    
    for init_cond in initial_conditions:
        trajectory = [init_cond.copy()]
        x = init_cond.copy()
        
        for _ in range(num_steps):
            # Simple Euler method for integration
            dx = A @ x
            x = x + dx * dt
            trajectory.append(x.copy())
        
        trajectories.append(np.array(trajectory))
    
    # Return simulation results and parameters
    return {
        "region_type": region_type,
        "k1": k1,
        "k2": k2,
        "trajectories": trajectories,
        "dt": dt,
        "num_steps": num_steps
    }

def plot_trajectories(simulation_result, title=None):
    """
    Visualizes trajectories of the dynamic system.
    
    Parameters:
        simulation_result (dict): Result from the simulate_system function
        title (str, optional): Plot title
    """
    # Настройка стиля для качественных надписей
    # Используем стандартные шрифты вместо Computer Modern Roman
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',  # Стандартный шрифт, доступный почти везде
        'font.size': 14,
        'axes.titlesize': 14, 
        'axes.labelsize': 14,
        'font.weight': 'normal',
        'text.usetex': False,     # Отключаем использование LaTeX
        'mathtext.default': 'regular'  # Используем обычный шрифт для математики
    })
    
    # Создаем фигуру и оси - явно указываем оси для избежания ошибки с colorbar
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract trajectories from simulation results
    trajectories = simulation_result["trajectories"]
    region_type = simulation_result["region_type"]
    k1 = simulation_result["k1"]
    k2 = simulation_result["k2"]
    
    # Calculate system matrix and eigenvalues
    A = np.array([
        [0, 1],
        [-(1 - k1), k2]
    ])
    eigenvalues = np.linalg.eigvals(A)
    
    # Set colors for initial and final points
    start_color = 'blue'
    end_color = 'orange'  # Используем оранжевый как в collage_generator.py
    
    # Отображаем траектории с градиентом по времени
    for traj in trajectories:
        # Устанавливаем цвет в зависимости от времени (от начала к концу)
        cmap = plt.cm.cool  # Меняем на схему cool (голубой-розовый градиент)
        for j in range(len(traj)-1):
            t = j / len(traj)  # Нормализованное время (0-1)
            color = cmap(t)
            ax.plot(traj[j:j+2, 0], traj[j:j+2, 1], '-', color=color, linewidth=1.5, alpha=0.7)
        
        # Начальная точка
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=start_color, markersize=6, alpha=0.7)
        # Конечная точка
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=end_color, markersize=6, alpha=0.7)
    
    # Add equilibrium point with larger size
    ax.plot(0, 0, 'ro', markersize=12, markeredgecolor='black')
    
    # Configure axes and titles
    if title is None:
        region_names = {
            1: "Stablefocus", 
            2: "Stablenode",
            3: "Criticaldamping",
            4: "Center",
            5: "Unstablefocus",
            6: "Saddlepoint",
            7: "Saddle-node",
            8: "Degeneratecase"
        }
        
        # Форматируем собственные значения для отображения
        eig_text = ""
        for j, eig in enumerate(eigenvalues):
            if abs(eig.imag) < 1e-10:  # Действительное собственное значение
                eig_text += f"$\\lambda_{{{j+1}}}={eig.real:.2f}$"
            else:  # Комплексное собственное значение
                eig_text += f"$\\lambda_{{{j+1}}}={eig.real:.2f}{'+' if eig.imag >= 0 else ''}{eig.imag:.2f}i$"
            if j < len(eigenvalues) - 1:
                eig_text += ", "
        
        # Создаем заголовок с жирным регионом и названием
        region_title = f"Region{region_type}:{region_names.get(region_type, 'Unknown')}"
        params_text = f"$k_1={k1:.2f}$, $k_2={k2:.2f}$"
        
        # Добавляем заголовок с жирным шрифтом для названия региона
        title = f"$\\mathbf{{{region_title}}}$\n{params_text}\n{eig_text}"
    
    ax.set_title(title, pad=10)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Создаем элементы для легенды
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue', markersize=6, 
               label='Initial Point', linestyle='None'),
        Line2D([0], [0], marker='s', color='orange', markerfacecolor='orange', markersize=6, 
               label='Final Point', linestyle='None'),
        Line2D([0], [0], marker='o', color='red', markerfacecolor='red', markersize=6, 
               markeredgecolor='black', label='Equilibrium Point', linestyle='None')
    ]
    
    # Добавляем легенду - меньшего размера и в углу
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right', framealpha=0.7, 
              markerscale=0.8, handletextpad=0.5, borderpad=0.4)
    
    # Добавляем градиентную шкалу времени - передаем ax для привязки colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=plt.cm.cool, norm=Normalize(vmin=0, vmax=1))  # Используем схему cool
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label='Time evolution', shrink=0.8)
    
    # Автоматически определяем границы графика с учетом всех траекторий
    max_x, max_y = 0, 0
    for traj in trajectories:
        region_max_x = np.max(np.abs(traj[:, 0]))
        region_max_y = np.max(np.abs(traj[:, 1]))
        max_x = max(max_x, region_max_x)
        max_y = max(max_y, region_max_y)
    
    # Добавляем небольшой отступ
    max_x = max_x * 1.2
    max_y = max_y * 1.2
    
    # Устанавливаем границы
    ax.set_xlim(-max_x, max_x)
    ax.set_ylim(-max_y, max_y)
    
    return plt

# Example usage:
if __name__ == "__main__":
    # Create directory for images if it doesn't exist
    images_dir = "simulation_images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Simulate the system for each region
    for region in range(1, 9):
        result = simulate_system(region)
        plt = plot_trajectories(result)
        plt.savefig(os.path.join(images_dir, f"region_{region}_simulation.png"), dpi=150)
        plt.close()
        print(f"Simulation for region {region} saved to file {images_dir}/region_{region}_simulation.png")
