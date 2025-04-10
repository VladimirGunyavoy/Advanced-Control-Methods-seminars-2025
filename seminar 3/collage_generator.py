import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def generate_phase_portrait_collage(save_path=None, show_plot=True):
    """
    Generates a collage of phase portraits for all equilibrium types.
    
    Parameters:
    -----------
    save_path : str, optional
        Path to save the collage. If None, the collage is not saved.
    show_plot : bool, default=True
        Whether to show the plot.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object with the collage.
    """
    # Импортируем необходимые функции для симуляции
    from phase import get_k_values, simulate_system
    
    # Настройка стиля для качественных надписей
    # Используем стандартные шрифты вместо Computer Modern Roman
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',  # Стандартный шрифт, доступный почти везде
        'font.size': 14,
        'axes.titlesize': 14, 
        'axes.labelsize': 14,
        'figure.titlesize': 22,
        'font.weight': 'normal',  # Для цифр на осях - нормальный вес
        'text.usetex': False,     # Отключаем использование LaTeX
        'mathtext.default': 'regular'  # Используем обычный шрифт для математики
    })

    # Создаем фигуру большого размера с местом для легенды
    fig = plt.figure(figsize=(32, 15), dpi=300)
    gs = GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.3], figure=fig, wspace=0.3, hspace=0.3)

    # Названия регионов для заголовков
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

    # Параметры для симуляции
    num_points = 50  # Уменьшаем для лучшей читаемости
    num_steps = 100
    dt = 0.05
    
    # Симулируем все траектории и определяем границы для каждого региона
    all_data = []
    region_limits = []
    
    for i in range(1, 9):
        # Симулируем систему для текущего региона
        result = simulate_system(i, num_points=num_points, num_steps=num_steps, dt=dt)
        all_data.append(result)
        
        # Находим границы для текущего региона
        max_x, max_y = 0, 0
        for traj in result["trajectories"]:
            region_max_x = np.max(np.abs(traj[:, 0]))
            region_max_y = np.max(np.abs(traj[:, 1]))
            max_x = max(max_x, region_max_x)
            max_y = max(max_y, region_max_y)
        
        # Добавляем небольшой отступ
        max_x = max_x * 1.2
        max_y = max_y * 1.2
        
        # Сохраняем границы для этого региона
        region_limits.append((max_x, max_y))
    
    # Теперь отрисовываем все графики с индивидуальными границами
    for i in range(1, 9):
        # Получаем параметры для этого региона
        k1, k2 = get_k_values(i)
        
        # Определяем позицию подграфика
        row = (i-1) // 4
        col = (i-1) % 4
        
        # Создаем подграфик
        ax = fig.add_subplot(gs[row, col])
        
        # Получаем данные из предварительно симулированных результатов
        result = all_data[i-1]
        trajectories = result["trajectories"]
        
        # Получаем границы для этого региона
        max_x, max_y = region_limits[i-1]
        
        # Задаем цвета для начальных и конечных точек
        start_color = 'blue'
        end_color = 'orange'
        
        # Отображаем траектории
        for traj in trajectories:
            # Устанавливаем цвет в зависимости от времени (от начала к концу)
            cmap = plt.cm.cool  # Меняем на схему cool (голубой-розовый градиент)
            for j in range(len(traj)-1):
                t = j / len(traj)  # Нормализованное время (0-1)
                color = cmap(t)
                ax.plot(traj[j:j+2, 0], traj[j:j+2, 1], '-', color=color, linewidth=1, alpha=0.7)
            
            # Начальная точка
            ax.plot(traj[0, 0], traj[0, 1], 'o', color=start_color, markersize=4)
            # Конечная точка
            ax.plot(traj[-1, 0], traj[-1, 1], 's', color=end_color, markersize=4)
        
        # Добавляем точку равновесия
        ax.plot(0, 0, 'ro', markersize=10, markeredgecolor='black')
        
        # Добавляем оси
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        # Добавляем подписи осей
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\dot{\theta}$')
        
        # Устанавливаем индивидуальные границы для графика
        ax.set_xlim(-max_x, max_x)
        ax.set_ylim(-max_y, max_y)
        
        # Отключаем равные масштабы осей, чтобы лимиты были разными
        # при сохранении одинакового размера графиков
        # ax.set_aspect('equal')  # Отключаем равные масштабы
        
        # Получаем собственные значения матрицы для отображения
        A = np.array([
            [0, 1],
            [-1 + k1, k2]
        ])
        eigenvalues = np.linalg.eigvals(A)
        
        # Форматируем собственные значения для отображения
        eig_text = ""
        for j, eig in enumerate(eigenvalues):
            if abs(eig.imag) < 1e-10:  # Действительное собственное значение
                eig_text += f"$\\lambda_{{{j+1}}}={eig.real:.2f}$"
            else:  # Комплексное собственное значение
                eig_text += f"$\\lambda_{{{j+1}}}={eig.real:.2f}{'+' if eig.imag >= 0 else ''}{eig.imag:.2f}i$"
            if j < len(eigenvalues) - 1:
                eig_text += ", "
        
        # Создаем заголовок с жирным регионом и названием (без пробелов для лучшего вида)
        region_title = f"Region{i}:{region_names[i]}"
        params_text = f"$k_1={k1:.2f}$, $k_2={k2:.2f}$"
        
        # Добавляем заголовок с жирным шрифтом для названия региона
        ax.set_title(f"$\\mathbf{{{region_title}}}$\n{params_text}\n{eig_text}", pad=10)
        
        # Добавляем сетку
        ax.grid(True, alpha=0.3)

    # Добавляем легенду в отдельный subplot справа
    legend_ax = fig.add_subplot(gs[:, 4])
    legend_ax.axis('off')  # Убираем оси

    # Создаем элементы для легенды (только точки старта, финиша и равновесия)
    legend_elements = [
        Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue', markersize=10, 
               label='Initial Point', linestyle='None'),
        Line2D([0], [0], marker='s', color='orange', markerfacecolor='orange', markersize=10, 
               label='Final Point', linestyle='None'),
        Line2D([0], [0], marker='o', color='red', markerfacecolor='red', markersize=10, 
               markeredgecolor='black', label='Equilibrium Point', linestyle='None')
    ]

    # Добавляем временной градиент
    sm = ScalarMappable(cmap=plt.cm.cool, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=legend_ax, orientation='horizontal', aspect=10, shrink=0.8)
    cbar.set_label('Time evolution')

    # Добавляем легенду
    legend_ax.legend(handles=legend_elements, title="Legend", loc='center', fontsize=12)

    # Настраиваем общий заголовок
    plt.suptitle("Phase Portraits for Different Equilibrium Types", y=0.98)

    # Сохраняем коллаж в высоком разрешении, если указан путь
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Показываем результат, если требуется
    if show_plot:
        plt.show()
        
    return fig

# Пример использования, если файл запускается напрямую
if __name__ == "__main__":
    generate_phase_portrait_collage(save_path="equilibrium_types_collage.png") 