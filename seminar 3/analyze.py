from phase import *
from check_k import check_k
import matplotlib.pyplot as plt
import os

def analyze_region(region_type):
    """
    Analyze a specific equilibrium region.
    
    Parameters:
    -----------
    region_type : int
        Region type (1-8)
        
    Returns:
    --------
    plt : matplotlib.pyplot
        Plot object with trajectory visualization
    """
    # Run check_k to analyze parameters
    check_k(region_type)
    
    # Simulate and visualize trajectories
    result = simulate_system(region_type)
    return plot_trajectories(result)

def save_all_regions(output_dir='analysis_results'):
    """
    Save analysis for all equilibrium regions to a directory.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the results
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save all region analyses
    for region_type in range(1, 9):
        # Get plt object for current region
        plt_obj = analyze_region(region_type)
        
        # Save image
        plt_obj.savefig(os.path.join(output_dir, f'region_{region_type}_simulation.png'), 
                       dpi=150, bbox_inches='tight')
        
        # Close figure to free memory
        plt_obj.close()
        
        print(f"Region {region_type} analysis saved to {output_dir}/region_{region_type}_simulation.png")

# Run this code if the script is executed directly
if __name__ == "__main__":
    # Create directory for output
    output_dir = "analysis_results"
    
    # Save all region analyses
    save_all_regions(output_dir) 