import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import seaborn as sns
from datetime import datetime

class FilteredCSIVisualizer:
    """
    A class for visualizing filtered CSI (Channel State Information) data.
    Provides enhanced visualization methods for analyzing subcarrier characteristics and patterns.
    """
    
    def __init__(self, base_dir: str = 'data_capture/csi'):
        """
        Initialize the visualizer with directory configuration.
        
        Args:
            base_dir (str): Base directory containing CSI files
        """
        self.base_dir = base_dir
        self.filtered_dir = os.path.join(base_dir, 'filtered')
        self.plot_dir = os.path.join(base_dir, 'filtered_plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Enhanced plotting configuration
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        self.colors = plt.cm.viridis(np.linspace(0, 1, 20))
        
    def load_filtered_data(self, filepath: str) -> Optional[Dict]:
        """
        Load filtered CSI data and its metadata.
        
        Args:
            filepath (str): Path to the filtered CSV file
            
        Returns:
            Optional[Dict]: Dictionary containing the data and metadata, or None if error
        """
        try:
            if os.path.getsize(filepath) == 0:
                print(f"Warning: {filepath} is empty")
                return None
                
            # Try to read the CSV file with error handling
            try:
                data = pd.read_csv(filepath)
            except pd.errors.EmptyDataError:
                print(f"Error: {filepath} is empty or has no valid data")
                return None
            except Exception as e:
                print(f"Error reading {filepath}: {str(e)}")
                return None
            
            if data.empty or len(data.columns) == 0:
                print(f"Warning: No valid data found in {filepath}")
                return None
                
            subcarrier_cols = [col for col in data.columns if col.startswith('subcarrier_')]
            if not subcarrier_cols:
                print(f"Warning: No subcarrier columns found in {filepath}")
                return None
            
            # Load metadata
            metadata_path = filepath.replace('filtered_', 'metadata_')
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            metadata[key.strip()] = value.strip()
            
            # Calculate additional metadata
            metadata['Mean Amplitude'] = f"{data[subcarrier_cols].mean().mean():.2f}"
            metadata['Max Amplitude'] = f"{data[subcarrier_cols].max().max():.2f}"
            metadata['Min Amplitude'] = f"{data[subcarrier_cols].min().min():.2f}"
            metadata['Total Duration'] = f"{(pd.to_numeric(data['timestamp'].iloc[-1]) - pd.to_numeric(data['timestamp'].iloc[0])) / 1e6:.2f} seconds"
            
            return {
                'data': data,
                'metadata': metadata,
                'filename': os.path.basename(filepath)
            }
        except Exception as e:
            print(f"Error loading file {filepath}: {str(e)}")
            return None
            
    def plot_active_subcarriers(self, data: pd.DataFrame, metadata: Dict, output_path: str):
        """
        Plot the time series of active subcarriers with enhanced visualization.
        
        Args:
            data (pd.DataFrame): The filtered CSI data
            metadata (Dict): Metadata about the filtering
            output_path (str): Where to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        subcarrier_cols = [col for col in data.columns if col.startswith('subcarrier_')]
        
        # Create time axis
        if 'timestamp' in data.columns:
            time = pd.to_numeric(data['timestamp']) - pd.to_numeric(data['timestamp'].iloc[0])
            time = time / 1e6  # Convert to seconds
        else:
            time = np.arange(len(data))
            
        # Plot each subcarrier with enhanced styling
        for i, col in enumerate(subcarrier_cols):
            color = self.colors[i % len(self.colors)]
            plt.plot(time, data[col], color=color, alpha=0.8, linewidth=1.5, 
                    label=f'Subcarrier {i}')
            
        plt.title('Active Subcarriers Time Series Analysis', fontsize=16, pad=20)
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel('CSI Amplitude', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add enhanced metadata
        info_text = (
            f"Active Subcarriers: {len(subcarrier_cols)}\n"
            f"Mean Amplitude: {metadata.get('Mean Amplitude', 'N/A')}\n"
            f"Max Amplitude: {metadata.get('Max Amplitude', 'N/A')}\n"
            f"Min Amplitude: {metadata.get('Min Amplitude', 'N/A')}\n"
            f"Duration: {metadata.get('Total Duration', 'N/A')}"
        )
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add legend with improved visibility
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_subcarrier_heatmap(self, data: pd.DataFrame, metadata: Dict, output_path: str):
        """
        Create an enhanced heatmap visualization of subcarrier activities.
        
        Args:
            data (pd.DataFrame): The filtered CSI data
            metadata (Dict): Metadata about the filtering
            output_path (str): Where to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        subcarrier_cols = [col for col in data.columns if col.startswith('subcarrier_')]
        heatmap_data = data[subcarrier_cols].T
        
        # Create time axis
        if 'timestamp' in data.columns:
            time = pd.to_numeric(data['timestamp']) - pd.to_numeric(data['timestamp'].iloc[0])
            time = time / 1e6  # Convert to seconds
            extent = [time.min(), time.max(), 0, len(subcarrier_cols)]
        else:
            extent = [0, len(data), 0, len(subcarrier_cols)]
            
        if extent[0] == extent[1]:
            extent[1] = extent[0] + 1
            
        # Enhanced heatmap with better colormap and annotations
        im = plt.imshow(heatmap_data, aspect='auto', cmap='plasma', extent=extent)
        plt.colorbar(im, label='CSI Amplitude', pad=0.02)
        
        plt.title('Subcarrier Activity Heatmap Analysis', fontsize=16, pad=20)
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel('Subcarrier Index', fontsize=14)
        
        # Add enhanced metadata
        info_text = (
            f"Active Subcarriers: {len(subcarrier_cols)}\n"
            f"Duration: {metadata.get('Total Duration', 'N/A')}\n"
            f"Mean Amplitude: {metadata.get('Mean Amplitude', 'N/A')}"
        )
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_subcarrier_statistics(self, data: pd.DataFrame, metadata: Dict, output_path: str):
        """
        Plot enhanced statistical information about each subcarrier.
        
        Args:
            data (pd.DataFrame): The filtered CSI data
            metadata (Dict): Metadata about the filtering
            output_path (str): Where to save the plot
        """
        subcarrier_cols = [col for col in data.columns if col.startswith('subcarrier_')]
        
        # Enhanced statistical calculations
        stats_data = pd.DataFrame()
        stats_data['Subcarrier'] = range(len(subcarrier_cols))
        stats_data['Standard Deviation'] = data[subcarrier_cols].std()
        stats_data['Mean'] = data[subcarrier_cols].mean()
        stats_data['Range'] = data[subcarrier_cols].max() - data[subcarrier_cols].min()
        stats_data['Variance'] = data[subcarrier_cols].var()
        stats_data['Skewness'] = data[subcarrier_cols].skew()
        
        # Create enhanced subplot figure with adjusted spacing
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2])
        fig.suptitle('Comprehensive Subcarrier Statistics Analysis', fontsize=20, y=0.95)
        
        # Plot 1: Standard deviation with confidence intervals
        ax1 = fig.add_subplot(gs[0, 0])
        sns.barplot(data=stats_data, x='Subcarrier', y='Standard Deviation', ax=ax1, 
                   color='skyblue', errorbar=('ci', 95))
        ax1.set_title('Standard Deviation by Subcarrier', fontsize=14)
        ax1.set_xlabel('Subcarrier Index', fontsize=12)
        ax1.set_ylabel('Standard Deviation', fontsize=12)
        
        # Plot 2: Mean values with error bars
        ax2 = fig.add_subplot(gs[0, 1])
        sns.barplot(data=stats_data, x='Subcarrier', y='Mean', ax=ax2, 
                   color='lightgreen', errorbar=('ci', 95))
        ax2.set_title('Mean Value by Subcarrier', fontsize=14)
        ax2.set_xlabel('Subcarrier Index', fontsize=12)
        ax2.set_ylabel('Mean', fontsize=12)
        
        # Plot 3: Range analysis
        ax3 = fig.add_subplot(gs[1, 0])
        sns.barplot(data=stats_data, x='Subcarrier', y='Range', ax=ax3, 
                   color='salmon')
        ax3.set_title('Amplitude Range by Subcarrier', fontsize=14)
        ax3.set_xlabel('Subcarrier Index', fontsize=12)
        ax3.set_ylabel('Range (Max - Min)', fontsize=12)
        
        # Plot 4: Variance analysis
        ax4 = fig.add_subplot(gs[1, 1])
        sns.barplot(data=stats_data, x='Subcarrier', y='Variance', ax=ax4, 
                   color='purple')
        ax4.set_title('Variance by Subcarrier', fontsize=14)
        ax4.set_xlabel('Subcarrier Index', fontsize=12)
        ax4.set_ylabel('Variance', fontsize=12)
        
        # Plot 5: Distribution analysis with violin plot
        ax5 = fig.add_subplot(gs[2, :])
        data_array = [data[col] for col in subcarrier_cols]
        sns.violinplot(data=data_array, ax=ax5, inner='box', color='lightgray')
        ax5.set_title('Subcarrier Value Distribution', fontsize=14)
        ax5.set_xlabel('Subcarrier Index', fontsize=12)
        ax5.set_ylabel('CSI Amplitude', fontsize=12)
        
        # Add summary statistics with adjusted position
        summary_text = (
            f"Total Subcarriers: {len(subcarrier_cols)}\n"
            f"Mean Amplitude: {metadata.get('Mean Amplitude', 'N/A')}\n"
            f"Duration: {metadata.get('Total Duration', 'N/A')}"
        )
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Adjust layout with more space for the bottom
        plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0.3, wspace=0.2)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def process_file(self, filepath: str):
        """
        Process a single filtered CSI file and generate visualizations.
        
        Args:
            filepath (str): Path to the filtered CSV file
        """
        print(f"Processing {os.path.basename(filepath)}...")
        
        # Load the data
        result = self.load_filtered_data(filepath)
        if result is None:
            return
            
        data = result['data']
        metadata = result['metadata']
        base_name = os.path.splitext(result['filename'])[0]
        
        # Generate plots
        self.plot_active_subcarriers(
            data, metadata,
            os.path.join(self.plot_dir, f"{base_name}_active_subcarriers.png")
        )
        
        self.plot_subcarrier_heatmap(
            data, metadata,
            os.path.join(self.plot_dir, f"{base_name}_heatmap.png")
        )
        
        self.plot_subcarrier_statistics(
            data, metadata,
            os.path.join(self.plot_dir, f"{base_name}_statistics.png")
        )
        
    def process_all_files(self):
        """
        Process all filtered CSI files in the filtered directory.
        """
        print(f"Looking for filtered CSI files in {self.filtered_dir}...")
        
        for filename in os.listdir(self.filtered_dir):
            if filename.startswith('filtered_') and filename.endswith('.csv'):
                filepath = os.path.join(self.filtered_dir, filename)
                self.process_file(filepath)
                
        print("Processing complete!")

def main():
    """
    Main execution function.
    """
    visualizer = FilteredCSIVisualizer()
    visualizer.process_all_files()

if __name__ == "__main__":
    main() 