import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import seaborn as sns
from datetime import datetime

class TopSubcarrierVisualizer:
    """
    A class for visualizing the top 8 selected subcarriers from CSI data.
    Provides enhanced visualization methods for analyzing the selected subcarriers.
    """
    
    def __init__(self, base_dir: str = 'data_capture/csi'):
        """
        Initialize the visualizer with directory configuration.
        
        Args:
            base_dir (str): Base directory containing CSI files
        """
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, '8_csi_filtered')
        self.plot_dir = os.path.join(base_dir, 'top8_plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Enhanced plotting configuration
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        self.colors = plt.cm.viridis(np.linspace(0, 1, 8))  # 8 colors for 8 subcarriers
        
    def load_data(self, filepath: str) -> Optional[Dict]:
        """
        Load top 8 subcarrier data and its metadata.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            Optional[Dict]: Dictionary containing the data and metadata
        """
        try:
            if os.path.getsize(filepath) == 0:
                print(f"Warning: {filepath} is empty")
                return None
                
            data = pd.read_csv(filepath)
            if data.empty or len(data.columns) == 0:
                print(f"Warning: No valid data found in {filepath}")
                return None
                
            # Load corresponding metadata
            metadata_path = filepath.replace('top8_', 'top8_metadata_').replace('.csv', '.txt')
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            metadata[key.strip()] = value.strip()
            
            return {
                'data': data,
                'metadata': metadata,
                'filename': os.path.basename(filepath)
            }
        except Exception as e:
            print(f"Error loading file {filepath}: {str(e)}")
            return None
            
    def plot_time_series(self, data: pd.DataFrame, metadata: Dict, output_path: str):
        """
        Plot time series of the top 8 subcarriers.
        
        Args:
            data (pd.DataFrame): The subcarrier data
            metadata (Dict): Metadata about the subcarriers
            output_path (str): Where to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Get subcarrier columns
        subcarrier_cols = [col for col in data.columns if col.startswith('subcarrier_')]
        
        # Create time axis
        if 'timestamp' in data.columns:
            time = pd.to_numeric(data['timestamp']) - pd.to_numeric(data['timestamp'].iloc[0])
            time = time / 1e6  # Convert to seconds
        else:
            time = np.arange(len(data))
            
        # Plot each subcarrier
        for i, col in enumerate(subcarrier_cols):
            plt.plot(time, data[col], color=self.colors[i], alpha=0.8, linewidth=1.5,
                    label=f'Subcarrier {col.split("_")[1]}')
            
        plt.title('Top 8 Subcarriers Time Series', fontsize=16, pad=20)
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel('CSI Amplitude', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add metadata
        duration = time.iloc[-1] if hasattr(time, 'iloc') else time[-1]
        info_text = (
            f"Selected Subcarriers: {len(subcarrier_cols)}\n"
            f"Total Samples: {len(data)}\n"
            f"Duration: {duration:.2f} seconds"
        )
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_heatmap(self, data: pd.DataFrame, metadata: Dict, output_path: str):
        """
        Create a heatmap visualization of the top 8 subcarriers.
        
        Args:
            data (pd.DataFrame): The subcarrier data
            metadata (Dict): Metadata about the subcarriers
            output_path (str): Where to save the plot
        """
        plt.figure(figsize=(15, 8))
        
        subcarrier_cols = [col for col in data.columns if col.startswith('subcarrier_')]
        heatmap_data = data[subcarrier_cols].T
        
        # Create time axis
        if 'timestamp' in data.columns:
            time = pd.to_numeric(data['timestamp']) - pd.to_numeric(data['timestamp'].iloc[0])
            time = time / 1e6  # Convert to seconds
            extent = [time.min(), time.max(), 0, len(subcarrier_cols)]
        else:
            extent = [0, len(data), 0, len(subcarrier_cols)]
            
        # Plot heatmap
        im = plt.imshow(heatmap_data, aspect='auto', cmap='plasma', extent=extent)
        plt.colorbar(im, label='CSI Amplitude', pad=0.02)
        
        plt.title('Top 8 Subcarriers Activity Heatmap', fontsize=16, pad=20)
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel('Subcarrier Index', fontsize=14)
        
        # Add metadata
        duration = time.iloc[-1] if hasattr(time, 'iloc') else time[-1]
        info_text = (
            f"Selected Subcarriers: {len(subcarrier_cols)}\n"
            f"Duration: {duration:.2f} seconds"
        )
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_statistics(self, data: pd.DataFrame, metadata: Dict, output_path: str):
        """
        Plot statistical information about the top 8 subcarriers.
        
        Args:
            data (pd.DataFrame): The subcarrier data
            metadata (Dict): Metadata about the subcarriers
            output_path (str): Where to save the plot
        """
        subcarrier_cols = [col for col in data.columns if col.startswith('subcarrier_')]
        
        # Calculate statistics
        stats_data = pd.DataFrame()
        stats_data['Subcarrier'] = [col.split('_')[1] for col in subcarrier_cols]
        stats_data['Standard Deviation'] = data[subcarrier_cols].std()
        stats_data['Mean'] = data[subcarrier_cols].mean()
        stats_data['Range'] = data[subcarrier_cols].max() - data[subcarrier_cols].min()
        stats_data['Variance'] = data[subcarrier_cols].var()
        
        # Create subplot figure
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(2, 2, figure=fig)
        fig.suptitle('Top 8 Subcarriers Statistics Analysis', fontsize=20, y=0.95)
        
        # Plot 1: Standard deviation
        ax1 = fig.add_subplot(gs[0, 0])
        sns.barplot(data=stats_data, x='Subcarrier', y='Standard Deviation', ax=ax1,
                   color='skyblue', errorbar=('ci', 95))
        ax1.set_title('Standard Deviation by Subcarrier', fontsize=14)
        ax1.set_xlabel('Subcarrier Index', fontsize=12)
        ax1.set_ylabel('Standard Deviation', fontsize=12)
        
        # Plot 2: Mean values
        ax2 = fig.add_subplot(gs[0, 1])
        sns.barplot(data=stats_data, x='Subcarrier', y='Mean', ax=ax2,
                   color='lightgreen', errorbar=('ci', 95))
        ax2.set_title('Mean Value by Subcarrier', fontsize=14)
        ax2.set_xlabel('Subcarrier Index', fontsize=12)
        ax2.set_ylabel('Mean', fontsize=12)
        
        # Plot 3: Range
        ax3 = fig.add_subplot(gs[1, 0])
        sns.barplot(data=stats_data, x='Subcarrier', y='Range', ax=ax3,
                   color='salmon')
        ax3.set_title('Amplitude Range by Subcarrier', fontsize=14)
        ax3.set_xlabel('Subcarrier Index', fontsize=12)
        ax3.set_ylabel('Range (Max - Min)', fontsize=12)
        
        # Plot 4: Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        data_array = [data[col] for col in subcarrier_cols]
        sns.violinplot(data=data_array, ax=ax4, inner='box', color='lightgray')
        ax4.set_title('Subcarrier Value Distribution', fontsize=14)
        ax4.set_xlabel('Subcarrier Index', fontsize=12)
        ax4.set_ylabel('CSI Amplitude', fontsize=12)
        
        # Add summary statistics
        duration = time.iloc[-1] if 'time' in locals() and hasattr(time, 'iloc') else (time[-1] if 'time' in locals() else len(data))
        summary_text = (
            f"Total Subcarriers: {len(subcarrier_cols)}\n"
            f"Total Samples: {len(data)}\n"
            f"Duration: {duration:.2f} seconds"
        )
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0.3, wspace=0.2)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def process_file(self, filepath: str):
        """
        Process a single file and generate visualizations.
        
        Args:
            filepath (str): Path to the CSV file
        """
        print(f"Processing {os.path.basename(filepath)}...")
        
        # Load the data
        result = self.load_data(filepath)
        if result is None:
            return
            
        data = result['data']
        metadata = result['metadata']
        base_name = os.path.splitext(result['filename'])[0]
        
        # Generate plots
        self.plot_time_series(
            data, metadata,
            os.path.join(self.plot_dir, f"{base_name}_time_series.png")
        )
        
        self.plot_heatmap(
            data, metadata,
            os.path.join(self.plot_dir, f"{base_name}_heatmap.png")
        )
        
        self.plot_statistics(
            data, metadata,
            os.path.join(self.plot_dir, f"{base_name}_statistics.png")
        )
        
    def process_all_files(self):
        """
        Process all files in the data directory.
        """
        print(f"Looking for top 8 subcarrier files in {self.data_dir}...")
        
        for filename in os.listdir(self.data_dir):
            if filename.startswith('top8_') and filename.endswith('.csv'):
                filepath = os.path.join(self.data_dir, filename)
                self.process_file(filepath)
                
        print("Processing complete!")

def main():
    """
    Main execution function.
    """
    visualizer = TopSubcarrierVisualizer()
    visualizer.process_all_files()

if __name__ == "__main__":
    main() 