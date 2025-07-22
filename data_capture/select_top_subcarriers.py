import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import shutil

class TopSubcarrierSelector:
    """
    A class for selecting and storing the 8 most active subcarriers from filtered CSI data.
    Uses variance and range metrics to determine subcarrier activity.
    """
    
    def __init__(self, base_dir: str = 'data_capture/csi'):
        """
        Initialize the selector with directory configuration.
        
        Args:
            base_dir (str): Base directory containing CSI files
        """
        self.base_dir = base_dir
        self.filtered_dir = os.path.join(base_dir, 'filtered')
        self.output_dir = os.path.join(base_dir, '8_csi_filtered')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parameters for subcarrier selection
        self.n_top_subcarriers = 8
        self.min_samples = 20  # Minimum samples needed for analysis
        
    def calculate_subcarrier_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate activity metrics for each subcarrier.
        
        Args:
            data (pd.DataFrame): The filtered CSI data
            
        Returns:
            pd.DataFrame: DataFrame containing subcarrier metrics
        """
        subcarrier_cols = [col for col in data.columns if col.startswith('subcarrier_')]
        
        metrics = pd.DataFrame()
        metrics['Subcarrier'] = subcarrier_cols
        metrics['Variance'] = data[subcarrier_cols].var()
        metrics['Range'] = data[subcarrier_cols].max() - data[subcarrier_cols].min()
        metrics['Mean'] = data[subcarrier_cols].mean()
        metrics['Std'] = data[subcarrier_cols].std()
        
        # Calculate combined activity score
        # Normalize each metric to [0,1] range
        for col in ['Variance', 'Range', 'Std']:
            metrics[f'{col}_normalized'] = (metrics[col] - metrics[col].min()) / (metrics[col].max() - metrics[col].min())
        
        # Calculate weighted activity score
        metrics['Activity_Score'] = (
            metrics['Variance_normalized'] * 0.4 +  # Variance is most important
            metrics['Range_normalized'] * 0.3 +     # Range is second most important
            metrics['Std_normalized'] * 0.3         # Standard deviation is third most important
        )
        
        return metrics
        
    def select_top_subcarriers(self, data: pd.DataFrame) -> List[str]:
        """
        Select the top N most active subcarriers based on activity metrics.
        
        Args:
            data (pd.DataFrame): The filtered CSI data
            
        Returns:
            List[str]: List of selected subcarrier column names
        """
        if len(data) < self.min_samples:
            print(f"Warning: Not enough samples ({len(data)}) for reliable analysis")
            return []
            
        metrics = self.calculate_subcarrier_metrics(data)
        
        # Sort by activity score and select top N
        top_subcarriers = metrics.nlargest(self.n_top_subcarriers, 'Activity_Score')['Subcarrier'].tolist()
        
        return top_subcarriers
        
    def process_file(self, filepath: str) -> Optional[Dict]:
        """
        Process a single filtered CSI file to select top subcarriers.
        
        Args:
            filepath (str): Path to the filtered CSV file
            
        Returns:
            Optional[Dict]: Dictionary containing the selected data and metadata
        """
        try:
            # Load the data
            data = pd.read_csv(filepath)
            if data.empty or len(data.columns) == 0:
                print(f"Warning: No valid data found in {filepath}")
                return None
                
            # Select top subcarriers
            top_subcarriers = self.select_top_subcarriers(data)
            if not top_subcarriers:
                print(f"Warning: Could not select subcarriers from {filepath}")
                return None
                
            # Create output data with selected subcarriers
            output_data = data[top_subcarriers].copy()
            
            # Add timestamp if available
            if 'timestamp' in data.columns:
                output_data['timestamp'] = data['timestamp']
            
            # Calculate metadata
            metadata = {
                'Original_File': os.path.basename(filepath),
                'Selected_Subcarriers': len(top_subcarriers),
                'Total_Samples': len(output_data),
                'Subcarrier_Indices': [int(col.split('_')[1]) for col in top_subcarriers]
            }
            
            return {
                'data': output_data,
                'metadata': metadata,
                'filename': os.path.basename(filepath)
            }
            
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")
            return None
            
    def save_results(self, result: Dict, output_path: str):
        """
        Save the selected subcarrier data and metadata.
        
        Args:
            result (Dict): Dictionary containing the data and metadata
            output_path (str): Base path for saving the results
        """
        try:
            # Save data
            data_path = os.path.join(self.output_dir, f"top8_{result['filename']}")
            result['data'].to_csv(data_path, index=False)
            
            # Save metadata
            metadata_path = os.path.join(self.output_dir, f"top8_metadata_{result['filename'].replace('.csv', '.txt')}")
            with open(metadata_path, 'w') as f:
                for key, value in result['metadata'].items():
                    f.write(f"{key}: {value}\n")
                    
            print(f"Saved selected subcarriers to {data_path}")
            print(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            
    def process_all_files(self):
        """
        Process all filtered CSI files in the filtered directory.
        """
        print(f"Looking for filtered CSI files in {self.filtered_dir}...")
        
        for filename in os.listdir(self.filtered_dir):
            if filename.startswith('filtered_') and filename.endswith('.csv'):
                filepath = os.path.join(self.filtered_dir, filename)
                print(f"\nProcessing {filename}...")
                
                result = self.process_file(filepath)
                if result is not None:
                    self.save_results(result, self.output_dir)
                    
        print("\nProcessing complete!")

def main():
    """
    Main execution function.
    """
    selector = TopSubcarrierSelector()
    selector.process_all_files()

if __name__ == "__main__":
    main() 