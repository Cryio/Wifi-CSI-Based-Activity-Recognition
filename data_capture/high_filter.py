import numpy as np
import pandas as pd
import os
from datetime import datetime
import glob
from typing import Dict, List

class CSIFilter:
    """
    A class for analyzing CSI (Channel State Information) data and selecting active subcarriers
    based on their variation patterns.
    """
    
    def __init__(self, base_dir: str = 'data_capture/csi'):
        """
        Initialize the CSIFilter with directory configuration.
        
        Args:
            base_dir (str): Base directory containing CSI files
        """
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, 'filtered')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parameters for activity detection
        self.variance_threshold = 1.0   # Base threshold for variance
        self.change_threshold = 0.02    # 2% changes are sufficient
        self.window_size = 5           # Window for analyzing changes
        self.min_samples = 20          # Minimum samples needed
        self.std_threshold = 1.2       # Threshold for detecting changes
        
        # Parameters for relative deviation analysis
        self.relative_std_threshold = 0.1  # Minimum relative standard deviation (10% of max)
        self.relative_range_threshold = 0.15  # Minimum relative range (15% of max)
        self.noise_floor = 1e-6  # Minimum acceptable variation

    def parse_csi_data(self, csi_data_str: str) -> np.ndarray:
        """
        Parse the CSI_Data string into a numpy array of subcarrier values.
        
        Args:
            csi_data_str (str): The CSI_Data string from the CSV
            
        Returns:
            np.ndarray: Array of subcarrier values
        """
        try:
            if isinstance(csi_data_str, float):
                return np.array([])
                
            # Handle different data formats
            if isinstance(csi_data_str, str):
                # Remove any square brackets and split by spaces or commas
                cleaned_str = csi_data_str.replace('[', '').replace(']', '')
                values = []
                
                # Try different separators
                if ',' in cleaned_str:
                    parts = cleaned_str.split(',')
                else:
                    parts = cleaned_str.split()
                
                # Convert values
                for v in parts:
                    try:
                        val = float(v.strip())
                        if not np.isnan(val) and not np.isinf(val):
                            values.append(val)
                    except (ValueError, TypeError):
                        continue
                        
                if not values:
                    return np.array([])
                    
                return np.array(values, dtype=np.float32)
            elif isinstance(csi_data_str, (list, np.ndarray)):
                # Handle array-like input
                return np.array([x for x in csi_data_str if isinstance(x, (int, float))], dtype=np.float32)
            else:
                return np.array([])
                
        except Exception as e:
            print(f"Error parsing CSI data: {str(e)}")
            return np.array([])

    def analyze_subcarrier_activity(self, subcarrier_data: np.ndarray) -> bool:
        """
        Analyze if a subcarrier shows significant activity.
        
        Args:
            subcarrier_data (np.ndarray): Time series data for one subcarrier
            
        Returns:
            bool: True if the subcarrier shows significant activity
        """
        if len(subcarrier_data) < self.min_samples:
            return False
            
        # Remove any NaN or inf values
        valid_data = subcarrier_data[~np.isnan(subcarrier_data) & ~np.isinf(subcarrier_data)]
        if len(valid_data) < self.min_samples:
            return False
            
        # Normalize the data to handle different scales
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        if std_val == 0:
            return False
            
        normalized_data = (valid_data - mean_val) / std_val
        
        # Calculate basic statistics
        variance = np.var(normalized_data)
        
        # If variance is too low, subcarrier is likely static
        if variance < self.variance_threshold:
            return False
            
        # Calculate temporal changes using rolling windows
        significant_changes = 0
        total_windows = len(normalized_data) - self.window_size
        
        if total_windows <= 0:
            return False
        
        for i in range(total_windows):
            window = normalized_data[i:i + self.window_size]
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            if window_std == 0:
                continue
                
            # Check if there's significant change in the next sample
            if i + self.window_size < len(normalized_data):
                next_value = normalized_data[i + self.window_size]
                if abs(next_value - window_mean) > self.std_threshold * window_std:
                    significant_changes += 1
        
        # Calculate ratio of significant changes
        change_ratio = significant_changes / total_windows
        
        return change_ratio >= self.change_threshold

    def analyze_relative_deviations(self, filtered_data: Dict[str, np.ndarray]) -> List[str]:
        """
        Analyze relative deviations between subcarriers to identify those with meaningful variations.
        
        Args:
            filtered_data (Dict[str, np.ndarray]): Dictionary of filtered subcarrier data
            
        Returns:
            List[str]: List of subcarrier keys to keep
        """
        if not filtered_data:
            return []
            
        # Calculate statistics for each subcarrier
        stats = {}
        for key, data in filtered_data.items():
            if len(data) < self.min_samples:
                continue
                
            # Remove any NaN or inf values
            valid_data = data[~np.isnan(data) & ~np.isinf(data)]
            if len(valid_data) < self.min_samples:
                continue
                
            stats[key] = {
                'std': np.std(valid_data),
                'range': np.ptp(valid_data),  # Peak-to-peak (max - min)
                'mean': np.mean(valid_data),
                'median': np.median(valid_data)
            }
        
        if not stats:
            return list(filtered_data.keys())
            
        # Find maximum statistics across all subcarriers
        max_std = max(s['std'] for s in stats.values())
        max_range = max(s['range'] for s in stats.values())
        
        # Only keep subcarriers with significant relative deviation
        keep_subcarriers = []
        for key, stat in stats.items():
            # Calculate relative metrics
            relative_std = stat['std'] / max_std if max_std > self.noise_floor else 0
            relative_range = stat['range'] / max_range if max_range > self.noise_floor else 0
            
            # Check if subcarrier shows significant variation
            if (relative_std >= self.relative_std_threshold or 
                relative_range >= self.relative_range_threshold):
                keep_subcarriers.append(key)
        
        return keep_subcarriers if keep_subcarriers else list(filtered_data.keys())

    def process_file(self, filepath: str) -> Dict:
        """
        Process a single CSI file to identify active subcarriers based on variation.
        
        Args:
            filepath (str): Path to the CSI file
            
        Returns:
            dict: Dictionary containing processed data and metadata
        """
        try:
            # Load data
            df = pd.read_csv(filepath)
            if 'CSI_Data' not in df.columns:
                raise ValueError("CSI_Data column not found in the file")
            
            # Process each row's CSI data
            all_subcarriers = []
            n_subcarriers = None
            
            # First pass to determine consistent number of subcarriers
            for csi_data_str in df['CSI_Data']:
                subcarriers = self.parse_csi_data(csi_data_str)
                if len(subcarriers) > 0:
                    if n_subcarriers is None:
                        n_subcarriers = len(subcarriers)
                    elif len(subcarriers) != n_subcarriers:
                        continue  # Skip inconsistent rows
                    all_subcarriers.append(subcarriers)
            
            if not all_subcarriers:
                raise ValueError("No valid CSI data found")
            
            # Convert to numpy array
            all_subcarriers = np.array(all_subcarriers)
            
            # Process each subcarrier
            filtered_data = {}
            active_subcarriers = []
            
            print(f"\nAnalyzing {n_subcarriers} subcarriers for activity...")
            
            for i in range(n_subcarriers):
                subcarrier_data = all_subcarriers[:, i]
                
                # Check if this subcarrier shows significant activity
                if self.analyze_subcarrier_activity(subcarrier_data):
                    filtered_data[f"subcarrier_{i}"] = subcarrier_data
                    active_subcarriers.append(i)
            
            print(f"Found {len(active_subcarriers)} active subcarriers out of {n_subcarriers}")
            
            if not filtered_data:
                print("Warning: No active subcarriers detected. Keeping all subcarriers.")
                for i in range(n_subcarriers):
                    filtered_data[f"subcarrier_{i}"] = all_subcarriers[:, i]
                    active_subcarriers.append(i)
            else:
                # Analyze relative deviations to further filter subcarriers
                keep_subcarriers = self.analyze_relative_deviations(filtered_data)
                if len(keep_subcarriers) < len(filtered_data):
                    print(f"Removing {len(filtered_data) - len(keep_subcarriers)} subcarriers with minimal deviation")
                    filtered_data = {k: filtered_data[k] for k in keep_subcarriers}
                    active_subcarriers = [int(k.split('_')[1]) for k in keep_subcarriers]
            
            # Create output DataFrame with consistent indexing
            result_df = pd.DataFrame(filtered_data, index=range(len(all_subcarriers)))
            
            # Add timestamps if available
            if 'Real_Timestamp' in df.columns:
                result_df['timestamp'] = df['Real_Timestamp'].iloc[:len(all_subcarriers)]
            
            return {
                'data': result_df,
                'active_subcarriers': active_subcarriers,
                'n_samples': len(result_df),
                'activity_stats': {
                    'total_subcarriers': n_subcarriers,
                    'active_subcarriers': len(active_subcarriers),
                    'activity_ratio': len(active_subcarriers) / n_subcarriers if n_subcarriers else 0
                }
            }
            
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")
            return None

    def batch_process_directory(self) -> None:
        """
        Process all CSI files in the directory to identify active subcarriers.
        """
        # Find all CSI files
        csi_files = glob.glob(os.path.join(self.base_dir, 'csi_data_*.csv'))
        
        for filepath in csi_files:
            filename = os.path.basename(filepath)
            print(f"\nProcessing {filename}...")
            
            # Process file
            result = self.process_file(filepath)
            
            if result is not None:
                # Generate output filename
                output_filename = f"filtered_{filename}"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Save filtered data
                result['data'].to_csv(output_path, index=False)
                
                # Save metadata
                metadata_filename = f"metadata_{filename.replace('.csv', '.txt')}"
                metadata_path = os.path.join(self.output_dir, metadata_filename)
                
                with open(metadata_path, 'w') as f:
                    f.write(f"Original File: {filename}\n")
                    f.write(f"Number of samples: {result['n_samples']}\n")
                    f.write(f"Number of subcarriers: {len(result['active_subcarriers'])}\n")
                
                print(f"Saved filtered data to {output_path}")
                print(f"Saved metadata to {metadata_path}")
            else:
                print(f"Skipped {filename} due to processing errors")

def main():
    """
    Main execution function to process CSI files.
    """
    filter_processor = CSIFilter()
    print("Starting CSI data processing...")
    filter_processor.batch_process_directory()
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
