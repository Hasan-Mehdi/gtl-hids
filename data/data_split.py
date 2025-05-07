import pandas as pd
import numpy as np
import os

# Set random seed
np.random.seed(42)

def combine_csv_files(input_files, output_file):
    """
    Combine multiple CSV files into a single output file.
    
    Args:
        input_files (list): List of input CSV file paths
        output_file (str): Output file path
    """
    # Create an empty list to store dataframes
    dfs = []
    
    # Read each input file and append to the list
    for file_path in input_files:
        print(f"Reading {file_path}...")
        try:
            # Using tab as separator based on the sample data
            df = pd.read_csv(file_path, sep='\t')
            print(f" Found {len(df)} rows and {len(df.columns)} columns")
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Concatenate all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Write to output file
        print(f"Writing {len(combined_df)} rows to {output_file}...")
        combined_df.to_csv(output_file, index=False, sep='\t')
        print("Done!")
        return combined_df
    else:
        print("No data to combine!")
        return None

# Main execution
if __name__ == "__main__":
    # Input files
    input_files = ["monday.csv", "tuesday.csv", "wednesday.csv"]  # Dataset B
    # input_files = ["monday.csv", "tuesday.csv", "wednesday.csv", "thursday.csv", "friday.csv"]  # Dataset A
    
    # Check if files exist
    for file in input_files:
        if not os.path.exists(file):
            print(f"Warning: {file} does not exist in the current directory")
    
    # Output file
    output_file = "finetune_dataset_B.csv"
    
    # Combine files
    combined_data = combine_csv_files(input_files, output_file)
    
    # Print summary
    if combined_data is not None:
        print("\nSummary:")
        print(f"Total rows in combined file: {len(combined_data)}")
        print(f"Columns: {', '.join(combined_data.columns)}")