import pandas as pd
import pickle
import json
import pathlib
from pathlib import Path
import os
import glob

def convert_to_serializable(obj):
    """Recursively convert objects to JSON-serializable formats."""
    if isinstance(obj, (pathlib.Path, pathlib.PurePath)):
        return str(obj)
    elif hasattr(obj, 'tolist'):  # Handle numpy arrays
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    return obj

def main():
    print("Converting metadata files for Colab compatibility...")
    
    # 1. Convert dataset_df.pkl -> dataset_df.csv
    try:
        df_path = Path('data/processed/task-1.1/dataset_df.pkl')
        if df_path.exists():
            print(f"Loading {df_path}...")
            df = pd.read_pickle(df_path)
            
            # Convert any Path objects in dataframe to strings
            # Check object columns
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (pathlib.Path, pathlib.PurePath)) else x)
                
            csv_path = df_path.with_suffix('.csv')
            df.to_csv(csv_path, index=True) # Keep index
            print(f"Saved {csv_path}")
        else:
            print(f"Warning: {df_path} not found.")
    except Exception as e:
        print(f"Error converting dataset_df.pkl: {e}")

    # 2. Convert fold*.pkl -> fold*.json
    try:
        fold_files = glob.glob('data/processed/task-1.1/fold*.pkl')
        for pkl_path in fold_files:
            print(f"Converting {pkl_path}...")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Convert Paths and Arrays
            data_clean = convert_to_serializable(data)
            
            json_path = pkl_path.replace('.pkl', '.json')
            with open(json_path, 'w') as f:
                json.dump(data_clean, f)
            print(f"Saved {json_path}")
            
    except Exception as e:
        print(f"Error converting fold files: {e}")

    print("\nConversion complete!")
    print("Please upload the generated .csv and .json files to Colab.")

if __name__ == "__main__":
    main()
