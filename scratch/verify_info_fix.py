import os
import pandas as pd
import numpy as np
from indexed_parquet import IndexedParquetDataset

def test_info_fix(tmp_path_dir):
    # Setup
    df1 = pd.DataFrame({"id": range(10), "val": ["a"]*10})
    df1.to_parquet(os.path.join(tmp_path_dir, "file1.parquet"))
    
    df2 = pd.DataFrame({"id": range(10, 20), "val": ["b"]*10})
    df2.to_parquet(os.path.join(tmp_path_dir, "file2.parquet"))
    
    dataset = IndexedParquetDataset.from_folder(tmp_path_dir)
    
    print("\n--- FULL DATASET INFO ---")
    dataset.info()
    
    print("\n--- FILTERED BY PATH (file1 only) ---")
    filtered = dataset.filter(path_pattern="file1.parquet")
    filtered.info()
    
    print("\n--- FILTERED BY INDEX (first 5 rows) ---")
    subset = dataset.limit(5)
    subset.info()

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_info_fix(tmp)
