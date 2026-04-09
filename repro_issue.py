import os
import pandas as pd
import numpy as np
from indexed_parquet import IndexedParquetDataset

def test_repro():
    tmp_dir = "repro_issue"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # CASE A: Remote col that exists ONLY in one file
    df1 = pd.DataFrame({"id": [1], "score": [0.5]})
    f1 = os.path.abspath(os.path.join(tmp_dir, "f1.parquet"))
    df1.to_parquet(f1)
    
    ds = IndexedParquetDataset.from_folder(tmp_dir)
    print(f"Original Schema: {ds.schema}")
    
    mapped_ds = ds.set_file_mapping(f1, {"score": "quality"})
    print(f"Mapped Schema (Case A): {mapped_ds.schema}")
    print(f"Row 0 keys: {list(mapped_ds[0].keys())}")
    
    # CASE B: Remap col that exists in OTHER files too
    df2 = pd.DataFrame({"id": [2], "score": [0.8]})
    f2 = os.path.abspath(os.path.join(tmp_dir, "f2.parquet"))
    df2.to_parquet(f2)
    
    ds_b = IndexedParquetDataset.from_folder(tmp_dir)
    print(f"\nOriginal Schema B: {ds_b.schema}")
    
    mapped_ds_b = ds_b.set_file_mapping(f1, {"score": "quality"})
    print(f"Mapped Schema B (Case B): {mapped_ds_b.schema}")
    print(f"Row 0 keys (Case B): {list(mapped_ds_b[0].keys())}")
    print(f"Row 1 keys (Case B): {list(mapped_ds_b[1].keys())}")

if __name__ == "__main__":
    test_repro()
