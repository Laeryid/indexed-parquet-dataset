import os
import pandas as pd
import numpy as np
import pytest
from torch.utils.data import DataLoader
from indexed_parquet import IndexedParquetDataset

def test_updates():
    # Setup test data
    os.makedirs("test_verify", exist_ok=True)
    
    # File 1: has 'a' and 'b' (int)
    df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    df1.to_parquet("test_verify/file1.parquet")
    
    # File 2: has 'a' (int) and 'c' (string)
    df2 = pd.DataFrame({"a": [3], "c": ["hello"]})
    df2.to_parquet("test_verify/file2.parquet")
    
    # File 3: has 'b' as string (to test concat mismatch)
    df3 = pd.DataFrame({"b": ["30"]})
    os.makedirs("test_verify/other", exist_ok=True)
    df3.to_parquet("test_verify/other/file3.parquet")

    print("\n1. Testing FileNotFoundError")
    try:
        IndexedParquetDataset.from_folder("non_existent_folder")
    except FileNotFoundError as e:
        print(f"Success: {e}")

    print("\n2. Testing Schema Evolution with Defaults")
    # Missing 'c' in file1 will be "N/A", missing 'b' in file2 will be 0
    ds = IndexedParquetDataset.from_folder(
        "test_verify", 
        pattern="file[12].parquet",
        default_fill_value=-1,
        fill_values_by_column={"c": "N/A"},
        fill_values_by_type={"int64": 0}
    )
    
    print(f"Dataset length: {len(ds)}")
    print(f"Row 0 (file1): {ds[0]}") # Should have c="N/A"
    print(f"Row 2 (file2): {ds[2]}") # Should have b=0
    
    assert ds[0]['c'] == "N/A"
    assert ds[2]['b'] == 0

    print("\n3. Testing info() with Types")
    ds.info()

    print("\n4. Testing concat() with type upcasting")
    ds_other = IndexedParquetDataset.from_folder("test_verify/other")
    # ds has 'b' as int64, ds_other has 'b' as string
    import warnings
    with warnings.catch_warnings(record=True) as w:
        ds_combined = ds.concat(ds_other)
        print(f"Combined length: {len(ds_combined)}")
        if len(w) > 0:
            print(f"Warning caught: {w[-1].message}")
    
    print(f"Row 0 'b' type: {type(ds_combined[0]['b'])} value: {ds_combined[0]['b']}")
    print(f"Row 3 'b' type: {type(ds_combined[3]['b'])} value: {ds_combined[3]['b']}")
    assert isinstance(ds_combined[0]['b'], str)

    print("\n5. Testing DataLoader with generate_collate_fn")
    loader = DataLoader(ds, batch_size=2, collate_fn=ds.generate_collate_fn())
    batch = next(iter(loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch 'a' tensor: {batch['a']}")

    # Clean up
    import shutil
    shutil.rmtree("test_verify")
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_updates()
