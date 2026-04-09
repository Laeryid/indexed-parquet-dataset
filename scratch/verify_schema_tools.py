import os
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from indexed_parquet import IndexedParquetDataset

def create_demo_data(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
    table1 = pa.Table.from_pydict({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
        "c": [10.5, 20.5, 30.5]
    })
    pq.write_table(table1, os.path.join(path, "file1.parquet"))
    
    table2 = pa.Table.from_pydict({
        "a": [4, 5],
        "b": ["u", "v"],
        "c": [40.5, 50.5]
    })
    pq.write_table(table2, os.path.join(path, "file2.parquet"))

def main():
    data_dir = "scratch/test_data"
    create_demo_data(data_dir)
    
    # 1. Load dataset
    ds = IndexedParquetDataset.from_folder(data_dir)
    print(f"Initial schema: {ds.schema}")
    print(f"Row 0: {ds[0]}")
    
    # 2. Add computed column
    ds2 = ds.alias("sum_abc", lambda row: row["a"] + row["c"])
    print(f"\nAfter alias (computed column): {ds2.schema}")
    print(f"Row 0 with sum_abc: {ds2[0]}")
    
    # 3. Cast column
    ds3 = ds2.cast("a", str)
    print(f"\nAfter cast 'a' to str: {ds3.schema}")
    row0 = ds3[0]
    print(f"Row 0 'a' type: {type(row0['a'])} (Value: {row0['a']})")
    
    # 4. Replace existing column
    ds4 = ds3.alias("b", lambda row: row["b"].upper())
    print(f"\nAfter replacing 'b' with upper case: {ds4.schema}")
    print(f"Row 0 'b': {ds4[0]['b']}")
    
    # 5. Clone (Materialize)
    clone_path = "scratch/materialized.parquet"
    if os.path.exists(clone_path): os.remove(clone_path)
    
    print("\nCloning (Materializing)...")
    fast_ds = ds4.clone(clone_path)
    
    print(f"Cloned schema: {fast_ds.schema}")
    print(f"Cloned row 0: {fast_ds[0]}")
    print(f"Cloned row 0 'b' should be uppercase: {fast_ds[0]['b']}")
    print(f"Cloned row 0 'a' should be str: {type(fast_ds[0]['a'])}")
    
    # Verify file exists
    if os.path.exists(clone_path):
        print(f"\nSuccess! Materialized file created at {clone_path}")
    else:
        print("\nFailure! Materialized file NOT created.")

if __name__ == "__main__":
    main()
