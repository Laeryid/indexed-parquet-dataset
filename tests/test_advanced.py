import os
import pytest
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pyarrow as pa
import pyarrow.parquet as pq
from indexed_parquet import IndexedParquetDataset

@pytest.fixture
def type_mismatch_dir(tmp_path):
    d = tmp_path / "mismatch_data"
    d.mkdir()
    
    # File 1: id as int
    df1 = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    df1.to_parquet(d / "f1.parquet")
    
    # File 2: id as float
    df2 = pd.DataFrame({"id": [3.5, 4.5], "val": [30, 40]})
    df2.to_parquet(d / "f2.parquet")
    
    return str(d)

def test_concat_type_upcasting(type_mismatch_dir):
    """[POSITIVE TEST] Verifies automatic type upcasting during concatenation."""
    ds1 = IndexedParquetDataset.from_folder(type_mismatch_dir, pattern="f1.parquet")
    ds2 = IndexedParquetDataset.from_folder(type_mismatch_dir, pattern="f2.parquet")
    
    with pytest.warns(UserWarning, match="Type mismatch for column 'id'"):
        ds_merge = ds1.merge(ds2)
    
    assert len(ds_merge) == 4
    # All values should be read as floats now
    assert isinstance(ds_merge[0]["id"], float)
    assert ds_merge[0]["id"] == 1.0
    assert ds_merge[2]["id"] == 3.5

@pytest.fixture
def hierarchy_dir(tmp_path):
    d = tmp_path / "hierarchy_data"
    d.mkdir()
    # File with only 'id'
    df = pd.DataFrame({"id": [1, 2]})
    df.to_parquet(d / "f1.parquet")
    return str(d)

def test_fill_values_hierarchy(hierarchy_dir):
    """[POSITIVE TEST] Verifies hierarchical fill value resolution (Default < Type < Column)."""
    # Setup dataset with 3 levels of fill values
    # Default < By Type < By Column
    ds = IndexedParquetDataset.from_folder(
        hierarchy_dir,
        default_fill_value="global_default",
        fill_values_by_type={"int64": "type_default"},
        fill_values_by_column={"missing_col": "col_default"}
    )
    
    # We need to simulated missing columns that are in the index but not in file
    # Let's manually add some columns to the index metadata to test this
    # OR better: use concat with a dataset that has those columns
    
    df2 = pd.DataFrame({"id": [3], "missing_col": [100], "other_col": [200]})
    # Create a dummy dataset just to get its index
    import shutil
    d2 = os.path.join(hierarchy_dir, "temp_concat")
    os.makedirs(d2, exist_ok=True)
    df2.to_parquet(os.path.join(d2, "f2.parquet"))
    ds2 = IndexedParquetDataset.from_folder(d2)
    
    ds_final = ds.merge(ds2)
    
    # Row 0 (from f1) belongs to ds (hierarchy_dir)
    # It doesn't have 'missing_col' or 'other_col'
    row0 = ds_final[0]
    
    # 1. 'missing_col' should be 'col_default' (overrides type and global)
    assert row0["missing_col"] == "col_default"
    
    # 2. 'other_col' should be 'type_default' if we knew its type
    # In concat, other_col will have type 'int64' from ds2
    assert row0["other_col"] == "type_default"
    
    # 3. Some totally unknown column? Only global default
    # (Actually it won't be in schema unless we alias it or it's in index)

def test_cast_method(hierarchy_dir):
    """[POSITIVE TEST] Verifies explicit column type casting."""
    ds = IndexedParquetDataset.from_folder(hierarchy_dir)
    # Cast int id to string
    cast_ds = ds.cast("id", str)
    assert isinstance(cast_ds[0]["id"], str)
    assert cast_ds[0]["id"] == "1"
    
    # Cast to float
    cast_ds_float = ds.cast("id", float)
    assert isinstance(cast_ds_float[0]["id"], float)
    assert cast_ds_float[0]["id"] == 1.0

def test_info_smoke(type_mismatch_dir, capsys):
    """[POSITIVE TEST] Smoke test for info() method."""
    ds = IndexedParquetDataset.from_folder(type_mismatch_dir)
    ds.info()
    captured = capsys.readouterr()
    assert "IndexedParquetDataset Summary" in captured.out
    assert "Rows (Visible/Total)" in captured.out
    
    # Test with empty dataset
    empty_ds = ds.filter(lambda x: False)
    empty_ds.info()
    captured = capsys.readouterr()
    assert "Rows (Visible/Total):    0/" in captured.out

def test_to_parquet_and_materialization(type_mismatch_dir, tmp_path):
    """[POSITIVE TEST] Verifies dataset materialization to a single parquet file."""
    dataset = IndexedParquetDataset.from_folder(type_mismatch_dir)
    # Apply some logic
    processed = dataset.alias("id_plus_1", lambda x: x["id"] + 1).filter(lambda x: x["id"] > 1)
    
    out_path = str(tmp_path / "materialized.parquet")
    processed.to_parquet(out_path)
    
    assert os.path.exists(out_path)
    
    # Verify content
    df = pd.read_parquet(out_path)
    assert len(df) == 3 # 2.0, 3.5, 4.5
    assert "id_plus_1" in df.columns
    assert list(df["id_plus_1"]) == [3.0, 4.5, 5.5]

def test_dataloader_integration(type_mismatch_dir):
    """[POSITIVE TEST] Verifies basic PyTorch DataLoader integration with type upcasting."""
    dataset = IndexedParquetDataset.from_folder(type_mismatch_dir)
    # Since we have strings/floats, we might need a custom collate or just check basic
    # default_collate handles dicts of tensors/numbers
    
    loader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=dataset.generate_collate_fn()
    )
    
    batch = next(iter(loader))
    assert "id" in batch
    assert len(batch["id"]) == 2
    assert isinstance(batch["id"], torch.Tensor)

def test_complex_fluent_chaining(type_mismatch_dir):
    """[POSITIVE TEST] Verifies complex fluent API chaining (alias -> cast -> filter -> shuffle -> limit)."""
    # From folder -> alias -> cast -> filter -> shuffle -> limit
    result = (
        IndexedParquetDataset.from_folder(type_mismatch_dir)
        .alias("str_id", lambda x: f"ID-{x['id']}")
        .cast("val", float)
        .filter(lambda x: x["id"] < 4)
        .shuffle(seed=42)
        .limit(2)
    )
    
    assert len(result) == 2
    row = result[0]
    assert row["str_id"].startswith("ID-")
    assert isinstance(row["val"], float)
    assert row["id"] < 4
