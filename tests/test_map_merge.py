import os
import pytest
import pandas as pd
import numpy as np
from indexed_parquet_dataset import IndexedParquetDataset

@pytest.fixture
def complex_ds(tmp_path):
    d = tmp_path / "complex"
    d.mkdir()
    df1 = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
    df1.to_parquet(d / "f1.parquet")
    df2 = pd.DataFrame({"id": [4, 5, 6], "val": ["d", "e", "f"]})
    df2.to_parquet(d / "f2.parquet")
    return IndexedParquetDataset.from_folder(str(d))

def test_map_basic(complex_ds):
    """[POSITIVE TEST] Verifies basic row transformation via map()."""
    ds = complex_ds.map(lambda row: {**row, "new_col": row["id"] * 10})
    
    assert len(ds) == 6
    assert ds[0]["new_col"] == 10
    assert ds[5]["new_col"] == 60
    assert "new_col" in ds[0]

def test_map_remove_columns(complex_ds):
    """[POSITIVE TEST] Verifies map() with explicit column removal."""
    ds = complex_ds.map(lambda row: {"id_only": row["id"]}, remove_columns=["id", "val"])
    
    row = ds[0]
    assert "id_only" in row
    assert "id" not in row
    assert "val" not in row
    assert row["id_only"] == 1

def test_map_chained(complex_ds):
    """[POSITIVE TEST] Verifies that multiple map() calls chain correctly."""
    ds = (complex_ds
          .map(lambda row: {**row, "a": row["id"] + 1})
          .map(lambda row: {**row, "b": row["a"] * 2}))
    
    assert ds[0]["a"] == 2
    assert ds[0]["b"] == 4

def test_merge_deduplication(complex_ds):
    """[POSITIVE TEST] Verifies that merging datasets with overlapping rows deduplicates them."""
    ds1 = complex_ds.filter(column_conditions={"id": ("<=", 3)}) # rows 1, 2, 3
    ds2 = complex_ds.filter(column_conditions={"id": (">=", 2)}) # rows 2, 3, 4, 5, 6
    
    merged = ds1.merge(ds2)
    
    # Rows 2 and 3 are in both. 
    # Total unique rows: 1, 2, 3, 4, 5, 6 => count should be 6
    assert len(merged) == 6
    
    # Verify order: ds1 rows first (1, 2, 3), then new rows from ds2 (4, 5, 6)
    ids = [row["id"] for row in merged]
    assert ids == [1, 2, 3, 4, 5, 6]

def test_merge_disjoint(tmp_path):
    """[POSITIVE TEST] Verifies merging of datasets from completely different files."""
    d1 = tmp_path / "d1"
    d1.mkdir()
    pd.DataFrame({"x": [1]}).to_parquet(d1 / "f1.parquet")
    ds1 = IndexedParquetDataset.from_folder(str(d1))
    
    d2 = tmp_path / "d2"
    d2.mkdir()
    pd.DataFrame({"x": [2]}).to_parquet(d2 / "f2.parquet")
    ds2 = IndexedParquetDataset.from_folder(str(d2))
    
    merged = ds1.merge(ds2)
    assert len(merged) == 2
    assert merged[0]["x"] == 1
    assert merged[1]["x"] == 2

def test_merge_identity(complex_ds):
    """[POSITIVE TEST] Merging a dataset with itself should result in the same dataset."""
    merged = complex_ds.merge(complex_ds)
    assert len(merged) == len(complex_ds)
    assert [r["id"] for r in merged] == [r["id"] for r in complex_ds]
