import os
import pytest
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from indexed_parquet import IndexedParquetDataset, SchemaMapper

@pytest.fixture
def dataset1(tmp_path):
    d = tmp_path / "ds1"
    d.mkdir()
    df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    df.to_parquet(d / "f1.parquet")
    return IndexedParquetDataset.from_folder(str(d))

@pytest.fixture
def dataset2(tmp_path):
    d = tmp_path / "ds2"
    d.mkdir()
    df = pd.DataFrame({"a": [3, 4], "c": [30, 40]})
    df.to_parquet(d / "f2.parquet")
    return IndexedParquetDataset.from_folder(str(d))

def test_copy(dataset1):
    """[POSITIVE TEST] Verifies in-memory copy operation independence."""
    ds_copy = dataset1.copy()
    assert ds_copy is not dataset1
    assert np.array_equal(ds_copy.indices, dataset1.indices)
    assert len(ds_copy) == len(dataset1)
    
    # Modify copy indices and check independence
    ds_copy.indices = ds_copy.indices[:1]
    assert len(ds_copy) == 1
    assert len(dataset1) == 2

def test_clone(dataset1, tmp_path):
    """[POSITIVE TEST] Verifies dataset cloning (materialization) to a new parquet file."""
    dataset1 = dataset1.rename("a", "new_a")
    clone_file = str(tmp_path / "clone.parquet")
    cloned_ds = dataset1.clone(clone_file)
    
    assert os.path.exists(clone_file)
    assert len(cloned_ds) == 2
    assert "new_a" in cloned_ds.schema
    
    # Verify content via pandas
    df = pd.read_parquet(clone_file)
    assert "new_a" in df.columns
    assert list(df["new_a"]) == [1, 2]

def test_merge_simple(dataset1, dataset2):
    """[POSITIVE TEST] Verifies simple merging of two datasets with overlapping schemas."""
    ds_merge = dataset1.merge(dataset2)
    assert len(ds_merge) == 4
    
    # Check data from first part
    assert ds_merge[0]["a"] == 1
    assert ds_merge[0]["b"] == 10
    assert ds_merge[0]["c"] is None
    
    # Check data from second part
    assert ds_merge[2]["a"] == 3
    assert ds_merge[2]["b"] is None
    assert ds_merge[2]["c"] == 30

def test_merge_with_aliases(dataset1, dataset2):
    """[POSITIVE TEST] Verifies merging of datasets with complex alias mappings."""
    # ds1: a -> alias1
    # ds2: a -> alias2
    ds1 = dataset1.rename("a", "alias1")
    ds2 = dataset2.rename("a", "alias2")
    
    ds_merge = ds1.merge(ds2)
    
    # Schema should have both aliases
    assert "alias1" in ds_merge.schema
    assert "alias2" in ds_merge.schema
    
    # Row 0 (from ds1) should have alias1 but None for alias2
    assert ds_merge[0]["alias1"] == 1
    assert ds_merge[0]["alias2"] is None
    
    # Row 2 (from ds2) should have alias2 but None for alias1
    assert ds_merge[2]["alias2"] == 3
    assert ds_merge[2]["alias1"] is None

def test_clone_shuffled(dataset1, tmp_path):
    """[POSITIVE TEST] Verifies that cloning a shuffled dataset preserves the shuffled order in the file."""
    shuffled = dataset1.shuffle(seed=42)
    clone_file = str(tmp_path / "shuffled.parquet")
    shuffled.clone(clone_file) # Returns dataset, but we check the file
    
    df = pd.read_parquet(clone_file)
    assert len(df) == 2
    original_data = {0: 1, 1: 2} # id -> a
    expected_order = [original_data[idx] for idx in shuffled.indices]
    assert list(df["a"]) == expected_order
