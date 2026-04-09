import os
import pytest
import pandas as pd
import numpy as np
from indexed_parquet import IndexedParquetDataset

@pytest.fixture
def multi_file_dir(tmp_path):
    test_dir = tmp_path / "multi_file_data"
    test_dir.mkdir()
    
    # Create subdirectories
    sub1 = test_dir / "202601"
    sub2 = test_dir / "202602"
    sub1.mkdir()
    sub2.mkdir()
    
    # File 1
    df1 = pd.DataFrame({"id": [1], "val": ["v1"]})
    df1.to_parquet(sub1 / "part_1.parquet")
    
    # File 2
    df2 = pd.DataFrame({"id": [2], "val": ["v2"]})
    df2.to_parquet(sub1 / "part_2.parquet")
    
    # File 3
    df3 = pd.DataFrame({"id": [3], "val": ["v3"]})
    df3.to_parquet(sub2 / "part_1.parquet")
    
    return str(test_dir)

def test_source_file_column(multi_file_dir):
    # Enabled
    dataset = IndexedParquetDataset.from_folder(multi_file_dir)
    dataset.include_source_column = True
    
    # Check schema
    assert "__source_file__" in dataset.schema
    
    # Check row content
    row = dataset[0]
    assert "__source_file__" in row
    assert row["__source_file__"].endswith("part_1.parquet")
    assert os.path.isabs(row["__source_file__"])

def test_source_file_column_custom_name(multi_file_dir):
    dataset = IndexedParquetDataset.from_folder(multi_file_dir)
    dataset.include_source_column = True
    dataset.source_column_name = "path_to_source"
    
    assert "path_to_source" in dataset.schema
    row = dataset[0]
    assert "path_to_source" in row
    assert "__source_file__" not in row

def test_path_filter_single_glob(multi_file_dir):
    dataset = IndexedParquetDataset.from_folder(multi_file_dir)
    
    # All files matching 202601
    filtered = dataset.filter(path_filter="**/202601/*.parquet")
    assert len(filtered) == 2
    
    # All rows should be from 202601
    for i in range(len(filtered)):
        # We can temporarily enable source column to check
        item = filtered.copy()[i] # copy creates new ds instance
        # Actually better check via indices
        f_idx, _ = filtered._get_file_and_local_idx(i)
        assert "202601" in filtered.index.files[f_idx].path

def test_path_filter_multi_glob(multi_file_dir):
    dataset = IndexedParquetDataset.from_folder(multi_file_dir)
    
    # Specific files by different masks
    filtered = dataset.filter(path_filter=[
        "**/202601/part_1.parquet",
        "**/202602/part_1.parquet"
    ])
    assert len(filtered) == 2
    
    ids = [filtered[i]["id"] for i in range(len(filtered))]
    assert set(ids) == {1, 3}

def test_path_filter_and_column_filter(multi_file_dir):
    dataset = IndexedParquetDataset.from_folder(multi_file_dir)
    
    filtered = dataset.filter(
        path_filter="**/202601/*.parquet",
        column_conditions={"id": 2}
    )
    assert len(filtered) == 1
    assert filtered[0]["id"] == 2

def test_persistence_of_settings(multi_file_dir, tmp_path):
    dataset = IndexedParquetDataset.from_folder(multi_file_dir)
    dataset.include_source_column = True
    dataset.source_column_name = "src"
    
    # Test copy/shuffle/select
    ds_copy = dataset.copy()
    assert ds_copy.include_source_column is True
    assert ds_copy.source_column_name == "src"
    
    # Test save/load
    idx_path = tmp_path / "index.pkl"
    dataset.save_index(str(idx_path))
    
    ds_loaded = IndexedParquetDataset.load_index(str(idx_path))
    assert ds_loaded.include_source_column is True
    assert ds_loaded.source_column_name == "src"
    assert "src" in ds_loaded[0]
