import os
import shutil
import pytest
import pandas as pd
import numpy as np
from indexed_parquet import IndexedParquetDataset

@pytest.fixture
def test_data_dir(tmp_path):
    """
    Fixture to create temporary parquet files for testing.
    tmp_path is a built-in pytest fixture for temporary directories.
    """
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    
    # Create 3 dummy parquet files
    # File 1: 10 rows
    df1 = pd.DataFrame({"id": range(0, 10), "val": [f"a_{i}" for i in range(10)], "label": [0]*10})
    df1.to_parquet(test_dir / "file1.parquet", row_group_size=5)
    
    # File 2: 20 rows
    df2 = pd.DataFrame({"id": range(10, 30), "val": [f"b_{i}" for i in range(10, 30)], "label": [1]*20})
    df2.to_parquet(test_dir / "file2.parquet", row_group_size=10)
    
    # File 3: 5 rows
    df3 = pd.DataFrame({"id": range(30, 35), "val": [f"c_{i}" for i in range(30, 35)], "label": [2]*5})
    df3.to_parquet(test_dir / "file3.parquet")
    
    return str(test_dir)

def test_basic_indexing(test_data_dir):
    dataset = IndexedParquetDataset.from_folder(test_data_dir)
    assert len(dataset) == 35
    
    # Check first and last elements
    assert dataset[0]["id"] == 0
    assert dataset[0]["val"] == "a_0"
    assert dataset[10]["id"] == 10
    assert dataset[34]["id"] == 34

def test_shuffle(test_data_dir):
    dataset = IndexedParquetDataset.from_folder(test_data_dir)
    shuffled = dataset.shuffle(seed=42)
    
    assert len(shuffled) == 35
    # Shuffled order should be different
    assert not np.array_equal(shuffled.indices, dataset.indices)
    
    # But data should be correct
    for i in range(len(shuffled)):
        row = shuffled[i]
        idx = shuffled.indices[i]
        if idx < 10:
            expected_val = f"a_{idx}"
        elif idx < 30:
            expected_val = f"b_{idx}"
        else:
            expected_val = f"c_{idx}"
        assert row["val"] == expected_val

def test_filter(test_data_dir):
    dataset = IndexedParquetDataset.from_folder(test_data_dir)
    filtered = dataset.filter(lambda x: x["label"] == 1)
    assert len(filtered) == 20
    for i in range(len(filtered)):
        assert filtered[i]["label"] == 1

def test_limit(test_data_dir):
    dataset = IndexedParquetDataset.from_folder(test_data_dir)
    limited = dataset.limit(5)
    assert len(limited) == 5
    assert limited[4]["id"] == 4

def test_map(test_data_dir):
    dataset = IndexedParquetDataset.from_folder(test_data_dir)
    mapped = dataset.map(lambda x: {**x, "val_len": len(x["val"])})
    assert "val_len" in mapped[0]
    assert mapped[0]["val_len"] == 3 # "a_0"

def test_rename(test_data_dir):
    dataset = IndexedParquetDataset.from_folder(test_data_dir)
    renamed = dataset.rename_column("val", "text")
    assert "text" in renamed[0]
    assert "val" not in renamed[0]
    assert renamed[0]["text"] == "a_0"

def test_batch_reading(test_data_dir):
    dataset = IndexedParquetDataset.from_folder(test_data_dir)
    batch = dataset[[0, 10, 30]] # uses __getitems__
    assert len(batch) == 3
    assert batch[0]["id"] == 0
    assert batch[1]["id"] == 10
    assert batch[2]["id"] == 30

def test_save_load_index(test_data_dir, tmp_path):
    dataset = IndexedParquetDataset.from_folder(test_data_dir)
    index_path = str(tmp_path / "index.pkl")
    dataset.save_index(index_path)
    
    loaded = IndexedParquetDataset.load_index(index_path)
    assert len(loaded) == len(dataset)
    assert loaded[0] == dataset[0]
    assert np.array_equal(loaded.indices, dataset.indices)
