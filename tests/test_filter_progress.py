import os
import pytest
import pandas as pd
import numpy as np
from indexed_parquet_dataset import IndexedParquetDataset

@pytest.fixture
def test_data_dir(tmp_path):
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    
    # Create 5 parquet files to see progress better
    for i in range(5):
        df = pd.DataFrame({
            "id": range(i*10, (i+1)*10),
            "val": [f"val_{j}" for j in range(i*10, (i+1)*10)],
            "group": [i] * 10
        })
        df.to_parquet(test_dir / f"file_{i}.parquet")
    
    return str(test_dir)

def test_filter_with_progress_conditions(test_data_dir):
    """[POSITIVE TEST] Verifies filter with show_progress=True for column_conditions."""
    dataset = IndexedParquetDataset.from_folder(test_data_dir)
    
    # Filter by column conditions
    filtered = dataset.filter(column_conditions={"group": 2}, show_progress=True)
    
    assert len(filtered) == 10
    for i in range(len(filtered)):
        assert filtered[i]["group"] == 2

def test_filter_with_progress_predicate(test_data_dir):
    """[POSITIVE TEST] Verifies filter with show_progress=True for predicate."""
    dataset = IndexedParquetDataset.from_folder(test_data_dir)
    
    # Filter by predicate
    filtered = dataset.filter(predicate=lambda x: x["group"] == 3, show_progress=True)
    
    assert len(filtered) == 10
    for i in range(len(filtered)):
        assert filtered[i]["group"] == 3

def test_filter_with_progress_both(test_data_dir):
    """[POSITIVE TEST] Verifies filter with both conditions and predicate."""
    dataset = IndexedParquetDataset.from_folder(test_data_dir)
    
    # Both: first group > 1 (2, 3, 4), then filter for group 3
    filtered = dataset.filter(
        column_conditions={"group": (">", 1)},
        predicate=lambda x: x["group"] == 3,
        show_progress=True
    )
    
    assert len(filtered) == 10
    for i in range(len(filtered)):
        assert filtered[i]["group"] == 3
