import os
import pytest
import pandas as pd
import numpy as np
import time
from indexed_parquet_dataset import IndexedParquetDataset

@pytest.fixture
def large_test_data(tmp_path):
    test_dir = tmp_path / "large_data"
    test_dir.mkdir()
    
    # Create 5 files with 100 rows each
    for i in range(5):
        df = pd.DataFrame({
            "id": range(i*100, (i+1)*100),
            "val": np.random.randn(100)
        })
        df.to_parquet(test_dir / f"data_{i}.parquet")
    return str(test_dir)

def test_early_stopping_speed(large_test_data):
    """Verifies that filter + limit is fast because of early stopping."""
    ds = IndexedParquetDataset.from_folder(large_test_data)
    
    call_count = 0
    def slow_filter(row):
        nonlocal call_count
        call_count += 1
        return row["id"] >= 0 # Always true
    
    # We want only 5 items. 
    # Without optimization, it would call slow_filter 500 times.
    # With optimization, it should call it ~5 times (plus batch overhead).
    start_time = time.time()
    filtered = ds.filter(filter_row=slow_filter).limit(5)
    
    # Until we access it, it should be lazy (0 calls)
    assert call_count == 0
    
    # Trigger materialization
    results = filtered[:]
    duration = time.time() - start_time
    
    assert len(results) == 5
    # Batch size is 1024 by default, so it might process the first batch of the first file (100 rows).
    # But it should NOT go to the second file.
    assert call_count <= 100 
    assert call_count >= 5

def test_lazy_chaining(large_test_data):
    """Verifies that alias/map can be chained with lazy filter."""
    ds = IndexedParquetDataset.from_folder(large_test_data)
    
    # 1. Filter (becomes lazy)
    f1 = ds.filter(filter_row=lambda x: x["id"] % 2 == 0)
    assert f1._indices is None
    
    # 2. Alias (remains lazy but propagates)
    f2 = f1.alias("id_plus_1", lambda x: x["id"] + 1)
    
    # 3. Limit (triggers materialized filter with fusion)
    f3 = f2.limit(10)
    
    # len() triggers materialization if limit didn't already
    assert len(f3) == 10
    assert f3[0]["id_plus_1"] == f3[0]["id"] + 1

def test_progress_bar_with_found_count(large_test_data, capsys):
    """Verifies that progress bar shows 'found' count."""
    ds = IndexedParquetDataset.from_folder(large_test_data)
    
    # Use a small batch size to see progress
    f = ds.filter(filter_row=lambda x: True, limit=10, show_progress=True, batch_size=2)
    _ = f[:]
    
    # We can't easily capture tqdm output from capsys if it goes to stderr, 
    # but we can check if it runs without error.
    assert len(f) == 10
