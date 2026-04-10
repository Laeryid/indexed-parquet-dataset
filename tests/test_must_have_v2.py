import os
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from indexed_parquet_dataset import IndexedParquetDataset

@pytest.fixture
def sample_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create two parquet files with different columns
    df1 = pa.table({
        "id": [1, 2, 3],
        "text": ["foo", "bar", "baz"],
        "val": [10, 20, 30]
    })
    pq.write_table(df1, data_dir / "f1.parquet")
    
    df2 = pa.table({
        "id": [4, 5],
        "text": ["extra", "more"],
        "other": [1, 2]
    })
    pq.write_table(df2, data_dir / "f2.parquet")
    
    return str(data_dir)

def test_select_columns(sample_data_dir):
    ds = IndexedParquetDataset.from_folder(sample_data_dir)
    original_schema = ds.schema
    assert "id" in original_schema
    assert "text" in original_schema
    assert "val" in original_schema
    assert "other" in original_schema
    
    # Select subset
    ds_sub = ds.select_columns(["id", "text"])
    assert ds_sub.schema == ["id", "text"]
    
    # Check data
    row = ds_sub[0]
    assert list(row.keys()) == ["id", "text"]
    assert row["id"] == 1
    assert row["text"] == "foo"

def test_sample(sample_data_dir):
    ds = IndexedParquetDataset.from_folder(sample_data_dir)
    sampled = ds.sample(n=2, seed=42)
    assert len(sampled) == 2
    
    # Repeated sample with same seed should be same
    sampled2 = ds.sample(n=2, seed=42)
    assert [r["id"] for r in sampled] == [r["id"] for r in sampled2]

def test_iter_batches(sample_data_dir):
    ds = IndexedParquetDataset.from_folder(sample_data_dir)
    batches = list(ds.iter_batches(batch_size=2))
    assert len(batches) == 3 # 1:[1,2], 2:[3,4], 3:[5]
    assert len(batches[0]) == 2
    assert len(batches[2]) == 1
    
    # With shuffle
    batches_shuffled = list(ds.iter_batches(batch_size=2, shuffle=True, seed=42))
    assert len(batches_shuffled) == 3

def test_to_parquet_sharding(sample_data_dir, tmp_path):
    ds = IndexedParquetDataset.from_folder(sample_data_dir)
    out_dir = tmp_path / "shards"
    
    # Export with sharding: 2 rows per shard
    ds.to_parquet(str(out_dir), shard_size=2)
    
    # Check files
    shard_files = sorted(os.listdir(out_dir))
    assert "part_0000.parquet" in shard_files
    assert "part_0001.parquet" in shard_files
    assert "part_0002.parquet" in shard_files
    
    # Total rows in shards should match
    total_rows = 0
    for f in shard_files:
        total_rows += pq.read_metadata(str(out_dir / f)).num_rows
    assert total_rows == len(ds)

def test_io_optimization(sample_data_dir):
    from unittest.mock import patch, MagicMock
    import pyarrow.parquet as pq
    
    ds = IndexedParquetDataset.from_folder(sample_data_dir)
    ds_sub = ds.select_columns(["id"]) # Only 'id'
    
    # We want to catch the call to pq.ParquetFile.read_row_group
    with patch("pyarrow.parquet.ParquetFile") as mock_pf_class:
        mock_pf = MagicMock()
        mock_pf_class.return_value = mock_pf
        
        # Mock read_row_group to return a dummy table with 'id'
        mock_pf.read_row_group.return_value = pa.table({"id": [99]})
        
        # Trigger read
        _ = ds_sub[0]
        
        # Verify read_row_group was called with columns=['id']
        # Note: mapping might affect the name, but here 'id' matches 'id'
        args, kwargs = mock_pf.read_row_group.call_args
        assert "columns" in kwargs
        assert kwargs["columns"] == ["id"]

if __name__ == "__main__":
    # Manual run if not using pytest
    import sys
    pytest.main([__file__])
