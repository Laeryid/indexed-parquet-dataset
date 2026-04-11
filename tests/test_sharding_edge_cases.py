import os
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from indexed_parquet_dataset import IndexedParquetDataset

@pytest.fixture
def large_sample_dir(tmp_path):
    data_dir = tmp_path / "large_data"
    data_dir.mkdir()
    
    # Create a dataset with 10 rows
    table = pa.table({
        "id": np.arange(10),
        "val": np.random.rand(10)
    })
    pq.write_table(table, data_dir / "data.parquet")
    return str(data_dir)

def test_chunk_larger_than_shard(large_sample_dir, tmp_path):
    """Test that sharding works correctly even if chunk_size is larger than shard_size."""
    ds = IndexedParquetDataset.from_folder(large_sample_dir)
    out_dir = tmp_path / "shards_large_chunk"
    
    # shard_size=3, chunk_size=10. 
    # The whole dataset (10 rows) will be buffered, then should be split into 4 shards: 3, 3, 3, 1
    ds.to_parquet(str(out_dir), shard_size=3, chunk_size=10, optimize_by_reorder=True)
    
    shard_files = sorted(os.listdir(out_dir))
    assert len(shard_files) == 4
    assert shard_files == ["part_0000.parquet", "part_0001.parquet", "part_0002.parquet", "part_0003.parquet"]
    
    # Verify row counts
    assert pq.read_metadata(str(out_dir / "part_0000.parquet")).num_rows == 3
    assert pq.read_metadata(str(out_dir / "part_0001.parquet")).num_rows == 3
    assert pq.read_metadata(str(out_dir / "part_0002.parquet")).num_rows == 3
    assert pq.read_metadata(str(out_dir / "part_0003.parquet")).num_rows == 1

def test_chunk_smaller_than_shard(large_sample_dir, tmp_path):
    """Test that sharding works correctly if chunk_size is smaller than shard_size."""
    ds = IndexedParquetDataset.from_folder(large_sample_dir)
    out_dir = tmp_path / "shards_small_chunk"
    
    # shard_size=7, chunk_size=2.
    # Should result in 2 shards: 7 rows and 3 rows
    ds.to_parquet(str(out_dir), shard_size=7, chunk_size=2, optimize_by_reorder=True)
    
    shard_files = sorted(os.listdir(out_dir))
    assert len(shard_files) == 2
    assert pq.read_metadata(str(out_dir / "part_0000.parquet")).num_rows == 7
    assert pq.read_metadata(str(out_dir / "part_0001.parquet")).num_rows == 3

if __name__ == "__main__":
    pytest.main([__file__])
