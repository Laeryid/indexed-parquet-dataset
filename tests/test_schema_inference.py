import os
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from indexed_parquet_dataset import IndexedParquetDataset

@pytest.fixture
def complex_data_dir(tmp_path):
    data_dir = tmp_path / "complex_data"
    data_dir.mkdir()
    
    # Create a schema with nested types
    schema = pa.schema([
        ("id", pa.int64()),
        ("text", pa.string()),
        ("meta", pa.struct([
            ("source", pa.string()),
            ("score", pa.float64())
        ]))
    ])
    
    # Batch 1: All values present
    data1 = {
        "id": [1],
        "text": ["hello"],
        "meta": [{"source": "web", "score": 0.9}]
    }
    # Batch 2: Some None values
    data2 = {
        "id": [2],
        "text": [None], # This is the problem column
        "meta": [None]
    }
    
    table1 = pa.Table.from_pydict(data1, schema=schema)
    table2 = pa.Table.from_pydict(data2, schema=schema)
    
    pq.write_table(table1, data_dir / "f1.parquet")
    pq.write_table(table2, data_dir / "f2.parquet")
    
    return str(data_dir)

def test_to_parquet_with_nulls_first(complex_data_dir, tmp_path):
    """
    Test that to_parquet doesn't fail when the first batch contains only None values
    for a column that has a specific type in the source.
    """
    ds = IndexedParquetDataset.from_folder(complex_data_dir)
    
    # We reorder so that the row with None comes FIRST in the output
    # Row 0 (from f1): data
    # Row 1 (from f2): nulls
    ds.indices = np.array([1, 0]) # Row 1 then Row 0
    
    out_file = tmp_path / "shuffled.parquet"
    
    # Without the fix, if chunk_size=1:
    # Batch 1 (Row 1): text=None -> inferred type 'null'
    # Batch 2 (Row 0): text='hello' -> type 'string' -> ValueError: Table schema does not match
    
    # With shard_size=1 we trigger writer creation on each row, 
    # but the error reported was about write_table mismatch within the SAME file.
    # So let's use chunk_size=1 to generate two separate calls to write_table.
    
    ds.to_parquet(str(out_file), chunk_size=1, optimize_by_reorder=False)
    
    assert os.path.exists(str(out_file))
    
    # Verify data
    ds_new = IndexedParquetDataset.from_folder(str(out_file))
    assert len(ds_new) == 2
    assert ds_new[0]["text"] is None
    assert ds_new[1]["text"] == "hello"
    assert ds_new[0]["meta"] is None
    assert ds_new[1]["meta"] == {"source": "web", "score": 0.9}

if __name__ == "__main__":
    pytest.main([__file__])
