import os
import pytest
import pandas as pd
import numpy as np
import time
from indexed_parquet_dataset import IndexedParquetDataset

@pytest.fixture
def test_data(tmp_path):
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    
    df = pd.DataFrame({
        "id": range(100),
        "text": [f"sentence with number {i}" for i in range(100)],
        "val": np.random.randn(100)
    })
    df.to_parquet(test_dir / "data.parquet")
    return str(test_dir)

def test_new_filter_naming(test_data):
    """Verifies that filter_row and filter_batch work."""
    ds = IndexedParquetDataset.from_folder(test_data)
    
    # filter_row (new name for predicate)
    f1 = ds.filter(filter_row=lambda x: x["id"] < 10)
    assert len(f1) == 10
    
    # filter_batch (new name for predicate_batch)
    f2 = ds.filter(filter_batch=lambda b: [x["id"] >= 90 for x in b])
    assert len(f2) == 10

def test_legacy_naming_warning(test_data):
    """Verifies that legacy naming still works but warns."""
    ds = IndexedParquetDataset.from_folder(test_data)
    
    with pytest.warns(DeprecationWarning, match="deprecated"):
        f = ds.filter(predicate=lambda x: x["id"] < 5)
    
    assert len(f) == 5

def test_transform_batch_in_filter(test_data):
    """Verifies transform_batch + filter_batch workflow."""
    ds = IndexedParquetDataset.from_folder(test_data)
    
    def my_tokenizer(batch):
        # Mocks a tokenizer that adds 'tokens' column
        for row in batch:
            row["tokens"] = row["text"].split()
        return batch
    
    def my_filter(batch):
        # Filter rows where tokens count > 3
        return [len(row["tokens"]) > 3 for row in batch]
    
    filtered = ds.filter(
        transform_batch=my_tokenizer,
        filter_batch=my_filter
    )
    
    # "sentence with number X" has 4 tokens
    # But wait, 0-9 are single digits, 10-99 are double digits. All have 4 words.
    # Actually "sentence", "with", "number", "X" -> 4 words.
    assert len(filtered) == 100
    
    # Change criteria
    filtered = ds.filter(
        transform_batch=my_tokenizer,
        filter_row=lambda x: int(x["tokens"][-1]) < 5
    )
    assert len(filtered) == 5

def test_alias_batch(test_data):
    """Verifies ds.alias(..., is_batch=True) works."""
    ds = IndexedParquetDataset.from_folder(test_data)
    
    def batch_identity(batch):
        # Returns a list of values for the new column
        return [f"id_{row['id']}" for row in batch]
    
    ds_aliased = ds.alias("new_id", batch_identity, is_batch=True)
    
    assert "new_id" in ds_aliased.schema
    assert ds_aliased[0]["new_id"] == "id_0"
    assert ds_aliased[99]["new_id"] == "id_99"

def test_map_batch(test_data):
    """Verifies ds.map(..., is_batch=True) works."""
    ds = IndexedParquetDataset.from_folder(test_data)
    
    def batch_transform(batch):
        for row in batch:
            row["processed"] = True
        return batch
    
    # We provide output_schema because map_batches can't know new columns statically
    new_schema = ds.schema + ["processed"]
    ds_mapped = ds.map(batch_transform, is_batch=True, output_schema=new_schema)
    
    assert "processed" in ds_mapped.schema
    assert ds_mapped[0]["processed"] is True
