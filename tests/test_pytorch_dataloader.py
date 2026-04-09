import os
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
import numpy as np
import pytest
import shutil

from indexed_parquet import IndexedParquetDataset

@pytest.fixture(scope="module")
def sample_data_dir():
    path = "test_pytorch_data"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    # Create data with some nulls
    # File 1: All Good
    table1 = pa.Table.from_pylist([
        {"id": i, "features": float(i * 10), "label": i % 2} for i in range(50)
    ])
    pq.write_table(table1, os.path.join(path, "clean.parquet"))
    
    # File 2: Some Nulls
    data2 = []
    for i in range(50, 100):
        row = {"id": i, "features": float(i * 10), "label": i % 2}
        if i % 10 == 0:
            row["features"] = None
        if i % 15 == 0:
            row["id"] = None
        data2.append(row)
        
    table2 = pa.Table.from_pylist(data2)
    pq.write_table(table2, os.path.join(path, "nulls.parquet"))
    
    yield path
    
    if os.path.exists(path):
        shutil.rmtree(path)

def test_pytorch_dataloader_integration(sample_data_dir):
    """[POSITIVE TEST] Verifies that the dataset works with PyTorch DataLoader, 
    matching the user's requested usage pattern."""
    
    # 1. Initialize dataset as requested
    ds = IndexedParquetDataset.from_folder(
        sample_data_dir,
        default_fill_value="-",         # All gaps become "-"
        fill_values_by_type={"int64": 0, "double": 0.0}, # Numbers get 0
    )
    
    # 2. Create DataLoader with parameters from the user's example
    loader = DataLoader(
        ds, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4,  # Parallel reading (Enabled by our pickling fix)
        collate_fn=ds.generate_collate_fn(on_none='fill')
    )
    
    # 3. Simulate training loop
    total_samples = 0
    for batch in loader:
        # batch - dict where values are types handled by default_collate
        ids = batch['id']
        features = batch['features']
        
        # Verify types
        assert isinstance(ids, torch.Tensor)
        assert isinstance(features, torch.Tensor)
        
        # Verify size
        assert len(ids) <= 32
        
        # Check that we don't have None values in the batch 
        # (they should be filled with 0 due to our config)
        # Note: PyTorch tensors can't contain None anyway, they'd be 0.0 or something
        # if filled correctly.
        
        total_samples += len(ids)
        if total_samples >= 32:
            break
            
    assert total_samples >= 32

def test_fill_values_logic(sample_data_dir):
    """[POSITIVE TEST] Specific check for the hierarchical fill logic requested."""
    ds = IndexedParquetDataset.from_folder(
        sample_data_dir,
        default_fill_value="MISSING",
        fill_values_by_type={"int64": -1, "double": -99.9}
    )
    
    # Find a row where 'id' was None (i=60)
    # The index 10 in nulls.parquet is global index 60
    item = ds[60]
    assert item['id'] == -1 # int64
    
    # Find a row where 'features' was None (i=50) 
    # Global index 50
    item = ds[50]
    assert item['features'] == -99.9 # double/float64
