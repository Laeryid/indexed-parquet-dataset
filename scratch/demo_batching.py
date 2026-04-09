import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from indexed_parquet import IndexedParquetDataset

def demo_batching():
    # 1. Create dummy data
    df = pd.DataFrame({
        "id": np.arange(100),
        "val": np.random.randn(100),
        "label": np.random.randint(0, 2, 100)
    })
    os.makedirs("demo_data", exist_ok=True)
    df.to_parquet("demo_data/data.parquet", row_group_size=20) # 5 row groups
    
    # 2. Initialize dataset
    dataset = IndexedParquetDataset.from_folder("demo_data")
    print(f"Dataset length: {len(dataset)}")
    
    # 3. Method A: Standard PyTorch DataLoader
    print("\n--- Method A: DataLoader ---")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    batch = next(iter(loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch 'id' type: {type(batch['id'])}")
    print(f"Batch 'id' shape: {batch['id'].shape}")
    
    # 4. Method B: Manual Batching (using slices/lists)
    print("\n--- Method B: Manual Batching ---")
    manual_batch = dataset[0:8]
    print(f"Manual batch type: {type(manual_batch)}")
    print(f"First item in manual batch: {manual_batch[0]}")
    
    # Clean up
    import shutil
    shutil.rmtree("demo_data")

if __name__ == "__main__":
    demo_batching()
