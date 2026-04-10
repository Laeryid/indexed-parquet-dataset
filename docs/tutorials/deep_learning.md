# Deep Learning Pipeline

This tutorial shows how to build a reliable and fast neural network training pipeline using `IndexedParquetDataset` and PyTorch.

## Why is it better than standard approaches?

1.  **RAM-efficiency**: Even if your dataset is 1 TB, you consume only a few hundred MB of RAM.
2.  **No Lag**: O(1) random access ensures that the `DataLoader` won't wait for an entire file to be parsed to retrieve a single row.
3.  **Flexibility**: You can filter and augment data on the fly without regenerating heavy files.

## Pipeline Example

Below is a complete example: from initialization to the training loop.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from indexed_parquet_dataset import IndexedParquetDataset

# 1. Initialization with automatic gap handling
# auto_fill=True fills missing values with defaults (0 for numbers, "" for strings)
ds = IndexedParquetDataset.from_folder("./data", auto_fill=True)

# 2. Data Preparation (Fluent API)
dataset = (ds
    .filter(lambda x: x["label"] is not None) # Remove rows without labels
    .shuffle(seed=42)
    .alias("image_tensor", lambda x: torch.tensor(x["pixels"])) # Convert to tensor
    .cast("label", "int") # Ensure label is an integer
)

# 3. Train/Val split
train_ds, val_ds = dataset.train_test_split(test_size=0.1, seed=42)

# 4. DataLoader setup
# generate_collate_fn ensures that None values don't end up in a batch,
# even if they somehow slipped into the data.
loader = DataLoader(
    train_ds, 
    batch_size=64, 
    shuffle=True, 
    num_workers=4,
    collate_fn=train_ds.generate_collate_fn(on_none='fill')
)

# 5. Training loop
model = nn.Linear(784, 10)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for batch in loader:
        # Data is already in tensor format thanks to alias and collate_fn
        images = batch["image_tensor"]
        labels = batch["label"]
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch} completed")
```

## Handling Gaps in Deep Learning

One of the main problems when using `DataLoader` with file-based data is the presence of `None`. If the standard `default_collate` encounters `None`, it crashes with an error.

`IndexedParquetDataset` offers three resolution strategies in `generate_collate_fn(on_none=...)`:

-   `'raise'` (default): Throw an error indicating the column where `None` was found.
-   `'fill'`: Replace `None` with a value from the dataset configuration (see [Schema Evolution](../how-to/schema-evolution.md)).
-   `'drop'`: Simply drop this row from the batch (the batch will be slightly smaller than `batch_size`).

## Parallel Reading (num_workers)

Because `IndexedParquetDataset` correctly implements `__getstate__` and `__setstate__` methods, it is fully compatible with multi-process reading in PyTorch (`num_workers > 0`). 

Each worker opens its own file descriptors without conflicting with others.

## Performance Tip

If you use very complex transformations in `.alias(lambda ...)` or heavy filtering, it may slow down reading. In such cases, it is recommended to "materialize" the dataset once:

```python
# All calculations will be performed once and saved to a new file
fast_ds = dataset.clone("processed_data.parquet")
```

Read more about this in the [Materialization](../how-to/materialization.md) section.
