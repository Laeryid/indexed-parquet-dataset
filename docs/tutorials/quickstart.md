# Quick Start Guide

This tutorial will walk you through the basic steps of using `indexed-parquet-dataset` to manage large Parquet files for machine learning.

## Step 1: Prepare your data

Suppose you have a folder structure with multiple Parquet files:

```text
data/
  part1.parquet
  part2.parquet
  subdir/
    part3.parquet
```

## Step 2: Initialize the Dataset

The easiest way to get started is by using `from_folder`. It scans the directory, indexes all files, and creates a dataset object.

```python
from indexed_parquet import IndexedParquetDataset

dataset = IndexedParquetDataset.from_folder("data", pattern="*.parquet", recursive=True)

print(f"Total rows: {len(dataset)}")
print(f"Columns: {dataset.schema}")
```

## Step 3: Accessing Data

You can access rows by index just like a regular Python list. This operation is **O(1)** and does not depend on the dataset size.

```python
# Single row access
row = dataset[0]  # {'id': 1, 'name': 'Item A', ...}

# Slidcing
subset = dataset[10:20]  # Returns a list of dictionaries

# Fancy indexing
items = dataset[[1, 5, 100]] # Returns a list of dictionaries
```

## Step 4: Shuffling and Splitting

Working with training and validation sets is easy with built-in methods.

```python
# Shuffle the whole dataset
train_ds = dataset.shuffle(seed=42)

# Split into 80% train and 20% test
train_ds, test_ds = dataset.train_test_split(test_size=0.2, seed=42)

print(f"Training rows: {len(train_ds)}")
print(f"Testing rows: {len(test_ds)}")
```

## Step 5: Integration with PyTorch

The `IndexedParquetDataset` class inherits from `torch.utils.data.Dataset` (if torch is installed), so it works out-of-the-box with `DataLoader`.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    train_ds, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4
)

for batch in loader:
    # batch is a dictionary of tensors/lists
    images = batch['image']
    labels = batch['label']
    ...
```

## Next Steps

- Learn about handling [Schema Evolution](../how-to/schema-evolution.md) when your Parquet files have different structures.
- Dive into the [API Reference](../reference/api.md) for detailed class documentation.
