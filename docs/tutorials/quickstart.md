# Quickstart

This tutorial will help you master the basics of `indexed-parquet-dataset` in 5 minutes.

## Step 1: Data Preparation

Suppose you have a folder with several Parquet files:

```text
data/
  part1.parquet
  part2.parquet
  subdir/
    part3.parquet
```

## Step 2: Dataset Initialization

Use the `from_folder` method. It scans the directory, indexes the files, and creates a dataset object.

```python
from indexed_parquet_dataset import IndexedParquetDataset

# Recursively scan all .parquet files
dataset = IndexedParquetDataset.from_folder("data", pattern="*.parquet", recursive=True)

print(f"Total rows: {len(dataset):,}")
print(f"Columns: {dataset.schema}")
```

## Step 3: Data Access

You can access rows by index, just like a regular list. This operation is performed in **O(1)** and does not depend on the data size.

```python
# Access a single row
row = dataset[0]  # {'id': 1, 'name': 'Item A', ...}

# Slices (return a list of dictionaries)
subset = dataset[10:20]  

# Fancy indexing (access by list of indices)
items = dataset[[1, 5, 100]] 
```

## Step 4: Transformations (Fluent API)

The library supports method chaining for data preparation:

```python
processed_ds = (dataset
                .filter(lambda x: x["category"] == "electronics")
                .shuffle(seed=42)
                .alias("price_usd", lambda x: x["price"] * 0.01) # New column
                .limit(5000))

print(f"Processed rows: {len(processed_ds)}")
```

## Step 5: Dataset Analysis

The `.info()` method displays a detailed table with statistics for each file and column coverage. This is very useful for debugging.

```python
dataset.info()
```

## Step 6: Saving the Index

Scanning millions of rows can take time. To avoid doing this every time, save the index to a file:

```python
# Save
dataset.save_index("my_index.pkl")

# Load instantly next time
dataset = IndexedParquetDataset.load_index("my_index.pkl")
```

## Step 7: PyTorch Integration

`IndexedParquetDataset` inherits from `torch.utils.data.Dataset`, so it works "out of the box" with `DataLoader`.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    processed_ds, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4
)

for batch in loader:
    # batch is a dictionary of tensors/lists
    print(batch['price_usd'])
    break
```

---

**What's next?**
- Learn about [Schema Evolution](../how-to/schema-evolution.md) if your files have different structures.
- See how to build a complete [Deep Learning Pipeline](deep_learning.md).
- Explore [Column Operations](../how-to/column-ops.md) for complex preprocessing.
