# Filtering and Sampling

`IndexedParquetDataset` provides several ways to restrict and shuffle data. Remember that all filtering methods return a new dataset object (lazy) without copying the data itself.

## Random Shuffling (shuffle)

The `.shuffle()` method shuffles the order of indices. You can fix the result by passing a `seed`.

```python
# Shuffle everything
dataset = dataset.shuffle(seed=42)
```

## Filtering by Conditions (filter)

The `.filter()` method is the most powerful data selection tool. It supports three modes:

### 1. Server-side Filtering (PyArrow-side)

This mode is the fastest because it is performed at the C++ level via PyArrow before the data reaches Python.

```python
# Filtering by exact value
dataset = dataset.filter(column_conditions={"status": "active"})

# Filtering by range (using a tuple)
dataset = dataset.filter(column_conditions={
    "score": (">", 0.8),
    "age": ("<=", 30)
})
```

### 2. Filtering via Predicate (Python-side)

If the conditions in `column_conditions` are not enough, you can pass a predicate function. It will be called for each row. This is slower but more flexible.

```python
dataset = dataset.filter(predicate=lambda x: len(x["text"]) > 100 and x["label"] in [1, 5])
```

### 3. File Filtering

You can keep data only from specific files using glob patterns or a list of paths.

```python
# Only files from the 2023 folder
dataset = dataset.filter(path_filter="**/2023/*.parquet")
```

## Selection and Limitation

### Limitation (limit)

```python
# Take only the first 1000 rows
dataset = dataset.limit(1000)
```

### Random Sampling (sample)

```python
# Select 500 random rows without replacement
dataset = dataset.sample(500, seed=123)
```

### Index Selection (select)

```python
# Keep only rows with specific indices
dataset = dataset.select([0, 10, 50, 100])

# Slices can be used
dataset = dataset.select(slice(0, 500, 2)) # Every second row from the first 500
```

## Train/Test Split

The `train_test_split` method is a "Swiss Army knife" for training preparation. It supports stratification!

```python
# Regular 80/20 split
train_ds, val_ds = dataset.train_test_split(test_size=0.2, seed=42)

# Split with stratification by the 'category' column
# (will preserve category proportions in both samples)
train_ds, val_ds = dataset.train_test_split(
    test_size=0.2, 
    stratify_by="category"
)
```

> [!NOTE]
> Stratification requires reading the specified column for **all** rows in the dataset at the time of the method call. On huge datasets, this may take some time.
