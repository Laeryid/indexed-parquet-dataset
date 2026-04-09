# Materialization and Export

If you use complex transformations (via `.map()` or `.alias(lambda ...)`), every row read will trigger the execution of your Python code. On large datasets, this can become a bottleneck.

To solve this problem, the library provides **materialization** mechanisms — recording the current state of the dataset into a new physical Parquet file.

## `to_parquet` Method

The `to_parquet` method simply saves the current state of the dataset to disk.

```python
dataset.to_parquet("processed_data.parquet")
```

### Sharding

If your dataset is too large for a single file, you can split it into parts (shards):

```python
# Splits the dataset into files with 50,000 rows each in the 'shards/' directory
dataset.to_parquet("shards/", shard_size=50000)
```

## `clone` Method (Recommended)

The `clone` method does the same as `to_parquet`, but returns a **new instance** of `IndexedParquetDataset` that immediately points to the newly created file.

This is the best way to "bake" filters and transformations and continue working with maximum speed (zero-overhead).

```python
# 1. Apply slow transformations
ds = (IndexedParquetDataset.from_folder("./raw")
      .filter(heavy_logic)
      .map(complex_transform))

# 2. Clone to a fast file
fast_ds = ds.clone("baked_data.parquet")

# 3. Now fast_ds works orders of magnitude faster
row = fast_ds[0] 
```

## When should you materialize?

1.  **Complex Python Transformations**: If there is a `.map()` or `.alias(callable)` in the chain.
2.  **Heavy Filtering**: If you filtered out 90% of data from a huge dataset. Direct access to the remaining 10% in the original files will still be fast, but the in-memory index will contain many "holes", and file descriptors will be held for all original files.
3.  **Preparation for Another Tool**: If you want to use cleaned data in another application (e.g., load into Apache Spark).
4.  **Final Version**: Before starting long model training on a cluster.

## Write Parameters

You can adjust the chunk size during writing:

```python
dataset.to_parquet(
    "output.parquet", 
    chunk_size=1024 # How many rows to accumulate in memory before flushing to the file
)
```
