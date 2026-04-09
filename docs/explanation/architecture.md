# Architecture and Performance

This page explains how `indexed-parquet-dataset` achieves O(1) random access and handles large-scale data efficiently.

## The Indexing Strategy

Unlike standard Parquet readers that often require scanning row groups to find a specific row index, this library pre-calculates a global index map during initialization.

### Initialization Scan
When you call `from_folder`, the library:
1.  Iterates through all Parquet files found.
2.  Reads only the **metadata** (not the data) of each file.
3.  Records the number of rows in every **Row Group** of every file.
4.  Calculates cumulative offsets.

### Instant Lookup
When you request an index `i`:
1.  We use binary search (`numpy.searchsorted`) on cumulative offsets to find which file contains row `i`.
2.  We find the specific Row Group within that file.
3.  We open the file and read exactly that Row Group, fetching only the requested row.

## Memory Management

### Lazy Handle Loading
Opening thousands of Parquet files simultaneously would exceed system file descriptor limits. To prevent this, the library uses an **LRU (Least Recently Used) Cache** for file handles.

By default, it keeps up to 128 file handles open. When a 129th file is needed, the least recently used one is closed automatically. You can adjust this via the `max_open_files` parameter.

### Efficient Data Transfer
When reading a row, `indexed-parquet-dataset` uses PyArrow's low-level `read_row_group` API. If you have selected specific columns, only the data for those columns is read from disk, significantly reducing I/O.

## Schema Evolution Engine

The library acts as a virtualization layer between the raw Parquet files and your application.

1.  **Uniform View**: Even if `File A` has columns `[X, Y]` and `File B` has `[X, Z]`, the dataset provides a unified schema `[X, Y, Z]`.
2.  **Mapping Layer**: A `SchemaMapper` object handles name translations and missing value injections in real-time as rows are fetched.
3.  **Recursion for Nested Data**: For complex types like `Struct` columns, the library recursively sanitizes the data to ensure no `None` values break training loops (especially important for PyTorch's `default_collate`).
