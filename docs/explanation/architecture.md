# Architecture and Performance

This page explains how `indexed-parquet-dataset` provides O(1) random access and efficiently works with terabytes of data.

## Indexing Strategy

Unlike standard Parquet reading libraries, which often require scanning all Row Groups to find a specific record, our library builds a global index map during initialization.

### 1. Initialization (Scanning)
When you call `from_folder`, the library:
1.  Walks through all Parquet files in the specified path.
2.  Reads only the **metadata** (not the data itself) of each file.
3.  Extracts the number of rows in each Row Group of each file.
4.  Calculates cumulative offsets (prefix sums).

**Result**: A lightweight in-memory map that says: *"row #1,500,000 is in file X, in row group Y, at position Z"*.

### 2. Instant Lookup
When `dataset[i]` is requested:
1.  Binary search (`numpy.searchsorted`) over offsets is used to find the required file in **O(log N_files)**.
2.  The local index within the file is calculated.
3.  The file is opened (or taken from the cache), and **only one** required Row Group is read.

## Resource Management

### Lazy Handle Loading (LRU Cache)
Opening thousands of files simultaneously would exceed the OS file descriptor limit. To prevent this, the library uses an **LRU (Least Recently Used) cache** for open files.

By default, up to 128 open files are held. When the 129th is required, the "oldest" file is closed automatically. You can control this via the `max_open_files` parameter.

### I/O Efficiency
When reading a row, the library requests only those columns from PyArrow that are necessary for the final result. If you used `select_columns`, the data of other columns will not even be read from the disk.

## Multi-process Modes (PyTorch)

Efficient model training in PyTorch uses `DataLoader` with `num_workers > 0`. This means data is read in parallel in several processes.

`indexed-parquet-dataset` is designed to be fully **Pickle-compatible**:
1.  The index itself (BaseIndex) is easily serialized.
2.  Open file handles **are not serialized**. Instead, each worker (process) opens its own files upon first data access.
3.  This ensures no file access conflicts between processes.

## Data Flow

```mermaid
graph LR
    Disk[(Parquet Files)] -- "1. Read Metadata" --> Indexer
    Indexer -- "2. Build Presums" --> Index
    
    User[dataset[idx]] --> Index
    Index -- "3. File + RowGroup" --> LRUCache
    LRUCache -- "4. Open/Reuse" --> PF[PyArrow File]
    PF -- "5. Read RowGroup" --> Mapper
    
    Mapper -- "6. Rename/Cast/Map" --> Output[Row Dict]
```

## When can O(1) slow down?

While index lookup is always instant, row retrieval time can increase if:
-   **Slow storage**: HDD or network drives with high latency.
-   **Too many transformations**: If you have a long chain of `.map()` and `.alias(lambda)`, Python spends time executing them.
-   **Very small Row Groups**: If a file is split into thousands of tiny groups, the overhead for reading group metadata can become noticeable.

In such cases, it is recommended to use [materialization](../how-to/materialization.md) via `.clone()`.
