# Indexed Parquet Dataset

[![PyPI version](https://img.shields.io/pypi/v/indexed-parquet-dataset.svg)](https://pypi.org/project/indexed-parquet-dataset/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Indexed Parquet Dataset** is a high-performance library for fast random access to Parquet data.

## Why Use It?

Standard libraries (pandas, pyarrow) work great for reading an entire file, but they are **not designed for efficient random access by row index**, especially when data is distributed across hundreds of Parquet files.

`indexed-parquet-dataset` solves this problem by offering:

1.  **True O(1) access**: We build a lightweight index once and instantly navigate to the desired row in any file.
2.  **Memory efficiency**: You don't load the entire dataset. Only the required file chunk is opened at the moment of access.
3.  **Schema flexibility**: Your files can have different columns or types — the library normalizes everything "on the fly".

## Comparison

| Feature | Pandas / PyArrow | HF Datasets | IterableDataset | **Indexed Parquet** |
| :--- | :---: | :---: | :---: | :---: |
| Random Access | ❌ Slow/RAM | ✅ Good | ❌ No | ✅ **O(1) / RAM-lite** |
| Cloud/Network Read | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Via FUSE only |
| Schema Evolution | ❌ Hard | ⚠️ Partial | ⚠️ Hard | ✅ **Native** |
| Lazy Loading | ❌ No | ✅ Yes | ✅ Yes | ✅ **Yes (LRU cache)** |

## Quick Example

```python
from indexed_parquet import IndexedParquetDataset

# Create dataset from folder
ds = IndexedParquetDataset.from_folder("path/to/data")

# Access like a regular list
row = ds[12345]  
print(row) # {'column_a': 10.5, 'column_b': 'text', ...}

# Direct PyTorch integration
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=32, shuffle=True)
```

## Sections

- [Quickstart](tutorials/quickstart.md) — learn the basics in 5 minutes.
- [Schema Evolution](how-to/schema-evolution.md) — how to work with "dirty" data.
- [Deep Learning Pipeline](tutorials/deep_learning.md) — best way to train models.
- [Architecture](explanation/architecture.md) — how it works under the hood.
