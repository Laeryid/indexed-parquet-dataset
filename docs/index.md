# Indexed Parquet Dataset

[![PyPI version](https://badge.fury.io/py/indexed-parquet-dataset.svg)](https://badge.fury.io/py/indexed-parquet-dataset)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Indexed Parquet Dataset** is a high-performance Python library designed for **O(1) random access** to large-scale Parquet datasets. 

It is specifically optimized for Deep Learning workflows, providing a seamless `Dataset` class for **PyTorch** while maintaining minimal memory overhead and supporting complex features like **Schema Evolution**.

## Key Features

- ⚡ **O(1) Random Access**: Instantly jump to any row in a multi-gigabyte dataset without scanning.
- 🔄 **Schema Evolution**: Handle datasets where files have different schemas, missing columns, or renamed fields.
- 📦 **Lazy Loading**: Only opens file handles and reads data when requested, with an efficient LRU handle cache.
- 🔥 **PyTorch Integration**: Native `torch.utils.data.Dataset` support with batching and shuffling.
- 🛠️ **Powerful API**: Built-in support for filtering, mapping, splitting, and sharding.

## Installation

```bash
pip install indexed-parquet-dataset
```

To include PyTorch support:

```bash
pip install "indexed-parquet-dataset[torch]"
```

## Quick Example

```python
from indexed_parquet import IndexedParquetDataset

# Scan a directory and index all parquet files
ds = IndexedParquetDataset.from_folder("path/to/data")

# Access any row instantly
row = ds[12345]  # Returns a dictionary
print(row)

# Integration with PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=32, shuffle=True)
```

Check out the [Quick Start](tutorials/quickstart.md) guide to learn more!
