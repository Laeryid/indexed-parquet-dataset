<p align="center">
  <img src="logo.png" width="400" alt="Indexed Parquet Dataset Logo">
</p>

<p align="center">
  <a href="https://pypi.org/project/indexed-parquet-dataset/"><img src="https://img.shields.io/pypi/v/indexed-parquet-dataset.svg" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  <a href="https://laeryid.github.io/indexed-parquet-dataset/"><img src="https://img.shields.io/badge/docs-Material%20MkDocs-blue.svg" alt="Documentation"></a>
</p>

# Indexed Parquet Dataset

**Indexed Parquet Dataset** is a high-performance Python library for **O(1) random access** to massive datasets in Parquet format.

It is specifically optimized for Deep Learning (PyTorch), consumes minimal memory, and supports advanced features such as **Schema Evolution** (working with files of different schemas in a single dataset).

## Key Features

- ⚡ **O(1) Random Access**: Instantly navigate to any row in a multi-gigabyte dataset without scanning files.
- 🔄 **Schema Evolution**: Work with datasets where files have different schemas, missing columns, or renamed fields.
- 📦 **Lazy Loading**: Files are opened only when data is requested. Features an efficient LRU handle cache.
- 🔥 **PyTorch Integration**: Native support for `torch.utils.data.Dataset`, including adaptive `collate_fn` generation.
- 🛠️ **Fluent API**: Method chaining: `shuffle`, `filter`, `alias`, `split`, `limit`, `rename`, `cast`, `map`.
- 💾 **Index Persistence**: Save and fast-load the index from a file.
- 🏗️ **Materialization**: "Bake" all transformations into new Parquet files via `clone()`.

## Architecture

The library remains lightweight, storing only metadata and a row map in RAM:

```mermaid
graph TD
    subgraph RAM ["Application (RAM - Lightweight)"]
        direction TB
        subgraph DS ["IndexedParquetDataset"]
            Indices["Indices Array [np.ndarray]<br/>(Shuffled/Filtered indices)"]
            Meta["Metadata & Schema<br/>(File offsets, column mapping)"]
            Cache["File Handle Cache<br/>(Lazy Loading LRU)"]
        end
        
        User["User Code / PyTorch DataLoader"] -- "dataset[idx]" --> Indices
        Indices -- "Global Index" --> Meta
        Meta -- "Find File & Row Offset" --> Cache
    end
    
    subgraph Storage ["Storage (HDD/SSD/S3-over-FUSE)"]
        F1["data_part_1.parquet"]
        F2["data_part_2.parquet"]
        FN["data_part_N.parquet"]
    end
    
    Cache -- "Lazy Read" --> F1
    Cache -- "Lazy Read" --> F2
    Cache -- "Lazy Read" --> FN
    
    F1 -. "O(1) Row Retrieval" .-> User
    F2 -. "O(1) Row Retrieval" .-> User
    FN -. "O(1) Row Retrieval" .-> User
```

## Installation

From PyPI:
```bash
pip install indexed-parquet-dataset
```

For PyTorch support:
```bash
pip install "indexed-parquet-dataset[torch]"
```

## Quickstart

### Basic Initialization

```python
from indexed_parquet import IndexedParquetDataset

# Scans the folder and builds a global row index
ds = IndexedParquetDataset.from_folder("./path/to/data")

print(f"Total rows: {len(ds)}")
print(f"First row: {ds[0]}") # {'id': 1, 'text': '...', ...}

# Random access to any row is instant
sample = ds[999_999]
```

### Transformations (Fluent API)

```python
ds = (IndexedParquetDataset.from_folder("./data")
      .filter(lambda x: x["score"] > 0.5)
      .shuffle(seed=42)
      .alias("text_len", lambda x: len(x["text"]))
      .limit(10000))

# Each row now has a virtual 'text_len' column
print(ds[0]["text_len"])
```

### Usage with PyTorch

```python
from torch.utils.data import DataLoader

ds = IndexedParquetDataset.from_folder("./data", auto_fill=True)
train_ds, val_ds = ds.train_test_split(test_size=0.1)

loader = DataLoader(
    train_ds, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4,
    collate_fn=ds.generate_collate_fn(on_none='fill')
)
```

## Documentation

Full documentation is available on [GitHub Pages](https://laeryid.github.io/indexed-parquet-dataset/).

## License

[Apache 2.0 License](LICENSE)
