# API Reference

This section provides detailed documentation for the main classes and methods of the `indexed-parquet-dataset` library.

## IndexedParquetDataset

The main class for indexing and data access. Inherits from `torch.utils.data.Dataset` (if PyTorch is installed).

::: indexed_parquet_dataset.dataset.IndexedParquetDataset
    options:
      members:
        - from_folder
        - load_index
        - save_index
        - info
        - shuffle
        - train_test_split
        - select
        - limit
        - sample
        - select_columns
        - map
        - alias
        - rename
        - cast
        - filter
        - merge
        - concat
        - copy
        - clone
        - to_parquet
        - iter_batches
        - generate_collate_fn
        - schema

## CollateHandler

A helper class for correct batch collection in `DataLoader`. Use it via the `ds.generate_collate_fn()` method.

::: indexed_parquet_dataset.dataset.CollateHandler

## SchemaMapper

The internal class responsible for schema transformations and calculated columns.

::: indexed_parquet_dataset.schema.SchemaMapper

## BaseIndex

The class that stores index metadata (file offsets, row counts).

::: indexed_parquet_dataset.indexer.BaseIndex
