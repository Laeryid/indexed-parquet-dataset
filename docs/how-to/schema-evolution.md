# Schema Evolution

One of the most powerful features of `indexed-parquet-dataset` is its ability to work with datasets whose structure has changed over time. This happens when new features are added, old ones are removed, or column names are changed during the data collection process.

## Problem: Different Schemas in Files

Typically, when trying to combine files with different schemas, libraries (like Pandas or PyArrow) throw an error or require manual normalization.

`IndexedParquetDataset` creates a **virtual merged schema**, automatically substituting values for missing columns.

## Handling Gaps

When a column is present in one file but missing in another, the library must return some value for the rows of the "empty" file.

### Automatic Filling (auto_fill)

The easiest way is to enable `auto_fill`. It will fill gaps with reasonable defaults (0 for numbers, an empty string for text, False for bool).

```python
ds = IndexedParquetDataset.from_folder("data", auto_fill=True)
```

### Fill Value Hierarchy

You can fine-tune the filling very precisely. The library looks for a value in the following order:

1.  `fill_values_by_column`: Value for a specific column.
2.  `fill_values_by_type`: Value for a specific PyArrow data type.
3.  `default_fill_value`: Global fallback (default is `None`).

```python
ds = IndexedParquetDataset.from_folder(
    "data",
    default_fill_value="N/A",  
    fill_values_by_type={'int64': -1, 'double': 0.0},
    fill_values_by_column={'priority': 1}
)
```

## Renaming and Mapping

If a column has changed its name (e.g., from `label` to `target`), you can normalize it in two ways:

### Global Renaming (rename)

Forces the library to assume that `label` is now called `target` in all files.

```python
ds = ds.rename("label", "target")
```

### File-Specific Mapping (set_file_mapping)

If only in one specific file the column name is unusual, you can fix it locally:

```python
ds = ds.set_file_mapping(
    "data/v1/bad_file.parquet", 
    {"old_name": "correct_name"}
)
```

## Merging Different Datasets (merge)

If you have two different `IndexedParquetDataset` objects, you can merge them into one. The library automatically:
1. Calculates a common schema.
2. Performs type **upcasting** (e.g., if one dataset's `id` column is `int32` and the other's is `int64`, the resulting one will be `int64`).
3. Removes row duplicates (those referring to the same file and physical row index).

```python
combined_ds = ds1.merge(ds2)
```

## Nested Structures (Structs)

The library can recursively descend into columns of type `Struct`. If a field is missing inside a structure, it will be filled according to the rules above. This is critical for PyTorch, which cannot assemble batches containing `None` at any nesting level.
