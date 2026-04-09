# Handling Schema Evolution

One of the most powerful features of `indexed-parquet-dataset` is its ability to handle **Schema Evolution** across a collection of Parquet files. This is common when your data ingestion pipeline changes over time, adding new columns or renaming old ones.

## Missing Columns and Auto-fill

By default, if a column exists in some files but not others, the dataset will return `None` (or a default fill value) for rows in files where the column is missing.

### Using Auto-fill

You can enable automatic filling of missing values with reasonable defaults (0 for numbers, "" for strings, False for bools):

```python
ds = IndexedParquetDataset.from_folder("data", auto_fill=True)
```

### Manual Fill Values

For more control, specify fill values via a hierarchy:

```python
ds = IndexedParquetDataset.from_folder(
    "data",
    default_fill_value="N/A",  # Global fallback
    fill_values_by_type={'int64': -1},  # Fallback for specific PyArrow types
    fill_values_by_column={'priority': 1} # Specific column fallback
)
```

## Renaming Columns

If a column was renamed in newer files (e.g., from `label` to `target`), you can normalize it globally:

```python
ds = IndexedParquetDataset.from_folder("data")
# Normalize 'label' to be visible as 'target' everywhere
ds = ds.rename("label", "target")
```

## Custom Transformations

You can add computed columns or transform existing ones on-the-fly:

```python
def normalize_image(row):
    row['image'] = row['image'] / 255.0
    return row

ds = ds.map(normalize_image)
```

## Filtering by Schema or Metadata

You can filter rows based on file paths or column existence:

```python
# Only use data from certain files
ds = ds.filter(path_filter="*/2023_data/*.parquet")

# Only use rows where a specific column condition is met (server-side via PyArrow)
ds = ds.filter(column_conditions={'status': 'completed'})
```
