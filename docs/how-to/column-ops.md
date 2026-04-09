# Column Operations

`IndexedParquetDataset` allows you to change the data schema virtually. You can rename columns, create new ones based on existing ones, or cast types — all without changing the original files.

## Column Selection (select_columns)

By default, the dataset returns all columns found. To speed up reading (especially over the network) and reduce memory consumption, restrict the selection:

```python
# Only 'id' and 'text' will be read and returned
dataset = dataset.select_columns(["id", "text"])
```

## Renaming and Aliases (rename / alias)

The `.alias()` method is a universal tool for working with column names.

### Simple Renaming

```python
# Now 'raw_text' column will be available as 'text'
dataset = dataset.alias("text", "raw_text")

# Synonym for alias
dataset = dataset.rename("raw_text", "text")
```

### Calculated Columns

You can create a new column based on row data.

```python
# Create 'char_count' column that counts text length
dataset = dataset.alias("char_count", lambda row: len(row["text"]))
```

## Arbitrary Transformations (map)

The `.map()` method allows you to change the entire row or add/remove multiple columns at once. This is more efficient than several `.alias()` calls.

```python
def preprocess(row):
    row["text"] = row["text"].lower().strip()
    row["is_long"] = len(row["text"]) > 100
    return row

# Apply the function to each row
dataset = dataset.map(preprocess)

# Unnecessary columns can be removed at the same time
dataset = dataset.map(preprocess, remove_columns=["raw_metadata"])

# And explicitly fix the result schema (recommended for stability)
dataset = dataset.map(preprocess, output_schema=["id", "text", "is_long"])
```

## Explicit Casting (cast)

Sometimes data in Parquet is stored in an inconvenient format (e.g., numbers stored as strings). The `.cast()` method fixes this.

```python
# Cast a column to a specific type
dataset = dataset.cast("price", "float") # 'int', 'float', 'str' are supported

# Or use your own casting function
dataset = dataset.cast("timestamp", pd.to_datetime)
```

## Application Order

It's important to understand the order in which transformations occur when reading a row:

1.  **Read from file**: Only the required source columns are read.
2.  **Name mapping**: Global and file-specific renames are applied.
3.  **Calculated columns**: Functions passed to `.alias(name, callable)` are executed.
4.  **Row Transforms**: Functions passed to `.map()` are executed.
5.  **Schema filtering**: Columns not included in the final `output_schema` or `selected_columns` are removed.

If this order interferes with your goal (e.g., you want to perform `.map()` on already renamed columns), you can always call `.clone()` to fix the current state and start a new chain.
