from __future__ import annotations
import os
import numpy as np
import warnings
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Callable, Union
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import fnmatch
from .indexer import scan_directory, BaseIndex, FileInfo
from .schema import SchemaMapper

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.n = 0
            self.desc = desc
        def __iter__(self):
            for i in self.iterable:
                yield i
                self.n += 1
        def update(self, n=1): self.n += n
        def close(self): pass
        def set_description(self, desc): self.desc = desc

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    Dataset = object  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False

class IndexedParquetDataset(Dataset):
    """High-performance Parquet dataset with O(1) random access and Schema Evolution support."""
    
    def __init__(
        self,
        index: BaseIndex,
        indices: Optional[np.ndarray] = None,
        mapper: Optional[SchemaMapper] = None,
        include_source_column: bool = False,
        source_column_name: str = "__source_file__",
        default_fill_value: Any = None,
        fill_values_by_type: Optional[Dict[str, Any]] = None,
        fill_values_by_column: Optional[Dict[str, Any]] = None,
        auto_fill: bool = False,
        max_open_files: int = 128,
        _type_casts: Optional[Dict[str, type]] = None,
        selected_columns: Optional[List[str]] = None,
    ):
        """Initializes the dataset.

        Args:
            index: A BaseIndex object containing dataset metadata.
            indices: Array of global indices to expose (for subsetting/shuffling).
            mapper: SchemaMapper for column renaming.
            include_source_column: If True, adds a virtual column with the source file path.
            source_column_name: Name of the virtual source column.
            default_fill_value: Value to use for missing data if no specific rule matches.
            fill_values_by_type: Dict mapping PyArrow types to default values.
            fill_values_by_column: Dict mapping column names to default values.
            auto_fill: If True, automatically populates fill_values_by_type with defaults.
                Note: auto_fill does NOT overwrite values already present in fill_values_by_type.
            max_open_files: Maximum number of simultaneously open Parquet file handles (LRU cache).
            _type_casts: Internal. Per-column cast functions used by concat upcasting.
        """
        self.index = index
        self.mapper = mapper or SchemaMapper()
        self.include_source_column = include_source_column
        self.source_column_name = source_column_name

        self.default_fill_value = default_fill_value
        self.fill_values_by_type = fill_values_by_type or {}

        if auto_fill:
            self._apply_auto_fill()

        self.fill_values_by_column = fill_values_by_column or {}
        self._type_casts: Dict[str, type] = _type_casts or {}  # Internal: concat upcasting
        self.selected_columns = selected_columns

        # indices allows for shuffling, filtering, and subsets without modifying the base index
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.index.total_rows)

        # Cumulative row counts for fast lookups
        self.file_offsets = np.cumsum([0] + [f.num_rows for f in self.index.files])

        # LRU cache for open file handles (Lazy Loading)
        self.max_open_files = max_open_files
        self._file_handles: OrderedDict[int, pq.ParquetFile] = OrderedDict()

    def __getstate__(self):
        """Returns the state for pickling, excluding non-picklable file handles."""
        state = self.__dict__.copy()
        state['_file_handles'] = {} # Don't pickle open handles
        return state

    def __setstate__(self, state):
        """Restores the state after unpickling."""
        self.__dict__.update(state)
        self._file_handles = {} # Re-initialize empty cache

    @classmethod
    def from_folder(
        cls, 
        directory: str, 
        pattern: str = "*.parquet", 
        recursive: bool = True, 
        strict_schema: bool = False,
        auto_fill: bool = False,
        **kwargs
    ) -> 'IndexedParquetDataset':
        """Creates an IndexedParquetDataset by scanning a directory."""
        index = scan_directory(directory, pattern, recursive, strict_schema)
        return cls(index, auto_fill=auto_fill, **kwargs)

    @property
    def schema(self) -> List[str]:
        """Returns the list of column names available in the dataset (after mapping)."""
        if self.selected_columns is not None:
            return self.selected_columns
            
        all_cols = set()
        
        # 1. Base columns from original files
        for col in self.index.all_columns:
            target = self.mapper.mapping.get(col, col)
            if target != col:
                # Global mapping shadows the original name
                all_cols.add(target)
            else:
                # Local mapping check: is it shadowed in ALL files it appears in?
                is_visible_as_original = False
                for f_info in self.index.files:
                    if col in f_info.columns:
                        f_map = self.mapper.file_mappings.get(f_info.path, {})
                        if col not in f_map:
                            is_visible_as_original = True
                            break
                if is_visible_as_original:
                    all_cols.add(col)
                    
        # 2. Add file-specific mapping targets
        for f_map in self.mapper.file_mappings.values():
            for target_col in f_map.values():
                all_cols.add(target_col)
                
        # 3. Add computed columns
        for col in self.mapper.transforms.keys():
            all_cols.add(col)
            
        if self.include_source_column:
            all_cols.add(self.source_column_name)
            
        return sorted(list(all_cols))

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        return (
            f"IndexedParquetDataset("
            f"rows={len(self):,}, "
            f"files={len(self.index.files)}, "
            f"columns={len(self.schema)}"
            f")"
        )

    def _get_file_and_local_idx(self, global_idx: int) -> tuple[int, int]:
        actual_idx = self.indices[global_idx]
        file_idx = np.searchsorted(self.file_offsets, actual_idx, side='right') - 1
        local_idx = actual_idx - self.file_offsets[file_idx]
        return int(file_idx), int(local_idx)

    def _get_file_handle(self, file_idx: int) -> pq.ParquetFile:
        if file_idx in self._file_handles:
            self._file_handles.move_to_end(file_idx)  # LRU touch
        else:
            self._file_handles[file_idx] = pq.ParquetFile(self.index.files[file_idx].path)
            if len(self._file_handles) > self.max_open_files:
                self._file_handles.popitem(last=False)  # evict least recently used
        return self._file_handles[file_idx]

    def _get_fill_value(self, column_name: str) -> Any:
        """Determines the fill value for a missing column based on hierarchy."""
        if column_name in self.fill_values_by_column:
            return self.fill_values_by_column[column_name]
        
        # Find original name (best effort)
        orig_name = None
        for k, v in self.mapper.mapping.items():
            if v == column_name:
                orig_name = k
                break
        if orig_name is None: orig_name = column_name
        
        col_type = self.index.column_types.get(orig_name)
        if col_type in self.fill_values_by_type:
            return self.fill_values_by_type[col_type]
            
        return self.default_fill_value

    def _deep_fill_nones(self, value: Any, fill: Any) -> Any:
        """Recursively replaces None values inside nested dicts and lists.

        PyArrow struct columns with null-typed fields (e.g. ``seed: null``)
        yield Python dicts containing ``None`` values at arbitrary depth.
        ``default_collate`` cannot handle ``NoneType`` anywhere in a batch,
        so we must sanitize the entire nested structure.

        Args:
            value: The value returned by PyArrow (may be dict, list, scalar or None).
            fill:  The replacement value to use wherever None is found.

        Returns:
            A sanitized copy of *value* with all Nones replaced by *fill*.
        """
        if value is None:
            return fill
        if isinstance(value, dict):
            return {k: self._deep_fill_nones(v, fill) for k, v in value.items()}
        if isinstance(value, list):
            return [self._deep_fill_nones(v, fill) for v in value]
        return value

    def _apply_auto_fill(self):
        """Populates fill_values_by_type with default values for common types."""
        defaults = {
            # Integers
            'int8': 0, 'int16': 0, 'int32': 0, 'int64': 0,
            'uint8': 0, 'uint16': 0, 'uint32': 0, 'uint64': 0,
            # Floats
            'float16': 0.0, 'float32': 0.0, 'float64': 0.0, 'double': 0.0, 'halffloat': 0.0,
            # Strings
            'string': "", 'large_string': "", 'utf8': "", 'large_utf8': "",
            # Bool
            'bool': False,
            # Dictionary/Categorical (best effort)
            'dictionary': ""
        }
        for t in set(self.index.column_types.values()):
            clean_t = t.lower().split('[')[0] # Handle complex types like list[int64]
            if clean_t in defaults and t not in self.fill_values_by_type:
                self.fill_values_by_type[t] = defaults[clean_t]

    def _read_rows_from_file(self, file_idx: int, local_indices: List[int]) -> List[Dict[str, Any]]:
        """Reads multiple rows from a single file efficiently."""
        pf = self._get_file_handle(file_idx)
        file_info = self.index.files[file_idx]
        file_path = file_info.path
        
        rg_to_indices = {}
        cumulative_rg_rows = 0
        for i, rg_rows in enumerate(file_info.row_groups):
            mask = (np.array(local_indices) >= cumulative_rg_rows) & (np.array(local_indices) < cumulative_rg_rows + rg_rows)
            rg_indices = np.array(local_indices)[mask]
            if len(rg_indices) > 0:
                rg_to_indices[i] = rg_indices - cumulative_rg_rows
            cumulative_rg_rows += rg_rows

        local_idx_to_result = {}
        target_schema = self.schema
        
        # Performance optimization: determine source columns to read from disk
        # We only read columns that are present in the target_schema and have a 
        # direct mapping to original columns, OR if we have transforms (which might
        # depend on anything, so we read all if transforms are active).
        # Actually, let's just map target_schema back to source columns.
        requested_source_cols = None
        if self.selected_columns is not None and not self.mapper.transforms:
            # We can optimize only if there are no arbitrary row transforms
            requested_source_cols = []
            for col in target_schema:
                if col == self.source_column_name:
                    continue
                src_col = self.mapper.get_source_column(col, file_path)
                if src_col in file_info.columns:
                    requested_source_cols.append(src_col)
            
            # If no columns were found but we need some, we must read at least one 
            # to keep the row count correct, but PyArrow handles empty column lists.
            # However, check for virtual columns.
            if not requested_source_cols and target_schema:
                # Fallback to reading all if optimization feels risky
                requested_source_cols = None

        for rg_idx, rg_local_indices in rg_to_indices.items():
            table = pf.read_row_group(rg_idx, columns=requested_source_cols)
            rg_start_offset = sum(file_info.row_groups[:rg_idx])
            
            for l_idx_in_rg in rg_local_indices:
                row_dict = table.slice(l_idx_in_rg, 1).to_pydict()
                item = {k: v[0] for k, v in row_dict.items()}
                
                # 0. Ensure all columns from index are present (as None/Fill)
                for col in self.index.all_columns:
                    if col not in item:
                        item[col] = self._get_fill_value(self.mapper.mapping.get(col, col))
                
                # 1. Inject virtual source column if needed
                if self.include_source_column:
                    item[self.source_column_name] = file_path
                
                # 2. Apply global then file-specific mapping logic & Computed Columns (via Mapper)
                item = self.mapper.map_columns(item, file_path)
                
                # Ensure all columns in final schema are present
                mapped_item = {}
                for col in target_schema:
                    if col not in item:
                        val = self._get_fill_value(col)
                    else:
                        val = item[col]
                    
                    # Handle None if still None and default is set
                    if val is None:
                        val = self._get_fill_value(col)
                    
                    if col in self._type_casts and val is not None:
                        try:
                            val = self._type_casts[col](val)
                        except (ValueError, TypeError):
                            pass

                    # Recursively sanitize nested dicts/lists (e.g. struct columns
                    # with null-typed fields like `seed: null` in PyArrow schemas).
                    # default_collate cannot handle NoneType anywhere in the tree.
                    if isinstance(val, (dict, list)):
                        fill = self._get_fill_value(col)
                        val = self._deep_fill_nones(val, fill)

                    mapped_item[col] = val

                # 3. Apply row-level transformations (new)
                # These are applied AFTER schema mapping to allow adding new columns
                for row_fn in self.mapper.row_transforms:
                    mapped_item = row_fn(mapped_item)
                
                local_idx_to_result[l_idx_in_rg + rg_start_offset] = mapped_item

        return [local_idx_to_result[l_idx] for l_idx in local_indices]

    def __getitem__(self, idx: Union[int, List[int], slice, np.ndarray]) -> Any:
        if isinstance(idx, (int, np.integer)):
            if idx < 0: idx += len(self)
            if idx < 0 or idx >= len(self): raise IndexError("Index out of range")
            return self.__getitems__([int(idx)])[0]
        elif isinstance(idx, (list, np.ndarray)):
            return self.__getitems__(list(idx))
        elif isinstance(idx, slice):
            return self.select(idx)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __getitems__(self, indices: List[int]) -> List[Dict[str, Any]]:
        file_to_local_indices = {}
        for i, global_idx in enumerate(indices):
            f_idx, l_idx = self._get_file_and_local_idx(global_idx)
            if f_idx not in file_to_local_indices: file_to_local_indices[f_idx] = []
            file_to_local_indices[f_idx].append((i, l_idx))
            
        results = [None] * len(indices)
        for f_idx, indexed_l_indices in file_to_local_indices.items():
            original_positions = [x[0] for x in indexed_l_indices]
            l_indices = [x[1] for x in indexed_l_indices]
            file_results = self._read_rows_from_file(f_idx, l_indices)
            for pos, res in zip(original_positions, file_results):
                results[pos] = res
        return results # type: ignore

    def shuffle(self, seed: Optional[int] = None, rg_buffer: Optional[int] = None) -> 'IndexedParquetDataset':
        """Shuffles the dataset indices.
        
        Args:
            seed: Random seed for reproducibility.
            rg_buffer: If provided, enables locality-aware shuffling. Instead of a global 
                shuffle, it shuffles row groups and then shuffles rows within a window 
                of `rg_buffer` row groups. This significantly Improves I/O performance 
                by reducing the number of row groups that need to be read/cached 
                simultaneously.
                
        Returns:
            A new IndexedParquetDataset instance with shuffled indices.
        """
        rng = np.random.default_rng(seed)
        
        if rg_buffer is None:
            new_indices = self.indices.copy()
            rng.shuffle(new_indices)
            return self._clone_with_indices(new_indices)

        if rg_buffer < 1:
            raise ValueError("rg_buffer must be at least 1")

        # 1. Map current active indices to their respective row groups
        actual_indices = self.indices
        
        # Find which file each index belongs to
        file_indices = np.searchsorted(self.file_offsets, actual_indices, side='right') - 1
        
        # Grouping by (file_idx, rg_idx)
        # Note: dict preserves insertion order of first appearance in Python 3.7+
        from collections import defaultdict
        rg_to_indices = defaultdict(list)
        
        # Pre-calculate RG cumulative boundaries for each file for speed
        file_rg_bounds = [np.cumsum([0] + f.row_groups) for f in self.index.files]
        
        # Process indices by file to use vector searchsorted logic
        unique_f_indices = np.unique(file_indices)
        for f_idx in unique_f_indices:
            f_mask = (file_indices == f_idx)
            f_global_indices = actual_indices[f_mask]
            f_local_indices = f_global_indices - self.file_offsets[f_idx]
            
            # Map local row index to row group index within this file
            f_rg_indices = np.searchsorted(file_rg_bounds[f_idx], f_local_indices, side='right') - 1
            
            # Group global indices by RG
            for rg_idx, global_idx in zip(f_rg_indices, f_global_indices):
                rg_to_indices[(f_idx, rg_idx)].append(global_idx)
        
        # 2. Get the list of all unique row group IDs present in current indices
        rg_keys = list(rg_to_indices.keys())
        # Shuffle the sequence of row groups themselves
        rng.shuffle(rg_keys)
        
        # 3. Flatten and shuffle within windows of rg_buffer
        final_indices = []
        for i in range(0, len(rg_keys), rg_buffer):
            window_slice = rg_keys[i : i + rg_buffer]
            window_pool = []
            for key in window_slice:
                window_pool.extend(rg_to_indices[key])
            
            # Shuffle rows within the current window of row groups
            window_pool_arr = np.array(window_pool)
            rng.shuffle(window_pool_arr)
            final_indices.append(window_pool_arr)
            
        if not final_indices:
            return self._clone_with_indices(np.array([], dtype=int))
            
        return self._clone_with_indices(np.concatenate(final_indices))

    def select(self, range_or_indices: Union[slice, List[int], np.ndarray]) -> 'IndexedParquetDataset':
        new_indices = self.indices[range_or_indices]
        return self._clone_with_indices(new_indices)

    def limit(self, n: int) -> 'IndexedParquetDataset':
        return self.select(slice(0, n))

    def _clone_with_indices(self, new_indices: np.ndarray) -> 'IndexedParquetDataset':
        return IndexedParquetDataset(
            self.index, new_indices, self.mapper,
            self.include_source_column, self.source_column_name,
            self.default_fill_value, self.fill_values_by_type, self.fill_values_by_column,
            max_open_files=self.max_open_files,
            _type_casts=self._type_casts.copy(),
            selected_columns=self.selected_columns,
        )

    def _clone_with_mapper(self, new_mapper: SchemaMapper) -> 'IndexedParquetDataset':
        return IndexedParquetDataset(
            self.index, self.indices.copy(), new_mapper,
            self.include_source_column, self.source_column_name,
            self.default_fill_value, self.fill_values_by_type, self.fill_values_by_column,
            max_open_files=self.max_open_files,
            _type_casts=self._type_casts.copy(),
            selected_columns=self.selected_columns,
        )

    def map(
        self, 
        fn: Callable[[dict], dict], 
        *, 
        remove_columns: Optional[List[str]] = None,
        output_schema: Optional[List[str]] = None
    ) -> 'IndexedParquetDataset':
        """Applies a row-level transformation to the dataset.
        
        Args:
            fn: A function that takes a row (dict) and returns a transformed row (dict).
            remove_columns: Optional list of columns to remove from the result.
            output_schema: Optional explicit list of columns for the new schema.
            
        Returns:
            A new IndexedParquetDataset instance.
        """
        new_mapper = SchemaMapper(
            mapping=self.mapper.mapping.copy(),
            file_mappings=self.mapper.file_mappings.copy(),
            transforms=self.mapper.transforms.copy(),
            row_transforms=self.mapper.row_transforms.copy()
        )
        effective_fn = fn
        if remove_columns:
            def _wrapped_fn(row, _fn=fn, _cols=remove_columns):
                result = _fn(row)
                for c in _cols:
                    result.pop(c, None)
                return result
            effective_fn = _wrapped_fn
            
        new_mapper.row_transforms.append(effective_fn)
        ds = self._clone_with_mapper(new_mapper)
        if output_schema:
            ds.selected_columns = output_schema
        return ds


    def train_test_split(
        self, 
        test_size: Union[float, int], 
        shuffle: bool = True, 
        seed: Optional[int] = None, 
        stratify_by: Optional[str] = None
    ) -> tuple['IndexedParquetDataset', 'IndexedParquetDataset']:
        """Splits the dataset into train and test sets."""
        n = len(self)
        if isinstance(test_size, float):
            n_test = int(n * test_size)
        else:
            n_test = test_size
        n_train = n - n_test

        if stratify_by:
            # Read labels for all indices (required for stratification)
            labels = []
            for i in range(len(self)):
                labels.append(self[i][stratify_by])
            labels = np.array(labels)
            
            unique_labels, inverse = np.unique(labels, return_inverse=True)
            train_indices_list = []
            test_indices_list = []
            
            rng = np.random.default_rng(seed)
            
            for i in range(len(unique_labels)):
                idx_in_group = np.where(inverse == i)[0]
                if shuffle:
                    rng.shuffle(idx_in_group)
                
                group_n_test = int(len(idx_in_group) * (n_test / n))
                test_indices_list.extend(idx_in_group[:group_n_test])
                train_indices_list.extend(idx_in_group[group_n_test:])
            
            train_indices = self.indices[np.array(train_indices_list)]
            test_indices = self.indices[np.array(test_indices_list)]
        else:
            indices = self.indices.copy()
            if shuffle:
                rng = np.random.default_rng(seed)
                rng.shuffle(indices)
            
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]

        return self._clone_with_indices(train_indices), self._clone_with_indices(test_indices)

    def copy(self) -> 'IndexedParquetDataset':
        """Returns a copy of the dataset."""
        return self._clone_with_indices(self.indices.copy())

    def select_columns(self, columns: List[str]) -> 'IndexedParquetDataset':
        """Selects a subset of columns to be returned.
        
        Args:
            columns: List of column names to keep.
            
        Returns:
            A new IndexedParquetDataset instance with updated schema.
        """
        # Validate columns exist in current schema
        current_schema = self.schema
        for col in columns:
            if col not in current_schema:
                warnings.warn(f"Column '{col}' not found in current schema.")
                
        ds = self.copy()
        ds.selected_columns = columns
        return ds

    def sample(self, n: int, seed: Optional[int] = None) -> 'IndexedParquetDataset':
        """Returns a random sample of n rows from the dataset.
        
        Args:
            n: Number of rows to sample.
            seed: Random seed for reproducibility.
            
        Returns:
            A new IndexedParquetDataset instance.
        """
        if n > len(self):
            n = len(self)
        
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self), size=n, replace=False)
        return self.select(indices)

    def iter_batches(self, batch_size: int, shuffle: bool = False, seed: Optional[int] = None):
        """Yields batches of data from the dataset.
        
        Args:
            batch_size: Number of rows per batch.
            shuffle: Whether to shuffle before iterating.
            seed: Random seed for shuffling.
        """
        ds = self.shuffle(seed) if shuffle else self
        n = len(ds)
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            yield ds[i:end]

    def alias(self, name: str, source: Union[str, Callable]) -> 'IndexedParquetDataset':
        """Creates a new alias for a column or a new computed column.
        
        Args:
            name: The target name of the column.
            source: Either an original column name (string) or a function function(row) -> value.
            
        Returns:
            A new IndexedParquetDataset instance.
        """
        new_mapper = SchemaMapper(
            mapping=self.mapper.mapping.copy(),
            file_mappings=self.mapper.file_mappings.copy(),
            transforms=self.mapper.transforms.copy()
        )
        if isinstance(source, str):
            new_mapper.mapping[source] = name
            # Remove transform if we are re-aliasing to a source column
            if name in new_mapper.transforms:
                del new_mapper.transforms[name]
        elif callable(source):
            new_mapper.transforms[name] = source
        else:
            raise TypeError("Alias source must be a string or a callable.")
            
        return self._clone_with_mapper(new_mapper)

    def set_file_mapping(self, file_path: str, mapping: Dict[str, str]) -> 'IndexedParquetDataset':
        """Sets a file-specific column mapping.
        
        Args:
            file_path: Absolute path to the file.
            mapping: Dict mapping source column names to target names.
        """
        new_mapper = SchemaMapper(
            mapping=self.mapper.mapping.copy(),
            file_mappings=self.mapper.file_mappings.copy(),
            transforms=self.mapper.transforms.copy()
        )
        # Ensure absolute path
        abs_path = os.path.abspath(file_path)
        new_mapper.file_mappings[abs_path] = mapping
        return self._clone_with_mapper(new_mapper)

    def rename(self, old_name: str, new_name: str) -> 'IndexedParquetDataset':
        """Renames a column."""
        return self.alias(new_name, old_name)


    def cast(self, column: str, target_type: Union[type, str, Callable]) -> 'IndexedParquetDataset':
        """Changes the type of a column using an alias transformation.
        
        Args:
            column: The name of the column to cast.
            target_type: The target type (int, float, str, etc.) or a callable.
        """
        if isinstance(target_type, str):
            if target_type == 'int': cast_fn = int
            elif target_type in ('float', 'double'): cast_fn = float
            elif target_type in ('str', 'string'): cast_fn = str
            else: raise ValueError(f"Unsupported type string: {target_type}")
        elif callable(target_type):
            cast_fn = target_type # type: ignore
        else:
            raise TypeError("target_type must be a string (int, float, str) or a callable.")

        def transform(row):
            val = row.get(column)
            if val is None:
                return None
            try:
                return cast_fn(val)
            except (ValueError, TypeError):
                return val
            
        return self.alias(column, transform)

    def get_arrow_schema(self) -> pa.Schema:
        """Derives the PyArrow schema for the dataset from source files and metadata.
        
        This is used during materialize operations (to_parquet) to ensure consistent 
        types even if some batches contain only None values (avoiding null-type inference).
        """
        fields = []
        file_schemas = {}
        
        # Mapping from common type strings to Arrow types
        # Note: mapping 'float' and 'double' to pa.float64()
        basic_types = {
            'int64': pa.int64(),
            'int32': pa.int32(),
            'float64': pa.float64(),
            'double': pa.float64(),
            'float': pa.float64(),
            'float32': pa.float32(),
            'string': pa.string(),
            'bool': pa.bool_(),
            'binary': pa.binary(),
            'timestamp[ns]': pa.timestamp('ns'),
            'timestamp[us]': pa.timestamp('us'),
            'timestamp[ms]': pa.timestamp('ms'),
            'timestamp[s]': pa.timestamp('s'),
        }
        
        for col in self.schema:
            dataType = None
            
            # 1. Check current index column types (already accounts for merge upcasts)
            # Find the original column name if mapped
            src_col = None
            for k, v in self.mapper.mapping.items():
                if v == col:
                    src_col = k
                    break
            if src_col is None: src_col = col
            
            type_str = self.index.column_types.get(src_col)
            if type_str and type_str.lower() in basic_types:
                dataType = basic_types[type_str.lower()]
            
            # 2. If not a basic type or not found in index, look in source files
            if dataType is None:
                for f_info in self.index.files:
                    if src_col in f_info.columns:
                        if f_info.path not in file_schemas:
                            try:
                                file_schemas[f_info.path] = pq.read_schema(f_info.path)
                            except: continue
                        f_schema = file_schemas.get(f_info.path)
                        if f_schema and src_col in f_schema.names:
                            found_type = f_schema.field(src_col).type
                            # If it's a null type, keep looking for a better one
                            if not pa.types.is_null(found_type):
                                dataType = found_type
                                break
                            else:
                                dataType = found_type # Fallback if no better found
            
            # 3. Virtual, computed or null column?
            # If we only found a 'null' type, or found nothing, try to infer from sample
            if dataType is None or pa.types.is_null(dataType):
                # Sample rows to infer type.
                sample_size = min(100, len(self))
                if sample_size > 0:
                    try:
                        # Using list slicing to get a batch
                        sample_batch = self[:sample_size]
                        sample_vals = [row.get(col) for row in sample_batch]
                        non_nones = [v for v in sample_vals if v is not None]
                        if non_nones:
                            # Use ALL non-nones in sample to infer the BROADEST type
                            inferred_type = pa.array(non_nones).type
                            if not pa.types.is_null(inferred_type):
                                dataType = inferred_type
                    except:
                        pass
            
            # 4. Final safety fallbacks
            if dataType is None or pa.types.is_null(dataType):
                if col == self.source_column_name:
                    dataType = pa.string()
                else:
                    dataType = pa.string() # pa.null() is dangerous for ParquetWriter
            
            fields.append(pa.field(col, dataType))
        
        return pa.schema(fields)

    def to_parquet(
        self, 
        path: str, 
        chunk_size: int = 1024, 
        shard_size: Optional[int] = None,
        optimize_by_reorder: bool = True,
        progress: bool = True
    ):
        """Materializes the dataset to one or more Parquet files.
        
        Args:
            path: Output file path or directory (if sharding).
            chunk_size: Cache size for intermediate batch collection.
            shard_size: If set, splits the dataset into multiple files of this many rows.
            optimize_by_reorder: If True, reads data in source-linear order (fastest),
                but potentially changes the row order in the output file.
            progress: Whether to show a progress bar.
        """
        # Ensure .parquet extension for single file output
        effective_path = str(path)
        if not shard_size and not effective_path.lower().endswith('.parquet'):
            effective_path += '.parquet'

        if shard_size:
            os.makedirs(effective_path, exist_ok=True)
            
        writer = None
        rows_in_current_shard = 0
        shard_idx = 0
        
        # Derive target schema upfront to avoid type inference issues
        target_schema = self.get_arrow_schema()
        
        def get_shard_path():
            if shard_size:
                return os.path.join(effective_path, f"part_{shard_idx:04d}.parquet")
            return effective_path

        def write_table_with_shards(table):
            nonlocal writer, rows_in_current_shard, shard_idx
            
            if not shard_size:
                if writer is None:
                    writer = pq.ParquetWriter(get_shard_path(), target_schema)
                writer.write_table(table)
                rows_in_current_shard += len(table)
                return

            offset = 0
            while offset < len(table):
                # If current shard is full, move to next
                if rows_in_current_shard >= shard_size:
                    if writer: writer.close()
                    shard_idx += 1
                    writer = None
                    rows_in_current_shard = 0
                
                remaining_in_shard = shard_size - rows_in_current_shard
                to_write = min(len(table) - offset, remaining_in_shard)
                
                chunk = table.slice(offset, to_write)
                if writer is None:
                    writer = pq.ParquetWriter(get_shard_path(), target_schema)
                
                writer.write_table(chunk)
                rows_in_current_shard += to_write
                offset += to_write

        total_rows = len(self)
        pbar = tqdm(total=total_rows, desc="Writing Parquet", disable=not progress)
        
        try:
            if not optimize_by_reorder:
                # Original slow path (preserves order)
                effective_chunk_size = chunk_size 
                for i in range(0, total_rows, effective_chunk_size):
                    batch_indices = list(range(i, min(i + effective_chunk_size, total_rows)))
                    batch_data = self[batch_indices]
                    if not batch_data: continue
                    
                    table = pa.Table.from_pylist(batch_data, schema=target_schema)
                    write_table_with_shards(table)
                    pbar.update(len(batch_data))
            else:
                # Optimized path: Group by (file, RG) to minimize reads
                # ...
                from collections import defaultdict
                file_to_rg_to_rows = defaultdict(lambda: defaultdict(list))
                
                # Pre-calculate RG boundaries for speed
                file_rg_bounds = [np.cumsum([0] + f.row_groups) for f in self.index.files]
                
                # Process active indices
                for idx in self.indices:
                    f_idx = np.searchsorted(self.file_offsets, idx, side='right') - 1
                    l_idx = idx - self.file_offsets[f_idx]
                    rg_idx = np.searchsorted(file_rg_bounds[f_idx], l_idx, side='right') - 1
                    l_idx_in_rg = l_idx - file_rg_bounds[f_idx][rg_idx]
                    file_to_rg_to_rows[f_idx][rg_idx].append(l_idx_in_rg)

                # 2. Iterate and write
                buffer = []
                
                # Sort files and RGs for linear access
                for f_idx in sorted(file_to_rg_to_rows.keys()):
                    pf = self._get_file_handle(f_idx)
                    file_info = self.index.files[f_idx]
                    
                    # Columns optimization
                    requested_columns = None
                    if self.selected_columns is not None and not self.mapper.transforms:
                        requested_columns = []
                        for col in self.schema:
                            if col == self.source_column_name: continue
                            src_col = self.mapper.get_source_column(col, file_info.path)
                            if src_col in file_info.columns: requested_columns.append(src_col)
                        if not requested_columns and self.schema: requested_columns = None

                    rgs = file_to_rg_to_rows[f_idx]
                    for rg_idx in sorted(rgs.keys()):
                        rows_in_rg = rgs[rg_idx]
                        table = pf.read_row_group(rg_idx, columns=requested_columns)
                        
                        for l_idx_in_rg in rows_in_rg:
                            row_dict = table.slice(l_idx_in_rg, 1).to_pydict()
                            item = {k: v[0] for k, v in row_dict.items()}
                            
                            for col in self.index.all_columns:
                                if col not in item: item[col] = self._get_fill_value(self.mapper.mapping.get(col, col))
                            if self.include_source_column: item[self.source_column_name] = file_info.path
                            item = self.mapper.map_columns(item, file_info.path)
                            
                            mapped_item = {}
                            for col in self.schema:
                                val = item.get(col, self._get_fill_value(col))
                                if val is None: val = self._get_fill_value(col)
                                if col in self._type_casts and val is not None:
                                    try: val = self._type_casts[col](val)
                                    except: pass
                                if isinstance(val, (dict, list)):
                                    val = self._deep_fill_nones(val, self._get_fill_value(col))
                                mapped_item[col] = val
                            for row_fn in self.mapper.row_transforms:
                                mapped_item = row_fn(mapped_item)
                            
                            buffer.append(mapped_item)
                            
                            if len(buffer) >= chunk_size:
                                out_table = pa.Table.from_pylist(buffer, schema=target_schema)
                                write_table_with_shards(out_table)
                                pbar.update(len(buffer))
                                buffer = []

                if buffer:
                    out_table = pa.Table.from_pylist(buffer, schema=target_schema)
                    write_table_with_shards(out_table)
                    pbar.update(len(buffer))
        finally:
            if writer: writer.close()
            pbar.close()

    def clone(
        self, 
        path: str, 
        optimize_by_reorder: bool = True, 
        progress: bool = True
    ) -> 'IndexedParquetDataset':
        """Materializes all computations and returns a new dataset instance."""
        # to_parquet will append .parquet if needed, but we need to know the effective path
        effective_path = str(path)
        if not effective_path.lower().endswith('.parquet'):
            effective_path += '.parquet'
            
        self.to_parquet(effective_path, optimize_by_reorder=optimize_by_reorder, progress=progress)
        return IndexedParquetDataset.from_folder(effective_path)

    def filter(
        self, 
        path_pattern: Optional[Union[str, Callable]] = None,
        path_filter: Optional[Union[str, List[str]]] = None,
        column_conditions: Optional[Dict[str, Any]] = None,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
        show_progress: bool = False
    ) -> 'IndexedParquetDataset':
        """Filters the dataset based on file paths, column conditions, or a custom predicate.

        This method is lazy and returns a new IndexedParquetDataset instance with updated indices.
        It supports three levels of filtering:
        1. File-level (fastest): via `path_pattern` or `path_filter`.
        2. Column-level (fast, PyArrow-based): via `column_conditions`.
        3. Row-level (flexible, Python-based): via `predicate`.

        Args:
            path_pattern: A string to search for as a substring in the absolute file path.
                If a callable is provided, it is automatically treated as the `predicate` argument.
            path_filter: A glob pattern (e.g., `"**/2023/*.parquet"`) or a list of glob patterns.
                Only files matching at least one pattern will be kept.
            column_conditions: A dictionary mapping column names to filter conditions.
                Supported formats:
                - `{"col": value}`: Exact match equality on PyArrow side.
                - `{"col": ("operator", value)}`: Comparison using operators: 
                  `"=="`, `">"`, `">="`, `"<"`, `"<="`.
                Example: `{"category": "A", "score": (">", 0.5)}`
            predicate: A callable that takes a row (dict) and returns a boolean.
                Example: `lambda row: len(row["text"]) > 100`
            show_progress: If True, displays progress bars (using tqdm) during file processing 
                for column conditions and predicates.

        Returns:
            A new IndexedParquetDataset instance containing only the matching rows.

        Note:
            If multiple filter types are provided, they are applied in the following order:
            File filters -> Column conditions -> Predicate.
        """
        if callable(path_pattern): predicate = path_pattern; path_pattern = None
        current_indices = self.indices.copy()

        if path_pattern or path_filter:
            valid_file_indices = []
            filters = [path_filter] if isinstance(path_filter, str) else (path_filter or [])
            for i, f in enumerate(self.index.files):
                match = (path_pattern and isinstance(path_pattern, str) and path_pattern in f.path)
                if not match:
                    for pattern in filters:
                        if fnmatch.fnmatch(f.path, pattern): match = True; break
                if match: valid_file_indices.append(i)
            
            mask = np.zeros(len(current_indices), dtype=bool)
            for f_idx in valid_file_indices:
                start, end = self.file_offsets[f_idx], self.file_offsets[f_idx + 1]
                mask |= (current_indices >= start) & (current_indices < end)
            current_indices = current_indices[mask]

        if len(current_indices) > 0 and column_conditions:
            file_to_indices = {}
            for idx in current_indices:
                f_idx = np.searchsorted(self.file_offsets, idx, side='right') - 1
                if f_idx not in file_to_indices: file_to_indices[f_idx] = []
                file_to_indices[f_idx].append(idx)
            
            new_indices_list = []
            iterable = file_to_indices.items()
            if show_progress:
                iterable = tqdm(iterable, desc="Filtering files by conditions")
            for f_idx, f_global_indices in iterable:
                f_info, pf = self.index.files[f_idx], self._get_file_handle(f_idx)
                # Simplified condition check via pyarrow
                table = pf.read(columns=[self.mapper.get_source_column(c) for c in column_conditions.keys() if self.mapper.get_source_column(c) in f_info.columns])
                file_mask = None
                for col, cond in column_conditions.items():
                    src_col = self.mapper.get_source_column(col, f_info.path)
                    if src_col not in table.column_names:
                        c_mask = pa.scalar(None, type=pa.bool_())
                    else:
                        arr = table.column(src_col)
                        if isinstance(cond, tuple):
                            op, val = cond
                            if op == '==': c_mask = pc.equal(arr, val)
                            elif op == '>': c_mask = pc.greater(arr, val)
                            elif op == '>=': c_mask = pc.greater_equal(arr, val)
                            elif op == '<': c_mask = pc.less(arr, val)
                            elif op == '<=': c_mask = pc.less_equal(arr, val)
                            else: c_mask = None
                        else:
                            c_mask = pc.equal(arr, cond)
                    
                    if c_mask is not None:
                        if file_mask is None: file_mask = c_mask
                        else: file_mask = pc.and_(file_mask, c_mask)
                
                if file_mask is None:
                    file_mask_np = np.ones(len(table), dtype=bool)
                else:
                    file_mask_np = pc.fill_null(file_mask, False).to_numpy().astype(bool)
                
                f_local_indices = (np.array(f_global_indices) - self.file_offsets[f_idx]).astype(int)
                new_indices_list.append(np.array(f_global_indices)[file_mask_np[f_local_indices]])
            current_indices = np.concatenate(new_indices_list) if new_indices_list else np.array([], dtype=int)

        if len(current_indices) > 0 and predicate:
            file_to_indices = {}
            for idx in current_indices:
                f_idx = np.searchsorted(self.file_offsets, idx, side='right') - 1
                if f_idx not in file_to_indices: file_to_indices[f_idx] = []
                file_to_indices[f_idx].append(idx)
            
            new_indices_list = []
            iterable = file_to_indices.items()
            if show_progress:
                iterable = tqdm(iterable, desc="Filtering files by predicate")
                
            for f_idx, f_global_indices in iterable:
                f_local_indices = (np.array(f_global_indices) - self.file_offsets[f_idx]).astype(int)
                rows = self._read_rows_from_file(f_idx, f_local_indices.tolist())
                mask = [predicate(row) for row in rows]
                new_indices_list.append(np.array(f_global_indices)[np.array(mask)])
            current_indices = np.concatenate(new_indices_list) if new_indices_list else np.array([], dtype=int)

        return self._clone_with_indices(current_indices)

    def merge(self, other: 'IndexedParquetDataset') -> 'IndexedParquetDataset':
        """Merges this dataset with another one, deduplicating identical rows.
        
        A row is considered identical if it comes from the same file and has
        the same local row index.
        """
        # 1. Unified unique files
        self_paths = {f.path: i for i, f in enumerate(self.index.files)}
        other_paths = {f.path: i for i, f in enumerate(other.index.files)}
        
        all_unique_paths = sorted(list(set(self_paths.keys()) | set(other_paths.keys())))
        path_to_new_idx = {path: i for i, path in enumerate(all_unique_paths)}
        
        new_files_info = []
        for path in all_unique_paths:
            if path in self_paths:
                new_files_info.append(self.index.files[self_paths[path]])
            else:
                new_files_info.append(other.index.files[other_paths[path]])
        
        # 2. Map indices to row identity (new_file_idx, local_idx)
        def get_row_identities(ds, path_map):
            ids = []
            for i in range(len(ds)):
                f_idx, l_idx = ds._get_file_and_local_idx(i)
                f_path = ds.index.files[f_idx].path
                new_f_idx = path_map[f_path]
                ids.append((new_f_idx, l_idx))
            return ids

        self_ids = get_row_identities(self, path_to_new_idx)
        other_ids = get_row_identities(other, path_to_new_idx)
        
        # 3. Deduplicate preserving order (self first, then new from other)
        # We use a dictionary as an ordered set
        seen = {}
        unified_ids = []
        for row_id in (self_ids + other_ids):
            if row_id not in seen:
                seen[row_id] = True
                unified_ids.append(row_id)
        
        # 4. Create new BaseIndex metadata
        new_total_rows_meta = sum(f.num_rows for f in new_files_info)
        new_all_columns = sorted(list(set(self.index.all_columns) | set(other.index.all_columns)))
        
        # 5. Type merging logic (upcasting conflicts)
        new_column_types = self.index.column_types.copy()
        type_casts = self._type_casts.copy()
        for col, o_type in other.index.column_types.items():
            if col in new_column_types:
                s_type = new_column_types[col]
                if s_type != o_type:
                    common_type = 'string'
                    s_is_float = 'float' in s_type or 'double' in s_type
                    o_is_float = 'float' in o_type or 'double' in o_type
                    s_is_int = 'int' in s_type
                    o_is_int = 'int' in o_type
                    
                    if (s_is_int and o_is_float) or (s_is_float and o_is_int) or (s_is_float and o_is_float):
                        common_type = 'double'
                    
                    warnings.warn(f"Type mismatch for column '{col}': {s_type} vs {o_type}. Upcasting to {common_type}.")
                    new_column_types[col] = common_type
                    cast_fn = str if common_type == 'string' else (float if common_type == 'double' else None)
                    if cast_fn: type_casts[col] = cast_fn
            else:
                new_column_types[col] = o_type

        new_index = BaseIndex(new_files_info, new_total_rows_meta, new_all_columns, new_column_types)
        
        # 6. Re-calculate global indices based on new file sequence
        new_file_offsets = np.zeros(len(new_files_info) + 1, dtype=int)
        current_offset = 0
        for i, f in enumerate(new_files_info):
            new_file_offsets[i] = current_offset
            current_offset += f.num_rows
        new_file_offsets[-1] = current_offset
        
        new_indices = np.array([new_file_offsets[f_idx] + l_idx for f_idx, l_idx in unified_ids], dtype=int)
        
        # 7. Merge mappers
        new_mapper = self.mapper.merge(
            other.mapper,
            list(self_paths.keys()),
            list(other_paths.keys())
        )

        return IndexedParquetDataset(
            new_index, new_indices, new_mapper,
            self.include_source_column, self.source_column_name,
            self.default_fill_value, self.fill_values_by_type, self.fill_values_by_column,
            max_open_files=self.max_open_files,
            _type_casts=type_casts,
            selected_columns=self.selected_columns
        )

    def get_supported_types(self) -> Dict[str, Any]:
        """Returns types and their current defaults."""
        res = {}
        for col, t in self.index.column_types.items():
            default = self.fill_values_by_column.get(col) or self.fill_values_by_type.get(t) or self.default_fill_value
            res[t] = {"example_column": col, "default": default}
        return res

    def generate_collate_fn(self, on_none: str = 'raise') -> Callable:
        """Returns a DataLoader-compatible collate function with robust None handling.

        Args:
            on_none: Strategy for handling None values.
                'raise' (default): Raises a descriptive TypeError.
                'drop': Drops samples containing None from the batch.
                'fill': Replaces None with 0/"" based on fill_values configuration.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for generate_collate_fn. "
                "Install it with: pip install torch"
            )
        # Pre-compute fill map: {column_name -> fill_value} for all known columns.
        # This avoids storing a reference to the full dataset inside CollateHandler.
        fill_map = {col: self._get_fill_value(col) for col in self.schema}
        return CollateHandler(fill_map, on_none)

    def info(self) -> None:
        """Prints summary statistics and metadata for the dataset."""
        total_indexed_rows = self.index.total_rows
        visible_rows = len(self)
        num_indexed_files = len(self.index.files)
        
        # Calculate visible count per file
        if visible_rows > 0:
            file_indices = np.searchsorted(self.file_offsets, self.indices, side='right') - 1
            unique_f_idx, counts = np.unique(file_indices, return_counts=True)
            f_idx_to_visible_count = dict(zip(unique_f_idx, counts))
        else:
            f_idx_to_visible_count = {}
            
        active_files = len(f_idx_to_visible_count)
        
        # Calculate storage size
        total_bytes = 0
        for f in self.index.files:
            if os.path.exists(f.path):
                total_bytes += os.path.getsize(f.path)
        
        if total_bytes < 1024**2:
            storage_str = f"{total_bytes / 1024:.2f} KB"
        elif total_bytes < 1024**3:
            storage_str = f"{total_bytes / (1024**2):.2f} MB"
        else:
            storage_str = f"{total_bytes / (1024**3):.2f} GB"
            
        print(f"\n{'='*95}")
        print(f" IndexedParquetDataset Summary")
        print(f"{'='*95}")
        print(f" Files (Active/Indexed):  {active_files}/{num_indexed_files:<6}  |  Storage Size:  {storage_str}")
        print(f" Rows (Visible/Total):    {visible_rows:,}/{total_indexed_rows:,} ({visible_rows/total_indexed_rows:.1%})")
        print(f"{'-'*95}")
        
        # Files Table
        print("\nFiles in Index:")
        file_header = f"{'#':<3} | {'Visible':>12} | {'Total':>12} | {'%':>6} | {'Groups':>6} | {'Path':<}"
        print(file_header)
        print("-" * 95)
        for i, f in enumerate(self.index.files):
            vis_count = f_idx_to_visible_count.get(i, 0)
            vis_ratio = (vis_count / f.num_rows) if f.num_rows > 0 else 0
            display_path = (f"...{f.path[-45:]}" if len(f.path) > 45 else f.path)
            print(f"{i:<3} | {vis_count:>12,} | {f.num_rows:>12,} | {vis_ratio:>6.1%} | {len(f.row_groups):>6} | {display_path}")
            
        # Columns Table
        print("\nColumn Statistics (Active State):")
        col_header = f"{'Column':<25} | {'Type':<12} | {'Files':>6} | {'Presence':>8} | {'Visible Rows':>12} | {'Coverage'}"
        print(col_header)
        print("-" * 95)
        
        visible_schema = self.schema
        for col in visible_schema:
            files_present = 0
            visible_rows_with_col = 0
            orig_name = next((k for k, v in self.mapper.mapping.items() if v == col), col)
            ctype = self.index.column_types.get(orig_name, "n/a")
            
            for f_idx, f in enumerate(self.index.files):
                src_col = self.mapper.get_source_column(col, f.path)
                if src_col in f.columns:
                    files_present += 1
                    visible_rows_with_col += f_idx_to_visible_count.get(f_idx, 0)
            
            presence_pct = files_present / num_indexed_files
            coverage_pct = visible_rows_with_col / visible_rows if visible_rows > 0 else 0
            
            # Truncate column name if too long
            col_display = (col[:22] + "...") if len(col) > 25 else col
            print(f"{col_display:<25} | {ctype:<12} | {files_present:>6} | {presence_pct:>8.1%} | {visible_rows_with_col:>12,} | {coverage_pct:>8.1%}")
        
        # Mappings
        if self.mapper.mapping or self.mapper.file_mappings or self.mapper.transforms:
            print("\nActive Mappings & Transforms:")
            if self.mapper.mapping:
                print(f"  Aliases: {self.mapper.mapping}")
            if self.mapper.transforms:
                print(f"  Computed Columns: {list(self.mapper.transforms.keys())}")
            if self.mapper.file_mappings:
                print(f"  File-specific overrides active for {len(self.mapper.file_mappings)} files")
        
        print(f"{'='*95}\n")

    def save_index(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "index": self.index, 
                "indices": self.indices, 
                "mapper": self.mapper.to_dict(), 
                "source": (self.include_source_column, self.source_column_name),
                "fill": (self.default_fill_value, self.fill_values_by_type, self.fill_values_by_column)
            }, f)

    @classmethod
    def load_index(cls, path: str):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        
        inc_source, source_name = d.get("source", (False, "__source_file__"))
        
        return cls(
            d['index'], 
            d['indices'], 
            SchemaMapper.from_dict(d['mapper']), 
            include_source_column=inc_source,
            source_column_name=source_name,
            default_fill_value=d['fill'][0], 
            fill_values_by_type=d['fill'][1], 
            fill_values_by_column=d['fill'][2]
        )

class CollateHandler:
    """Picklable helper for batch collation.

    Stores only a pre-computed fill_map dict instead of a reference to the
    full dataset, so DataLoader workers (num_workers > 0) receive a minimal
    copy rather than the entire dataset index.
    """

    def __init__(self, fill_map: Dict[str, Any], on_none: str):
        self.fill_map = fill_map  # {column_name -> fill_value}
        self.on_none = on_none

    def __call__(self, batch):
        from torch.utils.data._utils.collate import default_collate

        clean_batch = batch
        if self.on_none in ('drop', 'fill'):
            new_batch = []
            for item in batch:
                has_none = any(v is None for v in item.values())
                if has_none:
                    if self.on_none == 'drop':
                        continue
                    else:  # fill
                        item = item.copy()
                        for k, v in item.items():
                            if v is None:
                                item[k] = self.fill_map.get(k)
                new_batch.append(item)
            clean_batch = new_batch

        if not clean_batch:
            return {}

        try:
            return default_collate(clean_batch)
        except TypeError as e:
            if 'NoneType' in str(e):
                for i, item in enumerate(batch):
                    for k, v in item.items():
                        if v is None:
                            raise TypeError(
                                f"Batch collation failed: column '{k}' contains None at batch index {i}.\n"
                                f"PyTorch DataLoader cannot handle None values.\n"
                                f"Fix: Set 'auto_fill=True' or provide 'fill_values' when initializing IndexedParquetDataset."
                            ) from None
            raise e

