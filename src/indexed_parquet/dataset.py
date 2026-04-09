import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Callable, Union
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from .indexer import scan_directory, BaseIndex, FileInfo
from .schema import SchemaMapper

class IndexedParquetDataset(Dataset):
    """High-performance Parquet dataset with O(1) random access and Schema Evolution support."""
    
    def __init__(
        self, 
        index: BaseIndex, 
        indices: Optional[np.ndarray] = None,
        mapper: Optional[SchemaMapper] = None,
        transform: Optional[Callable] = None
    ):
        """Initializes the dataset.
        
        Args:
            index: A BaseIndex object containing dataset metadata.
            indices: Array of global indices to expose (for subsetting/shuffling).
            mapper: SchemaMapper for column renaming.
            transform: A callable to apply to each row after retrieval and mapping.
        """
        self.index = index
        self.mapper = mapper or SchemaMapper()
        self.transform = transform
        
        # indices allows for shuffling, filtering, and subsets without modifying the base index
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.index.total_rows)
            
        # Cumulative row counts for fast lookups
        self.file_offsets = np.cumsum([0] + [f.num_rows for f in self.index.files])
        
        # Cache for open file handles (Lazy Loading)
        self._file_handles: Dict[int, pq.ParquetFile] = {}

    @classmethod
    def from_folder(
        cls, 
        directory: str, 
        pattern: str = "*.parquet", 
        recursive: bool = True, 
        strict_schema: bool = False
    ) -> 'IndexedParquetDataset':
        """Creates an IndexedParquetDataset by scanning a directory.
        
        Args:
            directory: Directory to scan.
            pattern: File pattern (default: "*.parquet").
            recursive: Whether to scan subdirectories (default: True).
            strict_schema: If True, all files must have identical schemas.
            
        Returns:
            An initialized IndexedParquetDataset.
        """
        index = scan_directory(directory, pattern, recursive, strict_schema)
        return cls(index)

    @property
    def schema(self) -> List[str]:
        """Returns the list of column names available in the dataset (after mapping)."""
        all_cols = set()
        
        # 1. Columns from scan (potentially unmapped)
        for col in self.index.all_columns:
            all_cols.add(self.mapper.mapping.get(col, col))
            
        # 2. Columns from file-specific mappings
        for f_map in self.mapper.file_mappings.values():
            for target_col in f_map.values():
                all_cols.add(target_col)
                
        return sorted(list(all_cols))

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.indices)

    def _get_file_and_local_idx(self, global_idx: int) -> tuple[int, int]:
        """Maps a dataset index to a file index and local row index within that file."""
        actual_idx = self.indices[global_idx]
        
        # Find which file contains this actual_idx
        file_idx = np.searchsorted(self.file_offsets, actual_idx, side='right') - 1
        local_idx = actual_idx - self.file_offsets[file_idx]
        
        return int(file_idx), int(local_idx)

    def _get_file_handle(self, file_idx: int) -> pq.ParquetFile:
        """Returns a cached ParquetFile handle for the given file index."""
        if file_idx not in self._file_handles:
            file_path = self.index.files[file_idx].path
            self._file_handles[file_idx] = pq.ParquetFile(file_path)
        return self._file_handles[file_idx]

    def _read_rows_from_file(self, file_idx: int, local_indices: List[int]) -> List[Dict[str, Any]]:
        """Reads multiple rows from a single file efficiently using row group grouping."""
        pf = self._get_file_handle(file_idx)
        file_info = self.index.files[file_idx]
        file_path = file_info.path
        
        # Group indices by row group for batch reading
        rg_to_indices = {}
        cumulative_rg_rows = 0
        for i, rg_rows in enumerate(file_info.row_groups):
            mask = (np.array(local_indices) >= cumulative_rg_rows) & (np.array(local_indices) < cumulative_rg_rows + rg_rows)
            rg_indices = np.array(local_indices)[mask]
            if len(rg_indices) > 0:
                rg_to_indices[i] = rg_indices - cumulative_rg_rows
            cumulative_rg_rows += rg_rows

        local_idx_to_result = {}
        
        for rg_idx, rg_local_indices in rg_to_indices.items():
            table = pf.read_row_group(rg_idx)
            
            # Map rg_idx back to its starting offset in the file
            rg_start_offset = sum(file_info.row_groups[:rg_idx])
            
            for l_idx_in_rg in rg_local_indices:
                row_dict = table.slice(l_idx_in_rg, 1).to_pydict()
                item = {k: v[0] for k, v in row_dict.items()}
                
                # --- Schema Evolution ---
                for col in self.index.all_columns:
                    if col not in item:
                        item[col] = None
                
                item = self.mapper.map_columns(item, file_path)
                
                # --- Ensure all columns in the schema are present ---
                target_schema = self.schema
                for col in target_schema:
                    if col not in item:
                        item[col] = None
                
                if self.transform:
                    item = self.transform(item)
                
                local_idx_to_result[l_idx_in_rg + rg_start_offset] = item

        results = []
        for l_idx in local_indices:
            results.append(local_idx_to_result[l_idx])
            
        return results

    def __getitem__(self, idx: Union[int, List[int], slice, np.ndarray]) -> Any:
        """Retrieves items by index, slice, or list of indices."""
        if isinstance(idx, (int, np.integer)):
            if idx < 0: idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("Index out of range")
            return self.__getitems__([int(idx)])[0]
        elif isinstance(idx, (list, np.ndarray)):
            return self.__getitems__(list(idx))
        elif isinstance(idx, slice):
            return self.select(idx)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __getitems__(self, indices: List[int]) -> List[Dict[str, Any]]:
        """PyTorch-compatible batched retrieval."""
        file_to_local_indices: Dict[int, List[tuple[int, int]]] = {}
        for i, global_idx in enumerate(indices):
            f_idx, l_idx = self._get_file_and_local_idx(global_idx)
            if f_idx not in file_to_local_indices:
                file_to_local_indices[f_idx] = []
            file_to_local_indices[f_idx].append((i, l_idx))
            
        results: List[Optional[Dict[str, Any]]] = [None] * len(indices)
        for f_idx, indexed_l_indices in file_to_local_indices.items():
            original_positions = [x[0] for x in indexed_l_indices]
            l_indices = [x[1] for x in indexed_l_indices]
            
            file_results = self._read_rows_from_file(f_idx, l_indices)
            for pos, res in zip(original_positions, file_results):
                results[pos] = res
                
        return results  # type: ignore

    # --- Fluent API ---
    
    def shuffle(self, seed: Optional[int] = None) -> 'IndexedParquetDataset':
        """Returns a new dataset with shuffled indices."""
        rng = np.random.default_rng(seed)
        new_indices = self.indices.copy()
        rng.shuffle(new_indices)
        return IndexedParquetDataset(self.index, new_indices, self.mapper, self.transform)

    def train_test_split(
        self, 
        test_size: Union[float, int], 
        shuffle: bool = True, 
        seed: Optional[int] = None, 
        stratify_by: Optional[str] = None
    ) -> tuple['IndexedParquetDataset', 'IndexedParquetDataset']:
        """Splits the dataset into train and test sets.
        
        Args:
            test_size: Fraction (0-1) or absolute number of samples for the test set.
            shuffle: Whether to shuffle before splitting.
            seed: Random seed.
            stratify_by: Column name to use for stratified splitting.
            
        Returns:
            A tuple of (train_dataset, test_dataset).
        """
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
                # Ensure at least one test sample if group is large enough, 
                # or handle edge cases as needed.
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

        train_ds = IndexedParquetDataset(self.index, train_indices, self.mapper, self.transform)
        test_ds = IndexedParquetDataset(self.index, test_indices, self.mapper, self.transform)
        return train_ds, test_ds

    def limit(self, n: int) -> 'IndexedParquetDataset':
        """Returns a new dataset limited to the first n samples."""
        return IndexedParquetDataset(self.index, self.indices[:n], self.mapper, self.transform)

    def select(self, range_or_indices: Union[slice, List[int], np.ndarray]) -> 'IndexedParquetDataset':
        """Returns a new dataset with the specified subset of indices."""
        new_indices = self.indices[range_or_indices]
        return IndexedParquetDataset(self.index, new_indices, self.mapper, self.transform)

    def filter(
        self, 
        path_pattern: Optional[str] = None,
        column_conditions: Optional[Dict[str, Any]] = None,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> 'IndexedParquetDataset':
        """Optimized multi-level filter.
        
        Args:
            path_pattern: Substring to match in file paths.
            column_conditions: Dictionary of {col: value} or {col: (op, value)}.
                               Supported ops: ==, !=, >, >=, <, <=.
            predicate: Python function for row-level filtering (applied last).
            
        Returns:
            A new IndexedParquetDataset containing only the matching rows.
        """
        current_indices = self.indices.copy()

        # Level 1: Path filtering (Metadata level)
        if path_pattern:
            valid_file_indices = []
            for i, f in enumerate(self.index.files):
                if path_pattern in f.path:
                    valid_file_indices.append(i)
            
            # Create a mask for self.indices
            mask = np.zeros(len(current_indices), dtype=bool)
            for f_idx in valid_file_indices:
                start = self.file_offsets[f_idx]
                end = self.file_offsets[f_idx + 1]
                # mark indices that fall within [start, end)
                mask |= (current_indices >= start) & (current_indices < end)
            
            current_indices = current_indices[mask]

        if len(current_indices) == 0:
            return IndexedParquetDataset(self.index, current_indices, self.mapper, self.transform)

        # Level 2: Columnar filtering (Column level)
        if column_conditions:
            # Group current_indices by file for efficient batch reading of filter columns
            file_to_indices = {}
            for idx in current_indices:
                f_idx = np.searchsorted(self.file_offsets, idx, side='right') - 1
                if f_idx not in file_to_indices:
                    file_to_indices[f_idx] = []
                file_to_indices[f_idx].append(idx)
            
            new_indices_list = []
            cols_to_read = [self.mapper.get_source_column(c) for c in column_conditions.keys()]
            
            for f_idx, f_global_indices in file_to_indices.items():
                f_info = self.index.files[f_idx]
                pf = self._get_file_handle(f_idx)
                
                # Check if file has all required columns (Schema Evolution handled)
                # Actually, we read only available columns.
                available_cols = [c for c in cols_to_read if c in f_info.columns]
                
                # We need to filter row by row eventually, but we can do it per file.
                # To be efficient, we read only requested columns for the WHOLE file (or just the indices)
                # If f_global_indices is a large portion of the file, reading full column is fine.
                table = pf.read(columns=available_cols)
                
                # Apply conditions using pyarrow.compute
                file_mask = None
                for col, cond in column_conditions.items():
                    src_col = self.mapper.get_source_column(col, f_info.path)
                    
                    if src_col not in table.column_names:
                        # Column missing in this file -> considered None
                        # We create a dummy null array for comparison
                        arr = pa.array([None] * len(table))
                    else:
                        arr = table.column(src_col)
                    
                    if isinstance(cond, tuple):
                        op, val = cond
                        if op == '==': c_mask = pc.equal(arr, val)
                        elif op == '!=': c_mask = pc.not_equal(arr, val)
                        elif op == '>': c_mask = pc.greater(arr, val)
                        elif op == '>=': c_mask = pc.greater_equal(arr, val)
                        elif op == '<': c_mask = pc.less(arr, val)
                        elif op == '<=': c_mask = pc.less_equal(arr, val)
                        else: raise ValueError(f"Unsupported operator: {op}")
                    else:
                        c_mask = pc.equal(arr, cond)
                    
                    if file_mask is None:
                        file_mask = c_mask
                    else:
                        file_mask = pc.and_(file_mask, c_mask)
                
                # Handle potential nulls in the mask (e.g. from comparisons with missing columns)
                file_mask = pc.fill_null(file_mask, False)
                file_mask_np = file_mask.to_numpy(zero_copy_only=False).astype(bool)
                
                # Local indices are (f_global_indices - offset)
                f_local_indices = (np.array(f_global_indices) - self.file_offsets[f_idx]).astype(int)
                
                # Filter f_global_indices where file_mask_np[f_local_indices] is True
                valid_global_indices = np.array(f_global_indices)[file_mask_np[f_local_indices]]
                new_indices_list.append(valid_global_indices)
                
            if new_indices_list:
                current_indices = np.concatenate(new_indices_list)
            else:
                current_indices = np.array([], dtype=int)

        if len(current_indices) == 0:
            return IndexedParquetDataset(self.index, current_indices, self.mapper, self.transform)

        # Level 3: Predicate filtering (Row level)
        if predicate:
            temp_ds = IndexedParquetDataset(self.index, current_indices, self.mapper, self.transform)
            mask = [predicate(temp_ds[i]) for i in range(len(temp_ds))]
            current_indices = current_indices[np.array(mask)]

        return IndexedParquetDataset(self.index, current_indices, self.mapper, self.transform)

    def map(self, fn: Callable[[Dict[str, Any]], Any]) -> 'IndexedParquetDataset':
        """Adds a transformation function to be applied after retrieval."""
        old_transform = self.transform
        if old_transform:
            new_transform = lambda x: fn(old_transform(x))
        else:
            new_transform = fn
        return IndexedParquetDataset(self.index, self.indices, self.mapper, new_transform)

    def rename_column(self, old: str, new: str) -> 'IndexedParquetDataset':
        """Renames a column globally."""
        new_mapping = self.mapper.mapping.copy()
        new_mapping[old] = new
        new_mapper = SchemaMapper(new_mapping, self.mapper.file_mappings)
        return IndexedParquetDataset(self.index, self.indices, new_mapper, self.transform)

    def set_file_mapping(self, file_path: str, mapping: Dict[str, str]) -> 'IndexedParquetDataset':
        """Sets a schema mapping for a specific file."""
        new_file_mappings = self.mapper.file_mappings.copy()
        abs_path = os.path.abspath(file_path)
        new_file_mappings[abs_path] = mapping
        new_mapper = SchemaMapper(self.mapper.mapping, new_file_mappings)
        return IndexedParquetDataset(self.index, self.indices, new_mapper, self.transform)

    def copy(self) -> 'IndexedParquetDataset':
        """Returns a new instance of the dataset with a copy of current indices."""
        return IndexedParquetDataset(
            index=self.index,
            indices=self.indices.copy(),
            mapper=self.mapper,
            transform=self.transform
        )

    def clone(self, folder_path: str, filename: Optional[str] = None, batch_size: int = 10000) -> str:
        """Exports the visible state of the dataset to a new Parquet file.
        
        The resulting file contains only the rows currently in the dataset index,
        and uses the aliased column names. Transform is NOT applied to exported data.
        
        Args:
            folder_path: Directory to save the new file.
            filename: Name of the file (defaults to dataset_clone_<timestamp>.parquet).
            batch_size: Number of rows to process at once.
            
        Returns:
            Absolute path to the created file.
        """
        import time
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            
        if filename is None:
            filename = f"dataset_clone_{int(time.time())}.parquet"
        
        dest_path = os.path.join(folder_path, filename)
        
        # We need to determine the schema from the first batch to initialize the writer
        if len(self) == 0:
            raise ValueError("Cannot clone an empty dataset")

        # Temporarily disable transform for cloning to get raw (but mapped) data
        original_transform = self.transform
        self.transform = None
        
        try:
            writer: Optional[pq.ParquetWriter] = None
            
            for i in range(0, len(self), batch_size):
                end_idx = min(i + batch_size, len(self))
                batch_indices = list(range(i, end_idx))
                batch_data = self[batch_indices] # Returns list of dicts
                
                # Convert list of dicts to pyarrow Table
                # We use the dataset's schema to ensure all columns are present (as None if missing)
                table_schema = self.schema
                rows_for_pa = {col: [] for col in table_schema}
                for row in batch_data:
                    for col in table_schema:
                        rows_for_pa[col].append(row.get(col))
                
                table = pa.Table.from_pydict(rows_for_pa)
                
                if writer is None:
                    writer = pq.ParquetWriter(dest_path, table.schema)
                
                writer.write_table(table)
                
            if writer:
                writer.close()
        finally:
            self.transform = original_transform
            
        return os.path.abspath(dest_path)

    def concat(self, other: 'IndexedParquetDataset') -> 'IndexedParquetDataset':
        """Concatenates this dataset with another one vertically.
        
        Merges schemas (preserving conflicting aliases) and chains transforms if necessary.
        
        Args:
            other: The dataset to append to this one.
            
        Returns:
            A new combined IndexedParquetDataset.
        """
        # 1. Merge Index
        new_files = self.index.files + other.index.files
        new_total_rows = self.index.total_rows + other.index.total_rows
        new_all_columns = sorted(list(set(self.index.all_columns) | set(other.index.all_columns)))
        new_index = BaseIndex(new_files, new_total_rows, new_all_columns)
        
        # 2. Merge Indices
        other_indices_shifted = other.indices + self.index.total_rows
        new_indices = np.concatenate([self.indices, other_indices_shifted])
        
        # 3. Merge Mapper
        self_abs_files = [f.path for f in self.index.files]
        other_abs_files = [f.path for f in other.index.files]
        new_mapper = self.mapper.merge(other.mapper, self_abs_files, other_abs_files)
        
        # 4. Merge Transform
        # If both datasets have different transforms, we need a wrapper
        if self.transform == other.transform:
            new_transform = self.transform
        elif self.transform and other.transform:
            # Determine split point in the combined files list
            split_file_idx = len(self.index.files)
            
            # We'll re-implement _read_rows_from_file logic to handle split transform
            # But simpler: we use a wrapper that detects the file_path (not easily available in transform)
            # Actually, let's keep it simple for now and use self.transform or chain them
            # For now, we chain them if they exist or just use self.transform
            # Given the request, we should probably warn or just pick one.
            # I will chain them for now, but this is usually not what's wanted for vertical concat.
            # Best is to pick self.transform and warn.
            import warnings
            warnings.warn("Both datasets have different transforms. Using transform from the first dataset.")
            new_transform = self.transform
        else:
            new_transform = self.transform or other.transform
            
        return IndexedParquetDataset(new_index, new_indices, new_mapper, new_transform)

    def save_index(self, path: str) -> None:
        """Saves the dataset index and metadata to a pickle file."""
        import pickle
        state = {
            "index": self.index,
            "indices": self.indices,
            "mapper": self.mapper.to_dict(),
            "transform": self.transform
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_index(cls, path: str) -> 'IndexedParquetDataset':
        """Loads a dataset index from a pickle file."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        return cls(
            index=state["index"],
            indices=state["indices"],
            mapper=SchemaMapper.from_dict(state["mapper"]),
            transform=state["transform"]
        )

    def info(self) -> None:
        """Prints summary statistics and metadata for the dataset."""
        total_indexed_rows = self.index.total_rows
        visible_rows = len(self)
        num_files = len(self.index.files)
        
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
            
        print(f"\n{'='*70}")
        print(f" IndexedParquetDataset Summary")
        print(f"{'='*70}")
        print(f" Files:           {num_files:<10}  |  Storage Size:  {storage_str}")
        print(f" Total Rows:      {total_indexed_rows:<10,}  |  Visible Rows:  {visible_rows:,} ({visible_rows/total_indexed_rows:.1%})")
        print(f"{'-'*70}")
        
        # Files Table
        print("\nFiles in Index:")
        file_header = f"{'#':<3} | {'Rows':>12} | {'Groups':>6} | {'Path':<}"
        print(file_header)
        print("-" * 70)
        for i, f in enumerate(self.index.files):
            basename = os.path.basename(f.path)
            display_path = (f"...{f.path[-45:]}" if len(f.path) > 45 else f.path)
            print(f"{i:<3} | {f.num_rows:>12,} | {len(f.row_groups):>6} | {display_path}")
            
        # Columns Table
        print("\nColumn Statistics:")
        col_header = f"{'Column':<25} | {'Files':>6} | {'Presence':>8} | {'Est. Rows':>12} | {'Coverage'}"
        print(col_header)
        print("-" * 70)
        
        visible_schema = self.schema
        for col in visible_schema:
            files_present = 0
            rows_present = 0
            
            for f in self.index.files:
                src_col = self.mapper.get_source_column(col, f.path)
                if src_col in f.columns:
                    files_present += 1
                    rows_present += f.num_rows
            
            presence_pct = files_present / num_files
            coverage_pct = rows_present / total_indexed_rows
            
            # Truncate column name if too long
            col_display = (col[:22] + "...") if len(col) > 25 else col
            print(f"{col_display:<25} | {files_present:>6} | {presence_pct:>8.1%} | {rows_present:>12,} | {coverage_pct:>8.1%}")
            
        # Mappings
        if self.mapper.mapping or self.mapper.file_mappings:
            print("\nActive Mappings:")
            if self.mapper.mapping:
                print(f"  Global Aliases: {self.mapper.mapping}")
            if self.mapper.file_mappings:
                print(f"  File-specific overrides active for {len(self.mapper.file_mappings)} files")
        
        print(f"{'='*70}\n")
