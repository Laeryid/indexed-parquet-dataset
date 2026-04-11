from __future__ import annotations
import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
import pyarrow.parquet as pq

@dataclass
class FileInfo:
    """Metadata about a single Parquet file.
    
    Attributes:
        path: Absolute path to the file.
        num_rows: Total number of rows in the file.
        row_groups: List containing the number of rows in each row group.
        columns: List of column names present in this file.
    """
    path: str
    num_rows: int
    row_groups: List[int]
    columns: List[str]

@dataclass
class BaseIndex:
    """Metadata about the entire dataset.
    
    Attributes:
        files: List of FileInfo objects for all files in the dataset.
        total_rows: Combined number of rows across all files.
        all_columns: Sorted list of all unique columns found across all files.
        column_types: Dict mapping column names to their PyArrow type as a string.
    """
    files: List[FileInfo]
    total_rows: int
    all_columns: List[str]
    column_types: Dict[str, str]

def scan_directory(
    directory: str, 
    pattern: str = "*.parquet", 
    recursive: bool = True, 
    strict_schema: bool = False
) -> BaseIndex:
    """Scans a directory for Parquet files and extracts metadata for indexing.
    
    Args:
        directory: Path to the directory to scan.
        pattern: Glob pattern to match files (default: "*.parquet").
        recursive: Whether to search subdirectories recursively (default: True).
        strict_schema: If True, raises ValueError if files have different schemas.
        
    Returns:
        A BaseIndex object containing dataset metadata.
        
    Raises:
        FileNotFoundError: If no files matching the pattern are found.
        ValueError: If strict_schema is True and a file schema doesn't match the first file.
    """
    if os.path.isfile(directory):
        file_paths = [directory]
    else:
        search_path = os.path.join(directory, "**", pattern) if recursive else os.path.join(directory, pattern)
        file_paths = glob.glob(search_path, recursive=recursive)
    
    if not file_paths:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in '{directory}'")

    files_info: List[FileInfo] = []
    total_rows: int = 0
    all_columns_set: Set[str] = set()
    column_types: Dict[str, str] = {}
    first_schema: Optional[Set[str]] = None
    
    for path in sorted(file_paths):
        metadata = pq.read_metadata(path)
        num_rows = metadata.num_rows
        row_groups = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
        
        # Access PyArrow schema to get clean type names
        pa_schema = metadata.schema.to_arrow_schema()
        file_columns = pa_schema.names
        
        if first_schema is None:
            first_schema = set(file_columns)
        elif strict_schema and set(file_columns) != first_schema:
            raise ValueError(f"Schema mismatch in file {path} while strict_schema=True")
            
        for i, col in enumerate(file_columns):
            all_columns_set.add(col)
            # Store type name if not already seen
            if col not in column_types:
                column_types[col] = str(pa_schema.field(i).type)
            
        files_info.append(FileInfo(
            path=os.path.abspath(path),
            num_rows=num_rows,
            row_groups=row_groups,
            columns=file_columns
        ))
        total_rows += num_rows
        
    return BaseIndex(
        files=files_info,
        total_rows=total_rows,
        all_columns=sorted(list(all_columns_set)),
        column_types=column_types
    )
