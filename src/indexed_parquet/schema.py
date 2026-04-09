from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable
import json

class SchemaMapper:
    """Handles column mapping and aliasing for the dataset."""
    
    def __init__(
        self, 
        mapping: Optional[Dict[str, str]] = None, 
        file_mappings: Optional[Dict[str, Dict[str, str]]] = None,
        transforms: Optional[Dict[str, Callable]] = None,
        row_transforms: Optional[List[Callable[[dict], dict]]] = None
    ):
        """Initializes the SchemaMapper.
        
        Args:
            mapping: Global mapping (original name -> target name).
            file_mappings: File-specific mappings (file path -> {original -> target}).
            transforms: Global transformations (target name -> function(row)).
            row_transforms: Row-level transformations (list of functions function(row) -> row).
        """
        self.mapping = mapping if mapping is not None else {}
        self.file_mappings = file_mappings if file_mappings is not None else {}
        self.transforms = transforms if transforms is not None else {}
        self.row_transforms = row_transforms if row_transforms is not None else []
        self._rebuild_reverse_mapping()

    def _rebuild_reverse_mapping(self) -> None:
        """Rebuilds the reverse mapping for fast lookups."""
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def map_columns(self, data: Dict[str, Any], file_path: Optional[str] = None) -> Dict[str, Any]:
        """Renames columns in the input data according to global and file-specific mappings.
        
        Args:
            data: Raw dictionary of column values.
            file_path: Optional path to the file from which data was read.
            
        Returns:
            A dictionary with mapped column names.
        """
        # Apply file-specific mapping first if available
        current_data = data
        if file_path and file_path in self.file_mappings:
            f_map = self.file_mappings[file_path]
            current_data = {f_map.get(k, k): v for k, v in current_data.items()}
            
        if not self.mapping:
            mapped_data = current_data.copy()
        else:
            mapped_data = {}
            for col, val in current_data.items():
                target_name = self.mapping.get(col, col)
                mapped_data[target_name] = val
            
        # Apply transforms (computed columns)
        if not self.transforms:
            return mapped_data
            
        for target_name, transform in self.transforms.items():
            try:
                mapped_data[target_name] = transform(mapped_data)
            except Exception:
                pass
                
        return mapped_data

    def get_source_column(self, target_column: str, file_path: Optional[str] = None) -> str:
        """Returns the original column name for a given target name.
        
        Args:
            target_column: The mapped name of the column.
            file_path: Optional path to the file.
            
        Returns:
            The original column name.
        """
        return self.reverse_mapping.get(target_column, target_column)

    def select_source_columns(self, target_columns: List[str]) -> List[str]:
        """Returns a list of original column names required for the requested target columns."""
        return [self.get_source_column(col) for col in target_columns]

    def to_dict(self) -> Dict[str, Any]:
        """Converts the mapper state to a dictionary."""
        return {
            "mapping": self.mapping,
            "file_mappings": self.file_mappings,
            "transforms": self.transforms,
            "row_transforms": self.row_transforms
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaMapper':
        """Creates a SchemaMapper from a dictionary."""
        return cls(
            mapping=data.get("mapping"),
            file_mappings=data.get("file_mappings"),
            transforms=data.get("transforms"),
            row_transforms=data.get("row_transforms")
        )

    def merge(self, other: 'SchemaMapper', self_files: List[str], other_files: List[str]) -> 'SchemaMapper':
        """Merges this mapper with another one, preserving conflicting aliases via file-specific mappings.
        
        Args:
            other: The other SchemaMapper to merge.
            self_files: List of absolute file paths belonging to the current dataset.
            other_files: List of absolute file paths belonging to the other dataset.
            
        Returns:
            A new merged SchemaMapper.
        """
        new_file_mappings = self.file_mappings.copy()
        new_file_mappings.update(other.file_mappings)
        
        new_global_mapping = self.mapping.copy()
        
        for src_col, target_col in other.mapping.items():
            if src_col in new_global_mapping:
                if new_global_mapping[src_col] != target_col:
                    # Conflict: same source column, different targets.
                    # We preserve the alias for 'other' files by moving it to file_mappings.
                    for f_path in other_files:
                        if f_path not in new_file_mappings:
                            new_file_mappings[f_path] = {}
                        # Only set if not already present in file_mappings
                        if src_col not in new_file_mappings[f_path]:
                            new_file_mappings[f_path][src_col] = target_col
                # If the targets match, no conflict at global level.
            else:
                # No conflict with current global mapping, can add safely.
                new_global_mapping[src_col] = target_col
                
        new_transforms = self.transforms.copy()
        new_transforms.update(other.transforms)
        
        new_row_transforms = self.row_transforms + other.row_transforms
        
        return SchemaMapper(new_global_mapping, new_file_mappings, new_transforms, new_row_transforms)

    def __repr__(self) -> str:
        return f"SchemaMapper(mapping={self.mapping}, file_mappings={self.file_mappings})"
