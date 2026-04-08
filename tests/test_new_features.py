import os
import pytest
import pandas as pd
import numpy as np
from indexed_parquet import IndexedParquetDataset, scan_directory

@pytest.fixture
def evolution_data_dir(tmp_path):
    test_dir = tmp_path / "evolution_data"
    test_dir.mkdir()
    
    # File 1: Columns [id, val]
    df1 = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
    df1.to_parquet(test_dir / "file1.parquet")
    
    # File 2: Columns [id, score]
    df2 = pd.DataFrame({"id": [3, 4], "score": [0.5, 0.8]})
    df2.to_parquet(test_dir / "file2.parquet")
    
    return str(test_dir)

def test_schema_evolution(evolution_data_dir):
    dataset = IndexedParquetDataset.from_folder(evolution_data_dir)
    assert len(dataset) == 4
    
    # Check schema
    assert set(dataset.schema) == {"id", "val", "score"}
    
    # Rows from file 1 should have score=None
    row1 = dataset[0]
    assert row1["id"] == 1
    assert row1["val"] == "a"
    assert row1["score"] is None
    
    # Rows from file 2 should have val=None
    row3 = dataset[2]
    assert row3["id"] == 3
    assert row3["score"] == 0.5
    assert row3["val"] is None

def test_strict_schema(evolution_data_dir):
    with pytest.raises(ValueError, match="Schema mismatch"):
        IndexedParquetDataset.from_folder(evolution_data_dir, strict_schema=True)

def test_stratified_split(tmp_path):
    test_dir = tmp_path / "strat_data"
    test_dir.mkdir()
    
    # Create data with 100 rows, labels 0 and 1 in 80/20 proportion
    labels = [0] * 80 + [1] * 20
    df = pd.DataFrame({"id": range(100), "label": labels})
    df.to_parquet(test_dir / "data.parquet")
    
    dataset = IndexedParquetDataset.from_folder(str(test_dir))
    train, test = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42, stratify_by="label")
    
    assert len(train) == 80
    assert len(test) == 20
    
    # Check proportions in test set
    test_labels = [test[i]["label"] for i in range(len(test))]
    unique, counts = np.unique(test_labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    
    # Should have roughly 16 zeros and 4 ones (80% of 20 and 20% of 20)
    assert label_counts[0] == 16
    assert label_counts[1] == 4

def test_file_mapping(evolution_data_dir):
    dataset = IndexedParquetDataset.from_folder(evolution_data_dir)
    
    # File 2 has "score". Let's map it to "quality" for that file
    file2_path = os.path.abspath(os.path.join(evolution_data_dir, "file2.parquet"))
    mapped_ds = dataset.set_file_mapping(file2_path, {"score": "quality"})
    
    # Row from file 1 should be unchanged
    assert "val" in mapped_ds[0]
    
    # Row from file 2 should have "quality" instead of "score"
    row3 = mapped_ds[2]
    assert "quality" in row3
    assert row3["quality"] == 0.5
    assert "score" not in row3

def test_filter_optimized_path(evolution_data_dir):
    dataset = IndexedParquetDataset.from_folder(evolution_data_dir)
    
    # Filter only file1
    filtered = dataset.filter(path_pattern="file1.parquet")
    assert len(filtered) == 2
    for i in range(len(filtered)):
        assert "val" in filtered[i]
        assert filtered[i]["val"] is not None
        assert filtered[i]["score"] is None

def test_filter_optimized_column(evolution_data_dir):
    dataset = IndexedParquetDataset.from_folder(evolution_data_dir)
    
    # Equality
    filtered = dataset.filter(column_conditions={"id": 3})
    assert len(filtered) == 1
    assert filtered[0]["id"] == 3
    
    # Inequality
    filtered = dataset.filter(column_conditions={"id": (">", 2)})
    assert len(filtered) == 2 # 3 and 4
    for i in range(len(filtered)):
        assert filtered[i]["id"] > 2
        
    # AND logic
    filtered = dataset.filter(column_conditions={"id": (">", 1), "score": 0.5})
    assert len(filtered) == 1
    assert filtered[0]["id"] == 3

def test_filter_optimized_combined(evolution_data_dir):
    dataset = IndexedParquetDataset.from_folder(evolution_data_dir)
    
    # Path + Column + Predicate
    filtered = dataset.filter(
        path_pattern="file2.parquet",
        column_conditions={"id": (">=", 3)},
        predicate=lambda x: x["score"] > 0.6
    )
    # File 2 has IDs 3 and 4. ID 4 has score 0.8 (> 0.6).
    assert len(filtered) == 1
    assert filtered[0]["id"] == 4
