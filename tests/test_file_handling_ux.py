import os
import pytest
import pandas as pd
from indexed_parquet_dataset import IndexedParquetDataset

@pytest.fixture
def test_data_dir(tmp_path):
    test_dir = tmp_path / "test_data_ux"
    test_dir.mkdir()
    df = pd.DataFrame({"id": range(10), "val": [f"val_{i}" for i in range(10)]})
    df.to_parquet(test_dir / "data.parquet")
    return test_dir

def test_from_folder_with_single_file(test_data_dir):
    """Verifies that from_folder correctly handles a single file path."""
    file_path = str(test_data_dir / "data.parquet")
    
    # Positive test: load single file
    dataset = IndexedParquetDataset.from_folder(file_path)
    assert len(dataset) == 10
    assert dataset[0]["id"] == 0
    assert len(dataset.index.files) == 1
    assert dataset.index.files[0].path == os.path.abspath(file_path)

def test_to_parquet_adds_extension(test_data_dir, tmp_path):
    """Verifies that to_parquet adds .parquet extension if missing."""
    dataset = IndexedParquetDataset.from_folder(str(test_data_dir))
    
    output_path = str(tmp_path / "my_new_dataset") # No extension
    dataset.to_parquet(output_path)
    
    # Check that file with extension exists
    expected_file = output_path + ".parquet"
    assert os.path.exists(expected_file)
    
    # Check that file without extension does NOT exist (as a file)
    assert not os.path.isfile(output_path)
    
    # Verify content
    loaded = IndexedParquetDataset.from_folder(expected_file)
    assert len(loaded) == 10

def test_clone_adds_extension(test_data_dir, tmp_path):
    """Verifies that clone adds .parquet extension and returns working dataset."""
    dataset = IndexedParquetDataset.from_folder(str(test_data_dir))
    
    clone_path = str(tmp_path / "cloned_ux")
    cloned_ds = dataset.clone(clone_path)
    
    expected_file = clone_path + ".parquet"
    assert os.path.exists(expected_file)
    assert len(cloned_ds) == 10
    assert cloned_ds.index.files[0].path == os.path.abspath(expected_file)

def test_no_double_extension(test_data_dir, tmp_path):
    """Verifies that we don't get .parquet.parquet."""
    dataset = IndexedParquetDataset.from_folder(str(test_data_dir))
    
    output_path = str(tmp_path / "already_has.parquet")
    dataset.to_parquet(output_path)
    
    assert os.path.exists(output_path)
    assert not os.path.exists(output_path + ".parquet")
