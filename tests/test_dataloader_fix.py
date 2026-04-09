import os
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
import sys
import unittest

# Add src to path
sys.path.append(os.path.abspath("src"))
from indexed_parquet import IndexedParquetDataset

class TestDataLoaderFix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("test_data_fix", exist_ok=True)
        # Create a table with nulls
        table = pa.Table.from_pylist([
            {"id": 1, "val": 10.5, "cat": "A"},
            {"id": 2, "val": None, "cat": None},
            {"id": 3, "val": 30.5, "cat": "C"},
        ])
        pq.write_table(table, "test_data_fix/data.parquet")

    def test_auto_fill(self):
        """[POSITIVE TEST] Verifies that auto_fill=True correctly fills missing values."""
        print("\nChecking auto_fill=True...")
        ds = IndexedParquetDataset.from_folder("test_data_fix", auto_fill=True)
        loader = DataLoader(ds, batch_size=3)
        batch = next(iter(loader))
        
        # val should be filled with 0.0
        self.assertEqual(batch['val'][1].item(), 0.0)
        # cat should be filled with ""
        self.assertEqual(batch['cat'][1], "")
        print("auto_fill=True: OK")

    def test_collate_raise(self):
        """[NEGATIVE TEST] Verifies that 'on_none=raise' correctly raises TypeError on Nones."""
        print("\nChecking collate on_none='raise'...")
        ds = IndexedParquetDataset.from_folder("test_data_fix", auto_fill=False)
        collate_fn = ds.generate_collate_fn(on_none='raise')
        loader = DataLoader(ds, batch_size=3, collate_fn=collate_fn)
        
        with self.assertRaisesRegex(TypeError, "contains None at batch index 1"):
            next(iter(loader))
        print("collate on_none='raise': OK")

    def test_collate_drop(self):
        """[POSITIVE TEST] Verifies that 'on_none=drop' correctly filters out None items."""
        print("\nChecking collate on_none='drop'...")
        ds = IndexedParquetDataset.from_folder("test_data_fix", auto_fill=False)
        collate_fn = ds.generate_collate_fn(on_none='drop')
        loader = DataLoader(ds, batch_size=3, collate_fn=collate_fn)
        
        batch = next(iter(loader))
        # Original size was 3, but 1 row had None, so size should be 2
        self.assertEqual(len(batch['id']), 2)
        self.assertEqual(batch['id'].tolist(), [1, 3])
        print("collate on_none='drop': OK")

    def test_collate_fill(self):
        """[POSITIVE TEST] Verifies that 'on_none=fill' correctly replaces Nones with fill values."""
        print("\nChecking collate on_none='fill'...")
        ds = IndexedParquetDataset.from_folder("test_data_fix", auto_fill=False)
        collate_fn = ds.generate_collate_fn(on_none='fill')
        # We need to ensure we have defaults for filling
        ds._apply_auto_fill() 
        
        loader = DataLoader(ds, batch_size=3, collate_fn=collate_fn)
        batch = next(iter(loader))
        self.assertEqual(len(batch['id']), 3)
        self.assertEqual(batch['val'][1].item(), 0.0)
        print("collate on_none='fill': OK")

if __name__ == "__main__":
    unittest.main()
