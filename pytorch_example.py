import os
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
import sys
import shutil

# Add src to path
sys.path.append(os.path.abspath("src"))
from indexed_parquet_dataset import IndexedParquetDataset

def setup_dummy_data(path="./data"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    # Create a table with some missing values
    table = pa.Table.from_pylist([
        {"id": i, "features": float(i * 1.1)} if i % 5 != 0 else {"id": None, "features": None}
        for i in range(100)
    ])
    pq.write_table(table, os.path.join(path, "train.parquet"))
    print(f"Created dummy data in {path}")

def main():
    setup_dummy_data("./data")
    
    # Инициализируем датасет (точно как в запросе)
    ds = IndexedParquetDataset.from_folder("./data",
        default_fill_value="-",         # Все пропуски станут "-"
        fill_values_by_type={"int64": 0, "double": 0.0}, # Числа станут 0
    )
    
    # Создаем DataLoader
    loader = DataLoader(
        ds, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4,  # Для параллельного чтения (теперь работает на Windows!)
        collate_fn=ds.generate_collate_fn(on_none='fill') # Авто-заполнение None из конфига ds
    )
    
    print("Simulating training loop...")
    # Имитируем цикл тренировки
    for batch in loader:
        # batch — это словарь, где значения — тензоры PyTorch
        ids = batch['id']
        features = batch['features']
        print(f"Loaded batch size: {len(ids)}")
        print(f"Example ID: {ids[0].item()}")
        print(f"Example Features: {features[0].item()}")
        break
    
    print("\nVerification: SUCCESS")

if __name__ == "__main__":
    main()
