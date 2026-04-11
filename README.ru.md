<p align="center">
  <img src="logo.png" width="400" alt="Indexed Parquet Dataset Logo">
</p>

<p align="center">
  <a href="https://pypi.org/project/indexed-parquet-dataset/"><img src="https://img.shields.io/pypi/v/indexed-parquet-dataset.svg" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  <a href="https://laeryid.github.io/indexed-parquet-dataset/"><img src="https://img.shields.io/badge/docs-Material%20MkDocs-blue.svg" alt="Documentation"></a>
</p>

# Indexed Parquet Dataset

**Indexed Parquet Dataset** — это высокопроизводительная библиотека на Python для **O(1) случайного доступа** к огромным наборам данных в формате Parquet. 

Она специально оптимизирована для Deep Learning (PyTorch), потребляет минимум памяти и поддерживает сложные функции, такие как **Schema Evolution** (работа с файлами разных схем в одном датасете).

## Основные фишки

- ⚡ **O(1) Random Access**: Мгновенный переход к любой строке в многогигабайтном датасете без сканирования файлов.
- 🔄 **Schema Evolution**: Работа с наборами данных, где файлы имеют разные схемы, отсутствующие колонки или переименованные поля.
- 📦 **Lazy Loading**: Файлы открываются только при запросе данных. Эффективный LRU-кэш дескрипторов.
- 🔥 **PyTorch Integration**: Нативная поддержка `torch.utils.data.Dataset`, включая генерацию адаптивного `collate_fn`.
- 🛠️ **Fluent API**: Цепочки вызовов: `shuffle` (глобальный или оптимизированный), `filter`, `alias`, `split`, `limit`, `rename`, `cast`, `map`.
- 💾 **Index Persistence**: Сохранение и быстрая загрузка индекса из файла.
- 🏗️ **Materialization**: "Запекание" всех трансформаций в новые Parquet файлы через `clone()`.

## Архитектура

Библиотека остается легковесной, храня в оперативной памяти только метаданные и карту строк:

```mermaid
graph TD
    subgraph RAM ["Приложение (RAM - Легковесно)"]
        direction TB
        subgraph DS ["IndexedParquetDataset"]
            Indices["Indices Array [np.ndarray]<br/>(Перемешанные/Отфильтрованные индексы)"]
            Meta["Metadata & Schema<br/>(Оффсеты файлов, маппинг колонок)"]
            Cache["File Handle Cache<br/>(Lazy Loading LRU)"]
        end
        
        User["User Code / PyTorch DataLoader"] -- "dataset[idx]" --> Indices
        Indices -- "Global Index" --> Meta
        Meta -- "Find File & Row Offset" --> Cache
    end
    
    subgraph Storage ["Хранилище (HDD/SSD/S3-over-FUSE)"]
        F1["data_part_1.parquet"]
        F2["data_part_2.parquet"]
        FN["data_part_N.parquet"]
    end
    
    Cache -- "Lazy Read" --> F1
    Cache -- "Lazy Read" --> F2
    Cache -- "Lazy Read" --> FN
    
    F1 -. "O(1) Row Retrieval" .-> User
    F2 -. "O(1) Row Retrieval" .-> User
    FN -. "O(1) Row Retrieval" .-> User
```

## Установка

Из PyPI:
```bash
pip install indexed-parquet-dataset
```

Для поддержки PyTorch:
```bash
pip install "indexed-parquet-dataset[torch]"
```

## Быстрый старт

### Базовая инициализация

```python
from indexed_parquet_dataset import IndexedParquetDataset

# Сканирует папку и строит глобальный индекс строк
ds = IndexedParquetDataset.from_folder("./path/to/data")

print(f"Всего строк: {len(ds)}")
print(f"Первая строка: {ds[0]}") # {'id': 1, 'text': '...', ...}

# Случайный доступ к любой строке мгновенный
sample = ds[999_999]
```

### Трансформации (Fluent API)

```python
ds = (IndexedParquetDataset.from_folder("./data")
      .filter(lambda x: x["score"] > 0.5)
      .shuffle(seed=42, rg_buffer=32) # Оптимизированное перемешивание для быстрого I/O
      .alias("text_len", lambda x: len(x["text"]))
      .limit(10000))

# Теперь у каждой строки есть виртуальная колонка 'text_len'
print(ds[0]["text_len"])
```

### Использование с PyTorch

```python
from torch.utils.data import DataLoader

ds = IndexedParquetDataset.from_folder("./data", auto_fill=True)
train_ds, val_ds = ds.train_test_split(test_size=0.1)

loader = DataLoader(
    train_ds, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4,
    collate_fn=ds.generate_collate_fn(on_none='fill')
)
```

## Документация

Полная документация доступна на [GitHub Pages](https://laeryid.github.io/indexed-parquet-dataset/).

## Лицензия

[Apache 2.0 License](LICENSE)
