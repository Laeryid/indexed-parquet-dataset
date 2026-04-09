# План разработки библиотеки indexed-parquet-dataset

Создание Python-библиотеки для высокопроизводительного доступа к Parquet-файлам с поддержкой PyTorch Dataset API и цепочечных вызовов.

## Proposed Changes

### 1. Архитектура и структура проекта [NEW]

Создание стандартной структуры для Python-пакета:

```text
indexed-parquet-dataset/
├── pyproject.toml
├── LICENSE (Apache 2.0)
├── README.md
├── src/
│   └── indexed_parquet/
│       ├── __init__.py
│       ├── dataset.py      # Основной класс IndexedParquetDataset
│       ├── indexer.py      # Логика сканирования и построения индекса
│       └── schema.py       # Логика маппинга столбцов и схем
└── tests/
    └── test_dataset.py
```

### 2. Ядро библиотеки: `IndexedParquetDataset`

#### [NEW] [dataset.py](file:///c:/Experiments/indexed-parquet-dataset/src/indexed_parquet/dataset.py)
- Наследование от `torch.utils.data.Dataset`.
- Поддержка методов `__len__` и `__getitem__`.
- Хранение индекса в виде компактного массива (например, списка кортежей или структурированного `numpy` массива для экономии RAM).
- Реализация **Fluent Interface**:
  - `shuffle()`: перемешивание внутреннего массива индексов.
  - `filter(fn)`: создание нового объекта с отфильтрованными индексами.
  - `concat(other)`: объединение двух датасетов.
  - `select_columns(mapping)`: настройка алиасов для вывода.
  - `limit(n)`: ограничение выборки.

#### [NEW] [indexer.py](file:///c:/Experiments/indexed-parquet-dataset/src/indexed_parquet/indexer.py)
- Функция `scan_directory(path, pattern)`:
  - Использование `pyarrow.parquet.ParquetFile` для извлечения метаданных без чтения данных.
  - Сбор информации о Row Groups и количестве строк в них.
  - Формирование глобальной таблицы соответствия: `global_idx -> (file_id, row_group, row_in_group)`.

#### [NEW] [schema.py](file:///c:/Experiments/indexed-parquet-dataset/src/indexed_parquet/schema.py)
- Обработка `column_mapping`:
  - Возможность задать `dict` вида `{file_alias: {target_name: source_name}}`.
  - Обработка отсутствующих столбцов (заполнение `None` или дефолтными значениями).

### 3. Упаковка и PIP [NEW]

#### [NEW] [pyproject.toml](file:///c:/Experiments/indexed-parquet-dataset/pyproject.toml)
- Настройка метаданных (name, version, authors, license).
- Зависимости: `pyarrow`, `pandas`, `torch`, `numpy`.
- Описание путей к исходному коду.

## Расширенный функционал (Fluent API)

Пример использования, который должен стать возможным:
```python
dataset = (IndexedParquetDataset.from_folder("./data")
           .set_mapping({"old_file.parquet": {"text": "content"}})
           .filter(lambda x: x["lang"] == "ru")
           .shuffle()
           .limit(10000))
```

## Verification Plan

### Automated Tests
1.  **Unit-тесты**:
    - Проверка корректности индексации (сравнение с прямым чтением через pandas).
    - Тестирование `concat` на файлах с разными схемами.
    - Тестирование `shuffle` (проверка, что порядок меняется, а данные остаются верными).
    - Тестирование `filter` по метаданным и по контенту.
2.  **Performance-тесты**:
    - Замер времени доступа к случайной строке (O(1) verification).
    - Замер потребления RAM при индексации 1 млн+ строк.

### Manual Verification
- Создание тестового набора из 3-4 Parquet файлов с разными колонками.
- Проверка работы цепочки вызовов в интерактивном режиме.
- Попытка установки через `pip install -e .`.
