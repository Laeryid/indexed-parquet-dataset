# Indexed Parquet Dataset

[![PyPI version](https://img.shields.io/pypi/v/indexed-parquet-dataset.svg)](https://pypi.org/project/indexed-parquet-dataset/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Indexed Parquet Dataset** — это высокопроизводительная библиотека для быстрого случайного доступа к данным в формате Parquet. 

## Почему стоит использовать?

Стандартные библиотеки (pandas, pyarrow) работают отлично при чтении файла целиком, но они **не предназначены для эффективного случайного доступа по индексу строки**, особенно когда данные распределены по сотням Parquet файлов.

`indexed-parquet-dataset` решает эту проблему, предлагая:

1.  **Настоящий O(1) доступ**: Мы строим легкий индекс один раз и мгновенно переходим к нужной строке в любом файле.
2.  **Экономию памяти**: Вы не загружаете датасет целиком. Открывается только нужный кусок файла в момент обращения.
3.  **Гибкость схем**: Ваши файлы могут иметь разные колонки или их типы — библиотека нормализует всё "на лету".

## Сравнение

| Фича | Pandas / PyArrow | HF Datasets | IterableDataset | **Indexed Parquet** |
| :--- | :---: | :---: | :---: | :---: |
| Случайный доступ | ❌ Медленно/RAM | ✅ Хорошо | ❌ Нет | ✅ **O(1) / RAM-lite** |
| Чтение по сети | ✅ Да | ✅ Да | ✅ Да | ⚠️ Только через FUSE |
| Schema Evolution | ❌ Сложно | ⚠️ Частично | ⚠️ Сложно | ✅ **Нативно** |
| Lazy Loading | ❌ Нет | ✅ Да | ✅ Да | ✅ **Да (LRU кэш)** |

## Быстрый пример

```python
from indexed_parquet_dataset import IndexedParquetDataset

# Создаем датасет из папки
ds = IndexedParquetDataset.from_folder("path/to/data")

# Обращаемся как к обычному списку
row = ds[12345]  
print(row) # {'column_a': 10.5, 'column_b': 'text', ...}

# Прямая интеграция с PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=32, shuffle=True)
```

## Основные разделы

- [Быстрый старт](tutorials/quickstart.md) — освойте основы за 5 минут.
- [Эволюция схем](how-to/schema-evolution.md) — как работать с "грязными" данными.
- [Deep Learning Pipeline](tutorials/deep_learning.md) — лучший способ обучать модели.
- [Архитектура](explanation/architecture.md) — как это работает под капотом.
