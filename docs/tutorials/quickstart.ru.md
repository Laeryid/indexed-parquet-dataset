# Быстрый старт

Этот туториал поможет вам за 5 минут освоить базовые возможности `indexed-parquet-dataset`.

## Шаг 1: Подготовка данных

Допустим, у вас есть папка с несколькими Parquet файлами:

```text
data/
  part1.parquet
  part2.parquet
  subdir/
    part3.parquet
```

## Шаг 2: Инициализация датасета

Используйте метод `from_folder`. Он сканирует директорию, индексирует файлы и создает объект датасета.

```python
from indexed_parquet import IndexedParquetDataset

# Сканируем рекурсивно все .parquet файлы
dataset = IndexedParquetDataset.from_folder("data", pattern="*.parquet", recursive=True)

print(f"Всего строк: {len(dataset):,}")
print(f"Колонки: {dataset.schema}")
```

## Шаг 3: Доступ к данным

Вы можете обращаться к строкам по индексу, как к обычному списку. Эта операция выполняется за **O(1)** и не зависит от размера данных.

```python
# Доступ к одной строке
row = dataset[0]  # {'id': 1, 'name': 'Item A', ...}

# Слайсы (возвращают список словарей)
subset = dataset[10:20]  

# Выборка по списку индексов (Fancy indexing)
items = dataset[[1, 5, 100]] 
```

## Шаг 4: Трансформации (Fluent API)

Библиотека поддерживает цепочки вызовов для подготовки данных:

```python
processed_ds = (dataset
                .filter(lambda x: x["category"] == "electronics")
                .shuffle(seed=42)
                .alias("price_usd", lambda x: x["price"] * 0.01) # Новая колонка
                .limit(5000))

print(f"Обработано строк: {len(processed_ds)}")
```

## Шаг 5: Анализ датасета

Метод `.info()` выводит подробную таблицу со статистикой по каждому файлу и покрытию колонок. Это очень удобно для отладки.

```python
dataset.info()
```

## Шаг 6: Сохранение индекса

Сканирование миллионов строк может занять время. Чтобы не делать это каждый раз, сохраните индекс в файл:

```python
# Сохраняем
dataset.save_index("my_index.pkl")

# Загружаем мгновенно в следующий раз
dataset = IndexedParquetDataset.load_index("my_index.pkl")
```

## Шаг 7: Интеграция с PyTorch

`IndexedParquetDataset` наследуется от `torch.utils.data.Dataset`, поэтому он работает "из коробки" с `DataLoader`.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    processed_ds, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4
)

for batch in loader:
    # batch — это словарь тензоров/списков
    print(batch['price_usd'])
    break
```

---

**Что дальше?**
- Узнайте об [Эволюции схем](../how-to/schema-evolution.md), если ваши файлы имеют разную структуру.
- Посмотрите, как построить полный [Deep Learning Pipeline](deep_learning.md).
- Изучите [Операции с колонками](../how-to/column-ops.md) для сложной предобработки.
