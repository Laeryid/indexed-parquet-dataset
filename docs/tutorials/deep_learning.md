# Deep Learning Pipeline

Этот туториал показывает, как построить надежный и быстрый конвейер обучения нейронной сети, используя `IndexedParquetDataset` и PyTorch.

## Почему это лучше стандартных подходов?

1.  **RAM-efficiency**: Даже если ваш датасет занимает 1 ТБ, вы потребляете всего несколько сотен МБ оперативной памяти.
2.  **No Lag**: Случайный доступ O(1) гарантирует, что `DataLoader` не будет ждать, пока распарсится весь файл, чтобы достать одну строку.
3.  **Гибкость**: Вы можете фильтровать и дополнять данные на лету, не пересоздавая тяжелые файлы.

## Пример пайплайна

Ниже приведен полный пример: от инициализации до цикла обучения.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from indexed_parquet import IndexedParquetDataset

# 1. Инициализация с автоматической обработкой пропусков
# auto_fill=True заполнит None значениями по умолчанию (0 для чисел, "" для строк)
ds = IndexedParquetDataset.from_folder("./data", auto_fill=True)

# 2. Подготовка данных (Fluent API)
dataset = (ds
    .filter(lambda x: x["label"] is not None) # Убираем строки без меток
    .shuffle(seed=42)
    .alias("image_tensor", lambda x: torch.tensor(x["pixels"])) # Конвертируем в тензор
    .cast("label", "int") # Убеждаемся, что метка — целое число
)

# 3. Разделение на train/val
train_ds, val_ds = dataset.train_test_split(test_size=0.1, seed=42)

# 4. Настройка DataLoader
# generate_collate_fn гарантирует, что в батч не попадут None, 
# даже если они закрались в данные.
loader = DataLoader(
    train_ds, 
    batch_size=64, 
    shuffle=True, 
    num_workers=4,
    collate_fn=train_ds.generate_collate_fn(on_none='fill')
)

# 5. Цикл обучения
model = nn.Linear(784, 10)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for batch in loader:
        # Данные уже в виде тензоров благодаря alias и collate_fn
        images = batch["image_tensor"]
        labels = batch["label"]
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch} completed")
```

## Работа с пропусками в Deep Learning

Одной из главных проблем при использовании `DataLoader` с данными из файлов является наличие `None`. Если стандартный `default_collate` встречает `None`, он падает с ошибкой.

`IndexedParquetDataset` предлагает три стратегии решения в `generate_collate_fn(on_none=...)`:

-   `'raise'` (по умолчанию): Бросить ошибку с указанием колонки, где найден `None`.
-   `'fill'`: Заменить `None` на значение из конфигурации датасета (см. [Эволюцию схем](../how-to/schema-evolution.md)).
-   `'drop'`: Просто выкинуть эту строку из батча (батч будет чуть меньше `batch_size`).

## Параллельное чтение (num_workers)

Благодаря тому, что `IndexedParquetDataset` корректно реализует методы `__getstate__` и `__setstate__`, он полностью совместим с многопроцессорным чтением в PyTorch (`num_workers > 0`). 

Каждый воркер открывает свои собственные дескрипторы файлов, не конфликтуя с остальными.

## Совет по производительности

Если вы используете очень сложные трансформации в `.alias(lambda ...)` или тяжелую фильтрацию, это может замедлить чтение. В таком случае рекомендуется один раз "материализовать" датасет:

```python
# Все вычисления будут выполнены один раз и сохранены в новый файл
fast_ds = dataset.clone("processed_data.parquet")
```

Подробнее об этом в разделе [Материализация](../how-to/materialization.md).
