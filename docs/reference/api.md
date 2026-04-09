# Справочник API

В этом разделе приведена подробная документация по основным классам и методам библиотеки `indexed-parquet-dataset`.

## IndexedParquetDataset

Основной класс для индексации и доступа к данным. Наследуется от `torch.utils.data.Dataset` (если установлен PyTorch).

::: indexed_parquet.dataset.IndexedParquetDataset
    options:
      members:
        - from_folder
        - load_index
        - save_index
        - info
        - shuffle
        - train_test_split
        - select
        - limit
        - sample
        - select_columns
        - map
        - alias
        - rename
        - cast
        - filter
        - merge
        - concat
        - copy
        - clone
        - to_parquet
        - iter_batches
        - generate_collate_fn
        - schema

## CollateHandler

Вспомогательный класс для корректного сбора батчей в `DataLoader`. Используйте его через метод `ds.generate_collate_fn()`.

::: indexed_parquet.dataset.CollateHandler

## SchemaMapper

Внутренний класс, отвечающий за трансформации схем и вычисляемые колонки.

::: indexed_parquet.schema.SchemaMapper

## BaseIndex

Класс, хранящий метаданные индекса (оффсеты файлов, количество строк).

::: indexed_parquet.indexer.BaseIndex
