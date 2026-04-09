"""
Точное воспроизведение ошибки из вопроса пользователя.

Пользователь использует данные, где колонка "features" — это list<string> или
другой сложный тип. При null-значении PyArrow возвращает None для значения
в списке, а _get_fill_value для составных типов (list[...]) не находит
совпадение в fill_values_by_type и возвращает default_fill_value="-" (строку).

Но более вероятный сценарий: структурированные данные где значение является
словарём с некоторыми None-полями. Проверим несколько гипотез.
"""
import os
import shutil
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from indexed_parquet import IndexedParquetDataset


@pytest.fixture()
def data_with_list_col(tmp_path):
    """Parquet с колонкой типа list<string> (features=список токенов)."""
    rows = []
    for i in range(50):
        if i % 5 == 0:
            rows.append({"id": i, "features": None})  # null list
        else:
            rows.append({"id": i, "features": [f"tok_{j}" for j in range(3)]})
    
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(tmp_path / "data.parquet"))
    return str(tmp_path)


@pytest.fixture()
def data_with_struct_col(tmp_path):
    """Parquet с колонкой типа struct (вложенная структура)."""
    rows = []
    for i in range(50):
        if i % 5 == 0:
            rows.append({"id": i, "meta": None})  # null struct
        else:
            rows.append({"id": i, "meta": {"score": float(i), "tag": f"tag_{i}"}})
    
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(tmp_path / "data.parquet"))
    return str(tmp_path)


@pytest.fixture()
def data_with_mixed_nulls(tmp_path):
    """
    Точная копия данных из примера пользователя:
    - id: int64 с null
    - features: double/float с null
    Данные из двух разных файлов с разными схемами
    (schema evolution scenario).
    """
    # Файл 1: id и features
    rows1 = [{"id": i, "features": float(i * 1.5)} for i in range(30)]
    pq.write_table(pa.Table.from_pylist(rows1), str(tmp_path / "part1.parquet"))
    
    # Файл 2: другая схема — нет "id", есть null в "features"
    rows2 = []
    for i in range(30, 60):
        if i % 7 == 0:
            rows2.append({"features": None})  # нет "id", null в "features"
        else:
            rows2.append({"features": float(i * 2.0)})
    pq.write_table(pa.Table.from_pylist(rows2), str(tmp_path / "part2.parquet"))
    
    return str(tmp_path)


class TestExactUserScenario:
    """Воспроизводим точный сценарий пользователя."""
    
    def test_list_column_none_handled_after_fix(self, data_with_list_col):
        """[POSITIVE TEST] Verifies list column null handling after fix."""
        ds = IndexedParquetDataset.from_folder(
            data_with_list_col,
            default_fill_value="-",
            fill_values_by_type={"int64": 0},
        )
        print(f"\nColumn types: {ds.index.column_types}")
        item_null = ds[0]
        print(f"Item with null features: {item_null}")
        # null list -> "-" (default_fill_value)
        assert item_null['features'] == "-"
        
        loader = DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)
        
        # Не должно падать
        for batch in loader:
            pass
    
    def test_schema_evolution_missing_id_works_after_fix(self, data_with_mixed_nulls):
        """[POSITIVE TEST] Verifies schema evolution success with proper fill values."""
        ds = IndexedParquetDataset.from_folder(
            data_with_mixed_nulls,
            default_fill_value=0, # Для всех остальных
            fill_values_by_type={"int64": 0, "double": 0.0},
        )
        
        loader = DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)
        
        # Раньше тут был pytest.raises. Теперь должно работать!
        count = 0
        for batch in loader:
            count += len(batch['features'])
        assert count == 60
    
    def test_missing_column_fill_value_handles_custom_default(self, data_with_mixed_nulls):
        """[POSITIVE TEST] Verifies that default_fill_value is used when type is unknown."""
        ds = IndexedParquetDataset.from_folder(
            data_with_mixed_nulls,
            default_fill_value=0.0, # Корректный дефолт для числовых данных
            fill_values_by_type={},    # Типы не настроены специально
        )
        
        item = ds[30] # из файла без "id"
        assert item.get('id') == 0.0, "Должен сработать default_fill_value"
        
        loader = DataLoader(ds, batch_size=60, shuffle=False, num_workers=0)
        for batch in loader:
            pass


class TestCorrectBehaviorAfterFix:
    """После исправления — все эти тесты должны проходить."""
    
    def test_list_column_fill_with_empty_list(self, data_with_list_col):
        """
        После исправления: null list-колонка заполняется пустым списком [].
        fill_values_by_type={"list<string>": []} или auto_fill=True.
        """
        ds = IndexedParquetDataset.from_folder(
            data_with_list_col,
            fill_values_by_column={"features": []},  # явное заполнение для списков
        )
        item = ds[0]
        assert item['features'] == [], f"Ожидали [], получили {item['features']}"
    
    def test_schema_evolution_with_proper_fill(self, data_with_mixed_nulls):
        """
        После исправления: DataLoader работает с schema evolution
        когда fill_values покрывают все типы.
        """
        ds = IndexedParquetDataset.from_folder(
            data_with_mixed_nulls,
            default_fill_value=0,
            fill_values_by_type={"int64": 0, "double": 0.0},
        )
        
        loader = DataLoader(ds, batch_size=60, shuffle=False, num_workers=0)
        for batch in loader:
            assert 'features' in batch
            # Отсутствующий "id" должен быть заполнен 0
            if 'id' in batch:
                # Тензор не должен содержать NaN/None
                assert not torch.any(torch.isnan(batch['id'].float()))
            break
