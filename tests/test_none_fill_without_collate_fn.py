"""
Тесты, воспроизводящие ошибку из примера пользователя.

Пользователь использует:
    - default_fill_value="-"
    - fill_values_by_type={"int64": 0}
    - БЕЗ collate_fn (стандартный DataLoader)

Ожидание: DataLoader работает без TypeError.
Реальность: TypeError: found <class 'NoneType'>

ПРИЧИНА (баг):
    _get_fill_value() возвращает None когда:
    1. column_name не совпадает ни с одной колонкой в fill_values_by_column
    2. col_type не совпадает ни с одним ключом в fill_values_by_type
    3. default_fill_value равен None (по умолчанию!)

    НО при использовании with_folder(..., default_fill_value="-") пользователь
    ожидает, что "-" используется как fallback. Это работает корректно для
    строковых колонок, но вызывает другую ошибку для числовых (default_collate
    не может смешивать str и tensor).

    Более фундаментальный баг: __getitem__ НЕ применяет fill_values к значениям
    которые PyArrow читает как Python None из реального null в Parquet.
    Точнее, он ПРИМЕНЯЕТ (строки 256-257), но _get_fill_value может вернуть None
    снова, если default_fill_value=None (по умолчанию).

    В примере пользователя default_fill_value="-" и fill_values_by_type={"int64": 0}.
    Для колонки "features" (тип double/float64) ни одно правило не покрывает its.
    Значит _get_fill_value вернёт default_fill_value="-" (строку).
    Когда default_collate видит смесь float и str — возможна другая ошибка.
    
    НО реальный баг: при num_workers>0 на Windows используется spawn.
    _clone_with_indices и другие клоны НЕ передают fill_values при пикловании через
    __getstate__/__setstate__ (хотя они сохраняются в __dict__).
    
    Проверим гипотезы тестами.
"""
import os
import shutil
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from indexed_parquet import IndexedParquetDataset


@pytest.fixture(scope="module")
def null_data_dir(tmp_path_factory):
    """Создаёт директорию с parquet-файлом, содержащим null-значения."""
    path = tmp_path_factory.mktemp("null_data")
    
    # Точно воспроизводим данные из примера пользователя:
    # Некоторые строки содержат null в "id" и "features"
    rows = []
    for i in range(100):
        if i % 5 == 0:
            rows.append({"id": None, "features": None})
        else:
            rows.append({"id": i, "features": float(i * 1.1)})
    
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path / "train.parquet"))
    
    return str(path)


# === ТЕСТЫ, КОТОРЫЕ ДОЛЖНЫ ПАДАТЬ (воспроизводят баг) ===

class TestFillValuesBugReproduction:
    """Тесты, воспроизводящие реальную ошибку из примера пользователя."""
    
    def test_getitem_returns_none_for_null_int_when_fill_not_configured(self, null_data_dir):
        """
        БАГ: __getitem__ возвращает None для null int64 когда
        fill_values_by_type={} и default_fill_value=None (дефолт).
        
        Это напрямую ломает DataLoader без collate_fn.
        """
        ds = IndexedParquetDataset.from_folder(null_data_dir)  # нет fill-настроек
        # Строка 0 содержит null (i=0, 0%5==0)
        item = ds[0]
        # С bagagem fill-конфигурацией — None должен вернуться
        assert item['id'] is None, f"Ожидали None, получили {item['id']}"
    
    def test_dataloader_works_with_generic_fallback(self, null_data_dir):
        """[POSITIVE TEST] Verifies success when generic default_fill_value is provided."""
        ds = IndexedParquetDataset.from_folder(
            null_data_dir,
            default_fill_value=0.0,           # Fallback для всех
            fill_values_by_type={"int64": 0}, # Специфично для ID
        )
        loader = DataLoader(ds, batch_size=10, shuffle=False)
        for _ in loader: pass  # Должно работать

    def test_dataloader_must_fail_when_no_fill_provided(self, null_data_dir):
        """[NEGATIVE TEST] Verifies that DataLoader raises TypeError when no fill values are configured for nulls."""
        ds = IndexedParquetDataset.from_folder(
            null_data_dir,
            default_fill_value=None, # Отключаем автозаполнение
        )
        loader = DataLoader(ds, batch_size=10, shuffle=False)
        
        # Ожидаем ошибку NoneType, так как мы ничего не заполнили
        with pytest.raises(TypeError, match="NoneType"):
            for _ in loader:
                pass
    
    def test_getitem_does_not_fill_null_values_from_parquet(self, null_data_dir):
        """
        БАГ: __getitem__ должен подставлять fill value для значений,
        которые PyArrow вернул как None (реальные null из parquet).
        
        При default_fill_value="-" и fill_values_by_type={"int64": 0}:
        - id (null, int64) → должен стать 0
        - features (null, double) → должен стать "-" (default_fill_value)
        
        Но сейчас: features возвращается как None, 
        потому что _get_fill_value возвращает default_fill_value (строку "-"),
        которая несовместима с float tensor в DataLoader.
        
        Этот тест проверяет, что __getitem__ НЕ возвращает None.
        """
        ds = IndexedParquetDataset.from_folder(
            null_data_dir,
            default_fill_value="-",
            fill_values_by_type={"int64": 0},
        )
        
        # Строка 0: id=None, features=None в parquet
        item = ds[0]
        
        # Ни одно значение не должно быть None
        for col, val in item.items():
            assert val is not None, (
                f"Колонка '{col}' вернула None, хотя задан default_fill_value='-'.\n"
                f"Это ломает DataLoader без collate_fn."
            )
    
    def test_dataloader_user_exact_example_works_after_fix(self, null_data_dir):
        """
        РЕГРЕССИЯ: Код из вопроса пользователя с fill_values_by_type={"int64": 0}
        и default_fill_value="-" больше НЕ вызывает ошибку NoneType.

        Примечание: если пользователь задаёт "-" (строку) как fill для double-колонки,
        DataLoader может выдать "must be real number, not str" — это другая ошибка,
        не связанная с нашим багом. Для корректной работы нужно:
            fill_values_by_type={"int64": 0, "double": 0.0}
        """
        ds = IndexedParquetDataset.from_folder(
            null_data_dir,
            default_fill_value="-",
            fill_values_by_type={"int64": 0},
        )

        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        # Ошибка NoneType исправлена. Но "-" как fill для double-колонки
        # вызывает "must be real number, not str" — ожидаемо при несовместимых типах.
        try:
            for batch in loader:
                pass
        except TypeError as e:
            assert "NoneType" not in str(e), (
                f"РЕГРЕССИЯ: ошибка NoneType вернулась!\n{e}"
            )
            # "must be real number, not str" — допустимо (несовместимый fill value)



# === ТЕСТЫ ПРАВИЛЬНОГО ПОВЕДЕНИЯ (должны проходить после исправления) ===

class TestFillValuesCorrectBehavior:
    """Тесты ожидаемого поведения после исправления."""
    
    def test_getitem_fills_null_int_with_type_specific_value(self, null_data_dir):
        """
        После исправления: __getitem__ должен заменить null int64 на 0
        при fill_values_by_type={"int64": 0}.
        """
        ds = IndexedParquetDataset.from_folder(
            null_data_dir,
            fill_values_by_type={"int64": 0, "double": 0.0},
        )
        item = ds[0]  # id=None, features=None в parquet
        assert item['id'] == 0, f"Ожидали 0, получили {item['id']}"
        assert item['features'] == 0.0, f"Ожидали 0.0, получили {item['features']}"
    
    def test_dataloader_works_with_full_fill_config(self, null_data_dir):
        """
        После исправления: DataLoader работает когда ВСЕ типы покрыты.
        """
        ds = IndexedParquetDataset.from_folder(
            null_data_dir,
            default_fill_value="-",
            fill_values_by_type={"int64": 0, "double": 0.0},
        )
        
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        
        # Не должно быть исключений
        for batch in loader:
            assert 'id' in batch
            assert 'features' in batch
            break
    
    def test_dataloader_user_example_with_autofill_works(self, null_data_dir):
        """
        После исправления: DataLoader работает с auto_fill=True
        (все типы покрыты автоматически).
        """
        ds = IndexedParquetDataset.from_folder(
            null_data_dir,
            auto_fill=True,
        )
        
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        
        for batch in loader:
            assert 'id' in batch
            assert 'features' in batch
            # Значения должны быть заполнены
            assert batch['id'][0].item() == 0     # null → 0 (int default)
            assert batch['features'][0].item() == 0.0  # null → 0.0 (float default)
            break
