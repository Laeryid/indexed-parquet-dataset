"""
Тесты для воспроизведения бага с вложенными None в struct/list колонках.

БАГ: PyArrow колонки типа struct<..., field: null, ...> возвращают Python dict
с None-полями. Такие значения передаются в default_collate который не умеет
обрабатывать None внутри вложенных dict.

_get_fill_value() работает только на уровне top-level колонок и не обходит
вложенные структуры (dict, list-of-dict).

Воспроизводит точный сценарий:
    ds = IndexedParquetDataset.from_folder("./data",
        default_fill_value="-",
        fill_values_by_type={"int64": 0},
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    for batch in loader:  # <-- TypeError: found <class 'NoneType'>
        ...
"""
import os
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from indexed_parquet import IndexedParquetDataset


@pytest.fixture()
def struct_with_null_field(tmp_path):
    """
    Parquet с колонкой struct, где одно поле всегда null.
    Точно воспроизводит: gen_input_configs: struct<..., seed: null, ...>
    """
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("config", pa.struct([
            pa.field("temperature", pa.float64()),
            pa.field("seed", pa.null()),       # <-- ВОТ ЭТО проблема
            pa.field("generator", pa.string()),
        ])),
    ])
    
    arrays = [
        pa.array([1, 2, 3], type=pa.int64()),
        pa.StructArray.from_arrays(
            [
                pa.array([0.8, 0.9, 0.7], type=pa.float64()),
                pa.array([None, None, None], type=pa.null()),
                pa.array(["gpt-4", "gpt-4", "gpt-4"], type=pa.string()),
            ],
            names=["temperature", "seed", "generator"]
        ),
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, str(tmp_path / "data.parquet"))
    return str(tmp_path)


@pytest.fixture()
def list_of_struct_with_nulls(tmp_path):
    """
    Parquet с колонкой list<struct<...>> где некоторые элементы null.
    Воспроизводит: conversations: list<element: struct<from: string, value: string>>
    """
    rows = [
        {"id": i, "messages": [{"role": "user", "text": f"msg_{i}"}] if i % 3 != 0 else None}
        for i in range(20)
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(tmp_path / "data.parquet"))
    return str(tmp_path)


class TestNestedNoneBug:
    """Регрессионные тесты для бага с вложенными None в struct-колонках.
    
    ИСТОРИЯ: PyArrow struct-колонки с полями типа null (e.g. seed: null)
    возвращали Python dict с None-значениями. default_collate не умеет
    обрабатывать NoneType внутри вложенных dict.
    
    ИСПРАВЛЕНИЕ: _deep_fill_nones() рекурсивно заменяет None в dict/list
    на fill value (default_fill_value или значение из fill_values_by_type).
    """
    
    def test_struct_with_null_field_getitem_no_longer_returns_none(self, struct_with_null_field):
        """[POSITIVE TEST] Verifies that __getitem__ recursively fills Nones in structs."""
        ds = IndexedParquetDataset.from_folder(
            struct_with_null_field,
            default_fill_value="-",
            fill_values_by_type={"int64": 0},
        )
        item = ds[0]
        config = item['config']
        assert isinstance(config, dict), "config должен быть dict"
        # После исправления seed должен быть заполнен, а не None
        assert config['seed'] is not None, (
            "РЕГРЕССИЯ: seed снова None! _deep_fill_nones() не работает."
        )
        assert config['seed'] == "-", (
            f"seed должен быть '-' (default_fill_value), получили: {config['seed']!r}"
        )
    
    def test_struct_with_null_field_dataloader_no_longer_fails(self, struct_with_null_field):
        """[POSITIVE TEST] Verifies DataLoader success with nested null-typed fields after fix."""
        ds = IndexedParquetDataset.from_folder(
            struct_with_null_field,
            default_fill_value="-",
            fill_values_by_type={"int64": 0},
        )
        
        loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
        
        # Должно работать без исключений
        batch = next(iter(loader))
        assert 'id' in batch
        assert 'config' in batch
    
    def test_magpie_style_schema_works_after_fix(self, tmp_path):
        """[POSITIVE TEST] Regression for Magpie-style schema with nested None/null fields."""
        schema = pa.schema([
            pa.field("conversation_id", pa.string()),
            pa.field("instruction", pa.string()),
            pa.field("gen_input_configs", pa.struct([
                pa.field("temperature", pa.float64()),
                pa.field("seed", pa.null()),
                pa.field("input_generator", pa.string()),
            ])),
            pa.field("conversations", pa.list_(
                pa.struct([
                    pa.field("from", pa.string()),
                    pa.field("value", pa.string()),
                ])
            )),
        ])
        arrays = [
            pa.array([f"id_{i}" for i in range(10)], type=pa.string()),
            pa.array([f"question_{i}" for i in range(10)], type=pa.string()),
            pa.StructArray.from_arrays(
                [
                    pa.array([0.8] * 10, type=pa.float64()),
                    pa.array([None] * 10, type=pa.null()),
                    pa.array(["llama"] * 10, type=pa.string()),
                ],
                names=["temperature", "seed", "input_generator"]
            ),
            pa.array(
                [[{"from": "human", "value": f"q_{i}"}] for i in range(10)],
                type=pa.list_(pa.struct([
                    pa.field("from", pa.string()),
                    pa.field("value", pa.string()),
                ]))
            ),
        ]
        table = pa.Table.from_arrays(arrays, schema=schema)
        pq.write_table(table, str(tmp_path / "magpie.parquet"))
        
        ds = IndexedParquetDataset.from_folder(
            str(tmp_path),
            default_fill_value="-",
            fill_values_by_type={"int64": 0},
        )
        
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        
        # Должно работать без исключений
        batch = next(iter(loader))
        assert 'conversation_id' in batch
        assert 'gen_input_configs' in batch
        # default_collate для dict-колонок возвращает dict of lists
        configs = batch['gen_input_configs']
        # seed должен быть заполнен ('-'), а не None
        # configs['seed'] — это список значений по батчу
        seeds = configs.get('seed', []) if isinstance(configs, dict) else []
        assert all(s is not None for s in seeds), (
            f"РЕГРЕССИЯ: seed снова None! seeds={seeds}"
        )
    
    def test_no_nested_none_after_fix(self, struct_with_null_field):
        """[POSITIVE TEST] Verifies absence of None values in the result of __getitem__."""
        ds = IndexedParquetDataset.from_folder(
            struct_with_null_field,
            default_fill_value="-",
            fill_values_by_type={"int64": 0},
        )
        item = ds[0]
        
        config = item.get('config', {})
        has_nested_none = any(v is None for v in (config or {}).values())
        
        assert not has_nested_none, (
            f"РЕГРЕССИЯ: config dict всё ещё содержит None: {config}\n"
            "_deep_fill_nones() должен был заменить их на fill value."
        )


class TestExpectedBehaviorAfterFix:
    """Ожидаемое поведение после исправления."""
    
    def test_null_type_field_in_struct_replaced_by_fill_value(self, struct_with_null_field):
        """
        После исправления: поля типа null внутри struct должны заменяться на fill value.
        
        Стратегия: при чтении row из parquet, если значение является dict (struct),
        рекурсивно заменяем все None внутри него на подходящий fill value.
        """
        ds = IndexedParquetDataset.from_folder(
            struct_with_null_field,
            default_fill_value="",   # Пустая строка как дефолт
        )
        item = ds[0]
        config = item['config']
        
        # seed: null должен быть заменён на fill value (не None)
        assert config['seed'] is not None, (
            f"config['seed'] всё ещё None после исправления: {config}"
        )
        assert config['seed'] == "", f"Ожидали '', получили {config['seed']!r}"
    
    def test_dataloader_works_with_struct_null_field_after_fix(self, struct_with_null_field):
        """
        После исправления: DataLoader работает без ошибок для struct с null полями.
        """
        ds = IndexedParquetDataset.from_folder(
            struct_with_null_field,
            default_fill_value="",
        )
        
        # Должно работать без исключений
        loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        
        assert 'id' in batch
        assert 'config' in batch
