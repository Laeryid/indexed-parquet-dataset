# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.4] - 2026-09-02

### Added
- Added LRU caching for `Row Group` tables during dataset iteration.
- Dynamically scale `max_cached_row_groups` during `shuffle()` based on `rg_buffer` size to prevent I/O thrashing completely.

### Fixed
- Fixed extreme I/O thrashing and memory overhead when iterating over shuffled datasets by preventing multiple PyArrow reads of the same Row Group across index batches.

### Fixed
- Fixed CI release build issue where `setuptools_scm` generated dirty dev versions by removing `_version.py` from git tracking.

## [0.4.1] - 2026-09-02

### Fixed
- Fixed memory leak and I/O thrashing in PyArrow caused by sequential row-by-row fetching by implementing batched `__getitems__` in `__iter__`.
- Fixed slow sequential fetching in `train_test_split(stratify_by=...)` by switching to the optimized `__iter__` implementation.
