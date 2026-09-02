# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] - 2026-09-02

### Fixed
- Fixed CI release build issue where `setuptools_scm` generated dirty dev versions by removing `_version.py` from git tracking.

## [0.4.1] - 2026-09-02

### Fixed
- Fixed memory leak and I/O thrashing in PyArrow caused by sequential row-by-row fetching by implementing batched `__getitems__` in `__iter__`.
- Fixed slow sequential fetching in `train_test_split(stratify_by=...)` by switching to the optimized `__iter__` implementation.
