from .dataset import IndexedParquetDataset
from .indexer import scan_directory, BaseIndex, FileInfo
from .schema import SchemaMapper

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["IndexedParquetDataset", "scan_directory", "BaseIndex", "FileInfo", "SchemaMapper", "__version__"]
