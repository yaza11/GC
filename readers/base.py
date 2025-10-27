from typing import Any


class ReaderBaseClass:
    path_file: str = None

    # set lazy
    _data: Any = None
    _metadata: dict[str, Any] = None

    # set by children
    _x_name: str = None
    _x_unit: str = None

    _y_name: str = None
    _y_unit: str = None

    def __init__(self, path_file: str):
        self.path_file: str = path_file

    def _read_data(self, *args, **kwargs):
        """Sets _data. Needs to be implemented by children"""
        raise NotImplementedError()

    def _read_metadata(self, *args, **kwargs):
        """Sets _metadata. Needs to be implemented by children"""
        raise NotImplementedError()

    @property
    def data(self):
        if self._data is None:
            self._read_data()
        return self._data

    @property
    def metadata(self):
        if self._metadata is None:
            self._read_metadata()
        return self._metadata


