import pandas as pd

from readers.base import ReaderBaseClass


def read_chromeleon_data(path_file: str, decimal: str) -> pd.DataFrame:
    """Read a file exported from chromeleon"""
    with open(path_file, 'r') as f:
        lines = f.readlines()
    # find start line of data
    lines_new = []
    start_write = False
    for line in lines:
        if line.strip() == 'Chromatogram Data:':
            start_write = True
            continue
        if not start_write:
            continue
        line = line.replace('n.a.', 'NaN')
        if decimal == ',':
            line = line.replace('.', '_').replace(',', '.')
        lines_new.append(line.strip('\n').split('\t'))

    df = pd.DataFrame(data=lines_new[1:], columns=lines_new[0])
    return df.astype(float)


def read_chromeleon_metadata(path_file: str) -> dict:
    ...



class GcFidReader(ReaderBaseClass):
    _data: "GcSpectrum" = None

    def __init__(self, path_file: str, decimal=','):
        super().__init__(path_file)

        # call read data and metadata immediately
        self._read_data(decimal=decimal)
        self._read_metadata()

        # set x, y units and names from metadata

    def _read_data(self, decimal: str):
        self._data = read_chromeleon_data(self.path_file, decimal=decimal)

    def _read_metadata(self, *args, **kwargs):
        ...
