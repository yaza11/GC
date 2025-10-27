from chem_spectra.base import SpectrumBaseClass

from readers.chromeleon import GcFidReader


class GcSpectrum(SpectrumBaseClass):
    @classmethod
    def from_gc_fid_reader(cls, reader: GcFidReader):
        df = reader.data.drop(columns='Step (s)')
        return cls.from_dataframe(df)


