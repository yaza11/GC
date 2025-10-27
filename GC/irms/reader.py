import numpy as np
import pandas as pd
import os
import logging

from GC.cSpectrum import Spectra, GcSpectrum
from GC.irms.ratios_to_deltas import get_deltas

logger = logging.getLogger(__name__)


class GcIrmsReader:
    def __init__(
            self,
            *,
            list_path_files: list[str] | None = None,
            spectra: Spectra
    ):
        assert hasattr(spectra, 'kernel_params'), 'initialize spec up to set_kernels'

        self.list_path_files = list_path_files

        self.spectra = spectra

        columns = [os.path.basename(file).split('.')[0] for file in list_path_files]
        data = np.zeros((self.spectra.kernel_params.shape[0], len(columns)))

        self.ft44 = pd.DataFrame(data=data.copy(), columns=columns)
        self.ft45 = pd.DataFrame(data=data.copy(), columns=columns)
        self.ft46 = pd.DataFrame(data=data.copy(), columns=columns)

        self._set_tables()

    def _set_tables(self):
        fts = [self.ft44, self.ft45, self.ft46]
        columns = fts[0].columns

        self.d13 = pd.DataFrame().reindex_like(self.ft44).astype(float)
        self.d17 = pd.DataFrame().reindex_like(self.ft44).astype(float)
        self.d18 = pd.DataFrame().reindex_like(self.ft44).astype(float)

        traces = ('44', '45', '46')
        for j, (file, col) in enumerate(zip(self.list_path_files, columns)):
            # dxf: DXF = DXF(file)
            # resistances control amplification
            # higher resistance --> stronger amplification
            # to correct this, divide 45 and 46 by resistances
            # resistors_df: pd.DataFrame = dxf.method_info['resistors']
            # amplifications: pd.Series = (
            #         resistors_df.loc[:, 'R.Ohm'] / resistors_df.loc['1', 'R.Ohm']
            # )
            areas_calibrant: dict[str, np.ndarray[float]] = {}
            for trace, ft in zip(traces, fts):
                s = GcSpectrum(path_file=file, dxf_trace=trace)
                s.resample(delta_rt=self.spectra.delta_rt)
                # align traces
                time_shift = self.spectra.xcorr(s)
                logger.info(f'shifting {trace} by {time_shift * 60:.1f} seconds')
                s.rts += time_shift
                s.resample(self.spectra.rts)

                s.kernel_params = self.spectra.kernel_params
                s.peaks = self.spectra.peaks
                s.peak_properties = self.spectra.peak_properties
                s.peak_setting_parameters = self.spectra.peak_setting_parameters
                s.bin_spectrum()
                # rescale such that standards have desired isotope values
                areas = s.get_standard_areas()
                areas_calibrant[trace] = areas
                ft.loc[:, col] = s.line_spectrum
                ft.iloc[[0, 1, 2, -3, -2, -1], j] = areas
            areas45 = areas_calibrant['45'] / areas_calibrant['44']
            areas46 = areas_calibrant['46'] / areas_calibrant['44']
            d13, d17, d18 = get_deltas(
                I44=self.ft44.loc[:, col].copy(),
                I45=self.ft45.loc[:, col].copy(),
                I46=self.ft46.loc[:, col].copy(),
                areas45=areas45,
                areas46=areas46
            )
            self.d13.loc[:, col] = d13[:, 0]
            self.d17.loc[:, col] = d17[:, 0]
            self.d18.loc[:, col] = d18[:, 0]
