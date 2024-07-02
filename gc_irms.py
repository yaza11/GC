"""
Theory
------
Notation: R(n) is the ratio of an isotopic species of mass n to the base species.
Thus:
    R12 = [C12] / [C12] = 1
    R13 = [C13] / [C12]
    R16 = 1
    R17 = [O17] / [O16]
    R18 = [O18] / [O16]
    R44 = [(C12)(O16)(O16)] / [(C12)(O16)(O16)]
    R45 = {[(C13)(O16)(O16)] + [(C12)(O17)(O16)]} / [(C12)(O16)(O16)]
    R46 = {[(C13)(O17)(O16)] + [(C12)(O17)(O17)] + [(C12)(O18)(O16)]} / [(C12)(O16)(O16)]

Calculating R13, R17, R18 from R45, R46:
3 unknowns, 2 givens, no unique solution. But empirically:
    R17 = K * R18 ** a  (19)
with K = 9.2e-3 and a = 0.516

Further, from Santrock 1985 (eq. 10 + 11):
    R45 = R13 + 2 * R17  (10)
    R46 = 2 * R18 + 2 * R13 * R17 + R17 ** 2  (11)
Eliminating R13 by combining 10 and 11 and substituting R17, we get
    0 = - 3 * K ** 2 * R18 ** (2 * a)
        + 2 * K * R45 * R18 ** a            (20)
        + 2 * R18
        - R46
R18 can be obtained by finding the root of (20) numerically.

Before doing so, the signals of R45 and R46 have to be rescaled to match calibrations.
Gases with known isotopic compositions are injected at the start and end of each
run:
delta13C_injection_standard, delta18O_injection_standard

Through
    delta Xn / permil = 1000 * (Rn_sample / Rn_standard - 1)
It is possible to apply this calibration: First of all, we have to calculate the corresponding
R-values for the standards: The theoretical R18 and R13 values that should be
measured for the peaks are
    R13_theo = (delta13C_injection_standard / 1000 + 1) * R13_std
    R18_theo = (delta18O_injection_standard / 1000 + 1) * R18_std
where R13_std and R18_std are given in the dxf file corresponding to the VPDB
and SMOW standards. Consequently, using eq. 10, 11 and 19
    R17_theo = K * R18_theo ** a
    R45_theo = R13_theo + 2 * R17_theo
    R46_theo = 2 * R18_theo + 2 * R13_theo * R17_theo + R17_theo ** 2
Hence, the measured peaks have to be corrected by the factors
    c45_corrected = R45_theo / R45_meas
    c46_corrected = R46_theo / R46_meas
"""

import numpy as np
import pandas as pd
import os
import logging

from scipy.optimize import root_scalar, fsolve
from tqdm import tqdm

from cSpectrum import Spectra, Spectrum
from dxf_reader import DXF


logger = logging.getLogger(__name__)

K: float = 9.2e-3  # +/-.18
a: float = .516

# from dxf reference ratios
R13_STANDARD: float = 0.01118  # VPDB
R17_STANDARD: float = 0.00038  # VSMOW (redundant, can be calculated from K * R13_STANDARD ** a)
R18_STANDARD: float = 0.002005  # VSMOW

R2STD = {
    13: R13_STANDARD,
    17: R17_STANDARD,
    18: R18_STANDARD,
}

DELTA_C13_INJECTION_STANDARD: float = -33.4
DELTA_O18_INJECTION_STANDARD: float = 0.

# DELTA_C13_INJECTION_STANDARD = -43.411
# DELTA_O18_INJECTION_STANDARD = -21.097

k = {'S13': R13_STANDARD, 'S18': R18_STANDARD, 'K': K, 'A': a}

def find_R18(R45: float, R46: float) -> float:
    def func(R18: float) -> float:
        """Distance between current and actual solution"""
        eq = (
                - 3 * K ** 2 * R18 ** (2 * a)
                + 2 * K * R45 * R18 ** a
                + 2 * R18
                - R46
        )
        return eq

    solved_R18 = root_scalar(func, x0=.2e-2)
    return solved_R18


def find_R17(
        R18: float | np.ndarray | pd.DataFrame
) -> float | np.ndarray | pd.DataFrame:
    return K * R18 ** a


def find_R13(
        R17: float | np.ndarray | pd.DataFrame,
        R45: float | np.ndarray | pd.DataFrame
) -> float | np.ndarray | pd.DataFrame:
    return R45 - 2 * R17


def delta(
        isotope_mass: int,
        R_sample: float | int | np.ndarray | pd.DataFrame
) -> float | np.ndarray | pd.DataFrame:
    R_std = R2STD[isotope_mass]
    return (R_sample / R_std - 1) * 1e3


def R_theo():
    """Calculate the value of the R45 standard from R18 and R13 standards."""
    R13_theo = (DELTA_C13_INJECTION_STANDARD / 1000 + 1) * R13_STANDARD
    R18_theo = (DELTA_O18_INJECTION_STANDARD / 1000 + 1) * R18_STANDARD
    R17_theo = find_R17(R18_theo)
    R45_theo = R13_theo + 2 * R17_theo
    R46_theo = 2 * R18_theo + 2 * R13_theo * R17_theo + R17_theo ** 2
    return R45_theo, R46_theo


def factors_from_injection_peaks(R45, R46):
    """
    Use known composition of injection peaks to rescale R45 and R46 to
    desired compositions.

    Provide the R-values of the injection peaks and obtain the correction
    factors to apply for the entire spectrum
    """
    # use desired values
    R45_theo, R46_theo = R_theo()

    return np.array(R45_theo / R45).mean(), np.array(R46_theo / R46).mean()

def delta13c_santrock(r45sam, r46sam, d13cstd, r45std, r46std, d18ostd):
    """
    Given the measured isotope signals of a sample and a
    standard and the delta-13C of that standard, calculate
    the delta-13C of the sample.

    Algorithm from Santrock, Studley & Hayes 1985 Anal. Chem.
    """

    # function for calculating 17R from 18R
    def c17(r):
        return k['K'] * r ** k['A']
    rcpdb, rosmow = k['S13'], k['S18']

    # known delta values for the ref peak
    r13std = (d13cstd / 1000. + 1) * rcpdb
    r18std = (d18ostd / 1000. + 1) * rosmow

    # determine the correction factors
    c45 = r13std + 2 * c17(r18std)
    c46 = c17(r18std) ** 2 + 2 * r13std * c17(r18std) + 2 * r18std

    # correct the voltage ratios to ion ratios
    r45 = (r45sam / r45std) * c45
    r46 = (r46sam / r46std) * c46

    def rf(r18):
        return -3 * c17(r18) ** 2 + 2 * r45 * c17(r18) + 2 * r18 - r46
    # r18 = scipy.optimize.root(rf, r18std).x[0]  # use with scipy 0.11.0
    r18 = fsolve(rf, r18std)[0]
    r13 = r45 - 2 * c17(r18)
    return 1000 * (r13 / rcpdb - 1)

def get_deltas(I44, I45, I46, areas45, areas46):
    if not hasattr(I44, '__iter__'):
        I44 = np.array([[I44]])
        I45 = np.array([[I45]])
        I46 = np.array([[I46]])

    R45 = np.array(I45 / I44)
    R46 = np.array(I46 / I44)

    if R45.ndim == 1:
        R45 = R45[:, np.newaxis]
        R46 = R46[:, np.newaxis]

    f45, f46 = factors_from_injection_peaks(areas45, areas46)

    R45 *= f45
    R46 *= f46

    R18 = np.zeros_like(R45)
    h, w = R18.shape
    for i in tqdm(range(h), 'searching R18 values ...'):
        for j in range(w):
            R18[i, j] = find_R18(R45[i, j], R46[i, j]).root

    R17 = find_R17(R18)
    R13 = find_R13(R17, R45)

    # calculate delta values
    d13 = delta(13, R13)
    d17 = delta(17, R17)
    d18 = delta(18, R18)

    return d13, d17, d18


class GC_IRMS:
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
                s = Spectrum(path_file=file, dxf_trace=trace)
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
