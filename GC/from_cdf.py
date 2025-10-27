import numpy as np
import matplotlib.pyplot as plt
from chem_spectra.baseline_finder.main import Baseline
from chem_spectra.peaks.main import PeakList, Peak
from chem_spectra.peaks.peak_finder.main import PeakFinder

from scipy.io import netcdf_file

from GC.spectrum import SpectrumGc, SpectrumGcStandard
from scipy.signal import peak_widths


class GcCdfReader:
    # TODO: parse crl to add metadata

    filename: str = None

    is_standard_run: bool = None

    sample_id: str = None
    sample_amount: str = None
    sample_name: str = None
    sample_injection_volume: str = None

    detector_unit: str = None

    retention_unit: str = None
    retention_times: np.ndarray[float] = None

    variables: dict = None

    def __init__(self, file: str, is_standard_run: bool = None):
        with netcdf_file(file, 'r', mmap=False) as f:
            for attr, val in f.__dict__.items():
                if isinstance(val, bytes):
                    val_new = val.decode('utf-8')
                else:
                    val_new = val
                self.__setattr__(attr, val_new)

        if is_standard_run is not None:
            self.is_standard_run = is_standard_run

        # extract x, y information
        #  construct retention times from delay time and sampling interval
        self.rt0 = self.variables['actual_delay_time'].data
        self.drt = self.variables['actual_sampling_interval'].data
        self.rt_max = self.variables['actual_run_time_length'].data

        self.rts = np.arange(self.rt0, self.rt_max, self.drt)
        self.intensities = self.variables['ordinate_values'].data

        assert self.rts.shape == self.intensities.shape

    def plot_xy(self, try_add_processed: bool = True, ax: plt.Axes = None) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.rts, self.intensities, label='raw')
        ax.set_xlabel(f'RT in {self.retention_unit}')
        ax.set_ylabel(f'Intensity in {self.detector_unit}')
        ax.set_title(f'Chromatogram of {self.sample_name} (ID={self.sample_id})')

        if not try_add_processed:
            return ax

        # attempt to add peaks and baseline
        baseline_rts = np.concat(
            [self.variables['baseline_start_time'].data, self.variables['baseline_stop_time'].data])
        baseline_ints = np.concat(
            [self.variables['baseline_start_value'].data, self.variables['baseline_stop_value'].data])
        o = np.argsort(baseline_rts)
        baseline_rts = baseline_rts[o]
        baseline_ints = baseline_ints[o]
        ax.plot(baseline_rts, baseline_ints, alpha=.5, label='baseline')

        rts = self.variables['peak_retention_time'].data
        h = self.variables['peak_height'].data
        # add baseline to values
        b_at_peaks = np.interp(rts, baseline_rts, baseline_ints)

        ax.vlines(rts, b_at_peaks, h + b_at_peaks, colors='r', label='peaks')
        ax.legend()
        return ax

    def get_spectrum(
            self,
            use_available_baseline: bool = True,
            use_available_peaks: bool = True
    ) -> SpectrumGc | SpectrumGcStandard:
        spec = SpectrumGcStandard() if self.is_standard_run else SpectrumGc()

        spec.x = self.rts
        spec.y = self.intensities
        spec.x_unit = self.retention_unit
        spec.y_unit = self.detector_unit
        spec.x_name = 'RT'
        spec.y_name = 'Voltage at detector'

        if use_available_baseline:
            bl = Baseline(x=self.rts, y=self.intensities, fit_on_init=False)

            baseline_rts = np.concat(
                [self.variables['baseline_start_time'].data, self.variables['baseline_stop_time'].data])
            baseline_ints = np.concat(
                [self.variables['baseline_start_value'].data, self.variables['baseline_stop_value'].data])
            # interpolate to same points
            bl.y_baseline = np.interp(self.rts, xp=baseline_rts, fp=baseline_ints)
            bl.parameters['note'] = f'Baseline taken from cdf file {self.filename}'
            spec.baseline = bl

        if use_available_peaks:
            peak_rts = self.variables['peak_retention_time'].data.astype(float)
            peak_idcs = [np.argmin(np.abs(rt - self.rts)) for rt in peak_rts]
            peak_heights = self.variables['peak_height'].data.astype(float)
            properties = peak_widths(self.intensities, peaks=peak_idcs, rel_height=.5)
            properties = dict(zip(['widths', 'width_heights', 'left_ips', 'right_ips'], properties))

            peaks = []
            for idx in range(peak_rts.shape[0]):
                props = {}
                for k, v in properties.items():
                    props[k] = v[idx]
                p = Peak(x_idx=peak_idcs[idx], x=peak_rts[idx], y=peak_heights[idx], properties=props)
                peaks.append(p)

            peak_list = PeakList(peaks)
            spec.peak_list = peak_list
        return spec

if __name__ == '__main__':
    file = r"\\hlabstorage.dmz.marum.de\scratch\Yannick\gc-fid\fileformatsapolarfractions\1807rka.cdf"

    # f = netcdf_file(file, 'r')
    #
    # rts = f.variables['peak_retention_time'].data
    # h = f.variables['peak_height'].data
    #
    # plt.stem(rts, h, markerfmt='')
    # plt.savefig('temp.pdf')
    # plt.close()

    rdr = GcCdfReader(file)

    ax = rdr.plot_xy()
    ax.set_xlim([40, 50])
    # ax.set_ylim([0, 100])

    spec = rdr.get_spectrum()
    spec.plot_peaks(ax=ax)

    plt.savefig('temp.pdf')
