import logging

from GC.cSpectrum import Spectra
from GC.irms.gc_irms import GC_IRMS

logging.basicConfig(level=logging.INFO)

files = [
    ...  # your files here
]

s = Spectra(list_path_files=files, delta_rt=1e-4)

print('removing outliers')
s.remove_outliers_all(window_length=101, diff_threshold=.001, plts=True)

print('subtracting baseline')
s.subtract_base_line_all(plts=True)

print('aligning spectra')
s.align_spectra(plts=True, max_time_offset=None)

print('setting summed spectrum')
s.set_summed()

print('setting peaks')
# s.set_peaks(prominence=1e-3)
s.set_peaks(prominence=1e-2)

print('setting kernels')
s.set_kernels()

print('binning spectra')
s.bin_spectrum()

df = s.get_dataframe()

s.set_reconstruction_losses(idxs=None, plts=True)

gc_irms = GC_IRMS(list_path_files=files, spectra=s)

df = gc_irms.d13.copy()


