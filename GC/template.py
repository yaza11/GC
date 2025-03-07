import pandas as pd
import numpy as np
import logging

from GC.dxf_reader import DXF, get_dxf
from GC.cSpectrum import Spectrum, Spectra
from GC.gc_irms import GC_IRMS, get_deltas

logging.basicConfig(level=logging.INFO)

files = [
    "your files here"
]

# spec = Spectrum(path_file=files[0])
# spec.get_standard_areas(plts=True)

s = Spectra(list_path_files=files, delta_rt=1e-4)

print('removing outliers')
s.remove_outliers_all(window_length=101, diff_threshold=.001, plts=True)

print('subtracting baseline')
s.subtract_base_line_all(plts=True)

print('aligning spectra')
s.align_spectra(plts=True, max_time_offset=None)

# s.set_rt_window_all(window=(5, np.infty))

print('setting summed spectrum')
s.set_summed()

print('setting peaks')
# s.set_peaks(prominence=1e-3)
s.set_peaks(prominence=1e-2)

# thr 100 - 200 mV

print('setting kernels')
s.set_kernels()

print('binning spectra')
s.bin_spectrum()

df = s.get_dataframe()

s.set_reconstruction_losses(idxs=None, plts=True)

# import matplotlib.pyplot as plt
# for spec in s.spectra:
#     spec.get_standard_areas(plts=True)
#     plt.show()

gc_irms = GC_IRMS(list_path_files=files, spectra=s)

df = gc_irms.d13.copy()
df44 = gc_irms.ft44.copy()
df45 = gc_irms.ft45.copy()
df46 = gc_irms.ft46.copy()

df.loc[:, 'rts'] = s.rts[s.peaks] * 60
df44.loc[:, 'rts'] = s.rts[s.peaks] * 60
df45.loc[:, 'rts'] = s.rts[s.peaks] * 60
df46.loc[:, 'rts'] = s.rts[s.peaks] * 60

df.to_excel()
df44.to_excel()
df45.to_excel()
df46.to_excel()
