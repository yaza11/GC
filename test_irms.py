import pandas as pd
import numpy as np
import logging

from dxf_reader import DXF, get_dxf
from cSpectrum import Spectrum, Spectra
from gc_irms import GC_IRMS, get_deltas

logging.basicConfig(level=logging.INFO)

# files = [
#     "C:/Users/Yannick Zander/Downloads/GC-ICRMS/240202_Lennart_034__Inc_2_NL_TMSed_2-150.dxf",
#     "C:/Users/Yannick Zander/Downloads/GC-ICRMS/240202_Lennart_035__Inc_3_NL_TMSed_2-150.dxf",
#     "C:/Users/Yannick Zander/Downloads/GC-ICRMS/240202_Lennart_036__Inc_4_NL_TMSed_2-150.dxf"
# ]

files = [r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\240530_Lennart_246__BP_2_C13_E37_NCONCT3.dxf"]

# spec = Spectrum(files[0])

s = Spectra(list_path_files=files, delta_rt=1e-4)

print('removing outliers')
s.remove_outliers_all(window_length=101, diff_threshold=.001, plts=False)

print('subtracting baseline')
s.subtract_base_line_all(plts=True)

print('aligning spectra')
s.align_spectra(plts=False, max_time_offset=None)

# s.set_rt_window_all(window=(5, np.infty))

print('setting summed spectrum')
s.set_summed()

print('setting peaks')
# s.set_peaks(prominence=1e-3)
s.set_peaks(prominence=1e-2)

# thr 100 - 200 mV

print('setting kernels')
s.set_kernels()

# print('binning spectra')
# s.bin_spectrum()
#
# df = s.get_dataframe()
#
# s.set_reconstruction_losses(idxs=None, plts=True)

gc_irms = GC_IRMS(list_path_files=files, spectra=s)



