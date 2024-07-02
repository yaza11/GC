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

# files = [r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_375__AA_Inc7_2-500.dxf"]

files = [
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240617_Lennart_371__AA_Inc4_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_372__AA_Inc5_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_373__AA_Inc6_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_374__AA_Inc7_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_375__AA_Inc7_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_376__AA_Inc8_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_377__AA_Inc9_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_378__AA_Inc10_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_379__AA_Inc10_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_380__AA_Inc11_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240617_Lennart_366__AA_Inc1_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_381__AA_Inc12_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240618_Lennart_382__AA_Inc13_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240617_Lennart_367__AA_Inc1_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240617_Lennart_368__AA_Inc2_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240617_Lennart_369__AA_Inc3_2-500.dxf",
    r"C:\Users\Yannick Zander\Downloads\GC-ICRMS\AA\240617_Lennart_370__AA_Inc4_2-500.dxf"
]

# spec = Spectrum(path_file=files[0])
# spec.get_standard_areas(plts=True)

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

print('binning spectra')
s.bin_spectrum()

df = s.get_dataframe()

s.set_reconstruction_losses(idxs=None, plts=True)

gc_irms = GC_IRMS(list_path_files=files, spectra=s)

df = gc_irms.d13.copy()
df44 = gc_irms.ft44.copy()
df45 = gc_irms.ft45.copy()
df46 = gc_irms.ft46.copy()

df.loc[:, 'rts'] = s.rts[s.peaks] * 60
df44.loc[:, 'rts'] = s.rts[s.peaks] * 60
df45.loc[:, 'rts'] = s.rts[s.peaks] * 60
df46.loc[:, 'rts'] = s.rts[s.peaks] * 60

df.to_excel('AA_table.xlsx')
df44.to_excel('AA_table_area44.xlsx')
df45.to_excel('AA_table_area45.xlsx')
df46.to_excel('AA_table_area46.xlsx')
