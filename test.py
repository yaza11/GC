# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from cSpectrum import Spectra

xcol = 'RT'
ycol = 'counts'

files = [
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_21.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_22.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_23.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_24.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_25.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_26.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_27.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_28.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_29.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_30.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_31.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_32.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_34.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_35.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_36.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_37.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_40.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_41.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_42.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_43.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_44.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_45.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_46.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_47.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_48.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_49.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_50.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_51.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_52.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_53.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_54.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_55.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_56.RAW_.txt",
    "C:/Users/yanni/Downloads/GC-convert/GC-FID_Data/LennartStock_2025_57.RAW_.txt"
][:3]

# initiate Spectra object with a list of files 
s = Spectra(list_path_files=files)
# plotting the summed spectrum can be used at any point
s.plt_summed(plt_all=False)
# all keyword arguments from here on are optional, values chosen here are 
# the default values
# smooth the signals, remove outliers, bigger window_size gives smoother signals
s.remove_outliers_all(window_length=101, diff_threshold=.001, plts=False)
# remove the baseline from each spectrum by subtracting result from a minim filter
s.subtract_base_line_all(window_size_portion=1/50, plts=False)
# align all spectra to the first one, determined as the lag at maximum crosscorrelation
s.align_spectra(plts=False, max_time_offset=None)
# cut of injection peak after alignment
s.set_rt_window_all(window=(5, np.infty))
s.plt_summed()
# sum up the spectra
s.set_summed()
# search peaks in summed up spectrum
s.set_peaks(prominence=1e-3)
# search kernels as bigaussians in summed spectrum
s.set_kernels()
# integrate area under peaks by multiplying kernels with spectra
s.bin_spectrum()
# obtain the dataframe of binned spectra where columns represent the spectra and
# rows the picked times
# df = s.get_dataframe()
# save binned spectra to disc as csv or excel
# df.to_csv('your/path/here.csv')
# or
# df.to_excel('your/path/here.xlsx')

s.set_reconstruction_losses(plts=True, ylim=(0, 2e7))


