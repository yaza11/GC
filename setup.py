from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Analyse GC FID and IRMS data (dxf and txt files)'
LONG_DESCRIPTION = 'A package that allows integration of peaks in GC FID and IRMS data.'

# Setting up
setup(
    name="GC",
    version=VERSION,
    author="Yannick Zander",
    author_email="yzander@marum.de",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/yaza11/pyGC_FID_processing",
    packages=find_packages(),
    install_requires=['matplotlib', 'numpy', 'pandas', 'rpy2', 'scipy', 'tqdm', 'openpyxl', 'chem-spectra'],
    extras_require={'dev': 'twine'},
    keywords=['python', 'GC', 'FID', 'IRMS', 'dxf'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)