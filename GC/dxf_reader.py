from .helper import get_r_home

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# specify the R installation folder here (required by rpy2 package)
try:
    R_HOME = get_r_home()
except EnvironmentError as e:
    R_HOME = r"C:\Program Files\R\R-4.3.2"  # your installation path here
os.environ["R_HOME"] = R_HOME  # adding R_HOME folder to environment parameters
os.environ["PATH"] = R_HOME + ";" + os.environ["PATH"]  # and to system path

try:
    from rpy2.robjects.packages import importr, isinstalled
    from rpy2.robjects.vectors import ListVector, DataFrame, StrVector, IntVector, FloatVector, BoolVector
    import rpy2.robjects.vectors as rvec

    # install package if not found
    if not isinstalled("isoreader"):
        utils = importr("utils")
        utils.install_packages("isoreader")

    # import package
    isoreader = importr('isoreader')
except Exception as e:
    isoreader = ImportError(e)
    ListVector, DataFrame, StrVector, IntVector, FloatVector, BoolVector = ImportError(e), ImportError(e), ImportError(
        e), ImportError(e), ImportError(e), ImportError(e)


def get_dxf(path_file) -> ListVector:
    return isoreader.iso_read_continuous_flow(path_file)


def convert(data: ListVector) -> dict[str, dict | pd.DataFrame | np.ndarray]:
    """Convert rvec ListVector to nested dict of pd data frames and numpy vectors."""
    objs: dict[str, dict | pd.DataFrame | np.ndarray] = {}

    # convert to numpy and pandas
    for k, e in data.items():
        if isinstance(e, DataFrame):
            df = pd.DataFrame(e).T
            df.columns = e.colnames
            df.index = e.rownames
            objs[k] = df
        elif isinstance(e, StrVector | IntVector | FloatVector | BoolVector):
            vec = np.array(e)
            if len(vec) == 1:
                vec = vec[0]
            objs[k] = vec
        elif isinstance(e, ListVector):  # start recursion
            objs[k] = convert(e)
        else:
            print(f'no conversion for type {type(e)} implemented')

    return objs


class DXF:
    def __init__(self, path_file: str):
        """Use isoverse to read file."""
        data: ListVector = isoreader.iso_read_continuous_flow(path_file)
        data_py: dict = convert(data)
        self.__dict__ |= data_py

        self.data = pd.DataFrame({
            '44': self.v44,
            '45': self.v45,
            '46': self.v46,
            'time': self.time
        })

    @property
    def time(self) -> np.ndarray:
        return self.raw_data['time.s'].to_numpy()

    @property
    def v44(self) -> np.ndarray:
        return self.raw_data['v44.mV'].to_numpy()

    @property
    def v45(self) -> np.ndarray:
        return self.raw_data['v45.mV'].to_numpy()

    @property
    def v46(self) -> np.ndarray:
        return self.raw_data['v46.mV'].to_numpy()

    def plot(self):
        plt.plot(self.time, self.v44 / self.v44.max(), label='mz 44')
        plt.plot(self.time, self.v45 / self.v45.max(), label='mz 45')
        plt.plot(self.time, self.v46 / self.v46.max(), label='mz 46')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative intensity')
        plt.legend()
        plt.title('normalized signals')
        plt.show()
