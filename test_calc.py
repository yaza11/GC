# test data from santrock paper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from gc_irms import (get_deltas, find_R18, K, a, R13_STANDARD, R18_STANDARD,
                     DELTA_C13_INJECTION_STANDARD, DELTA_O18_INJECTION_STANDARD,
                     R_theo, delta13c_santrock
                     )


def test_R18(R45, R46):
    for i in range(R46.shape[0]):
        R45_ = R45.iat[i, 0]
        R46_ = R46.iat[i, 0]
    
        R18reg = find_R18(R45_, R46_).root

        R18 = np.linspace(0, .02, 1000)
        
        eq = (
            - 3 * K ** 2 * R18 ** (2 * a)
            + 2 * K * R45_ * R18 ** a
            + 2 * R18
            - R46_
        )
        
        plt.plot(R18, eq)
        plt.vlines(R18reg, eq.min(), eq.max(), colors='r')
        plt.grid(True)
        plt.show()


k = {'S13': R13_STANDARD, 'S18': R18_STANDARD, 'K': K, 'A': a}

r45sam, r46sam = 1.1399413, 1.5258049
# d13cstd = -43.411
# d18ostd = -21.097


r45std, r46std = 1.1331231, 1.4058630


d13c = delta13c_santrock(
    r45sam,
    r46sam,
    DELTA_C13_INJECTION_STANDARD,
    r45std,
    r46std,
    DELTA_O18_INJECTION_STANDARD
)
d13c2 = get_deltas(1, r45sam, r46sam, r45std, r46std)

print(d13c, d13c2)

# test_R18(R45, R46)
