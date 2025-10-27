
K: float = 9.2e-3  # +/-.18
a: float = .516

# from dxf reference ratios
R13_STANDARD: float = 0.01118  # VPDB
R17_STANDARD: float = 0.00038  # VSMOW (redundant, can be calculated from K * R13_STANDARD ** a)
R18_STANDARD: float = 0.002005  # VSMOW

R2STD: dict[int, float] = {
    13: R13_STANDARD,
    17: R17_STANDARD,
    18: R18_STANDARD,
}

DELTA_C13_INJECTION_STANDARD: float = -33.4
DELTA_O18_INJECTION_STANDARD: float = 0.

# DELTA_C13_INJECTION_STANDARD = -43.411
# DELTA_O18_INJECTION_STANDARD = -21.097

k: dict[str, float] = {'S13': R13_STANDARD, 'S18': R18_STANDARD, 'K': K, 'A': a}

"""Calculate the value of the R45 standard from R18 and R13 standards."""
R13_theo: float = (DELTA_C13_INJECTION_STANDARD / 1000 + 1) * R13_STANDARD
R18_theo: float = (DELTA_O18_INJECTION_STANDARD / 1000 + 1) * R18_STANDARD
R17_theo: float = K * R18_theo ** a
R45_theo: float = R13_theo + 2 * R17_theo
R46_theo: float = 2 * R18_theo + 2 * R13_theo * R17_theo + R17_theo ** 2
