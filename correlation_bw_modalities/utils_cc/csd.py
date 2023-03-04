import numpy as np


def inverse_csd(data, fs, spacing_um, conductivity=0.3, radius_um=None):
    """
    conductivity: S/m
    spacing: um
    Return units: <voltage>/m^3
    """
    n_chs = data.shape[0]
    spacing = spacing_um / 1E6
    if radius_um is None:
        radius = n_chs * spacing
    else:
        radius = radius_um / 1E6
    z = np.arange(n_chs)*spacing
    dmat = np.abs(z[:, None] - z[None, :])
    fmat = spacing/(2*conductivity)*(np.sqrt(dmat**2+radius**2)-dmat)
    return np.dot(np.linalg.inv(fmat), data)
