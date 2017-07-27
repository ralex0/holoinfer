import numpy as np
from holoinfer.model.math import dist


def rayleigh_gans_sphere_holo(position, k, center, m, a):
    r = (position - center).T
    r = dist(r[0], r[1], r[2])

    z = center[2]
    theta = np.arccos(z / r)

    V = 4 / 3 * np.pi * a ** 3

    x = k * a
    u = 2 * x * np.sin(.5 * theta)
    G = 3 / u ** 3 * (np.sin(u) - u * np.cos(u))

    re_m, im_m = m[0], m[1]

    s2_mag = k ** 3 * a ** 3 * np.sqrt(re_m * re_m + im_m * im_m - 2 * re_m + 1) * 2 / 3 * np.abs(np.cos(theta)) * g
    s2_phase = np.arctan((1 - re_m) / im_m) if im_m != 0 else np.pi / 2

    holo_scatt = (1 / (k * r) * s2_mag) ** 2
    holo_inter = 2 * 1/(k*r) * np.sqrt(s2_mag**2) * np.sin(k*(r-z) + s2_phase)

    holo_full = holo_scatt + holo_inter + 1

    return holo_full

