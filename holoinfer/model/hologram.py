import numpy as np
from holoinfer.model.math import dist


def rayleigh_gans_sphere_holo(position, k, center, m, a):
    r = (position - center).T
    r = dist(r[0], r[1], r[2])

    z = center[2]
    theta = np.arccos(z / r)

    x = k * a
    u = 2 * x * np.sin(.5 * theta)
    g = 3 / u ** 3 * (np.sin(u) - u * np.cos(u))

    re_m, im_m = m[0], m[1]

    s2_mag = k ** 3 * a ** 3 * np.sqrt(re_m * re_m + im_m * im_m - 2 * re_m + 1) * 2 / 3 * np.cos(theta) * g
    s2_phase = np.arctan((re_m - 1) / im_m)

    holo_scatt = (1 / (k * r) * s2_mag) ** 2
    holo_inter = 2 * 1 / (k * r) * s2_mag * np.sin(k * (r - z) + s2_phase)

    return holo_scatt + holo_inter
