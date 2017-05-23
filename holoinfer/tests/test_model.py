"""
Test functions for the generattive model part of the package
"""
from unittest import TestCase
import numpy as np

from holoinfer.model import hologram, math


class TestHolograms(TestCase):
    def test_rayleigh_gans_holo(self):
        """
        Generates a hologram from set of parameters and checks if it is an array with the correct number of points
        :return: 
        """
        resolution = 32
        x = y = np.linspace(0, 10, resolution)
        grid = math.cartesian(x, y, 0)

        illum_wavelength = 1.0
        illum_wavenumber = 2 * np.pi / illum_wavelength
        sc_center = np.array([3, 7, 5])
        sc_index = np.array([1.15, 0])
        sc_radius = illum_wavelength / 10

        holo = hologram.rayleigh_gans_sphere_holo(grid, illum_wavenumber, sc_center, sc_index, sc_radius)

        assert holo.size == resolution*resolution


if __name__ == '__main__':
    from unittest import main as runtests
    runtests()