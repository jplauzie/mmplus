import numpy as np
from mumaxplus import Ferromagnet, Grid, World

class TestThermalSeed:
    @staticmethod
    def generate_noise(seed=None):
        world = World((1e-9, 1e-9, 1e-9))
        magnet = Ferromagnet(world, Grid((16, 16, 4)))
        magnet.msat = 1e3
        magnet.alpha = 1
        magnet.temperature = 10
        if seed is not None:
            magnet.thermal_seed = seed
        return magnet.thermal_noise()

    def test_default_seed(self):
        noise1 = self.generate_noise()
        noise2 = self.generate_noise()
        assert not np.allclose(noise1, noise2)

    def test_same_set_seed(self):
        noise1 = self.generate_noise(1234567)
        noise2 = self.generate_noise(1234567)
        assert np.allclose(noise1, noise2)

    def test_different_set_seed(self):
        noise1 = self.generate_noise(1234567)
        noise2 = self.generate_noise(7654321)
        assert not np.allclose(noise1, noise2)