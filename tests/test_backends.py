# -*- coding: utf-8 -*-
import unittest
import pytest

from fastgrab.backends import (
    # BackendFactory,
    # AutopyBackend,
    LinuxX11Backend
)

class TestBackends(unittest.TestCase):

    # def test_that_autopy_backend_can_be_created(self):
    #     _ = AutopyBackend()

    # def test_that_autopy_availability_can_be_checked(self):
    #     backend = AutopyBackend()
    #     assert backend.is_available() in [True, False]

    # def test_that_autopy_backend_can_take_screenshot(self):
    #     backend = AutopyBackend()
    #     screenshot = backend.screenshot([0, 0, 100, 100])
    #     assert screenshot.shape == (100, 100, 3)

    # def test_that_osx_backend_can_be_created(self):
    #     pytest.skip('not implemented')

    def test_that_linux_x11_backend_can_be_created(self):
        backend = LinuxX11Backend()
        assert backend.is_available() in [True, False]

    def test_that_linux_x11_backend_can_take_screenshot(self):
        backend = LinuxX11Backend()
        screenshot = backend.screenshot(0, 0, 100, 100)
        assert screenshot.shape == (100, 100, 3)

    def test_that_vlc_x11_backend_can_be_created(self):
        py.test.skip('not implemented')


if __name__ == '__main__':
    unittest.main()
