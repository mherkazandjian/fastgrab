# -*- coding: utf-8 -*-
import unittest
import py.test


class TestMiscellaneous(unittest.TestCase):

    def test_that_scipy_can_be_imported(self):
        import scipy

    def test_that_pylab_can_be_imported(self):
        import pylab

    def test_that_shared_memory_directory_can_be_written_to(self):

        with open('/dev/shm/test.out', 'wb') as f_obj:
            pass

if __name__ == '__main__':
    unittest.main()
