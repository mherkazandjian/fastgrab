# -*- coding: utf-8 -*-
import unittest
import py.test
import numpy
from numpy.testing import assert_array_equal

from fastgrab.dataio import FrameBuffer


class TestFrameBuffer(unittest.TestCase):

    dtype = [('timestamp', 'S27'), ('frame', '(3, 20, 20)i4')]

    def test_that_frame_buffer_can_be_created(self):
        _ = FrameBuffer()

    def test_that_frame_buffer_is_initialized_correctly(self):
        fb = FrameBuffer(dtype=self.dtype)
        assert fb.data.size == 1
        assert fb.data['frame'][0].shape == (3, 20, 20)
        #     Sat Jun 25 02:49:45 2016
        # py.test.set_trace()

    def test_that_storage_doubling_can_be_done(self):
        fb = FrameBuffer(dtype=self.dtype)
        old_size = fb.data.size
        fb.double_storage()
        new_size = fb.data.size
        assert new_size == 2 * old_size
        assert fb.data[:]['frame'].shape == (2, 1, 3, 20, 20)

    def test_that_multiple_storage_doubling_is_done_correctly(self):
        fb = FrameBuffer(dtype=self.dtype)
        for i in range(10):
            fb.double_storage()
        assert fb.data.size == 1 << 10

    def test_that_setting_current_buffer_works(self):
        fb = FrameBuffer(dtype='3i4')
        fb.set_current_buffer([1, 2, 3])
        assert_array_equal(fb.data[0], [1, 2, 3])

    def test_that_data_can_be_appended(self):
        fb = FrameBuffer(dtype='3i4')
        fb.append([1, 2, 3])

    def test_that_data_can_be_appended_many_times(self):
        fb = FrameBuffer(dtype='3i4')
        data_rows = numpy.int32(numpy.random.rand(3*20).reshape(20, 3)*10)
        for row in data_rows:
            fb.append(row)
        assert data_rows.flatten().sum() == fb.data.flatten().sum()


if __name__ == '__main__':
    unittest.main()
