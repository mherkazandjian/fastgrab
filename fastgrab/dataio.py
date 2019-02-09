"""
<keywords>
test, python
</keywords>
<description>
</description>
<seealso>
</seealso>
"""
import numpy


class DataBufferBase(object):
    """
    a data buffer interface that expands as data is put into it.
    """
    def __init__(self, *args, **kwargs):
        """
        constructor
        """
        self.data = None
        """The container of the data"""

        self._current_buffer_index = None
        """The index of the last record with data"""

        self.buffer_size = None
        """The size of the buffer. Note: this is larger than the current
        buffer index"""

    def init_data(self, *args, **kwargs):
        """
        initialize the values of self.data (creates on element) and
        set _current_buffer_index to zero (the zeroth element)

        :param args:
        :param kwargs:
        :return:
        """
        self.data = self._data_like_array(1, *args, **kwargs)
        self._current_buffer_index = 0
        self.buffer_size = 1

    def append(self, value):
        """
        Set value to the back of the buffer. Value is set to the element
        pointed to at self._current_buffer_index. If the buffer does not
        have enough space, the storage is doubled.

        .. code-block:: python

            fb.append(element)

        :param value: The value to be copied to the back of the buffer
        :return: self
        """

        if self._current_buffer_index == self.buffer_size:
            self.double_storage()

        self.set_current_buffer(value)
        self._current_buffer_index += 1
        return self

    def set_current_buffer(self, value):
        """
        set value to the element pointed to by self._current_buffer_index.
        In this method no storage doubling is done.
        :param value: The value to be copied to the back of the buffer
        :return: self
        """
        raise NotImplementedError('''must be implemented by subclass''')

    def save(self, path):
        """
        save the content of self.data attribute into a numpy .npz file up
        to the index where it is filled with data.

        .. code-block:: python

            data_buf.save('/path/to/file.npz')

        :param string path: the path where the data will be saved.
        :return: self
        """
        numpy.savez(path, data=self.data[0:self._current_buffer_index])
        return self

    def load(self, path):
        """
        load data in the specified file and set the data to self.data

        .. code-block:: python

            data_buf.load('/path/to/file.npz')

        :param path: the path of the file containing the data
        :return: self
        """
        self.data = numpy.load(path)['data']
        return self

    def double_storage(self):
        """
         double the storage of self.data
        :return: self
        """
        current_size = self.data.size
        extra_data = self._data_like_array(current_size)

        self.data = self._join_data_arrays(self.data,
                                           extra_data.reshape(self.data.shape))
        self.buffer_size *= 2
        return self

    def _data_like_array(self, size, *args, **kwargs):
        """

        :param size:
        :return:
        """
        raise NotImplementedError('''must be implemented by subclass''')

    def _join_data_arrays(self, array1, array2):
        """

        :param array1:
        :param array2:
        :return:
        """
        raise NotImplementedError('''must be implemented by subclass''')


class FrameBuffer(DataBufferBase):
    """

    """
    def __init__(self, dtype=None, *args, **kwargs):
        """
        constructor
        """
        super(FrameBuffer, self).__init__(*args, **kwargs)

        if dtype is not None:
            self.init_data(dtype=dtype, *args, **kwargs)

    def _data_like_array(self, size, dtype=None, *args, **kwargs):
        """

        :param size:
        :param dtype:
        :return:
        """
        if dtype is None:
            assert self.data is not None, 'can not determine dtype'
            dtype = self.data.dtype

        return numpy.zeros(size, dtype)


    def _join_data_arrays(self, array1, array2):
        """

        :param array1:
        :param array2:
        :return:
        """
        # if array1.size == array2.size:
        return numpy.vstack((array1, array2))


    def set_current_buffer(self, value):
        """

        :param value:
        :return:
        """
        self.data[self._current_buffer_index] = value

