"""
<keywords>
test, python
</keywords>
<description>
</description>
<seealso>
</seealso>
"""
import threading


class RecorderBase(threading.Thread):
    """

    """
    def __init__(self):
        """

        """
        self._backend = None
        self._buffer = None

    def setup(self):
        """

        :return:
        """
        raise NotImplementedError('''this must be implemented''')

    def record(self):
        """

        :return:
        """
        raise NotImplementedError('''this must be implemented''')

    def pause_record(self):
        """

        :return:
        """
        raise NotImplementedError('''this must be implemented''')

    def resume_record(self):
        """

        :return:
        """
        raise NotImplementedError('''this must be implemented''')

    def stop_record(self):
        """

        :return:
        """
        raise NotImplementedError('''this must be implemented''')

    def capture_frame(self):
        """

        :return:
        """
        raise NotImplementedError('''this must be implemented''')

    def save(self):
        """

        :return:
        """
        raise NotImplementedError('''this must be implemented''')


class Recorder(RecorderBase):
    """

    """
    def __init__(self):
        """

        """
        pass


