"""
module that provide utility classes and functions
"""
import time
import logging
import datetime

__all__ = ['logger']

logger = logging.getLogger("fastgrab")
logger.setLevel(logging.DEBUG)

def formatTime(self, record, datefmt=None):
    ct = self.converter(record.created)
    if datefmt:
        s = time.strftime(datefmt, ct)
    else:
        t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        s = "%s %03d" % (t, record.msecs)

class MyFormatter(logging.Formatter):

    converter = datetime.datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):

        ct = self.converter(record.created)

        if datefmt:

            s = ct.strftime(datefmt)

        else:

            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)

        return s


console = logging.StreamHandler()
formatter = MyFormatter(fmt='%(asctime)s-%(name)s-%(levelname)-8s] - %(message)s', datefmt='%H:%M:%S.%f')
console.setFormatter(formatter)
logger.addHandler(console)

'''
logger.debug("aaaaaaa")
logger.info("bbbbbbb")
logger.error("ccccccc")
logger.warn("wwwwwww")
logger.fatal("ttttttt")
'''
