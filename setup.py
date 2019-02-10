from setuptools import setup, Extension
import numpy
from fastgrab import metadata

module_info = Extension(
    "fastgrab._linux_x11",
    include_dirs=[numpy.get_include()],
    libraries=['X11', 'gomp'],
    extra_compile_args=[
        '-fno-strict-aliasing',
        '-fopenmp',
        '-std=c11'
    ],
    sources=["fastgrab/linux_x11/screenshot.c"]
)

setup(
    name=metadata.package,
    version=metadata.version,
    description=metadata.description,
    author=metadata.authors,
    url=metadata.url,
    packages=[metadata.package],
    ext_modules=[module_info]
)
