from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from fastgrab import metadata

module_info = Extension(
    "fastgrab._linux_x11",
    include_dirs=[],  # Populated by build_ext().
    libraries=['X11', 'gomp'],
    extra_compile_args=[
        '-fno-strict-aliasing',
        # '-fopenmp',
        '-std=c11',
        '-mtune=native'
    ],
    sources=["fastgrab/linux_x11/screenshot.c"]
)

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it's still in its setup process.
        # NOTE: Doesn't exist in modern numpy versions.
        if "__NUMPY_SETUP__" in __builtins__:
            __builtins__.__NUMPY_SETUP__ = False
        # Add the numpy header directory to our build process.
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
    name=metadata.package,
    version=metadata.version,
    description=metadata.description,
    author=metadata.authors,
    url=metadata.url,
    packages=[metadata.package],
    cmdclass={'build_ext':build_ext},
    setup_requires=['numpy'],
    ext_modules=[module_info]
)
