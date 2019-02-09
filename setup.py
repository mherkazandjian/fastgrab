import os
from distutils.core import setup, Extension
import importlib.util
spec = importlib.util.spec_from_file_location(
    'metadata',
    os.path.join('fastgrab', 'metadata.py')
)
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)

module_info = Extension(
    "linux_x11",
    include_dirs=[],
    libraries=['X11', 'gomp'],
    extra_compile_args=['-fno-strict-aliasing', '-fopenmp', '-mavx2'],
    sources=["fastgrab/linux_x11/screenshot.c"]
)

setup(
    name=metadata.package,
    version=metadata.version,
    description=metadata.description,
    authors=metadata.authors,
    url=metadata.url,
    packages=[metadata.package],
    ext_modules=[module_info]
)
