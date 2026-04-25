"""Poetry build hook that wires the libX11 C extension into the wheel."""
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        import numpy
        self.include_dirs.append(numpy.get_include())


def build(setup_kwargs):
    setup_kwargs.update({
        "ext_modules": [
            Extension(
                "fastgrab._linux_x11",
                sources=["fastgrab/linux_x11/screenshot.c"],
                libraries=["X11", "gomp"],
                extra_compile_args=[
                    "-fno-strict-aliasing",
                    "-std=c11",
                    "-mtune=native",
                ],
            ),
        ],
        "cmdclass": {"build_ext": build_ext},
    })
