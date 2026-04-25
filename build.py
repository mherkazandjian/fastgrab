"""Poetry build hook + standalone in-place builder for the libX11 C extension.

When invoked by Poetry / pip during wheel build, ``build(setup_kwargs)`` is
called and injects the C extension into the setup kwargs.

When invoked as a script (``python build.py``), it runs setuptools'
``build_ext --inplace`` so the resulting ``.so`` lands next to the Python
source. The dev/test container's entrypoint uses this path because
poetry-core's PEP 660 editable install does not place compiled extensions
in-tree on its own.
"""
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        import numpy
        self.include_dirs.append(numpy.get_include())


EXT_MODULES = [
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
]


def build(setup_kwargs):
    setup_kwargs.update({
        "ext_modules": EXT_MODULES,
        "cmdclass": {"build_ext": build_ext},
    })


if __name__ == "__main__":
    setup(
        name="fastgrab",
        ext_modules=EXT_MODULES,
        cmdclass={"build_ext": build_ext},
        script_args=["build_ext", "--inplace"],
    )
