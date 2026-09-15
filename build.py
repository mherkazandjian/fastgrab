"""Poetry build hook + standalone in-place builder for the libX11 C extension.

When invoked by Poetry / pip during wheel build, ``build(setup_kwargs)`` is
called and injects the C extension into the setup kwargs — but only on
Linux. Windows and macOS get a pure-Python wheel because their backends
(``fastgrab.backends.windows`` / ``fastgrab.backends.macos``) are pure
ctypes against the OS-native APIs and have no C-side build step.

When invoked as a script (``python build.py``), it runs setuptools'
``build_ext --inplace`` so the resulting ``.so`` lands next to the Python
source. The dev/test container's entrypoint uses this path because
poetry-core's PEP 660 editable install does not place compiled extensions
in-tree on its own. On non-Linux hosts the script is a no-op.
"""
import os
import shutil
import sys
import tempfile

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


# Asked of the compiler rather than assumed. -fopenmp is not a universal
# spelling: it selects libgomp under gcc and libomp under clang, and a
# machine with one runtime but not the other cannot build with it at all.
# Measured on Debian 13 with clang 19 and no libomp packages, the build
# does not even reach the linker -- omp.h is missing and compilation
# fails. This project ships an sdist, so every Linux user compiles
# locally on whatever toolchain they happen to have, and install failures
# are already the largest category of issues against it. A serial build
# that works beats a parallel build that will not compile.
OPENMP_COMPILE_ARGS = ["-fopenmp"]
OPENMP_LINK_ARGS = ["-fopenmp"]

_OPENMP_PROBE = """\
#include <omp.h>
#ifndef _OPENMP
#error "the compiler accepted -fopenmp without defining _OPENMP"
#endif
int fastgrab_probe(void);
int fastgrab_probe(void)
{
    int total = 0;
    int i;
#pragma omp parallel for reduction(+:total)
    for (i = 0; i < 64; i++)
        total += i;
    return total + omp_get_max_threads();
}
"""


class build_ext(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        import numpy
        self.include_dirs.append(numpy.get_include())

    def build_extensions(self):
        """Add the OpenMP flags only if this toolchain can honour them."""
        if self._openmp_works():
            for ext in self.extensions:
                ext.extra_compile_args = (
                    list(ext.extra_compile_args) + OPENMP_COMPILE_ARGS)
                ext.extra_link_args = (
                    list(ext.extra_link_args) + OPENMP_LINK_ARGS)
        else:
            sys.stderr.write(
                "fastgrab: this toolchain cannot build with OpenMP, so the "
                "row copy will be serial. Capture still works, and still "
                "uses MIT-SHM; large frames are copied on one core. Install "
                "an OpenMP runtime for the compiler in use (libgomp for "
                "gcc, libomp for clang) and reinstall to get it back.\n")
        super().build_extensions()

    def _openmp_works(self):
        """Compile *and link* a real parallel region with this compiler.

        Through self.compiler rather than by inspecting $CC: setuptools
        has already merged the environment, the interpreter's own build
        flags and any wrapper such as ccache into that object, and
        second-guessing it is how a probe ends up testing a different
        compiler than the one that builds the extension.

        Linked as a shared object, not an executable: linker_exe and
        linker_so are configured separately -- an executable probe
        passes while the real link fails under, say, CC=clang with
        LDSHARED="gcc -shared". -Wl,-z,defs turns an unresolved runtime
        symbol into a link error here; it must not reach the extension
        itself, which legitimately leaves Python's symbols to the
        interpreter that loads it.
        """
        probe_dir = tempfile.mkdtemp(prefix="fastgrab-openmp-probe-")
        try:
            source = os.path.join(probe_dir, "probe.c")
            with open(source, "w") as handle:
                handle.write(_OPENMP_PROBE)
            objects = self.compiler.compile(
                [source], output_dir=probe_dir,
                extra_postargs=OPENMP_COMPILE_ARGS)
            self.compiler.link_shared_object(
                objects, os.path.join(probe_dir, "probe.so"),
                extra_postargs=OPENMP_LINK_ARGS + ["-Wl,-z,defs"])
        except Exception:
            # Broad on purpose, and safe because nothing outside this
            # temporary directory is involved: any failure at all means
            # this toolchain cannot give us OpenMP. A real failure to
            # build the extension is raised by super().build_extensions()
            # and is not caught here.
            return False
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)
        return True


EXT_MODULES = [
    Extension(
        "fastgrab._linux_x11",
        sources=["fastgrab/linux_x11/screenshot.c"],
        # No "gomp" here. -fopenmp already tells each compiler driver to
        # link its own runtime, so naming gomp explicitly was redundant
        # under gcc and actively wrong under clang, which links libomp as
        # well: measured, that produces an extension bound to *both*
        # runtimes, with omp_get_max_threads() answering from libgomp
        # while the parallel regions execute through libomp.
        #
        # No "-mtune=native" either. It only tunes within the selected
        # instruction set on x86, so it buys little, and it is rejected
        # outright when cross-compiling -- measured, clang 19 fails with
        # "unsupported argument 'native'" for aarch64. Anyone who wants
        # it can pass it in CFLAGS.
        libraries=["X11", "Xext"],
        extra_compile_args=[
            "-fno-strict-aliasing",
            "-std=c11",
        ],
        extra_link_args=[],
    ),
]


def build(setup_kwargs):
    if sys.platform != "linux":
        return  # pure-Python wheel on Windows / macOS
    setup_kwargs.update({
        "ext_modules": EXT_MODULES,
        "cmdclass": {"build_ext": build_ext},
    })


if __name__ == "__main__":
    if sys.platform != "linux":
        # No C extension to build on Windows / macOS — exit cleanly so the
        # docker entrypoint and CI scripts stay portable.
        raise SystemExit(0)
    setup(
        name="fastgrab",
        ext_modules=EXT_MODULES,
        cmdclass={"build_ext": build_ext},
        script_args=["build_ext", "--inplace"],
    )
