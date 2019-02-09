cmake_minimum_required(VERSION 3.8)
project(c_extention)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

#find_package(PkgConfig REQUIRED)

# python3
set(PYTHON_EXECUTABLE "/home/mher/progs/sw/miniconda/envs/conda3/bin/python3")
set(PYTHON_LIBRARIES "/home/mher/progs/sw/miniconda/envs/conda3/lib/libpython3.7m.so")
set(PYTHON_INCLUDE_DIRS "/home/mher/progs/sw/miniconda/envs/conda3/include/python3.7m")


find_package(PythonLibs 3.7 REQUIRED)

include_directories(
        ${PYTHON_INCLUDE_DIRS}
        "/home/mher/progs/sw/miniconda/envs/conda3/lib/python3.7/site-packages/numpy/core/include")

set(SOURCE_FILES screenshot.c)

add_executable(c_extention ${SOURCE_FILES})
