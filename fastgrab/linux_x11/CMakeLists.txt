cmake_minimum_required(VERSION 3.5)
project(c_extention)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

#find_package(PkgConfig REQUIRED)
#pkg_check_module(PYTHON python2)

#find_package(PythonLibs 2.7 EXACT REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
#find_package(PythonLibs 3 REQUIRED)

# python2
set(PYTHON_EXECUTABLE "/progs/prerequisites/bin/python2.7")
set(PYTHON_LIBRARIES "/progs/prerequisites/lib/libpython2.7.so")
set(PYTHON_INCLUDE_DIRS "/progs/prerequisites/include/python2.7")
set(SOURCE_FILES define_new_type_python_2/exmodmodule.c)

# python3
#set(PYTHON_EXECUTABLE "/progs/prerequisites/bin/python3.5m")
#set(PYTHON_LIBRARIES "/progs/prerequisites/lib/libpython3.5m.so")
#set(PYTHON_INCLUDE_DIRS "/progs/prerequisites/include/python3.5m")

set(SOURCE_FILES screenshot.c)

#pkg_search_module(PYTHON python-2.7)
#
include_directories(${PYTHON_INCLUDE_DIRS})


add_executable(c_extention ${SOURCE_FILES})