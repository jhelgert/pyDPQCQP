FIND_PACKAGE(PythonInterp 3)
FIND_PACKAGE(PythonLibs 3)

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/v2.6.2.tar.gz
)

FetchContent_GetProperties(pybind11)

if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()