cmake_minimum_required(VERSION 3.0.0)
project(pybind_mwe VERSION 0.1.0)
include(FetchContent)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
#find_package(CPLEX REQUIRED)
include(Blaze)
#include(fmt)
include(nlohmann-json)
include(pybind11)

add_library(Helpers STATIC ${SOURCE_DIR}/Helpers.cpp ${SOURCE_DIR}/Helpers.hpp)
target_include_directories(Helpers PUBLIC ${blaze_SOURCE_DIR})

add_library(QCQP STATIC ${SOURCE_DIR}/QCQP.cpp ${SOURCE_DIR}/QCQP.hpp)
target_include_directories(QCQP PUBLIC ${blaze_SOURCE_DIR})
target_link_libraries(QCQP PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(QCQP PUBLIC ${BLAS_LIBRARIES})
target_link_libraries(QCQP PUBLIC ${LAPACK_LIBRARIES})

pybind11_add_module(_dpqcqp_cpp_wrapper ${SOURCE_DIR}/python_binding.cpp)
target_include_directories(_dpqcqp_cpp_wrapper PUBLIC ${blaze_SOURCE_DIR})
target_link_libraries(_dpqcqp_cpp_wrapper PUBLIC Helpers QCQP)
target_link_libraries(_dpqcqp_cpp_wrapper PUBLIC ${BLAS_LIBRARIES})
target_link_libraries(_dpqcqp_cpp_wrapper PUBLIC ${LAPACK_LIBRARIES})
target_link_libraries(_dpqcqp_cpp_wrapper PRIVATE nlohmann_json::nlohmann_json)
