# Top-level CMakeLists for quik_bind Python module via pybind11 and scikit-build-core
cmake_minimum_required(VERSION 3.15...3.31)

# Require C++14 for Eigen compatibility
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Locate the quik library sources and include directories
add_subdirectory(thirdparty/quik EXCLUDE_FROM_ALL)

# Make Eigen header path available to binding code
get_target_property(EIGEN_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${eigen_SOURCE_DIR})

# Include quik headers
include_directories(thirdparty/quik/include)

python_add_library(quik_bind MODULE cpp/quik_bind/binding.cpp WITH_SOABI)
target_link_libraries(quik_bind PRIVATE pybind11::headers quik Eigen3::Eigen)

target_compile_definitions(quik_bind PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Install the Python module
install(TARGETS quik_bind DESTINATION quik_bind)

# Create and install the __init__.py file
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/quik_bind/__init__.py" "from .quik_bind import init_robot, fkn, ik\n\n__all__ = [\"init_robot\", \"fkn\", \"ik\"]\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/quik_bind/__init__.py" DESTINATION quik_bind)
