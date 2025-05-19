cmake_minimum_required(VERSION 3.15)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Add subdirectory with static lib
add_subdirectory(thirdparty/quik EXCLUDE_FROM_ALL)

# Build the Python module with pybind11
python_add_library(quik_bind MODULE cpp/quik_bind/binding.cpp WITH_SOABI)
target_link_libraries(quik_bind
    PRIVATE
        pybind11::headers
        quik
)
# Quik headers already transitively included, but add explicitly if needed
# target_include_directories(quik_bind PRIVATE thirdparty/quik/include)

target_compile_definitions(quik_bind PRIVATE VERSION_INFO="${PROJECT_VERSION}")

# Install Python module
install(TARGETS quik_bind DESTINATION quik_bind)

# Generate and install __init__.py
set(_init_py "${CMAKE_CURRENT_BINARY_DIR}/quik_bind/__init__.py")
file(WRITE "${_init_py}" "from .quik_bind import init_robot, fkn, ik\n\n__all__ = [\"init_robot\", \"fkn\", \"ik\"]\n")
install(FILES "${_init_py}" DESTINATION quik_bind)
