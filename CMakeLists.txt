cmake_minimum_required(VERSION 3.28.3)

project(_inferX)

set(PYTHON_PACKAGE_NAME inferX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

file(GLOB_RECURSE 
    SOURCE_FILES 
    ${CMAKE_CURRENT_LIST_DIR}/csrc/*.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/include/*.h
)

# Config mode looks in paths defined in: CMAKE_PREFIX_PATH set by the upstream scikit-build-core
# If running manually we'll have to set that up.
find_package(pybind11 CONFIG REQUIRED)

# We don't need target_link_libraries() this handles that
pybind11_add_module(${PROJECT_NAME} ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")

install(TARGETS ${PROJECT_NAME} DESTINATION ${PYTHON_PACKAGE_NAME})
