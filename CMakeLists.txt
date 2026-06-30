cmake_minimum_required(VERSION 3.28.3)

project(_inferX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYTHON_PACKAGE_NAME inferX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/external")

include(FetchContent)

file(GLOB_RECURSE 
    SOURCE_FILES 
    ${CMAKE_CURRENT_LIST_DIR}/csrc/*.cpp 
    ${CMAKE_CURRENT_LIST_DIR}/include/*.h
)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03 # 3.11.3
    GIT_PROGRESS   TRUE
    USES_TERMINAL_DOWNLOAD TRUE 
)
FetchContent_MakeAvailable(json)

# Config mode looks in paths defined in: CMAKE_PREFIX_PATH set by the upstream scikit-build-core
# If running manually we'll have to set that up.
find_package(CURL REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# We don't need target_link_libraries() this handles that
pybind11_add_module(${PROJECT_NAME} ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} 
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include"
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/csrc"
)

target_link_libraries(${PROJECT_NAME} 
    PRIVATE CURL::libcurl
    PRIVATE nlohmann_json::nlohmann_json
)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME}
        PRIVATE
        -fsanitize=address
        -fno-omit-frame-pointer
        -g
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wunreachable-code -Wunused
    )

    target_link_options(${PROJECT_NAME}
        PRIVATE
        -fsanitize=address
    )
endif()


install(TARGETS ${PROJECT_NAME} DESTINATION ${PYTHON_PACKAGE_NAME})
