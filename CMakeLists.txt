cmake_minimum_required(VERSION 3.0.0)
project("calicam" VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wpedantic -Wextra -fexceptions)

add_executable(calicam calicam.cpp)
target_include_directories(calicam PUBLIC ${PROJECT_SOURCE_DIR}/include)
