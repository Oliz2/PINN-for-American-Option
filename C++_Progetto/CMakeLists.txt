# Build of pinn.exe (PINN for American options, C++ / LibTorch).
# Author: Alberto Oliva Medin
#
# Requirements: CMake >= 3.18, a C++20 compiler (MSVC on Windows) and LibTorch (CPU) unpacked in C:/libtorch.
# Build:   cmake -S . -B build   then   cmake --build build --config Release
cmake_minimum_required(VERSION 3.18)
project(pinn_american_option)

# Folder where LibTorch was unpacked: change it if LibTorch is somewhere else
set(CMAKE_PREFIX_PATH "C:/libtorch")
find_package(Torch REQUIRED)

# All the .cpp files under src/ (src/main = library, src/test = main)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(pinn ${SOURCES})
target_include_directories(pinn PRIVATE "${CMAKE_SOURCE_DIR}/src/main")
target_link_libraries(pinn "${TORCH_LIBRARIES}")
set_property(TARGET pinn PROPERTY CXX_STANDARD 20)
set_property(TARGET pinn PROPERTY CXX_STANDARD_REQUIRED ON)

# Windows: copy the LibTorch DLLs next to pinn.exe, so that it runs without changing the PATH
if (MSVC)
  target_compile_options(pinn PRIVATE /wd4267 /wd4244 /wd4251)  # hides the LibTorch warnings
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pinn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:pinn>)
endif()