cmake_minimum_required(VERSION 3.18)
project(preproc_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find LibTorch ─────────────────────────────────────────────────────────────
# Expects CMAKE_PREFIX_PATH to point to the LibTorch install inside the active
# Python environment.  Set automatically by scripts/build_cpp.sh.
find_package(Torch REQUIRED)

# ── Binary ────────────────────────────────────────────────────────────────────
add_executable(preproc_runner preproc_runner/main.cpp)
target_link_libraries(preproc_runner "${TORCH_LIBRARIES}")
target_compile_options(preproc_runner PRIVATE ${TORCH_CXX_FLAGS})

# On macOS, rpath so the binary finds libtorch at runtime without setting
# DYLD_LIBRARY_PATH manually.
if(APPLE)
    set_target_properties(preproc_runner PROPERTIES
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )
endif()

message(STATUS "Torch version:    ${Torch_VERSION}")
message(STATUS "Torch libraries:  ${TORCH_LIBRARIES}")
message(STATUS "CXX flags:        ${TORCH_CXX_FLAGS}")
