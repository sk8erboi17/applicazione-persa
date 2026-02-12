cmake_minimum_required(VERSION 3.20)
project(latex_ocr_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

option(LATEX_OCR_METAL "Enable Metal GPU backend" ON)
option(LATEX_OCR_ACCELERATE "Enable Apple Accelerate framework" ON)

# ggml as submodule
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/ggml/CMakeLists.txt")
    set(GGML_METAL ${LATEX_OCR_METAL} CACHE BOOL "" FORCE)
    set(GGML_ACCELERATE ${LATEX_OCR_ACCELERATE} CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/ggml)
else()
    message(STATUS "ggml not found as submodule, using find_package")
    find_package(ggml REQUIRED)
endif()

# stb_image (header-only)
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${CMAKE_SOURCE_DIR}/third_party)

# cJSON
add_library(cjson STATIC third_party/cJSON.c)
target_include_directories(cjson PUBLIC ${CMAKE_SOURCE_DIR}/third_party)

# Main library
add_library(latex_ocr_lib STATIC
    src/tokenizer.cpp
    src/image_preprocess.cpp
    src/encoder.cpp
    src/decoder.cpp
    src/model.cpp
    src/latex_ocr.cpp
)

target_include_directories(latex_ocr_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_link_libraries(latex_ocr_lib PUBLIC
    ggml
    cjson
    stb_image
)

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(FOUNDATION_FRAMEWORK Foundation)

    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(latex_ocr_lib PUBLIC ${ACCELERATE_FRAMEWORK})
    endif()
    if(METAL_FRAMEWORK AND LATEX_OCR_METAL)
        target_link_libraries(latex_ocr_lib PUBLIC
            ${METAL_FRAMEWORK}
            ${METALKIT_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
        )
        target_compile_definitions(latex_ocr_lib PUBLIC LATEX_OCR_USE_METAL)
    endif()
endif()

target_compile_options(latex_ocr_lib PRIVATE
    -Wall -Wextra -O2 -march=native
)

# Main executable
add_executable(latex_ocr src/main.cpp)
target_link_libraries(latex_ocr PRIVATE latex_ocr_lib)
target_compile_options(latex_ocr PRIVATE -Wall -Wextra -O2 -march=native)

# Install
install(TARGETS latex_ocr RUNTIME DESTINATION bin)
install(FILES include/latex_ocr.h DESTINATION include)
