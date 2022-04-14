cmake_minimum_required(VERSION 3.12)
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

project(tiny_opencv C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")

add_library(
	tiny_opencv
	${CMAKE_CURRENT_LIST_DIR}/include/tiny_opencv.hpp
	${CMAKE_CURRENT_LIST_DIR}/include/tiny_types.hpp
	${CMAKE_CURRENT_LIST_DIR}/include/kalman_filter.hpp
	${CMAKE_CURRENT_LIST_DIR}/include/hungarian.hpp
	${CMAKE_CURRENT_LIST_DIR}/src/filter/threshold.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/filter/blur.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/filter/bilateral_filter.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/filter/box_filter.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/filter/median_filter.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/filter/gaussian_blur.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/filter/filter_2d.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/filter/equalize_hist.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/filter/clear_boundary.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/algorithm/kalman_filter.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/algorithm/hungarian.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/algorithm/get_rotation_matrix_2d.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/utils/split.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/utils/merge.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/cvtcolor/rgb2yuv.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/cvtcolor/yuv2rgb.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/cvtcolor/rgb2bgr.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/cvtcolor/rgb2hsv.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/cvtcolor/cvtcolor.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/mat.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/opencv.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/resize.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/randn.cpp
	${CMAKE_CURRENT_LIST_DIR}/src/3rd_party/quirc/quirc.h
	${CMAKE_CURRENT_LIST_DIR}/src/3rd_party/quirc/quirc_internal.h
	${CMAKE_CURRENT_LIST_DIR}/src/3rd_party/quirc/decode.c
	${CMAKE_CURRENT_LIST_DIR}/src/3rd_party/quirc/identify.c
	${CMAKE_CURRENT_LIST_DIR}/src/3rd_party/quirc/quirc.c
	${CMAKE_CURRENT_LIST_DIR}/src/3rd_party/quirc/version_db.c
)

target_include_directories(
	tiny_opencv
	PUBLIC
	${CMAKE_CURRENT_LIST_DIR}/include
)

target_compile_options(
	tiny_opencv
	PUBLIC
#	-fno-exceptions
)

target_link_libraries(
	tiny_opencv
)

# build the test program
add_executable(
	tiny_opencv_test
	${CMAKE_CURRENT_LIST_DIR}/test/unit_test.hpp
	${CMAKE_CURRENT_LIST_DIR}/test/unit_test_cvtcolor.cpp
	${CMAKE_CURRENT_LIST_DIR}/test/unit_test_filter.cpp
	${CMAKE_CURRENT_LIST_DIR}/test/unit_test_matrix.cpp
	${CMAKE_CURRENT_LIST_DIR}/test/unit_test_kalman.cpp
	${CMAKE_CURRENT_LIST_DIR}/test/util.cpp
	${CMAKE_CURRENT_LIST_DIR}/test/test.cpp
)

find_package(OpenCV REQUIRED)

target_include_directories(
	tiny_opencv_test
	PUBLIC
	${CMAKE_CURRENT_LIST_DIR}/include
	${OpenCV_INCLUDE_DIRS}
)

target_sources(
	tiny_opencv_test
	PUBLIC
)

target_link_libraries(
	tiny_opencv_test
	PUBLIC
	tiny_opencv
	${OpenCV_LIBS}
)
