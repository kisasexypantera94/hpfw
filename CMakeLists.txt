cmake_minimum_required(VERSION 3.11...3.16)

project(hpfw
        VERSION 0.0.1
        DESCRIPTION "Audio HashPrint framework"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hpfw INTERFACE)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_directories(hpfw INTERFACE /usr/local/lib)
target_include_directories(
        hpfw
        INTERFACE
        SYSTEM

        include/

        /usr/local/include
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/include

        thirdparty/cpp-taskflow-2.2.0
)

target_link_libraries(
        hpfw
        INTERFACE

        Eigen3::Eigen
        fftw3
        tbb
        essentia

        # MKL RELATED
        pthread
        m
        dl
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/lib/libmkl_intel_lp64.a
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/lib/libmkl_sequential.a
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/lib/libmkl_core.a
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/lib/libmkl_blacs_mpich_lp64.a
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/lib/libmkl_scalapack_ilp64.a
)
target_compile_definitions(
        hpfw
        INTERFACE

        -DUSE_FFTW
        -DEIGEN_USE_MKL_ALL
)

target_compile_options(
        hpfw
        INTERFACE

        -m64

        -ffast-math
        -Ofast
        -march=native

        -pedantic
        -Wall
        -Wextra
        -Wcast-align
        -Wcast-qual
        -Wctor-dtor-privacy
        -Wformat=2
        -Winit-self
        -Wmissing-declarations
        -Wmissing-include-dirs
        -Woverloaded-virtual
        -Wredundant-decls
        -Wsign-promo
        -Wstrict-overflow=5
        -Wswitch-default
        -Wundef
        -Wno-unused
)

add_subdirectory(examples)
