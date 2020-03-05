cmake_minimum_required(VERSION 3.15)

project(hpfw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-fsanitize=thread")

add_definitions(
        -DUSE_FFTW
        -DEIGEN_USE_MKL_ALL

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

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(
        SYSTEM

        /usr/local/include
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/include

        thirdparty/Gist/src
        thirdparty/cpp-taskflow-2.2.0
)

link_directories(/usr/local/lib)

add_executable(
        hpfw

        main.cpp
)

target_link_libraries(
        hpfw

        Eigen3::Eigen
        mpg123
        fftw3
        tbb
        essentia

        # MKL RELATED
        pthread
        m
        dl
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/lib/libmkl_intel_lp64.a
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/lib/libmkl_sequential.a
        #        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/lib/libmkl_intel_thread.a
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/lib/libmkl_core.a
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/lib/libmkl_blacs_mpich_lp64.a
        /opt/intel/compilers_and_libraries_2020.0.166/mac/mkl/lib/libmkl_scalapack_ilp64.a
        #        /opt/intel/compilers_and_libraries_2020.0.166/mac/compiler/lib/libiomp5.a
)
