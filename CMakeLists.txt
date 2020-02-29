cmake_minimum_required(VERSION 3.15)

project(hpfw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-fsanitize=thread")

add_definitions(
        -DUSE_FFTW
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

include_directories(
        SYSTEM

        /usr/local/include
        thirdparty/Gist/src
        thirdparty/cpp-taskflow-2.2.0
)

link_directories(/usr/local/lib)

link_libraries(
        mpg123
        fftw3
        tbb
)

add_library(
        Gist STATIC

        thirdparty/Gist/src/core/CoreFrequencyDomainFeatures.cpp
        thirdparty/Gist/src/core/CoreTimeDomainFeatures.cpp
        thirdparty/Gist/src/mfcc/MFCC.cpp
        thirdparty/Gist/src/onset-detection-functions/OnsetDetectionFunction.cpp
        thirdparty/Gist/src/fft/WindowFunctions.cpp
        thirdparty/Gist/src/pitch/Yin.cpp
        thirdparty/Gist/src/Gist.cpp
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(
        hpfw

        main.cpp
)

target_link_libraries(hpfw Gist Eigen3::Eigen)
