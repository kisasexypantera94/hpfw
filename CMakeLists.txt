cmake_minimum_required(VERSION 3.15)

project(audio-combiner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-fsanitize=thread")

add_definitions(
        -DUSE_FFTW
        -O3
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
        thirdparty/eigen-3.3.7
)

link_directories(/usr/local/lib)

link_libraries(
        mpg123
        tbb
        fftw3
)

add_library(
        Gist STATIC

        thirdparty/Gist/src/core/CoreFrequencyDomainFeatures.cpp
        thirdparty/Gist/src/core/CoreTimeDomainFeatures.cpp
        thirdparty/Gist/src/mfcc/MFCC.cpp
        thirdparty/Gist/src/onset-detection-functions/OnsetDetectionFunction.cpp
#        thirdparty/Gist/src/fft/AccelerateFFT.cpp
        thirdparty/Gist/src/fft/WindowFunctions.cpp
        thirdparty/Gist/src/pitch/Yin.cpp
        thirdparty/Gist/src/Gist.cpp
)

add_executable(
        audio-combiner

        main.cpp
        fingerprint/algo/tsai.h fingerprint/algo/combiner.h)

target_link_libraries(audio-combiner Gist)
