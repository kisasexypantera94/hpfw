cmake_minimum_required(VERSION 3.15)

project(audio-combiner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-fsanitize=thread -Xpreprocessor")

include_directories(/usr/local/include)

link_directories(/usr/local/lib)
link_libraries(mpg123 tbb fftw3)

add_definitions("-O3")
add_definitions("-pedantic -Wall -Wextra -Wcast-align -Wcast-qual
-Wctor-dtor-privacy -Wformat=2 -Winit-self -Wmissing-declarations
-Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls
-Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused")

add_executable(audio-combiner main.cpp fingerprint/handle.h fingerprint/algo/mpg123_wrapper.h fingerprint/algo/shazam.h fingerprint/algo/fft.h fingerprint/algo/spectro.h helpers.h)
