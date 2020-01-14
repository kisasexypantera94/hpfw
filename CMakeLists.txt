cmake_minimum_required(VERSION 3.5)

project(audio-combiner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-fsanitize=thread -O3 -Xpreprocessor -fopenmp")

include_directories(/usr/local/include /usr/local/include/eigen3)

link_directories(/usr/local/lib)
link_libraries(mpg123 omp)

add_definitions("-O3 -Xpreprocessor -fopenmp")
add_definitions("-pedantic -Wall -Wextra -Wcast-align -Wcast-qual
-Wctor-dtor-privacy -Wformat=2 -Winit-self -Wmissing-declarations
-Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls
-Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused")

add_executable(audio-combiner main.cpp fingerprint/handle.h fingerprint/mpg123_wrapper.h fingerprint/algo.h)
