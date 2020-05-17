#add_subdirectory(pybind11-2.5.0)
#pybind11_add_module(pyhpfw hello.cpp)
#target_link_libraries(pyhpfw PRIVATE hpfw)

add_library(pyhpfw SHARED parallel_collector_wrapper.cpp)
target_link_libraries(pyhpfw PRIVATE hpfw)
target_compile_options(
        pyhpfw
        PRIVATE

        -fPIC
)