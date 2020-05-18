add_library(pyhpfw SHARED parallel_collector_wrapper.cpp)
target_link_libraries(pyhpfw PRIVATE hpfw)
target_compile_options(
        pyhpfw
        PRIVATE

        -fPIC
)