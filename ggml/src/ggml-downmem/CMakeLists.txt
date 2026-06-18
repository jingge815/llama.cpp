if (NOT GGML_DOWNMEM_ROOT)
    set(GGML_DOWNMEM_ROOT "/home/fjg/src/downmem" CACHE PATH "ggml: downmem source root")
endif()

if (NOT GGML_DOWNMEM_LIB)
    set(GGML_DOWNMEM_LIB "${GGML_DOWNMEM_ROOT}/build-rv/libdmmShared.so" CACHE FILEPATH "ggml: downmem dmm library")
endif()

ggml_add_backend_library(ggml-downmem
                         ggml-downmem.cpp
                         ggml-downmem.h)

target_include_directories(ggml-downmem PRIVATE
    ${GGML_DOWNMEM_ROOT})

target_link_libraries(ggml-downmem PRIVATE
    ${GGML_DOWNMEM_LIB}
    m)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ggml-downmem PRIVATE dl)
    set_target_properties(ggml-downmem PROPERTIES
        BUILD_RPATH "${GGML_DOWNMEM_ROOT}/build-rv;${GGML_DOWNMEM_ROOT}/third-party/localdeps/usr/lib/x86_64-linux-gnu;/home/fjg/src/downmem-toolchains/installed-rv/lib/x86_64-pc-linux-gnu")
endif()
