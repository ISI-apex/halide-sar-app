# Provides macros to check for distributed support in Halide

macro(CHECK_HALIDE_SET_DISTRIBUTED VARIABLE LIBRARIES)
    if (NOT DEFINED "${VARIABLE}" OR "x${${VARIABLE}}" STREQUAL "x${VARIABLE}")
        if(NOT CMAKE_REQUIRED_QUIET)
            message(STATUS "Checking for Halide set_distributed")
        endif()
        # Not currently bothering to check CMAKE_REQUIRED_* vars
        try_compile(${VARIABLE}
                    ${CMAKE_BINARY_DIR}
                    SOURCES ${PROJECT_SOURCE_DIR}/cmake/CheckHalideDistributed/test_set_distributed.cpp
                    LINK_LIBRARIES ${LIBRARIES}
                    OUTPUT_VARIABLE OUTPUT)
        if(${VARIABLE})
            if(NOT CMAKE_REQUIRED_QUIET)
                message(STATUS "Checking for Halide set_distributed - found")
            endif()
            set(${VARIABLE} 1 CACHE INTERNAL "Have Halide set_distributed")
        else()
            if(NOT CMAKE_REQUIRED_QUIET)
                message(STATUS "Checking for Halide set_distributed - not found")
            endif()
            set(${VARIABLE} "" CACHE INTERNAL "Have Halide set_distributed")
            file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
                 "Checking for Halide set_distributed "
                 "failed with the following output:\n"
                 "${OUTPUT}\n\n")
        endif()
    endif()
endmacro()
