include_directories(${CMAKE_SOURCE_DIR})

# Generators and Filters (libraries)

add_executable(arange.generator arange.cpp)
target_link_libraries(arange.generator PRIVATE Halide::Generator)
add_halide_library(arange FROM arange.generator)

add_executable(cross.generator cross.cpp)
target_link_libraries(cross.generator PRIVATE Halide::Generator)
add_halide_library(cross FROM cross.generator)

add_executable(interp.generator interp.cpp)
target_link_libraries(interp.generator PRIVATE Halide::Generator)
add_halide_library(interp FROM interp.generator)

add_executable(linspace.generator linspace.cpp)
target_link_libraries(linspace.generator PRIVATE Halide::Generator)
add_halide_library(linspace FROM linspace.generator)

add_executable(norm.generator norm.cpp)
target_link_libraries(norm.generator PRIVATE Halide::Generator)
add_halide_library(norm1d FROM norm.generator)
add_halide_library(norm2d FROM norm.generator)

add_executable(stack.generator stack.cpp)
target_link_libraries(stack.generator PRIVATE Halide::Generator)
add_halide_library(hstack1 FROM stack.generator)
add_halide_library(hstack2 FROM stack.generator)
add_halide_library(vstack1 FROM stack.generator)
add_halide_library(vstack2 FROM stack.generator)

add_executable(repeat1.generator repeat.cpp)
target_link_libraries(repeat1.generator PRIVATE Halide::Generator)
add_halide_library(repeat1 FROM repeat1.generator)

add_executable(taylor.generator taylor.cpp)
target_link_libraries(taylor.generator PRIVATE Halide::Generator)
add_halide_library(taylor FROM taylor.generator)


# TODO: automate these tests
# Non-automated test executables

add_executable(norm_test norm_test.cpp)
target_link_libraries(norm_test PRIVATE Halide::Halide
                                        norm1d
                                        norm2d)

add_executable(taylor_test taylor_test.cpp)
target_link_libraries(taylor_test PRIVATE Halide::Halide
                                          taylor)


# Tests

add_executable(arange_test arange_test.cpp)
target_link_libraries(arange_test PRIVATE Halide::Halide
                                          arange)
add_test(arange arange_test)

add_executable(cross_test cross_test.cpp)
target_link_libraries(cross_test PRIVATE Halide::Halide
                                         cross)
add_test(cross cross_test)

add_executable(interp_test interp_test.cpp)
target_link_libraries(interp_test PRIVATE Halide::Halide
                                          interp)
add_test(interp interp_test)

add_executable(linspace_test linspace_test.cpp)
target_link_libraries(linspace_test PRIVATE Halide::Halide
                                            linspace)
add_test(linspace linspace_test)

add_executable(repeat_test repeat_test.cpp)
target_link_libraries(repeat_test PRIVATE Halide::Halide
                                          repeat1)
add_test(repeat repeat_test)

add_executable(stack_test stack_test.cpp)
target_link_libraries(stack_test PRIVATE Halide::Halide
                                         hstack1
                                         hstack2
                                         vstack1
                                         vstack2)
add_test(stack stack_test)
